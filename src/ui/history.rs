//! Undo and redo for an editor's value (#60, Overlands #1333).
//!
//! # Why snapshots
//!
//! The editors are immediate-mode: a widget is handed `&mut T` and mutates it
//! in place, and the frame learns an edit happened only afterwards, from
//! [`EditorResponse::rebake`](crate::ui::EditorResponse). There is no command
//! object to invert, so history here is a ring of whole values. An
//! [`crate::AudioPatch`] is a few hundred bytes and a
//! [`crate::SequenceRecipe`] a few kilobytes, and [`CAP`] of them is cheap
//! against the cost of describing every edit twice.
//!
//! # What a step is
//!
//! A step is a *committed* edit — `rebake`, the flag a drag raises once when
//! it ends rather than on every frame it moves. So dragging a slider from 100
//! to 800 Hz is one entry, not seven hundred. [`EditHistory::baseline`] holds
//! the value as of the last commit, which is what an undo returns to; it
//! survives across the frames of a drag, which is exactly why it cannot be
//! re-taken each frame.
//!
//! # One value, one history
//!
//! [`EditHistory::disable`] turns a history off for a value another history
//! already owns. The sequence editor's patch canvas is the case: an
//! instrument's patch lives *inside* the recipe, so an edit to it is an edit
//! to the recipe. Were both on, one Ctrl+Z would walk two stacks at once and
//! the recipe's would go stale.

use std::collections::VecDeque;

/// How many committed edits a history keeps. Past this the oldest is
/// dropped: an editing session is long and undo is used near its head.
pub(crate) const CAP: usize = 64;

/// Undo/redo over snapshots of one value. See the [module docs](self).
#[derive(Clone, Debug)]
pub(crate) struct EditHistory<T> {
    /// The value as of the last commit — what the next undo returns to.
    /// `None` until the first frame has seen the value.
    baseline: Option<T>,
    /// Committed values, oldest first; the newest is one undo away.
    past: VecDeque<T>,
    /// Values undone away from, newest last.
    future: Vec<T>,
    /// Off while another history owns this value.
    enabled: bool,
}

impl<T> Default for EditHistory<T> {
    fn default() -> Self {
        Self {
            baseline: None,
            past: VecDeque::new(),
            future: Vec::new(),
            enabled: true,
        }
    }
}

impl<T: Clone + PartialEq> EditHistory<T> {
    /// Hand this value's history to someone else: nothing is recorded and
    /// nothing can be undone here.
    ///
    /// Idempotent, and it drops whatever was already recorded — a value that
    /// has changed owner has a history that is no longer about it.
    pub(crate) fn disable(&mut self) {
        if self.enabled {
            *self = Self {
                enabled: false,
                ..Default::default()
            };
        }
    }

    /// Whether this history is the one that owns its value. Read by the
    /// test that holds the one-history rule; the editors themselves only
    /// ever set it.
    #[cfg(test)]
    pub(crate) fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Note the value at the top of a frame, before any widget sees it.
    ///
    /// Only the *first* such call after a commit takes a snapshot: the
    /// baseline has to be the value before the edit began, and a drag's edit
    /// begins several frames before it commits.
    pub(crate) fn begin(&mut self, current: &T) {
        if self.enabled && self.baseline.is_none() {
            self.baseline = Some(current.clone());
        }
    }

    /// Record a committed edit, if the value really moved.
    ///
    /// Call once per frame in which the editor reported `rebake`. A commit
    /// that changed nothing — a slider dropped where it was picked up — is
    /// not a step, and forgetting to check would fill the ring with entries
    /// an undo could not be seen to do anything to.
    pub(crate) fn commit(&mut self, current: &T) {
        if !self.enabled {
            return;
        }
        match self.baseline.take() {
            Some(base) if base != *current => {
                self.past.push_back(base);
                if self.past.len() > CAP {
                    self.past.pop_front();
                }
                self.future.clear();
                self.baseline = Some(current.clone());
            }
            other => self.baseline = other.or_else(|| Some(current.clone())),
        }
    }

    pub(crate) fn can_undo(&self) -> bool {
        self.enabled && !self.past.is_empty()
    }

    pub(crate) fn can_redo(&self) -> bool {
        self.enabled && !self.future.is_empty()
    }

    /// Step `current` back to the value before the last committed edit.
    /// `false` if there was nothing to undo.
    pub(crate) fn undo(&mut self, current: &mut T) -> bool {
        let Some(previous) = self.past.pop_back() else {
            return false;
        };
        self.future.push(current.clone());
        *current = previous;
        // An undo is not an edit to record, but it *is* the new baseline:
        // the next commit's step starts from here.
        self.baseline = Some(current.clone());
        true
    }

    /// Step `current` forward again after an undo. `false` if there was
    /// nothing to redo.
    pub(crate) fn redo(&mut self, current: &mut T) -> bool {
        let Some(next) = self.future.pop() else {
            return false;
        };
        self.past.push_back(current.clone());
        *current = next;
        self.baseline = Some(current.clone());
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive one committed edit: the frame sees the value, a widget changes
    /// it, the frame reports `rebake`.
    fn edit(h: &mut EditHistory<i32>, value: &mut i32, to: i32) {
        h.begin(value);
        *value = to;
        h.commit(value);
    }

    #[test]
    fn undo_and_redo_walk_the_committed_edits() {
        let mut h = EditHistory::default();
        let mut v = 0;
        for step in [1, 2, 3] {
            edit(&mut h, &mut v, step);
        }
        assert!(h.can_undo() && !h.can_redo());

        assert!(h.undo(&mut v));
        assert_eq!(v, 2);
        assert!(h.undo(&mut v));
        assert_eq!(v, 1);
        assert!(h.undo(&mut v));
        assert_eq!(v, 0, "back to the value before the first edit");
        assert!(!h.can_undo());
        assert!(!h.undo(&mut v), "nothing left to undo");

        for step in [1, 2, 3] {
            assert!(h.redo(&mut v));
            assert_eq!(v, step);
        }
        assert!(!h.can_redo());
        assert!(!h.redo(&mut v));
    }

    /// The frames of a drag are one step: `begin` takes the baseline once,
    /// and only the frame that commits records anything.
    #[test]
    fn a_drag_across_many_frames_is_one_step() {
        let mut h = EditHistory::default();
        let mut v = 100;
        for frame in 1..=7 {
            h.begin(&v);
            v = 100 + frame * 100;
            // Only the last frame of the drag commits.
            if frame == 7 {
                h.commit(&v);
            }
        }
        assert_eq!(v, 800);
        assert!(h.undo(&mut v));
        assert_eq!(v, 100, "one undo goes back past the whole drag");
        assert!(!h.can_undo());
    }

    /// A commit that moved nothing is not a step. A slider picked up and put
    /// down where it was would otherwise leave an undo that does nothing
    /// visible, which reads as a broken undo.
    #[test]
    fn a_commit_that_changed_nothing_is_not_a_step() {
        let mut h = EditHistory::default();
        let mut v = 5;
        h.begin(&v);
        h.commit(&v);
        assert!(!h.can_undo());

        edit(&mut h, &mut v, 6);
        h.begin(&v);
        h.commit(&v);
        assert!(h.can_undo());
        assert!(h.undo(&mut v));
        assert_eq!(v, 5);
        assert!(!h.can_undo(), "only the edit that moved it was recorded");
    }

    /// Editing after an undo drops the redo stack: the future that was
    /// undone away from is not reachable from here any more.
    #[test]
    fn an_edit_after_an_undo_drops_the_redo_stack() {
        let mut h = EditHistory::default();
        let mut v = 0;
        edit(&mut h, &mut v, 1);
        edit(&mut h, &mut v, 2);
        assert!(h.undo(&mut v));
        assert_eq!(v, 1);
        assert!(h.can_redo());

        edit(&mut h, &mut v, 9);
        assert!(!h.can_redo(), "9 replaced the branch that held 2");
        assert!(h.undo(&mut v));
        assert_eq!(v, 1);
    }

    /// The ring keeps the newest [`CAP`] steps and drops the oldest.
    #[test]
    fn the_ring_is_capped_and_drops_the_oldest_step() {
        let mut h = EditHistory::default();
        let mut v = 0;
        for step in 1..=(CAP as i32 + 10) {
            edit(&mut h, &mut v, step);
        }
        let mut undone = 0;
        while h.undo(&mut v) {
            undone += 1;
        }
        assert_eq!(undone, CAP, "exactly the cap is kept");
        assert_eq!(v, 10, "the ten oldest steps fell off the front");
    }

    /// A disabled history records nothing and undoes nothing — the value
    /// belongs to another history.
    #[test]
    fn a_disabled_history_records_nothing() {
        let mut h = EditHistory::default();
        let mut v = 0;
        edit(&mut h, &mut v, 1);
        assert!(h.can_undo());

        h.disable();
        assert!(!h.is_enabled());
        assert!(
            !h.can_undo(),
            "what it had recorded went with the ownership"
        );
        edit(&mut h, &mut v, 2);
        assert!(!h.can_undo());
        assert!(!h.undo(&mut v));
        assert_eq!(v, 2);

        // Idempotent: disabling twice is not a way to lose a second history.
        h.disable();
        assert!(!h.is_enabled());
    }
}
