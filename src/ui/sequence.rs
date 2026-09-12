//! Timeline editor for a [`SequenceRecipe`] — the "arrange view".
//!
//! Sits above the patch canvas the way the schema does: a recipe owns named
//! [`Instrument`]s (each an [`AudioPatch`]) and parallel [`Track`]s of
//! [`Event`]s scheduled in beats.  This module edits the *sequence* layer:
//!
//! - **Transport** — BPM, sample rate, total duration, and the optional loop
//!   window (start + crossfade), which is drawn as markers on the timeline.
//! - **Instruments** — add / remove / rename; pick one as *active* to edit its
//!   embedded patch in the Phase-2 canvas (see [`active_instrument_canvas`]).
//!   Events name their instrument by id, so a rename carries the instrument's
//!   events on every track, and its canvas layout, to the new id. It applies
//!   on Enter or when the field loses focus, never per keystroke, and only
//!   for a name that is non-empty after trimming, unique, and within
//!   this editor's [`EditorLimits`] byte cap — which defaults to
//!   symbios-audio's own [`crate::Envelope`], the bound the record
//!   sanitiser holds a recipe to. Until then the field shows the reason in
//!   the error colour, and Esc puts the name back.
//! - **Timeline** — one lane per track, named in its gutter after the
//!   instrument most of it plays. A note is a block whose x is `time_beats`
//!   and width is `gate_beats`, with a translucent tail for `release_beats`,
//!   painted in its *instrument's* colour — a hue per id at the saturation
//!   and luminance of the style's `note_fill` — faded by its volume, and
//!   labelled with as much of its instrument and pitch as the block has room
//!   for (#62, Overlands #1335 C3). It opens zoomed so the whole sequence is
//!   in view, with the loop markers labelled on the ruler ("loop start",
//!   "blend", "end"), and there is a Fit to go back to that zoom (C4). An
//!   event whose instrument id names no instrument bakes to silence; its
//!   block is drawn hatched in the error colour and labelled `missing: <id>`.
//! - **Editing notes** — click a block to pick it, shift-click or drag a box
//!   over the lanes to pick several. A drag moves everything picked and a
//!   drag on a block's right edge resizes that one's gate; Alt makes the
//!   drag carry copies and leave the originals. Ctrl+D duplicates, the arrows
//!   nudge by the grid (Shift for an eighth of it), Delete removes, and
//!   Escape lets go. The grid itself is the toolbar's [`Snap`] picker, and
//!   a new note comes from a double-click or a right-click on a lane and
//!   takes *that lane's* instrument (C6).
//! - **Inspector** — full numeric editing of the note the last pick landed
//!   on, plus delete. For a note with no instrument it says so and offers to
//!   reassign every note of that id to an existing instrument.
//!
//! Every label says what a field does rather than what the schema calls it,
//! with the time at the recipe's BPM beside the beats (C5), and what is not
//! wanted every minute — the genetics controls, the JSON box — is under the
//! timeline's More menu rather than above everything it edits (C9).
//!
//! The timeline publishes the geometry it painted, [`NoteGeom`] and
//! [`SequenceEditorState::timeline_rect`]: a host hanging an overlay on a
//! note, a scripted harness pressing one, and the marquee itself all read
//! the rects the paint used rather than working them out again from the zoom
//! and the scroll offset (crate #67).
//!
//! Every colour the timeline paints — its ground, grid and ruler, the lanes,
//! the loop markers, note blocks and their names, the error colour of a
//! missing note or a refused name — comes from the [`EditorStyle`] in effect
//! ([`crate::ui::style`]), so it follows the host's theme (#58).
//!
//! Like the rest of [`crate::ui`] this is pure egui returning an
//! [`EditorResponse`]; the host drives the bake-and-play monitor (Phase 3) off
//! `rebake`, baking the whole recipe with [`crate::mixdown::bake_sequence`].
//!
//! Editor-only view state (active instrument, per-instrument canvas layout,
//! selection, zoom, names being typed) lives in [`SequenceEditorState`],
//! never in the serialized recipe.

use std::collections::{HashMap, HashSet};

use bevy_egui::egui::{self, Align2, Color32, Id, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};

use symbios_genetics::Genotype;

use crate::patch::AudioPatch;
use crate::sequence::{Event, Instrument, PitchMode, SequenceRecipe, Track};

use super::evolve::fresh_rng;
use super::history::EditHistory;
use super::io::json_io;
use super::limits::{Cap, CapState, EditorLimits, cap_readout};
use super::style::{EditorStyle, editor_style, relative_luminance};
use super::{
    EditorResponse, JsonIoState, PatchEditorState, audio_patch_canvas, drag_value_debounced,
    ranged_slider,
};

/// The header above the lanes: the beat numbers on its first line, the loop
/// markers' labels on its second. Two lines because a marker's label over
/// the lanes sits on the notes it is about (#62, Overlands #1335 C4).
const RULER_H: f32 = 34.0;
/// Where in the ruler each of its two lines starts.
const RULER_BEATS_Y: f32 = 1.0;
const RULER_MARKS_Y: f32 = 17.0;
const LANE_H: f32 = 34.0;
/// Left margin inside the timeline: the lane's remove cross and its name.
/// Wide enough for an instrument name, which is what a lane is called.
const GUTTER: f32 = 104.0;
/// Points of timeline right of the last beat, so the end marker's label has
/// somewhere to go.
const TAIL_PAD: f32 = 16.0;
/// The zoom range the slider offers and a fit is held to, in points a beat.
const MIN_PPB: f32 = 4.0;
const MAX_PPB: f32 = 160.0;
/// Smallest gate a block can be resized to (beats).
const MIN_GATE: f32 = 0.1;
/// Beats in a bar, for the snap picker's coarsest division. The schema
/// carries no time signature, so this is common time and nothing reads it
/// but the picker.
const BEATS_PER_BAR: f32 = 4.0;
/// How opaque a note block is at volume 0. A silent note is faint, not
/// invisible, and its name still has to read on it — see
/// `tests::a_notes_name_reads_on_its_block_in_dark_and_light`.
const QUIET_ALPHA: f32 = 0.55;
/// Space either side of a marker's label, and inside a note block.
const LABEL_PAD: f32 = 4.0;

/// Whether an in-progress block drag is moving the event or resizing its gate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum DragMode {
    #[default]
    Move,
    /// Alt was held when the drag began: the notes being dragged are copies
    /// the drag made, and the originals stay where they were.
    Copy,
    Resize,
}

/// Where one note was drawn on the timeline's last frame, in screen points.
///
/// The timeline hands its geometry out rather than letting a reader work it
/// out again from the zoom and the scroll offset: a host hanging an overlay
/// on a note, a scripted harness pressing one, and the editor's own marquee
/// all read the same rects the paint used (crate #67).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoteGeom {
    /// Which track it is on.
    pub track: usize,
    /// Its index within that track's events.
    pub event: usize,
    /// The block: `time_beats` to `time_beats + gate_beats`.
    pub body: Rect,
    /// The block and its release tail together — what a pointer over the
    /// note can be said to be over.
    pub extent: Rect,
}

/// Editor-side view state for the sequence editor — kept out of the
/// serialized [`SequenceRecipe`].  Construct with [`Default`].
#[derive(Clone, Debug)]
pub struct SequenceEditorState {
    /// Index of the instrument whose patch is open in the canvas, if any.
    active_instrument: Option<usize>,
    /// Per-instrument patch-canvas layout, keyed by instrument id so each
    /// instrument keeps its node positions when you switch away and back.
    /// A rename moves the entry to the new id.
    canvas_states: HashMap<String, PatchEditorState>,
    /// The `(track, event)` the inspector edits: the last note picked, and
    /// always one of [`Self::selection`] while there is a selection at all.
    selected_event: Option<(usize, usize)>,
    /// Every picked note. A move, a duplicate and a nudge act on all of
    /// them; the inspector edits [`Self::selected_event`].
    selection: HashSet<(usize, usize)>,
    /// A marquee being dragged out on a lane's background: where it started
    /// and where the pointer is, in screen points.
    marquee: Option<(Pos2, Pos2)>,
    /// The track and beat the last right-click on a lane landed at.
    ///
    /// Remembered rather than read from the pointer while the menu is up:
    /// the pointer has to move *into* the menu to choose anything, and by
    /// then the lane is no longer under it. Asking the lane's response
    /// where the pointer is left "Add a note here" greyed out the moment it
    /// could be reached — which a picture caught and the test that drove
    /// the menu by name did not.
    menu_at: Option<(usize, f32)>,
    /// Timeline zoom in points a beat, or `None` to fit the width it is
    /// given — which is what it opens at, and what the Fit button goes back
    /// to (#62, Overlands #1335 C4).
    zoom: Option<f32>,
    /// The grid drags, nudges and new notes land on.
    snap: Snap,
    /// Move-vs-resize for the active block drag.
    drag_mode: DragMode,
    /// The note whose *widget* the active drag belongs to.
    ///
    /// Not the same as [`Self::selected_event`] once an Alt-drag has
    /// duplicated: egui ties a drag to the id of the widget the press
    /// landed on, and the copies have ids of their own that nothing has
    /// pressed. So the drag keeps reading the original's response and
    /// carries the copies by its delta.
    drag_anchor: Option<(usize, usize)>,
    /// Mutation rate for the More menu's Mutate recipe.
    mutate_rate: f32,
    /// Whether the JSON box is open, from the timeline's More menu.
    show_json: bool,
    /// Buffer + last error for the JSON import/export section.
    json: JsonIoState,
    /// The timeline's own rect as of the last frame drawn, in screen
    /// points, and every note on it.
    ///
    /// Recorded rather than re-derived: the timeline is the only thing that
    /// knows its zoom, its gutter and how far its `ScrollArea` is scrolled,
    /// and a reader that works those out again is a second answer waiting
    /// to disagree (crate #67).
    timeline_rect: Rect,
    notes: Vec<NoteGeom>,
    /// Names being typed into instrument rows, keyed by row. The recipe is
    /// renamed only when one is committed; see [`NameEdit`].
    name_edits: HashMap<usize, NameEdit>,
    /// Undo/redo over the whole recipe (#60). An instrument's patch is part
    /// of the recipe, so this is the *only* history in the sequence editor:
    /// the embedded canvas's own is turned off in
    /// [`active_instrument_canvas`].
    history: EditHistory<SequenceRecipe>,
    /// The sequence editor took this frame's keyboard.
    owns_keys: bool,
    /// It acted on an Escape this frame.
    took_escape: bool,
    /// The caps every Add is held to, and the byte limit the name field
    /// refuses past. [`EditorLimits::default`] is the record boundary's own
    /// envelope, so an editor nobody configured refuses exactly what the
    /// sanitiser would have deleted (#63, Overlands #1336 C8).
    limits: EditorLimits,
    /// A removal that has been asked for and not yet answered.
    ///
    /// State that outlives the frame it was asked in, so it lives here
    /// beside [`Self::menu_at`] and [`Self::marquee`] rather than in a
    /// widget: the click that asks happens deep inside the lane loop or an
    /// instrument's row, and the question is drawn at the top of that
    /// section on the *next* frame — a position that is the same in both of
    /// egui's passes (#1332) and nowhere near the cross that was pressed.
    pending_removal: Option<PendingRemoval>,
    /// Where every control that *asks* for a removal was last drawn, in
    /// screen points: each instrument row's cross, then each track
    /// gutter's, in the order they are drawn.
    ///
    /// Published rather than re-derived, the way [`Self::notes`] is: the
    /// gutter's cross is painted geometry that only the lane loop knows the
    /// position of, and a test or a harness that worked it out again would
    /// be a second answer waiting to disagree (crate #67).
    removal_asks: Vec<Rect>,
    /// Where the confirmation was last drawn, or [`Rect::NOTHING`] if none
    /// is up — so "the buttons are nowhere near the cross that asked" is a
    /// thing a test can measure rather than a thing a screenshot suggests.
    confirmation_rect: Rect,
}

/// A removal waiting to be confirmed.
///
/// Only removals that lose something ask: an empty track and an instrument
/// no note names go at once, because a confirmation for a no-op is the
/// dialog everyone learns to dismiss without reading.
#[derive(Clone, Debug, PartialEq, Eq)]
enum PendingRemoval {
    /// A track with notes on it.
    Track {
        /// Which track, by index.
        track: usize,
        /// How many notes go with it.
        notes: usize,
    },
    /// An instrument some notes name.
    Instrument {
        /// Which instrument, by row.
        row: usize,
        /// Its id, so a row that has moved or been renamed under the
        /// question can be recognised and the question dropped.
        id: String,
        /// How many notes across the recipe name it.
        notes: usize,
    },
}

impl Default for SequenceEditorState {
    fn default() -> Self {
        Self {
            active_instrument: None,
            canvas_states: HashMap::new(),
            selected_event: None,
            selection: HashSet::new(),
            marquee: None,
            menu_at: None,
            zoom: None,
            snap: Snap::default(),
            drag_mode: DragMode::Move,
            drag_anchor: None,
            mutate_rate: 0.3,
            show_json: false,
            json: JsonIoState::default(),
            timeline_rect: Rect::NOTHING,
            notes: Vec::new(),
            name_edits: HashMap::new(),
            history: EditHistory::default(),
            owns_keys: false,
            took_escape: false,
            limits: EditorLimits::default(),
            pending_removal: None,
            removal_asks: Vec::new(),
            confirmation_rect: Rect::NOTHING,
        }
    }
}

/// A name typed into an instrument's row, held here until it is committed.
///
/// Notes find their instrument by id, so a rename has to move them too, and
/// it must not do that per keystroke: typing "inst1" -> "inst" -> "inst2"
/// while an "inst2" exists would pour inst1's notes into inst2's, and no
/// later keystroke could tell them apart again (#55). So the field edits
/// this buffer, and Enter or leaving the field commits it once, if the name
/// is valid ([`check_name`]). Esc drops it.
#[derive(Clone, Debug)]
struct NameEdit {
    /// The row's id when typing began. A row that no longer has it (a
    /// removal moved the rows up, or the host swapped the recipe) drops the
    /// edit rather than apply it to another instrument.
    from: String,
    /// What the field shows.
    text: String,
    /// The field had the focus the last time it was drawn, so its next
    /// frame without focus is a commit. A refused commit clears this, so a
    /// refused name waits for the user and never applies itself later
    /// because some other instrument moved out of its way.
    typing: bool,
}

/// Why a typed name cannot be an instrument's id.
#[derive(Clone, Debug, PartialEq, Eq)]
enum NameRefusal {
    /// Nothing but whitespace.
    Empty,
    /// Longer than the byte limit.
    TooLong { bytes: usize, limit: usize },
    /// Another instrument already answers to it.
    Taken(String),
}

impl std::fmt::Display for NameRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "A name can't be empty"),
            Self::TooLong { bytes, limit } => {
                write!(f, "Too long: {bytes} bytes, the limit is {limit}")
            }
            Self::Taken(name) => write!(f, "Another instrument is called '{name}'"),
        }
    }
}

/// The id `text` would give instrument `row`, trimmed, or why it can't.
fn check_name<'a>(
    recipe: &SequenceRecipe,
    row: usize,
    text: &'a str,
    limit: usize,
) -> Result<&'a str, NameRefusal> {
    let name = text.trim();
    if name.is_empty() {
        Err(NameRefusal::Empty)
    } else if name.len() > limit {
        Err(NameRefusal::TooLong {
            bytes: name.len(),
            limit,
        })
    } else if recipe
        .instruments
        .iter()
        .enumerate()
        .any(|(i, inst)| i != row && inst.id == name)
    {
        Err(NameRefusal::Taken(name.to_owned()))
    } else {
        Ok(name)
    }
}

/// Point every note of `from`, on every track, at `to`.
fn reassign_notes(recipe: &mut SequenceRecipe, from: &str, to: &str) {
    for event in recipe.tracks.iter_mut().flat_map(|t| t.events.iter_mut()) {
        if event.instrument_id == from {
            event.instrument_id = to.to_owned();
        }
    }
}

/// Rename instrument `row` to `new`, which [`check_name`] accepted, in one
/// step: its notes on every track and its canvas layout go with it, so the
/// rename is a pure relabel and the recipe bakes the same.
fn rename_instrument(
    recipe: &mut SequenceRecipe,
    canvas_states: &mut HashMap<String, PatchEditorState>,
    row: usize,
    new: &str,
) {
    let old = std::mem::replace(&mut recipe.instruments[row].id, new.to_owned());
    // A recipe from outside the editor can hold two instruments with one id.
    // The other still answers to `old`, so the notes and the layout stay
    // with it: which of the two a note meant cannot be known.
    if recipe.instruments.iter().any(|inst| inst.id == old) {
        return;
    }
    reassign_notes(recipe, &old, new);
    match canvas_states.remove(&old) {
        Some(layout) => {
            canvas_states.insert(new.to_owned(), layout);
        }
        // Never opened, so it has no layout; one already under `new` was
        // left by a removed instrument and would scatter this one's nodes.
        None => {
            canvas_states.remove(new);
        }
    }
}

/// How the timeline and the inspector name a note whose instrument is gone.
fn missing_label(instrument_id: &str) -> String {
    if instrument_id.is_empty() {
        "missing: (none)".to_owned()
    } else {
        format!("missing: {instrument_id}")
    }
}

/// What is wrong with a note whose instrument is gone, as a sentence.
fn no_instrument_text(instrument_id: &str) -> String {
    if instrument_id.is_empty() {
        "This note names no instrument".to_owned()
    } else {
        format!("No instrument called '{instrument_id}'")
    }
}

/// A block for a note whose instrument is gone: the error colour, hatched
/// and outlined, so it reads as broken without relying on the hue alone.
/// Every colour is `error` (the host's `error_fg_color`) at some strength.
fn paint_missing_block(painter: &egui::Painter, body: Rect, selected: bool, error: Color32) {
    painter.rect_filled(
        body,
        3.0,
        error.gamma_multiply(if selected { 0.35 } else { 0.2 }),
    );
    let hatch = painter.with_clip_rect(body.intersect(painter.clip_rect()));
    let stroke = Stroke::new(1.0, error.gamma_multiply(0.45));
    let mut x = body.left() - body.height();
    while x < body.right() {
        hatch.line_segment(
            [
                Pos2::new(x, body.bottom()),
                Pos2::new(x + body.height(), body.top()),
            ],
            stroke,
        );
        x += 6.0;
    }
    painter.rect_stroke(
        body,
        3.0,
        Stroke::new(if selected { 2.0 } else { 1.0 }, error),
        StrokeKind::Inside,
    );
}

/// A note's block: its instrument's `tint`, faded by its volume, outlined
/// in `note_selected` when selected.
///
/// The tint does not change with the selection, so a note's name reads the
/// same either way and the selection is a shape as well as a colour. The
/// volume is the block's opacity, floored at [`QUIET_ALPHA`] so a quiet
/// note is faint rather than gone and its name still reads on it.
fn paint_note_block(
    painter: &egui::Painter,
    body: Rect,
    selected: bool,
    tint: Color32,
    volume: f32,
    style: &EditorStyle,
) {
    painter.rect_filled(body, 3.0, tint.gamma_multiply(volume_alpha(volume)));
    let edge = if selected {
        Stroke::new(2.0, style.note_selected)
    } else {
        // The ground's colour, so two notes that touch stay two.
        Stroke::new(1.0, style.timeline_ground)
    };
    painter.rect_stroke(body, 3.0, edge, StrokeKind::Inside);
}

impl SequenceEditorState {
    /// Index of the instrument currently open for patch editing, if any.
    pub fn active_instrument(&self) -> Option<usize> {
        self.active_instrument
    }

    /// Open instrument `index` in [`active_instrument_canvas`], or close the
    /// canvas with `None`. It does the same as clicking the instrument's
    /// Edit toggle, so a host can open an editor on a chosen instrument. An index
    /// past the end of the recipe's instruments is cleared on the next draw.
    pub fn set_active_instrument(&mut self, index: Option<usize>) {
        self.active_instrument = index;
    }

    /// The `(track, event)` open in the event inspector, if any.
    pub fn selected_event(&self) -> Option<(usize, usize)> {
        self.selected_event
    }

    /// Open event `event` of track `track` in the event inspector, or clear
    /// the selection with `None`. It does the same as clicking the event's
    /// block. A pair that names no event is cleared on the next draw.
    pub fn set_selected_event(&mut self, selected: Option<(usize, usize)>) {
        match selected {
            Some(at) => self.select_only(at),
            None => self.clear_selection(),
        }
    }

    /// Every `(track, event)` picked, not only the one the inspector edits.
    ///
    /// A move, a duplicate and an arrow nudge act on all of these;
    /// [`Self::selected_event`] is the last one picked, which is the one
    /// the inspector shows.
    pub fn selected_events(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.selection.iter().copied()
    }

    /// The grid drags, nudges and new notes land on.
    pub fn snap(&self) -> Snap {
        self.snap
    }

    /// Put the timeline on `snap`'s grid, as its picker does.
    pub fn set_snap(&mut self, snap: Snap) {
        self.snap = snap;
    }

    /// The timeline's rect as of the last frame drawn, in screen points —
    /// its ruler, its gutter and its lanes, as wide as the zoom made it.
    ///
    /// [`Rect::NOTHING`] before the editor has been drawn once. Re-measured
    /// every frame, so a rect held across frames should be asked for again.
    pub fn timeline_rect(&self) -> Rect {
        self.timeline_rect
    }

    /// Every note the timeline drew on its last frame, in screen points.
    /// See [`NoteGeom`].
    pub fn notes(&self) -> &[NoteGeom] {
        &self.notes
    }

    /// Pick `at` and nothing else, as a plain click on a note does.
    fn select_only(&mut self, at: (usize, usize)) {
        self.selection.clear();
        self.selection.insert(at);
        self.selected_event = Some(at);
    }

    /// Add `at` to the selection, or take it out if it was already in, as a
    /// shift-click does.
    fn select_also(&mut self, at: (usize, usize)) {
        if self.selection.remove(&at) {
            if self.selected_event == Some(at) {
                self.selected_event = self.selection.iter().copied().next();
            }
        } else {
            self.selection.insert(at);
            self.selected_event = Some(at);
        }
    }

    fn clear_selection(&mut self) {
        self.selection.clear();
        self.selected_event = None;
    }

    /// Drop selected notes that `recipe` no longer has, and keep
    /// [`Self::selected_event`] inside what is left.
    ///
    /// A removal moves the indices after it down, so a selection kept
    /// across one would otherwise come to mean different notes.
    fn forget_missing_notes(&mut self, recipe: &SequenceRecipe) {
        self.selection.retain(|(track, event)| {
            recipe
                .tracks
                .get(*track)
                .is_some_and(|t| *event < t.events.len())
        });
        if self
            .selected_event
            .is_none_or(|at| !self.selection.contains(&at))
        {
            self.selected_event = self.selection.iter().copied().next();
        }
    }
}

/// The grid a drag, a nudge and a new note land on.
///
/// One enum with one [`Snap::ALL`] is the whole roster: the picker walks it,
/// the arrow keys take their step from it, and the tests cover it by walking
/// the same array, so a division cannot be offered by the picker and unknown
/// to the maths (the #50 lesson, applied to the one thing here that is not
/// generated upstream).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Snap {
    /// No grid: a note lands where it was dropped.
    Off,
    /// An eighth of a beat.
    Eighth,
    /// The quarter beat every drag snapped to before there was a choice.
    #[default]
    Quarter,
    /// One whole beat.
    Beat,
    /// One bar: four beats. The schema carries no time signature, so
    /// this is common time.
    Bar,
}

impl Snap {
    /// Every division, coarsest last: what the picker offers and what the
    /// tests walk.
    pub const ALL: [Self; 5] = [
        Self::Off,
        Self::Eighth,
        Self::Quarter,
        Self::Beat,
        Self::Bar,
    ];

    /// The grid in beats, or `None` when snapping is off.
    pub fn beats(self) -> Option<f32> {
        match self {
            Self::Off => None,
            Self::Eighth => Some(0.125),
            Self::Quarter => Some(0.25),
            Self::Beat => Some(1.0),
            Self::Bar => Some(BEATS_PER_BAR),
        }
    }

    /// How the picker names it.
    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Eighth => "1/8",
            Self::Quarter => "1/4",
            Self::Beat => "beat",
            Self::Bar => "bar",
        }
    }

    /// How far one arrow press moves a note: the grid, or a quarter beat
    /// with the grid off, so an arrow always moves something.
    pub fn step(self) -> f32 {
        self.beats().unwrap_or(0.25)
    }
}

/// Snap a beat value to `snap`'s grid, clamped to `>= 0`.
///
/// Only the low end is clamped: a note before beat 0 cannot be baked, but
/// one past the end is fine — the timeline grows to hold it, and the
/// duration is the owner's to raise.
fn snap_beat(beats: f32, snap: Snap) -> f32 {
    match snap.beats() {
        Some(grid) => ((beats / grid).round() * grid).max(0.0),
        None => beats.max(0.0),
    }
}

/// The zoom that puts `total_beats` inside `width` points of panel, held to
/// the range the slider offers.
///
/// The editor opened at 48 points a beat whatever it was given, so the
/// seeded 34-beat recipe was 1 632 points of content in a 900-point slot
/// and its loop end, its crossfade and the second half of every lane were
/// off screen behind a scrollbar that only appears under the pointer (#62,
/// Overlands #1335 C4).
fn fit_px_per_beat(width: f32, total_beats: f32) -> f32 {
    let lanes = width - GUTTER - TAIL_PAD;
    (lanes / total_beats.max(1.0)).clamp(MIN_PPB, MAX_PPB)
}

/// How many hues the timeline hands out.
///
/// Twelve is thirty degrees apart, which is about as close as two colours of
/// one luminance can be and still be told apart on a note block, and more
/// instruments than a recipe usually has. Past twelve they start to share.
const HUE_BUCKETS: usize = 12;

/// FNV-1a over `id`'s bytes: the same number every frame, every run and
/// whatever order the instruments are in.
fn instrument_hash(id: &str) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for byte in id.as_bytes() {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

/// The hue `id` asks for, as a bucket index.
fn wanted_bucket(id: &str) -> usize {
    instrument_hash(id) as usize % HUE_BUCKETS
}

/// How far apart two buckets are, going round the wheel whichever way is
/// shorter.
fn bucket_distance(a: usize, b: usize) -> usize {
    let d = a.abs_diff(b);
    d.min(HUE_BUCKETS - d)
}

/// Every instrument's tint, keyed by id.
///
/// An id's own hash picks the hue it asks for, so an instrument keeps its
/// colour as tracks are re-ordered and as other instruments come and go. A
/// hash is not a spread, though, and that is not a detail: in the recipe
/// the `host_window` example seeds, `gust` and `pluck` hash to within four
/// degrees of each other, and their notes came out one colour — which is
/// the finding this is here to fix. So an id whose hue is already taken is
/// given the free one furthest from every hue in use, and which of two ids
/// keeps a contested hue is settled by sorting them: never by the order the
/// tracks, or the instruments, happen to be in.
fn instrument_tints<'a>(
    ids: impl IntoIterator<Item = &'a str>,
    style: &EditorStyle,
) -> HashMap<String, Color32> {
    let mut sorted: Vec<&str> = ids.into_iter().collect();
    sorted.sort_unstable();
    sorted.dedup();
    let mut taken: Vec<usize> = Vec::with_capacity(sorted.len());
    let mut out = HashMap::with_capacity(sorted.len());
    for id in sorted {
        let wanted = wanted_bucket(id);
        let bucket = if taken.contains(&wanted) {
            // The free hue with the most room around it. None free means
            // more instruments than hues, and it takes the one it asked
            // for and shares.
            (0..HUE_BUCKETS)
                .filter(|b| !taken.contains(b))
                .max_by_key(|b| taken.iter().map(|t| bucket_distance(*b, *t)).min())
                .unwrap_or(wanted)
        } else {
            wanted
        };
        taken.push(bucket);
        out.insert(id.to_owned(), tint_of_bucket(bucket, style));
    }
    out
}

/// The colour of hue bucket `bucket` under `style`.
fn tint_of_bucket(bucket: usize, style: &EditorStyle) -> Color32 {
    let base = egui::ecolor::Hsva::from(style.note_fill);
    tint_at_luminance(
        bucket as f32 / HUE_BUCKETS as f32,
        base.s,
        relative_luminance(style.note_fill),
    )
}

/// The colour of hue `hue` and saturation `sat` whose relative luminance is
/// `target`.
///
/// Bisected along a path that runs black -> the hue at full value -> white,
/// on which luminance only rises, so there is exactly one point to find.
///
/// Holding the luminance is the point. Contrast is a function of luminance
/// alone, so every tint built this way has the same contrast against the
/// note's text as the style's own `note_fill` does — one style test covers
/// every instrument's colour rather than each hue needing its own.
fn tint_at_luminance(hue: f32, sat: f32, target: f32) -> Color32 {
    let at = |t: f32| {
        let hsva = if t <= 0.5 {
            egui::ecolor::Hsva::new(hue, sat, 2.0 * t, 1.0)
        } else {
            egui::ecolor::Hsva::new(hue, sat * (2.0 - 2.0 * t), 1.0, 1.0)
        };
        Color32::from(hsva)
    };
    let (mut lo, mut hi) = (0.0_f32, 1.0_f32);
    for _ in 0..20 {
        let mid = 0.5 * (lo + hi);
        if relative_luminance(at(mid)) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    at(0.5 * (lo + hi))
}

/// The colour the notes of instrument `id` are painted when nothing else
/// wants its hue.
///
/// Every note was the style's one `note_fill`, so five instruments on five
/// lanes were one blue and a note moved to the wrong lane looked at home
/// (#62, Overlands #1335 C3). Each id now gets a hue, at the saturation and
/// luminance of the style's `note_fill`: the colour follows the host's
/// theme, and what reads on one instrument's notes reads on every
/// instrument's. Which hue each id in a recipe actually gets is
/// [`instrument_tints`], which pushes crowded ones apart; this is what an
/// id asks for, and what a note naming no instrument at all would get.
fn instrument_tint(id: &str, style: &EditorStyle) -> Color32 {
    tint_of_bucket(wanted_bucket(id), style)
}

/// How opaque a note of `volume` is drawn: loud notes solid, quiet ones
/// faint, never below [`QUIET_ALPHA`].
fn volume_alpha(volume: f32) -> f32 {
    QUIET_ALPHA + (1.0 - QUIET_ALPHA) * volume.clamp(0.0, 1.0)
}

/// The instrument most of `track`'s notes name, and how many name it.
///
/// Ties break on the instrument whose first note starts earliest and then on
/// the lower id, so a lane's name depends on what is on it and not on the
/// order the events happen to be stored in.
fn dominant_instrument(track: &Track) -> Option<(&str, usize)> {
    let mut tally: std::collections::BTreeMap<&str, (usize, f32)> =
        std::collections::BTreeMap::new();
    for event in &track.events {
        let entry = tally
            .entry(event.instrument_id.as_str())
            .or_insert((0, f32::INFINITY));
        entry.0 += 1;
        entry.1 = entry.1.min(event.time_beats);
    }
    tally
        .into_iter()
        .reduce(|best, next| {
            let (_, (best_n, best_t)) = best;
            let (_, (next_n, next_t)) = next;
            let better = next_n > best_n || (next_n == best_n && next_t < best_t);
            if better { next } else { best }
        })
        .map(|(id, (count, _))| (id, count))
}

/// A note's pitch as it is written on its block, or `None` at the patch's
/// native pitch, which is the default and would say nothing on every note.
fn pitch_label(pitch: f32) -> Option<String> {
    if (pitch - 1.0).abs() < 5e-3 {
        return None;
    }
    let mut digits = format!("{pitch:.2}");
    while digits.ends_with('0') {
        digits.pop();
    }
    if digits.ends_with('.') {
        digits.pop();
    }
    Some(format!("\u{00D7}{digits}"))
}

/// What a note's block can say, widest first: its instrument and its pitch,
/// then the pitch alone, then nothing.
///
/// The block draws the first of these that fits inside it. A label was
/// drawn at the block's left edge whatever the block's width, so a
/// quarter-beat note at the fitted zoom wrote its instrument's name across
/// the four notes after it.
fn note_labels(event: &Event, missing: bool) -> Vec<String> {
    if missing {
        return vec![missing_label(&event.instrument_id)];
    }
    let name = event.instrument_id.as_str();
    match pitch_label(event.pitch_multiplier) {
        Some(pitch) => vec![format!("{name} {pitch}"), pitch],
        None => vec![name.to_owned()],
    }
}

/// Everything a note is, as one line, for its hover: what the block would
/// say if it were wide enough, and what it never has room for.
fn note_tooltip(event: &Event, bpm: f32) -> String {
    let pitch = pitch_label(event.pitch_multiplier).unwrap_or_else(|| "native pitch".to_owned());
    format!(
        "{} \u{00B7} {pitch} \u{00B7} from beat {} for {} ({}), tail {}, volume {:.0}%",
        if event.instrument_id.is_empty() {
            "(no instrument)"
        } else {
            event.instrument_id.as_str()
        },
        trim_beats(event.time_beats),
        trim_beats(event.gate_beats),
        beats_as_time(event.gate_beats, bpm),
        trim_beats(event.release_beats),
        100.0 * event.volume.clamp(0.0, 1.0),
    )
}

/// A beat count without the trailing zeros a `{:.2}` leaves on it.
fn trim_beats(beats: f32) -> String {
    let mut digits = format!("{beats:.2}");
    while digits.ends_with('0') {
        digits.pop();
    }
    if digits.ends_with('.') {
        digits.pop();
    }
    if digits.is_empty() {
        "0".to_owned()
    } else {
        digits
    }
}

/// `beats` at `bpm`, in seconds or minutes and seconds.
///
/// Beats are what the schema stores and what the timeline rules, but nobody
/// hears beats: "34 beats" at 60 BPM is half a minute of audio and at 300
/// BPM is under seven seconds, and the field that set it said only
/// "duration (beats)" (#62, Overlands #1335 C5).
fn beats_as_time(beats: f32, bpm: f32) -> String {
    let secs = if bpm > 0.0 { beats * 60.0 / bpm } else { 0.0 };
    if secs >= 60.0 {
        let whole = secs.floor() as u32;
        format!("{}:{:02}", whole / 60, whole % 60)
    } else if secs >= 10.0 {
        format!("{secs:.0} s")
    } else {
        format!("{secs:.1} s")
    }
}

/// Where a marker's label goes: at `x` inside `bounds`, to the right of the
/// marker when there is room and to its left when there is not, and never
/// outside `bounds`.
///
/// A marker can sit at the very last beat — the end marker always does — so
/// a label simply placed to its right is a label off the end of the
/// timeline, which the acceptance of #1335 asks about by name.
fn marker_label_rect(x: f32, size: Vec2, top: f32, bounds: Rect) -> Rect {
    let room_right = bounds.right() - LABEL_PAD - (x + LABEL_PAD);
    let left = if room_right >= size.x {
        x + LABEL_PAD
    } else {
        x - LABEL_PAD - size.x
    };
    let left = left.clamp(
        bounds.left() + LABEL_PAD,
        (bounds.right() - LABEL_PAD - size.x).max(bounds.left() + LABEL_PAD),
    );
    Rect::from_min_size(Pos2::new(left, top), size)
}

/// A fresh `instN` id not already used by an instrument in `recipe`.
fn unique_instrument_id(recipe: &SequenceRecipe) -> String {
    let mut n = 1;
    loop {
        let candidate = format!("inst{n}");
        if !recipe.instruments.iter().any(|i| i.id == candidate) {
            return candidate;
        }
        n += 1;
    }
}

impl SequenceEditorState {
    /// Step the recipe back to before the last committed edit. `true` if
    /// anything moved.
    ///
    /// There is one history for the whole recipe, an open instrument's
    /// patch included: the patch is part of the recipe, so an edit made in
    /// the embedded canvas is a step on this stack and not on that canvas's
    /// (#60, Overlands #1333).
    pub fn undo(&mut self, recipe: &mut SequenceRecipe) -> bool {
        self.history.undo(recipe)
    }

    /// Step the recipe forward again after an [`Self::undo`].
    pub fn redo(&mut self, recipe: &mut SequenceRecipe) -> bool {
        self.history.redo(recipe)
    }

    /// Whether there is a committed edit to undo.
    pub fn can_undo(&self) -> bool {
        self.history.can_undo()
    }

    /// Whether there is an undone edit to redo.
    pub fn can_redo(&self) -> bool {
        self.history.can_redo()
    }

    /// Tell the editor its recipe was replaced from outside, so the swap
    /// becomes a step in its history. See
    /// [`crate::ui::PatchEditorState::note_external_change`].
    pub fn note_external_change(&mut self, recipe: &SequenceRecipe) {
        self.history.commit(recipe);
    }

    /// The caps this editor holds its recipe to.
    pub fn limits(&self) -> &EditorLimits {
        &self.limits
    }

    /// Hold this editor — and every instrument patch opened inside it — to
    /// `limits` instead of the record boundary's own envelope.
    ///
    /// The embedded canvases get them too, and so does every canvas opened
    /// later: an instrument's patch is part of the recipe, so a host that
    /// tightens the recipe's caps and leaves a patch on the defaults has
    /// two answers to one question.
    pub fn set_limits(&mut self, limits: EditorLimits) {
        self.limits = limits;
        for canvas in self.canvas_states.values_mut() {
            canvas.set_limits(limits);
        }
    }

    /// Whether a removal is waiting to be confirmed — the host's cue that
    /// the editor has a question on screen.
    pub fn awaiting_confirmation(&self) -> bool {
        self.pending_removal.is_some()
    }

    /// Where the confirmation was drawn on the last frame, if one is up.
    ///
    /// In screen points, re-measured every frame.
    pub fn confirmation_rect(&self) -> Option<Rect> {
        (self.confirmation_rect != Rect::NOTHING).then_some(self.confirmation_rect)
    }

    /// Where every control that asks for a removal was drawn on the last
    /// frame, in screen points: each instrument row's cross in row order,
    /// then each track gutter's in track order — the order they are drawn
    /// in.
    ///
    /// An affordance at the point the user clicks is hovered by that same
    /// click, so what a confirmation must not do is put its buttons where
    /// the cross that summoned it was. This is what makes that checkable.
    pub fn removal_asks(&self) -> &[Rect] {
        &self.removal_asks
    }

    /// Whether the sequence editor took this frame's keyboard. See
    /// [`crate::ui::PatchEditorState::wants_keyboard`].
    pub fn wants_keyboard(&self) -> bool {
        self.owns_keys
    }

    /// Whether it acted on an Escape this frame by clearing its selection.
    /// See [`crate::ui::PatchEditorState::took_escape`].
    pub fn took_escape(&self) -> bool {
        self.took_escape
    }
}

/// Edit a whole [`SequenceRecipe`]: transport, instruments, the track/event
/// timeline, and the selected-event inspector.
///
/// `state` persists view/selection across frames — pass the same instance each
/// frame.  Does **not** draw the active instrument's patch; call
/// [`active_instrument_canvas`] for that (typically in a separate panel).
pub fn sequence_recipe_editor(
    ui: &mut egui::Ui,
    recipe: &mut SequenceRecipe,
    state: &mut SequenceEditorState,
    id: Id,
) -> EditorResponse {
    let mut res = EditorResponse::NONE;
    let style = editor_style(ui);
    state.history.begin(recipe);
    state.removal_asks.clear();
    state.confirmation_rect = Rect::NOTHING;

    egui::CollapsingHeader::new("Transport")
        .default_open(true)
        .show(ui, |ui| res.merge(transport(ui, recipe)));

    ui.separator();
    res.merge(instruments_panel(ui, recipe, state, id, &style));

    ui.separator();
    res.merge(timeline(ui, recipe, state, id, &style));

    ui.separator();
    egui::CollapsingHeader::new("Event inspector")
        .default_open(true)
        .show(ui, |ui| {
            res.merge(event_inspector(ui, recipe, state, &style))
        });

    res.merge(recipe_keys(ui, recipe, state, ui.min_rect()));
    if res.rebake {
        state.history.commit(recipe);
    }
    res
}

/// The sequence editor's keyboard: undo and redo over the whole recipe, and
/// Escape to clear the selected event.
///
/// The same ownership rule as the canvas's — the pointer over the editor,
/// nothing holding text focus — so typing an instrument's name keeps its
/// own keys. The *canvas* half of the sequence editor reads its keys
/// through this too, via [`active_instrument_canvas`]: one recipe, one
/// history, one place the keys land.
fn recipe_keys(
    ui: &mut egui::Ui,
    recipe: &mut SequenceRecipe,
    state: &mut SequenceEditorState,
    region: egui::Rect,
) -> EditorResponse {
    use egui::{Key, Modifiers};

    let mut res = EditorResponse::NONE;
    state.took_escape = false;
    let typing = ui.memory(|m| m.focused()).is_some();
    state.owns_keys = !typing && ui.rect_contains_pointer(region);
    if !state.owns_keys {
        return res;
    }
    // Most specific first: `consume_key` ignores extra Shift, so Ctrl+Z
    // would swallow Ctrl+Shift+Z the other way round.
    let (redo, undo, escape, duplicate, nudge, fine) = ui.input_mut(|i| {
        (
            i.consume_key(Modifiers::COMMAND | Modifiers::SHIFT, Key::Z)
                | i.consume_key(Modifiers::COMMAND, Key::Y),
            i.consume_key(Modifiers::COMMAND, Key::Z),
            i.consume_key(Modifiers::NONE, Key::Escape),
            i.consume_key(Modifiers::COMMAND, Key::D),
            [(Key::ArrowLeft, -1.0), (Key::ArrowRight, 1.0)]
                .into_iter()
                .filter(|(key, _)| {
                    i.consume_key(Modifiers::SHIFT, *key) || i.consume_key(Modifiers::NONE, *key)
                })
                .map(|(_, dir)| dir)
                .sum::<f32>(),
            i.modifiers.shift,
        )
    });
    if redo {
        res.rebake = state.history.redo(recipe);
        res.changed = res.rebake;
        return res;
    }
    if undo {
        res.rebake = state.history.undo(recipe);
        res.changed = res.rebake;
        return res;
    }
    // One step per press, most recent gesture first: a question waiting to
    // be answered, then a marquee being dragged out, then the notes it or a
    // click picked (#61's ladder, three rungs deep on the timeline now).
    //
    // Escape means Keep — the reading every other dialog has taught — and
    // it spends exactly one press and says so through `took_escape`, or the
    // host's own Esc ladder loses a rung and the pop-out closes under the
    // question.
    if escape && state.pending_removal.take().is_some() {
        state.took_escape = true;
    } else if escape && (state.marquee.take().is_some() || !state.selection.is_empty()) {
        state.clear_selection();
        state.took_escape = true;
    }
    if duplicate && duplicate_selection(recipe, state) {
        res.changed = true;
        res.rebake = true;
    }
    if nudge != 0.0 && !state.selection.is_empty() {
        // Shift is the fine step, an eighth of the grid, for a note that
        // has to sit just off it.
        let step = if fine {
            0.125 * state.snap.step()
        } else {
            state.snap.step()
        };
        let picked: Vec<(usize, usize)> = state.selection.iter().copied().collect();
        for (ti, ei) in picked {
            if let Some(event) = recipe.tracks.get_mut(ti).and_then(|t| t.events.get_mut(ei)) {
                event.time_beats = (event.time_beats + nudge * step).max(0.0);
            }
        }
        res.changed = true;
        res.rebake = true;
    }
    res
}

/// Draw the active instrument's patch in the Phase-2 node canvas, or a hint if
/// none is selected.  Each instrument keeps its own canvas layout (keyed by
/// id) in `state`.
///
/// # The canvas claims the rest of the `Ui` — draw it last
///
/// With an instrument open this is [`audio_patch_canvas`], which takes all
/// the space left in `ui`. Anything drawn after it is laid out below that
/// space, and inside an [`egui::Window`] it also makes the window grow to its
/// constraint (the same heading on [`audio_patch_canvas`] has the mechanism).
/// This one is easy to miss because it only shows up once an instrument is
/// open: with none open, the hint label is short and the content after it
/// fits. Give it the last region, usually an `egui::CentralPanel` next to an
/// `egui::Panel::left` that holds [`sequence_recipe_editor`] in a
/// `ScrollArea`, as the `sequence_editor` and `host_window` examples do.
pub fn active_instrument_canvas(
    ui: &mut egui::Ui,
    recipe: &mut SequenceRecipe,
    state: &mut SequenceEditorState,
    id: Id,
) -> EditorResponse {
    let Some(i) = state
        .active_instrument
        .filter(|i| *i < recipe.instruments.len())
    else {
        state.active_instrument = None;
        ui.label("Press an instrument's \u{270F} Edit to open its patch here.");
        return EditorResponse::NONE;
    };
    let inst_id = recipe.instruments[i].id.clone();
    ui.label(format!("Patch for instrument \u{201C}{inst_id}\u{201D}"));
    state.history.begin(recipe);
    let limits = state.limits;
    let canvas_state = state.canvas_states.entry(inst_id).or_default();
    // A patch opened for the first time inherits the recipe's caps; one
    // opened before already has them from `set_limits`.
    canvas_state.set_limits(limits);
    // This patch is part of the recipe, so the recipe's history owns it.
    // With both on, one Ctrl+Z would walk two stacks at once and the
    // recipe's would go stale (#60, Overlands #1333).
    canvas_state.disable_history();
    let mut res = audio_patch_canvas(
        ui,
        &mut recipe.instruments[i].patch,
        canvas_state,
        id.with("inst_canvas"),
    );
    // The canvas's own keys are its own (delete, duplicate, nudge, fit);
    // undo and redo are the recipe's, and reach it from here.
    let region = ui.min_rect();
    res.merge(recipe_keys(ui, recipe, state, region));
    if res.rebake {
        state.history.commit(recipe);
    }
    res
}

// ---------------------------------------------------------------------------
// Transport
// ---------------------------------------------------------------------------

fn transport(ui: &mut egui::Ui, recipe: &mut SequenceRecipe) -> EditorResponse {
    let mut res = EditorResponse::NONE;

    res.merge(ranged_slider(ui, &mut recipe.bpm, 20.0..=300.0, |s| {
        s.logarithmic(true).text("BPM")
    }));

    ui.horizontal(|ui| {
        ui.label("Sample rate");
        egui::ComboBox::from_id_salt("seq_sample_rate")
            .selected_text(recipe.sample_rate.to_string())
            .show_ui(ui, |ui| {
                for sr in [22_050_u32, 32_000, 44_100, 48_000, 96_000] {
                    if ui
                        .selectable_label(recipe.sample_rate == sr, sr.to_string())
                        .clicked()
                    {
                        recipe.sample_rate = sr;
                        res.changed = true;
                        res.rebake = true;
                    }
                }
            });
    });

    let bpm = recipe.bpm;
    res.merge(beat_field(
        ui,
        "Length",
        "How long the whole sequence is. Notes past it are still baked; \
         the loop and the crossfade are measured from it",
        &mut recipe.duration_beats,
        0.25,
        0.25..=512.0,
        bpm,
    ));

    let dur = recipe.duration_beats.max(0.0);
    let mut looping = recipe.loop_start_beats.is_some();
    let loop_toggle = ui.checkbox(&mut looping, "Loop").on_hover_text(
        "Bake the sequence to loop without a seam: it plays to the end and \
         returns to the loop point",
    );
    if loop_toggle.changed() {
        recipe.loop_start_beats = looping.then_some(0.0);
        res.changed = true;
        res.rebake = true;
    }
    if let Some(loop_start) = recipe.loop_start_beats.as_mut() {
        res.merge(beat_field(
            ui,
            "Loop from",
            "Where the loop returns to. Everything before it is a run-up, \
             played once",
            loop_start,
            0.25,
            0.0..=dur,
            bpm,
        ));
    }
    if recipe.loop_start_beats.is_some() {
        res.merge(beat_field(
            ui,
            "Blend into loop",
            "How much of the end is faded into the loop point, so the seam \
             cannot be heard",
            &mut recipe.loop_crossfade_beats,
            0.25,
            0.0..=dur.max(0.25),
            bpm,
        ));
    }

    res
}

/// A beats field that says what it is for and how long it is.
///
/// The label is words rather than the schema's name for the field, the
/// hover says what the value does, and the time at the recipe's BPM is
/// written beside the beats — nobody hears beats (#62, Overlands #1335 C5).
fn beat_field(
    ui: &mut egui::Ui,
    label: &str,
    hover: &str,
    value: &mut f32,
    speed: f32,
    range: std::ops::RangeInclusive<f32>,
    bpm: f32,
) -> EditorResponse {
    ui.horizontal(|ui| {
        ui.label(label).on_hover_text(hover);
        let res = drag_value_debounced(ui, value, speed, range, " beats");
        ui.label(egui::RichText::new(beats_as_time(*value, bpm)).weak());
        res
    })
    .inner
}

// ---------------------------------------------------------------------------
// Instruments
// ---------------------------------------------------------------------------

fn instruments_panel(
    ui: &mut egui::Ui,
    recipe: &mut SequenceRecipe,
    state: &mut SequenceEditorState,
    id: Id,
    style: &EditorStyle,
) -> EditorResponse {
    let mut res = EditorResponse::NONE;
    let mut add = false;
    let mut remove: Option<usize> = None;
    let mut remove_now: Option<usize> = None;
    let mut rename: Option<(usize, String)> = None;
    let limits = state.limits;

    let room = limits.at(Cap::Instruments, recipe.instruments.len());
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Instruments").strong());
        cap_readout(ui, room, style);
        if ui
            .add_enabled(room.has_room(), egui::Button::new("Add instrument"))
            .on_hover_text("A new instrument with an empty patch")
            // The only hover a disabled widget ever shows (#1289): its
            // `on_hover_text` never fires, so the reason goes here or
            // nowhere.
            .on_disabled_hover_text(room.full_reason())
            .clicked()
        {
            add = true;
        }
    });
    // Drawn here, at the top of the section, on the frame *after* the cross
    // was pressed: a fixed position in both of egui's passes, and one no
    // click of the user's is already sitting on.
    res.merge(confirm_instrument_removal(
        ui,
        recipe,
        state,
        style,
        &mut remove_now,
    ));
    let asking = state.pending_removal.is_some();
    let mut asks: Vec<Rect> = Vec::new();

    state.name_edits.retain(|row, edit| {
        recipe
            .instruments
            .get(*row)
            .is_some_and(|inst| inst.id == edit.from)
    });
    for i in 0..recipe.instruments.len() {
        let field = ui.horizontal(|ui| {
            let active = state.active_instrument == Some(i);
            if ui
                // The pencil and the word, as Overlands says it ("✏ Edit
                // audio…"): the one glyph the vocabulary keeps for editing
                // (#58). U+270F, the emoji-presentation pencil, is in egui's
                // embedded Noto Emoji, where the text-only U+270E is not, so
                // it draws in any host that ships egui's default fonts (#52).
                .selectable_label(active, "\u{270F} Edit")
                .on_hover_text("Open this instrument's patch in the canvas")
                .clicked()
            {
                state.active_instrument = if active { None } else { Some(i) };
            }
            let field = name_field(
                ui,
                recipe,
                i,
                &mut state.name_edits,
                id.with(("instrument_name", i)),
                style.error,
                limits.get(Cap::InstrumentIdBytes),
            );
            let nodes = recipe.instruments[i].patch.graph.nodes.len();
            ui.label(egui::RichText::new(format!("{nodes} node(s)")).weak());
            // Disabled while a question is up: the bar pushed these rows
            // down, so the pointer that pressed this cross is now resting
            // on a *different* instrument's, and the second click of an
            // impatient double would remove the wrong one.
            let cross = ui
                .add_enabled(!asking, egui::Button::new("\u{2716}"))
                .on_hover_text("Remove instrument")
                .on_disabled_hover_text("Answer the question above first");
            asks.push(cross.rect);
            if cross.clicked() {
                remove = Some(i);
            }
            field
        });
        let NameField {
            rect,
            focused,
            commit,
        } = field.inner;
        if let Some(name) = commit {
            rename = Some((i, name));
        }
        // Under the field: why the typed name is refused, or that the name
        // it shows is not the instrument's yet. The second happens only when
        // a refused name became valid because another instrument moved out
        // of its way; it waits for the user rather than apply itself.
        let note = state.name_edits.get(&i).and_then(|edit| {
            match check_name(
                recipe,
                i,
                &edit.text,
                state.limits.get(Cap::InstrumentIdBytes),
            ) {
                Err(refusal) => Some((refusal.to_string(), style.error)),
                Ok(_) if !focused => Some((
                    "Not applied yet: Enter in the field applies it, Esc puts the name back"
                        .to_owned(),
                    ui.visuals().weak_text_color(),
                )),
                Ok(_) => None,
            }
        });
        if let Some((note, colour)) = note {
            ui.horizontal(|ui| {
                ui.add_space(rect.left() - ui.cursor().left());
                ui.label(egui::RichText::new(note).small().color(colour));
            });
        }
    }

    if let Some((i, name)) = rename {
        rename_instrument(recipe, &mut state.canvas_states, i, &name);
        res.changed = true;
        res.rebake = true;
    }
    if add {
        let id = unique_instrument_id(recipe);
        recipe.instruments.push(Instrument {
            id,
            patch: AudioPatch::default(),
        });
        res.changed = true;
        res.rebake = true;
    }
    if let Some(i) = remove {
        // An instrument no note names is nothing to lose, so it goes at
        // once; one that notes depend on asks, and says how many (#63,
        // Overlands #1336 C2).
        let id = recipe.instruments[i].id.clone();
        let notes = notes_naming(recipe, &id);
        if notes == 0 {
            remove_now = Some(i);
        } else {
            state.pending_removal = Some(PendingRemoval::Instrument { row: i, id, notes });
        }
    }
    if let Some(i) = remove_now {
        state.pending_removal = None;
        recipe.instruments.remove(i);
        match state.active_instrument {
            Some(a) if a == i => state.active_instrument = None,
            Some(a) if a > i => state.active_instrument = Some(a - 1),
            _ => {}
        }
        // The rows below move up, and a name being typed moves with its row.
        state.name_edits = std::mem::take(&mut state.name_edits)
            .into_iter()
            .filter_map(|(row, edit)| match row.cmp(&i) {
                std::cmp::Ordering::Less => Some((row, edit)),
                std::cmp::Ordering::Equal => None,
                std::cmp::Ordering::Greater => Some((row - 1, edit)),
            })
            .collect();
        res.changed = true;
        res.rebake = true;
    }

    state.removal_asks.extend(asks);
    res
}

// ---------------------------------------------------------------------------
// Confirming a removal
// ---------------------------------------------------------------------------

/// How many notes across the whole recipe name instrument `id`.
///
/// The number the question is about: a lane's gutter already counts its own
/// notes, but an instrument is used by any track, so this walks all of them.
fn notes_naming(recipe: &SequenceRecipe, id: &str) -> usize {
    recipe
        .tracks
        .iter()
        .flat_map(|t| &t.events)
        .filter(|e| e.instrument_id == id)
        .count()
}

/// `n` notes, said as a sentence rather than as a field label.
///
/// The editor writes "2 node(s)" beside a count in a row, where the
/// brackets are read as shorthand. A question is a sentence, and "Its 1
/// note(s) go with it" is not one.
fn notes_phrase(n: usize) -> String {
    if n == 1 {
        "1 note".to_owned()
    } else {
        format!("{n} notes")
    }
}

/// What the user said to a pending removal.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Answer {
    /// Still on screen.
    Waiting,
    /// Put it back.
    Keep,
    /// Go ahead.
    Remove,
}

/// The question, and the two answers.
///
/// Drawn at the top of the section that owns the thing, on the frame after
/// the cross was pressed — a row that is in the same place in both of
/// egui's passes (#1332), and a long way from the 14 px cross the pointer
/// is still sitting on. While it is up, every removal control in the editor
/// is disabled: the bar pushes the rows below it down, so a pointer left
/// where it clicked would otherwise be resting on some *other* row's cross.
fn confirm_bar(
    ui: &mut egui::Ui,
    question: &str,
    remove_hover: &str,
    style: &EditorStyle,
) -> (Answer, Rect) {
    let mut answer = Answer::Waiting;
    let drawn = egui::Frame::group(ui.style())
        .stroke(egui::Stroke::new(1.0, style.warn))
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new(question).color(style.warn));
                if ui
                    .button("Keep")
                    .on_hover_text("Leave it where it is and put the question away")
                    .clicked()
                {
                    answer = Answer::Keep;
                }
                if ui.button("Remove").on_hover_text(remove_hover).clicked() {
                    answer = Answer::Remove;
                }
            });
        })
        .response
        .rect;
    (answer, drawn)
}

/// The pending instrument removal, if there is one and it still makes
/// sense; `remove_now` is set when it is confirmed.
///
/// Re-checked against the recipe every frame rather than trusted: the host
/// can put a different recipe under the editor between frames (#1333 A9),
/// and a row index that outlived its instrument would otherwise ask about
/// one thing and remove another.
fn confirm_instrument_removal(
    ui: &mut egui::Ui,
    recipe: &SequenceRecipe,
    state: &mut SequenceEditorState,
    style: &EditorStyle,
    remove_now: &mut Option<usize>,
) -> EditorResponse {
    let Some(PendingRemoval::Instrument { row, id, .. }) = &state.pending_removal else {
        return EditorResponse::NONE;
    };
    let (row, id) = (*row, id.clone());
    if recipe.instruments.get(row).map(|i| i.id.as_str()) != Some(id.as_str()) {
        state.pending_removal = None;
        return EditorResponse::NONE;
    }
    // Counted again now, not when the question was asked: an edit made
    // while it was up would otherwise have the bar quoting a stale number.
    let notes = notes_naming(recipe, &id);
    if notes == 0 {
        state.pending_removal = None;
        *remove_now = Some(row);
        return EditorResponse::NONE;
    }
    let question = format!(
        "Remove \u{201C}{id}\u{201D}? {} {} it, and will be left naming an \
         instrument that is not there.",
        notes_phrase(notes),
        if notes == 1 { "plays" } else { "play" }
    );
    let (answer, drawn) = confirm_bar(
        ui,
        &question,
        &format!(
            "Remove \u{201C}{id}\u{201D} and leave its {} naming nothing",
            notes_phrase(notes)
        ),
        style,
    );
    state.confirmation_rect = drawn;
    match answer {
        Answer::Remove => *remove_now = Some(row),
        Answer::Keep => state.pending_removal = None,
        Answer::Waiting => {}
    }
    EditorResponse::NONE
}

/// The pending track removal, if there is one and it still makes sense.
///
/// Pushes [`Action::RemoveTrackNow`] rather than removing here, so every
/// edit to `recipe.tracks` still goes through [`apply_actions`] — one place
/// that mutates the tracks, whichever of the three doors asked.
fn confirm_track_removal(
    ui: &mut egui::Ui,
    recipe: &SequenceRecipe,
    state: &mut SequenceEditorState,
    style: &EditorStyle,
    actions: &mut Vec<Action>,
) {
    let Some(PendingRemoval::Track { track, .. }) = &state.pending_removal else {
        return;
    };
    let track = *track;
    let Some(events) = recipe.tracks.get(track).map(|t| t.events.len()) else {
        state.pending_removal = None;
        return;
    };
    if events == 0 {
        state.pending_removal = None;
        actions.push(Action::RemoveTrackNow(track));
        return;
    }
    let question = format!(
        "Remove track {}? Its {} {} with it.",
        track + 1,
        notes_phrase(events),
        if events == 1 { "goes" } else { "go" }
    );
    let (answer, drawn) = confirm_bar(
        ui,
        &question,
        &format!(
            "Remove track {} and its {}",
            track + 1,
            notes_phrase(events)
        ),
        style,
    );
    state.confirmation_rect = drawn;
    match answer {
        Answer::Remove => actions.push(Action::RemoveTrackNow(track)),
        Answer::Keep => state.pending_removal = None,
        Answer::Waiting => {}
    }
}

/// What one instrument row's name field did this frame.
struct NameField {
    /// Where the field is, so the line under it can line up with it.
    rect: Rect,
    /// The field has the keyboard focus.
    focused: bool,
    /// A valid name committed this frame, to rename the instrument to.
    commit: Option<String>,
}

/// The name field of instrument `row`: it edits the row's [`NameEdit`], not
/// the id, and reports a commit only for a name [`check_name`] accepts.
/// While the typed name is refused, the text and the frame take `error`.
fn name_field(
    ui: &mut egui::Ui,
    recipe: &SequenceRecipe,
    row: usize,
    edits: &mut HashMap<usize, NameEdit>,
    field_id: Id,
    error: Color32,
    limit: usize,
) -> NameField {
    let current = &recipe.instruments[row].id;
    let refused = |edits: &HashMap<usize, NameEdit>| {
        edits
            .get(&row)
            .is_some_and(|edit| check_name(recipe, row, &edit.text, limit).is_err())
    };

    let mut text = edits
        .get(&row)
        .map_or_else(|| current.clone(), |edit| edit.text.clone());
    let mut field = egui::TextEdit::singleline(&mut text)
        .id(field_id)
        .desired_width(140.0);
    if refused(edits) {
        field = field.text_color(error);
    }
    let r = ui.add(field).on_hover_text(
        "Rename: Enter or clicking away applies it, and the notes follow; Esc puts the name back",
    );

    if r.changed() {
        let from = edits
            .get(&row)
            .map_or_else(|| current.clone(), |edit| edit.from.clone());
        edits.insert(
            row,
            NameEdit {
                from,
                text,
                typing: true,
            },
        );
        // The colour above was chosen before this keystroke.
        ui.ctx().request_repaint();
    }
    if refused(edits) {
        ui.painter().rect_stroke(
            r.rect,
            ui.visuals().widgets.inactive.corner_radius,
            Stroke::new(1.0, error),
            StrokeKind::Outside,
        );
    }

    let rect = r.rect;
    let focused = r.has_focus();
    let nothing = NameField {
        rect,
        focused,
        commit: None,
    };
    if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Escape)) {
        edits.remove(&row);
        return nothing;
    }
    if focused {
        if let Some(edit) = edits.get_mut(&row) {
            edit.typing = true;
        }
        return nothing;
    }
    // Not focused: Enter, Tab or a click elsewhere took the focus, or the
    // field was not drawn while it had it (a collapsed header, a closed
    // window). An edit typed into until now is committed, once.
    let Some(edit) = edits.get_mut(&row) else {
        return nothing;
    };
    if !std::mem::take(&mut edit.typing) {
        return nothing;
    }
    let commit = match check_name(recipe, row, &edit.text, limit) {
        // Refused: the edit and its reason stay until fixed or Esc'd.
        Err(_) => return nothing,
        Ok(name) if name == current.as_str() => None,
        Ok(name) => Some(name.to_owned()),
    };
    edits.remove(&row);
    NameField { commit, ..nothing }
}

// ---------------------------------------------------------------------------
// Timeline
// ---------------------------------------------------------------------------

/// A deferred edit to the recipe's tracks, applied after the lane loop has
/// let go of its borrow — the shape the canvas uses for the same reason.
enum Action {
    /// A note of the lane's dominant instrument at this beat.
    AddNote { track: usize, beat: f32 },
    /// The user asked to remove a track — from the gutter's cross or the
    /// track's own menu. Whether it goes at once or asks first is decided
    /// in [`apply_actions`], because both doors arrive here (#63, Overlands
    /// #1336 C2).
    RemoveTrack(usize),
    /// A removal the user has confirmed.
    RemoveTrackNow(usize),
    /// Copy every selected note in place, and drag the copies instead of
    /// the originals (an Alt-drag).
    DuplicateSelection,
}

fn timeline(
    ui: &mut egui::Ui,
    recipe: &mut SequenceRecipe,
    state: &mut SequenceEditorState,
    id: Id,
    style: &EditorStyle,
) -> EditorResponse {
    let mut res = EditorResponse::NONE;
    state.forget_missing_notes(recipe);

    let bpm = recipe.bpm;
    let dur = recipe.duration_beats.max(1.0);
    let loop_start = recipe.loop_start_beats;
    let crossfade = recipe.loop_crossfade_beats;
    // Notes whose id is not in here name no instrument and bake to nothing.
    let known: HashSet<String> = recipe.instruments.iter().map(|i| i.id.clone()).collect();
    let tints = instrument_tints(recipe.instruments.iter().map(|i| i.id.as_str()), style);
    let error = style.error;

    let mut total = dur;
    for track in &recipe.tracks {
        for event in &track.events {
            total = total.max(event.time_beats + event.gate_beats + event.release_beats);
        }
    }
    total = (total + 4.0).ceil();
    let lanes = recipe.tracks.len();

    // The zoom the timeline is drawn at: the user's, or the one that fits
    // what it has to show inside the room it has. `available_width` is the
    // width of the row the `ScrollArea` will be given, which is the
    // question a fit is the answer to.
    let room = ui.available_width();
    let fitted = fit_px_per_beat(room, total);
    let ppb = state.zoom.unwrap_or(fitted).clamp(MIN_PPB, MAX_PPB);

    let mut actions: Vec<Action> = Vec::new();
    // Set inside the lane loop, applied the moment it lets go of the tracks
    // and before the drag delta is read: an Alt-drag duplicates and then
    // carries the copies, so a duplicate deferred to the end of the frame
    // would let the first frame's movement land on the originals.
    let mut copy_now = false;

    res.merge(timeline_toolbar(
        ui,
        recipe,
        state,
        id,
        ppb,
        fitted,
        total,
        style,
        &mut actions,
    ));
    // The question, if one is up, between the toolbar and the ruler: a row
    // that is in the same place in both of egui's passes and is nowhere
    // near the gutter cross or the track menu that asked.
    confirm_track_removal(ui, recipe, state, style, &mut actions);
    // Read once, after the question has had its say: every removal control
    // below is inert while one is up, and every Add is held to these.
    let asking = state.pending_removal.is_some();
    let limits = state.limits;
    let mut asks: Vec<Rect> = Vec::new();
    if state.show_json {
        res.merge(json_io(ui, recipe, &mut state.json, id.with("recipe_json")));
    }

    // The lane names, taken before the loop borrows the tracks mutably: a
    // note says its instrument's name, and the lane says the name of the
    // instrument most of it plays (#62, Overlands #1335 C3).
    let lane_names: Vec<Option<(String, usize)>> = recipe
        .tracks
        .iter()
        .map(|track| dominant_instrument(track).map(|(id, n)| (id.to_owned(), n)))
        .collect();

    // Always visible: a scrollbar that appears under the pointer cannot say
    // that there is more timeline than the panel is showing, which is
    // exactly what it is for (#62, Overlands #1335 C4).
    egui::ScrollArea::horizontal()
        .id_salt(id.with("timeline_scroll"))
        .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysVisible)
        .show(ui, |ui| {
            let width = GUTTER + total * ppb + TAIL_PAD;
            let height = RULER_H + (lanes.max(1) as f32) * LANE_H;
            let (rect, _) = ui.allocate_exact_size(Vec2::new(width, height), Sense::hover());
            state.timeline_rect = rect;
            state.notes.clear();
            let painter = ui.painter_at(rect);
            let bx = |beat: f32| rect.left() + GUTTER + beat * ppb;
            let beat_at = |x: f32| (x - bx(0.0)) / ppb;

            painter.rect_filled(rect, 0.0, style.timeline_ground);

            paint_ruler(&painter, rect, total, ppb, bx(0.0), style);
            // The band goes under the lanes it shades; its markers' lines
            // go over them, after the notes.
            paint_crossfade_band(
                &painter,
                rect,
                dur,
                loop_start,
                crossfade,
                ppb,
                bx(0.0),
                style,
            );

            let lanes_top = rect.top() + RULER_H;
            for (ti, track) in recipe.tracks.iter_mut().enumerate() {
                let track_events = track.events.len();
                let lane_top = lanes_top + ti as f32 * LANE_H;
                let lane = Rect::from_min_max(
                    Pos2::new(rect.left(), lane_top),
                    Pos2::new(rect.right(), lane_top + LANE_H),
                );
                painter.rect_filled(
                    lane,
                    0.0,
                    if ti % 2 == 0 {
                        style.lane
                    } else {
                        style.lane_alt
                    },
                );

                let name = lane_names[ti].as_ref();
                res.merge(lane_gutter(
                    ui,
                    &painter,
                    lane,
                    ti,
                    name,
                    id,
                    style,
                    asking,
                    &mut asks,
                    &mut actions,
                ));

                // The lane's background: a plain drag pulls a marquee out
                // of it, a double-click adds a note, and a right-click
                // opens the lane's menu at the beat it was pressed at.
                //
                // Nothing that acts on a press is drawn where the press
                // lands: an affordance at the pointer takes the click meant
                // for what is under it, which is how 7b's remove cross ate
                // the click that was picking its wire (crate #67).
                let bg = Rect::from_min_max(
                    Pos2::new(rect.left() + GUTTER, lane_top),
                    Pos2::new(rect.right(), lane_top + LANE_H),
                );
                let bg_resp = ui.interact(bg, id.with(("lane_bg", ti)), Sense::click_and_drag());
                // `interact_pointer_pos`, not `press_origin`: a click is
                // reported on the release, and egui clears the press
                // origin on exactly that event.
                if bg_resp.secondary_clicked()
                    && let Some(p) = bg_resp.interact_pointer_pos()
                {
                    state.menu_at = Some((ti, beat_at(p.x)));
                }
                let menu_beat = state
                    .menu_at
                    .filter(|(track, _)| *track == ti)
                    .map(|(_, beat)| beat);
                let note_room = limits.at(Cap::TrackEvents, track_events);
                res.merge(lane_menu(
                    &bg_resp,
                    ti,
                    name,
                    menu_beat,
                    note_room,
                    asking,
                    &mut actions,
                ));
                // The other door to a new note, and it has to be held to
                // the same cap: a double-click that added a 4 097th note
                // would be truncated away by the sanitiser a moment later.
                if bg_resp.double_clicked()
                    && note_room.has_room()
                    && let Some(p) = bg_resp.interact_pointer_pos()
                {
                    actions.push(Action::AddNote {
                        track: ti,
                        beat: snap_beat(beat_at(p.x), state.snap),
                    });
                }
                // The marquee's two corners.
                //
                // `Response::interact_pointer_pos` is where the pointer is
                // *now*, not where the press landed, so a box anchored on
                // it is a box with no anchor: both corners follow the
                // pointer and it never covers anything. The anchor is
                // `press_origin`, which is a position on the context —
                // fine here because the timeline is a `ScrollArea`, whose
                // interact rects are already screen points, where on the
                // patch canvas's `Scene` the two are in different
                // coordinates and mixing them double-transforms (crate
                // #67, and 7b's first coordinate fact).
                if bg_resp.drag_started()
                    && let Some(origin) = ui.ctx().input(|i| i.pointer.press_origin())
                {
                    state.marquee = Some((origin, origin));
                }
                if bg_resp.dragged()
                    && let Some((from, _)) = state.marquee
                    && let Some(p) = bg_resp.interact_pointer_pos()
                {
                    state.marquee = Some((from, p));
                }
                if bg_resp.drag_stopped() {
                    state.marquee = None;
                }

                for (ei, event) in track.events.iter_mut().enumerate() {
                    let geom = note_geometry(event, lane_top, ppb, bx(0.0));
                    state.notes.push(NoteGeom {
                        track: ti,
                        event: ei,
                        body: geom.0,
                        extent: geom.1,
                    });
                    let (body, extent) = geom;
                    let selected = state.selection.contains(&(ti, ei));
                    let missing = !known.contains(&event.instrument_id);
                    let tint = tints
                        .get(&event.instrument_id)
                        .copied()
                        .unwrap_or_else(|| instrument_tint(&event.instrument_id, style));

                    if extent.right() > body.right() {
                        painter.rect_filled(
                            Rect::from_min_max(Pos2::new(body.right(), body.top()), extent.max),
                            2.0,
                            if missing {
                                error.gamma_multiply(0.15)
                            } else {
                                tint.gamma_multiply(0.35 * volume_alpha(event.volume))
                            },
                        );
                    }
                    if missing {
                        paint_missing_block(&painter, body, selected, error);
                    } else {
                        paint_note_block(&painter, body, selected, tint, event.volume, style);
                    }
                    paint_note_label(&painter, body, event, missing, error, style);

                    let resp = ui.interact(body, id.with(("ev", ti, ei)), Sense::click_and_drag());
                    let resp = if missing {
                        resp.on_hover_text(format!(
                            "{}, so this note is silent. Select it to reassign it.",
                            no_instrument_text(&event.instrument_id)
                        ))
                    } else {
                        resp.on_hover_text(note_tooltip(event, bpm))
                    };
                    if resp.clicked() {
                        if resp.ctx.input(|i| i.modifiers.shift) {
                            state.select_also((ti, ei));
                        } else {
                            state.select_only((ti, ei));
                        }
                    }
                    if resp.drag_started() {
                        // Where the press landed, not where the pointer has
                        // got to: `interact_pointer_pos` is the *current*
                        // position, and a drag only starts once it has
                        // moved, so on the frame this runs the pointer is
                        // already several points along. On a 24-point block
                        // that was enough to read every move as a resize of
                        // the right edge.
                        let near_right = resp
                            .ctx
                            .input(|i| i.pointer.press_origin())
                            .is_some_and(|p| p.x >= body.right() - 8.0);
                        let alt = resp.ctx.input(|i| i.modifiers.alt);
                        state.drag_mode = match (near_right, alt) {
                            (true, _) => DragMode::Resize,
                            (false, true) => DragMode::Copy,
                            (false, false) => DragMode::Move,
                        };
                        if !state.selection.contains(&(ti, ei)) {
                            state.select_only((ti, ei));
                        } else {
                            state.selected_event = Some((ti, ei));
                        }
                        state.drag_anchor = Some((ti, ei));
                        copy_now |= state.drag_mode == DragMode::Copy;
                    }
                }
            }

            if copy_now && duplicate_selection(recipe, state) {
                res.changed = true;
                res.rebake = true;
            }
            // Every selected note moves with the one under the pointer, so
            // the drag is read once, outside the lane loop, and applied to
            // all of them.
            res.merge(drag_selection(ui, recipe, state, id, ppb));
            // The marquee picks from the rects the paint just recorded, not
            // from a second reckoning of where the notes are.
            if let Some((from, to)) = state.marquee {
                let band = Rect::from_two_pos(from, to);
                let caught: HashSet<(usize, usize)> = state
                    .notes
                    .iter()
                    .filter(|note| band.intersects(note.extent))
                    .map(|note| (note.track, note.event))
                    .collect();
                if caught != state.selection {
                    state.selection = caught;
                    if state
                        .selected_event
                        .is_none_or(|at| !state.selection.contains(&at))
                    {
                        state.selected_event = state.selection.iter().copied().next();
                    }
                }
            }
            // Over the notes, not under them: a note as long as the
            // sequence — the seeded recipe has two — covered the loop and
            // end lines completely, so the markers C4 is about were
            // invisible at exactly the zoom that had just brought them on
            // screen.
            paint_markers(
                &painter,
                rect,
                dur,
                loop_start,
                crossfade,
                ppb,
                bx(0.0),
                style,
            );
            paint_marquee(&painter, state.marquee, style);
        });

    // A marquee that ended over a note, or outside the timeline, never sees
    // its lane's `drag_stopped`; the pointer coming up ends it either way.
    if state.marquee.is_some() && !ui.ctx().input(|i| i.pointer.any_down()) {
        state.marquee = None;
    }

    state.removal_asks.extend(asks);
    res.merge(apply_actions(recipe, state, actions, &lane_names));
    res
}

/// The row above the lanes: a track, the zoom and a Fit, the snap grid, how
/// many lanes there are, and the More menu that holds what is not wanted
/// every minute.
///
/// One row, the way #59 gave the canvas one: the genetics controls and the
/// JSON fold used to sit above the instruments and the timeline both, which
/// is two rows of chrome before the thing being edited (#62, Overlands
/// #1335 C9).
#[expect(
    clippy::too_many_arguments,
    reason = "one toolbar row's worth of state"
)]
fn timeline_toolbar(
    ui: &mut egui::Ui,
    recipe: &mut SequenceRecipe,
    state: &mut SequenceEditorState,
    id: Id,
    ppb: f32,
    fitted: f32,
    total: f32,
    style: &EditorStyle,
    actions: &mut Vec<Action>,
) -> EditorResponse {
    let mut res = EditorResponse::NONE;
    let limits = state.limits;
    ui.horizontal_wrapped(|ui| {
        let room = limits.at(Cap::Tracks, recipe.tracks.len());
        if ui
            .add_enabled(room.has_room(), egui::Button::new("Add track"))
            .on_hover_text("A new empty track at the bottom of the timeline")
            .on_disabled_hover_text(room.full_reason())
            .clicked()
        {
            recipe.tracks.push(Track::default());
            res.changed = true;
            res.rebake = true;
        }
        cap_readout(ui, room, style);

        ui.label("Zoom")
            .on_hover_text("How wide one beat is drawn, in points");
        // Narrower than egui's default, and the track count is off the row
        // altogether: at the sequence editor's panel width the row wrapped
        // and left More on a line of its own, which is two rows of chrome
        // again.
        ui.spacing_mut().slider_width = 64.0;
        let mut zoom = ppb;
        if ranged_slider(ui, &mut zoom, MIN_PPB..=MAX_PPB, |s| s.show_value(false)).changed {
            state.zoom = Some(zoom);
            res.changed = true;
        }
        if ui
            .add_enabled(state.zoom.is_some(), egui::Button::new("Fit"))
            .on_hover_text(format!(
                "Zoom so all {} beats are in view, and keep them there as the panel is resized",
                trim_beats(total)
            ))
            .on_disabled_hover_text(format!(
                "Already fitted: all {} beats are in view at {:.0} points a beat",
                trim_beats(total),
                fitted
            ))
            .clicked()
        {
            state.zoom = None;
            res.changed = true;
        }

        ui.label("Snap")
            .on_hover_text("The grid a dragged note, an arrow nudge and a new note land on");
        egui::ComboBox::from_id_salt(id.with("snap"))
            .selected_text(state.snap.label())
            .width(56.0)
            .show_ui(ui, |ui| {
                for snap in Snap::ALL {
                    if ui
                        .selectable_label(state.snap == snap, snap.label())
                        .clicked()
                    {
                        state.snap = snap;
                    }
                }
            })
            .response
            .on_hover_text("off, an eighth, a quarter, one beat, or one bar of four");

        ui.separator();
        res.merge(timeline_more_menu(ui, recipe, state, actions));
    });
    res
}

/// The timeline's More menu: the genetics controls and the JSON box, which
/// cost the editor two rows above everything it edits.
fn timeline_more_menu(
    ui: &mut egui::Ui,
    recipe: &mut SequenceRecipe,
    state: &mut SequenceEditorState,
    actions: &mut Vec<Action>,
) -> EditorResponse {
    let mut res = EditorResponse::NONE;
    ui.menu_button("More", |ui| {
        if ui
            .button("Mutate recipe")
            .on_hover_text("Nudge BPM and note volumes via symbios-genetics")
            .clicked()
        {
            recipe.mutate(&mut fresh_rng(), state.mutate_rate);
            res.changed = true;
            res.rebake = true;
        }
        ranged_slider(ui, &mut state.mutate_rate, 0.0..=1.0, |s| s.text("rate"));
        ui.separator();
        let picked = state.selection.len();
        if ui
            .add_enabled(picked > 0, egui::Button::new("Duplicate selected notes"))
            .on_hover_text(format!(
                "Copy the {picked} picked note(s) in place — Alt-drag one does the same"
            ))
            .on_disabled_hover_text("Click a note first; shift-click or drag a box picks more")
            .clicked()
        {
            actions.push(Action::DuplicateSelection);
            ui.close();
        }
        ui.separator();
        if ui
            .selectable_label(state.show_json, "Import / Export JSON")
            .on_hover_text("Show the JSON box under this row")
            .clicked()
        {
            state.show_json = !state.show_json;
        }
    });
    res
}

/// The beat numbers along the top of the timeline, and a line down the
/// whole height at every beat.
fn paint_ruler(
    painter: &egui::Painter,
    rect: Rect,
    total: f32,
    ppb: f32,
    origin: f32,
    style: &EditorStyle,
) {
    // Number every beat, every other one, or every fourth, so the numbers
    // never run into each other as the zoom comes down.
    let step = if ppb < 20.0 {
        4
    } else if ppb < 40.0 {
        2
    } else {
        1
    };
    let mut beat = 0i32;
    while (beat as f32) <= total {
        let x = origin + beat as f32 * ppb;
        painter.line_segment(
            [Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())],
            Stroke::new(1.0, style.timeline_grid),
        );
        if beat % step == 0 {
            painter.text(
                Pos2::new(x + 2.0, rect.top() + RULER_BEATS_Y),
                Align2::LEFT_TOP,
                beat.to_string(),
                egui::FontId::proportional(10.0),
                style.ground_text,
            );
        }
        beat += 1;
    }
}

/// The loop markers: a line down the lanes at each, the crossfade shaded,
/// and each one labelled on the ruler's second line.
///
/// The lines were there and said nothing — the accent line at beat 2 was
/// the loop point and the strong one at the end was the end, and nothing on
/// screen said either (#62, Overlands #1335 C4). Every label is placed with
/// [`marker_label_rect`], which keeps it inside the timeline: the end
/// marker sits at the last beat, where a label simply written to its right
/// would be off the edge.
/// The shade over the crossfade at the end of the sequence.
///
/// Under the lanes, unlike the marker lines: a wash over the note blocks
/// makes them look faded, which is what a quiet note looks like.
#[expect(clippy::too_many_arguments, reason = "the markers and their geometry")]
fn paint_crossfade_band(
    painter: &egui::Painter,
    rect: Rect,
    dur: f32,
    loop_start: Option<f32>,
    crossfade: f32,
    ppb: f32,
    origin: f32,
    style: &EditorStyle,
) {
    if crossfade <= 0.0 || loop_start.is_none() {
        return;
    }
    let lanes_top = rect.top() + RULER_H;
    painter.rect_filled(
        Rect::from_min_max(
            Pos2::new(origin + (dur - crossfade).max(0.0) * ppb, lanes_top),
            Pos2::new(origin + dur * ppb, rect.bottom()),
        ),
        0.0,
        style.crossfade_band,
    );
}

#[expect(clippy::too_many_arguments, reason = "the markers and their geometry")]
fn paint_markers(
    painter: &egui::Painter,
    rect: Rect,
    dur: f32,
    loop_start: Option<f32>,
    crossfade: f32,
    ppb: f32,
    origin: f32,
    style: &EditorStyle,
) {
    let lanes_top = rect.top() + RULER_H;
    let mut labels: Vec<(f32, &str, Color32)> = Vec::new();

    let x_end = origin + dur * ppb;
    if crossfade > 0.0 && loop_start.is_some() {
        labels.push((
            origin + (dur - crossfade).max(0.0) * ppb,
            "blend",
            style.loop_start,
        ));
    }
    if let Some(ls) = loop_start {
        let x = origin + ls * ppb;
        painter.line_segment(
            [Pos2::new(x, lanes_top), Pos2::new(x, rect.bottom())],
            Stroke::new(2.0, style.loop_start),
        );
        labels.push((x, "loop start", style.loop_start));
    }
    painter.line_segment(
        [Pos2::new(x_end, lanes_top), Pos2::new(x_end, rect.bottom())],
        Stroke::new(2.0, style.loop_end),
    );
    labels.push((x_end, "end", style.loop_end));

    // Left to right, each one clear of the one before it. Two markers can
    // be a couple of beats apart — the blend and the end of the seeded
    // recipe are — and their labels are wider than that, so placing each
    // one only against the timeline's edges drew them on top of each other
    // and neither could be read.
    labels.sort_by(|a, b| a.0.total_cmp(&b.0));
    let font = egui::FontId::proportional(10.0);
    let mut after: Option<f32> = None;
    for (x, text, colour) in labels {
        let galley = painter.layout_no_wrap(text.to_owned(), font.clone(), colour);
        let mut at = marker_label_rect(x, galley.size(), rect.top() + RULER_MARKS_Y, rect);
        if let Some(clear) = after.filter(|clear| at.left() < *clear) {
            // Just clear of the last one, and still inside the timeline: a
            // marker at the very last beat has nowhere further right to go,
            // and being inside is the promise the acceptance holds us to.
            let last_chance = (rect.right() - LABEL_PAD - galley.size().x).max(rect.left());
            at = Rect::from_min_size(Pos2::new(clear.min(last_chance), at.top()), galley.size());
        }
        after = Some(at.right() + LABEL_PAD);
        painter.galley(at.min, galley, colour);
    }
}

/// The rubber band of a marquee in progress.
fn paint_marquee(painter: &egui::Painter, marquee: Option<(Pos2, Pos2)>, style: &EditorStyle) {
    let Some((from, to)) = marquee else { return };
    let band = Rect::from_two_pos(from, to);
    painter.rect_filled(band, 0.0, style.note_selected.gamma_multiply(0.15));
    painter.rect_stroke(
        band,
        0.0,
        Stroke::new(1.0, style.note_selected),
        StrokeKind::Inside,
    );
}

/// A lane's gutter: the remove cross, and the lane's name.
///
/// The name is the discoverable end of everything the lane can do: its
/// hover says how to add a note, which used to be said only by the empty
/// inspector, where nobody who had selected a note could see it (#62,
/// Overlands #1335 C6).
#[expect(clippy::too_many_arguments, reason = "one gutter's worth of state")]
fn lane_gutter(
    ui: &mut egui::Ui,
    painter: &egui::Painter,
    lane: Rect,
    track: usize,
    name: Option<&(String, usize)>,
    id: Id,
    style: &EditorStyle,
    asking: bool,
    asks: &mut Vec<Rect>,
    actions: &mut Vec<Action>,
) -> EditorResponse {
    let cross = Rect::from_min_size(
        Pos2::new(lane.left() + 3.0, lane.center().y - 7.0),
        Vec2::splat(14.0),
    );
    asks.push(cross);
    // Inert while a question is up: the bar above pushed every lane down,
    // so the pointer that pressed this cross is now over the lane before
    // it, and a second click would ask about the wrong track.
    let cross_resp = ui
        .interact(
            cross,
            id.with(("rm_track", track)),
            if asking {
                Sense::hover()
            } else {
                Sense::click()
            },
        )
        .on_hover_text(if asking {
            "Answer the question above the timeline first".to_owned()
        } else {
            format!("Remove track {} and its notes", track + 1)
        });
    painter.text(
        cross.center(),
        Align2::CENTER_CENTER,
        "\u{2716}",
        egui::FontId::proportional(13.0),
        if asking {
            style.ground_text.gamma_multiply(0.4)
        } else if cross_resp.hovered() {
            style.error
        } else {
            style.ground_text
        },
    );
    if cross_resp.clicked() {
        actions.push(Action::RemoveTrack(track));
    }

    let text = Rect::from_min_max(
        Pos2::new(cross.right() + 4.0, lane.top()),
        Pos2::new(lane.left() + GUTTER - 4.0, lane.bottom()),
    );
    let (label, colour) = match name {
        Some((name, _)) => (name.clone(), style.ground_text),
        None => ("empty".to_owned(), style.ground_text.gamma_multiply(0.6)),
    };
    let galley = painter.layout(
        label,
        egui::FontId::proportional(12.0),
        colour,
        text.width().max(1.0),
    );
    painter.galley(
        Pos2::new(text.left(), text.center().y - 0.5 * galley.size().y),
        galley,
        colour,
    );
    let hover = match name {
        Some((name, n)) => format!(
            "Track {}: {n} note(s), most of them {name}. Double-click the track to add \
             one, or right-click it for the track's menu",
            track + 1
        ),
        None => format!(
            "Track {}: no notes yet. Double-click the track to add one, or right-click \
             it for the track's menu",
            track + 1
        ),
    };
    ui.interact(text, id.with(("lane_name", track)), Sense::hover())
        .on_hover_text(hover);
    EditorResponse::NONE
}

/// A lane's context menu, opened by a right-click on it: a note at the beat
/// pressed, and the lane's removal.
///
/// A menu at the pointer is the canvas's Add-node gesture (#61), and it is
/// how a lane gets an add control without putting a target where a click
/// already means something else.
fn lane_menu(
    bg: &egui::Response,
    track: usize,
    name: Option<&(String, usize)>,
    beat: Option<f32>,
    room: CapState,
    asking: bool,
    actions: &mut Vec<Action>,
) -> EditorResponse {
    let whose = match name {
        Some((name, _)) => format!("Add a {name} note here"),
        None => "Add a note here".to_owned(),
    };
    bg.context_menu(|ui| {
        ui.label(
            egui::RichText::new(format!("Track {}", track + 1))
                .weak()
                .small(),
        );
        let style = editor_style(ui);
        let can_add = beat.is_some() && room.has_room();
        ui.horizontal(|ui| {
            if ui
                .add_enabled(can_add, egui::Button::new(whose))
                .on_hover_text("At the beat you right-clicked, on this track's grid")
                .clicked()
                && let Some(beat) = beat
            {
                actions.push(Action::AddNote { track, beat });
                ui.close();
            }
            cap_readout(ui, room, &style);
        });
        // Written into the menu rather than left to a hover. #1289's rule
        // is that a disabled widget shows only `on_disabled_hover_text` —
        // but inside an open menu egui shows no tooltip at all, because a
        // popup in the layer suppresses them
        // (`Tooltip::should_show_tooltip`). So a refusal explained only by
        // a disabled hover is explained nowhere, which a probe caught and
        // no amount of hovering by hand would have.
        if !can_add {
            ui.label(
                egui::RichText::new(if room.has_room() {
                    "Right-click the track to say where the note goes".to_owned()
                } else {
                    room.full_reason()
                })
                .small()
                .color(style.error),
            );
        }
        ui.separator();
        if ui
            .add_enabled(!asking, egui::Button::new("Remove this track"))
            .on_disabled_hover_text("Answer the question above the timeline first")
            .clicked()
        {
            actions.push(Action::RemoveTrack(track));
            ui.close();
        }
    });
    EditorResponse::NONE
}

/// One note's block and its full extent (the block plus its release tail).
fn note_geometry(event: &Event, lane_top: f32, ppb: f32, origin: f32) -> (Rect, Rect) {
    let left = origin + event.time_beats * ppb;
    let top = lane_top + 4.0;
    let height = LANE_H - 8.0;
    let body = Rect::from_min_size(
        Pos2::new(left, top),
        Vec2::new((event.gate_beats * ppb).max(6.0), height),
    );
    let tail = (event.release_beats * ppb).max(0.0);
    let extent = Rect::from_min_size(body.min, Vec2::new(body.width() + tail, height));
    (body, extent)
}

/// What one note's block says: the widest of [`note_labels`] that fits
/// inside it, or nothing.
fn paint_note_label(
    painter: &egui::Painter,
    body: Rect,
    event: &Event,
    missing: bool,
    error: Color32,
    style: &EditorStyle,
) {
    let colour = if missing { error } else { style.note_text };
    let font = egui::FontId::proportional(11.0);
    let draw = |galley: std::sync::Arc<egui::Galley>| {
        painter.galley(
            Pos2::new(
                body.left() + LABEL_PAD,
                body.center().y - 0.5 * galley.size().y,
            ),
            galley,
            colour,
        );
    };
    // A broken note says so whatever its width: the label is the only
    // thing that names the instrument nothing answers to, so it runs past
    // the block the way every label used to. Making it fit inside is
    // Overlands #1342, which is not this step's.
    if missing {
        draw(painter.layout_no_wrap(missing_label(&event.instrument_id), font, colour));
        return;
    }
    // The text starts [`LABEL_PAD`] inside the left edge and needs only
    // clearance at the right, so a 4-beat note's pitch fits at the zoom
    // that fits the whole sequence.
    let room = body.width() - LABEL_PAD - 1.0;
    for label in note_labels(event, false) {
        if label.is_empty() {
            continue;
        }
        let galley = painter.layout_no_wrap(label, font.clone(), colour);
        if galley.size().x <= room {
            draw(galley);
            return;
        }
    }
}

/// Move or resize every selected note by this frame's drag.
///
/// Read once for the whole timeline rather than inside the lane loop: a
/// multi-note move is one gesture, and asking each note for its own drag
/// would move only the one under the pointer.
fn drag_selection(
    ui: &egui::Ui,
    recipe: &mut SequenceRecipe,
    state: &mut SequenceEditorState,
    id: Id,
    ppb: f32,
) -> EditorResponse {
    let mut res = EditorResponse::NONE;
    // The widget the press landed on, which after an Alt-drag is not one of
    // the notes being moved.
    let Some(anchor) = state.drag_anchor else {
        return res;
    };
    let resp = ui.ctx().read_response(id.with(("ev", anchor.0, anchor.1)));
    let Some(resp) = resp else { return res };

    let picked: Vec<(usize, usize)> = state.selection.iter().copied().collect();
    if resp.dragged() {
        let dx = resp.drag_delta().x / ppb;
        for (ti, ei) in &picked {
            let Some(event) = recipe
                .tracks
                .get_mut(*ti)
                .and_then(|t| t.events.get_mut(*ei))
            else {
                continue;
            };
            match state.drag_mode {
                DragMode::Move | DragMode::Copy => {
                    event.time_beats = (event.time_beats + dx).max(0.0);
                }
                // A resize is one note's: dragging one edge of five
                // different gates is not a gesture anyone means.
                DragMode::Resize if (*ti, *ei) == anchor => {
                    event.gate_beats = (event.gate_beats + dx).max(MIN_GATE);
                }
                DragMode::Resize => {}
            }
        }
        res.changed = true;
    }
    if resp.drag_stopped() {
        for (ti, ei) in &picked {
            let Some(event) = recipe
                .tracks
                .get_mut(*ti)
                .and_then(|t| t.events.get_mut(*ei))
            else {
                continue;
            };
            match state.drag_mode {
                DragMode::Move | DragMode::Copy => {
                    event.time_beats = snap_beat(event.time_beats, state.snap);
                }
                DragMode::Resize if (*ti, *ei) == anchor => {
                    event.gate_beats = snap_beat(event.gate_beats, state.snap).max(MIN_GATE);
                }
                DragMode::Resize => {}
            }
        }
        state.drag_anchor = None;
        res.rebake = true;
    }
    res
}

/// Apply the edits the draw loop collected, now that it has let go of the
/// tracks.
fn apply_actions(
    recipe: &mut SequenceRecipe,
    state: &mut SequenceEditorState,
    actions: Vec<Action>,
    lane_names: &[Option<(String, usize)>],
) -> EditorResponse {
    let mut res = EditorResponse::NONE;
    let limits = state.limits;
    for action in actions {
        match action {
            Action::AddNote { track, beat } => {
                if track >= recipe.tracks.len() {
                    continue;
                }
                // The last word on the cap. The two doors that offer a note
                // are both disabled at it, but an Action is a frame old by
                // the time it lands, and the recipe can have changed under
                // it — a queued add must not be the one that goes over.
                if !limits
                    .at(Cap::TrackEvents, recipe.tracks[track].events.len())
                    .has_room()
                {
                    continue;
                }
                // The lane's own instrument, not the recipe's first: a note
                // added to the bass lane used to arrive as a `bed` (#62,
                // Overlands #1335 C6).
                let instrument_id = lane_names
                    .get(track)
                    .and_then(|n| n.as_ref().map(|(name, _)| name.clone()))
                    .or_else(|| recipe.instruments.first().map(|i| i.id.clone()))
                    .unwrap_or_default();
                recipe.tracks[track].events.push(Event {
                    time_beats: snap_beat(beat, state.snap),
                    instrument_id,
                    volume: 0.8,
                    ..Event::default()
                });
                state.select_only((track, recipe.tracks[track].events.len() - 1));
                res.changed = true;
                res.rebake = true;
            }
            Action::RemoveTrack(track) => {
                let Some(events) = recipe.tracks.get(track).map(|t| t.events.len()) else {
                    continue;
                };
                // An empty track is nothing to lose, so it goes at once. A
                // track with notes asks, and the question is drawn at the
                // top of the timeline next frame.
                if events == 0 {
                    remove_track(recipe, state, track);
                    res.changed = true;
                    res.rebake = true;
                } else {
                    state.pending_removal = Some(PendingRemoval::Track {
                        track,
                        notes: events,
                    });
                }
            }
            Action::RemoveTrackNow(track) => {
                state.pending_removal = None;
                if track < recipe.tracks.len() {
                    remove_track(recipe, state, track);
                    res.changed = true;
                    res.rebake = true;
                }
            }
            Action::DuplicateSelection => {
                if duplicate_selection(recipe, state) {
                    res.changed = true;
                    res.rebake = true;
                }
            }
        }
    }
    res
}

/// Take track `track` out of the recipe.
///
/// One place, so the gutter's cross, the track's menu and a confirmed
/// removal all leave the editor in the same state.
fn remove_track(recipe: &mut SequenceRecipe, state: &mut SequenceEditorState, track: usize) {
    recipe.tracks.remove(track);
    state.clear_selection();
}

/// Copy every selected note, and leave the copies selected.
///
/// The copies land on top of the originals rather than beside them: an
/// Alt-drag is about to carry them somewhere, and an offset the drag then
/// adds to would put them where neither the pointer nor the grid says. The
/// menu's Duplicate leaves them there to be dragged or nudged off.
fn duplicate_selection(recipe: &mut SequenceRecipe, state: &mut SequenceEditorState) -> bool {
    let mut picked: Vec<(usize, usize)> = state.selection.iter().copied().collect();
    picked.sort_unstable();
    if picked.is_empty() {
        return false;
    }
    let anchor = state.selected_event;
    let limits = state.limits;
    let mut copies: HashSet<(usize, usize)> = HashSet::new();
    let mut new_anchor = None;
    for (ti, ei) in picked {
        let Some(track) = recipe.tracks.get_mut(ti) else {
            continue;
        };
        let Some(event) = track.events.get(ei).cloned() else {
            continue;
        };
        // Held to the cap per track, copy by copy: a duplicate of a large
        // selection is the likeliest way to walk over it, and the ones that
        // fit still land rather than the whole gesture being refused.
        if !limits.at(Cap::TrackEvents, track.events.len()).has_room() {
            continue;
        }
        track.events.push(event);
        let at = (ti, track.events.len() - 1);
        if anchor == Some((ti, ei)) {
            new_anchor = Some(at);
        }
        copies.insert(at);
    }
    if copies.is_empty() {
        return false;
    }
    state.selected_event = new_anchor.or_else(|| copies.iter().copied().next());
    state.selection = copies;
    true
}

// ---------------------------------------------------------------------------
// Event inspector
// ---------------------------------------------------------------------------

fn event_inspector(
    ui: &mut egui::Ui,
    recipe: &mut SequenceRecipe,
    state: &mut SequenceEditorState,
    style: &EditorStyle,
) -> EditorResponse {
    let mut res = EditorResponse::NONE;

    let Some((ti, ei)) = state.selected_event else {
        ui.label(
            "No note picked. Click a note to edit it, shift-click or drag a box to pick \
             more, or double-click a lane to add one.",
        );
        return res;
    };
    if ti >= recipe.tracks.len() || ei >= recipe.tracks[ti].events.len() {
        state.clear_selection();
        ui.label("No note picked.");
        return res;
    }

    let inst_ids: Vec<String> = recipe.instruments.iter().map(|i| i.id.clone()).collect();
    let dur = recipe.duration_beats.max(1.0);
    let bpm = recipe.bpm;
    let picked = state.selection.len();
    let error = style.error;
    let mut delete = false;
    // Set by the reassign offer: every note of the first id takes the second.
    let mut reassign_all: Option<(String, String)> = None;
    let sharing_id = {
        let id = &recipe.tracks[ti].events[ei].instrument_id;
        recipe
            .tracks
            .iter()
            .flat_map(|t| &t.events)
            .filter(|e| e.instrument_id == *id)
            .count()
    };

    {
        let ev = &mut recipe.tracks[ti].events[ei];
        // Counted from one, the way the lanes are named in their gutters
        // and the way anyone reading it counts (#62, Overlands #1335 C5).
        ui.label(
            egui::RichText::new(format!("Track {} \u{00B7} note {}", ti + 1, ei + 1)).strong(),
        );
        if picked > 1 {
            ui.label(
                egui::RichText::new(format!(
                    "{picked} notes picked \u{2014} a drag, an arrow or Ctrl+D moves or copies \
                     all of them; the fields below edit this one"
                ))
                .weak()
                .small(),
            );
        }
        let missing = !inst_ids.contains(&ev.instrument_id);
        if missing {
            ui.colored_label(error, no_instrument_text(&ev.instrument_id));
        }

        ui.horizontal_wrapped(|ui| {
            egui::ComboBox::from_id_salt("evt_instrument")
                .selected_text(if missing {
                    egui::RichText::new(missing_label(&ev.instrument_id)).color(error)
                } else {
                    egui::RichText::new(ev.instrument_id.as_str())
                })
                .show_ui(ui, |ui| {
                    for name in &inst_ids {
                        if ui
                            .selectable_label(ev.instrument_id == *name, name)
                            .clicked()
                            && ev.instrument_id != *name
                        {
                            ev.instrument_id = name.clone();
                            res.changed = true;
                            res.rebake = true;
                        }
                    }
                })
                .response
                .on_hover_text("The instrument this note plays");
            if missing {
                let whose = if ev.instrument_id.is_empty() {
                    "Reassign all notes with no instrument to\u{2026}".to_owned()
                } else {
                    format!("Reassign all notes of '{}' to\u{2026}", ev.instrument_id)
                };
                ui.add_enabled_ui(!inst_ids.is_empty(), |ui| {
                    ui.menu_button(whose, |ui| {
                        for name in &inst_ids {
                            if ui.button(name).clicked() {
                                reassign_all = Some((ev.instrument_id.clone(), name.clone()));
                                ui.close();
                            }
                        }
                    })
                    .response
                    .on_hover_text(format!(
                        "All {sharing_id} note(s) that name it, on every track, not only this one"
                    ));
                })
                .response
                .on_disabled_hover_text("There are no instruments yet: add one above");
            }
        });

        res.merge(beat_field(
            ui,
            "Start",
            "Where the note begins, in beats from the start of the sequence",
            &mut ev.time_beats,
            0.05,
            0.0..=dur * 4.0,
            bpm,
        ));
        res.merge(beat_field(
            ui,
            "Hold",
            "How long the note is held down. The instrument's envelope \
             sustains for this long and then starts to release",
            &mut ev.gate_beats,
            0.05,
            MIN_GATE..=dur * 4.0,
            bpm,
        ));
        res.merge(beat_field(
            ui,
            "Tail",
            "Extra time baked after the note is let go, for the release to \
             ring out. 0 cuts it dead",
            &mut ev.release_beats,
            0.05,
            0.0..=32.0,
            bpm,
        ));
        ui.horizontal(|ui| {
            ui.label("Pitch").on_hover_text(
                "How far the note is shifted from the instrument's own \
                 pitch: 2 is an octave up, 0.5 an octave down",
            );
            res.merge(ranged_slider(
                ui,
                &mut ev.pitch_multiplier,
                0.25..=4.0,
                |s| s.logarithmic(true).show_value(true),
            ));
        });
        // Pitch mode: tape varispeed (pitch and time coupled) vs.
        // synthesis-time retune (the note keeps its slot whatever its
        // pitch). Said as what happens, not as the technique's name.
        ui.horizontal_wrapped(|ui| {
            ui.label("Pitch mode");
            for (variant, label, hover) in [
                (
                    PitchMode::Varispeed,
                    "pitch and length linked",
                    "Resampled like tape: pitching it up makes it shorter, \
                     and down makes it longer",
                ),
                (
                    PitchMode::TimePreserving,
                    "pitch and length independent",
                    "Retuned as it is synthesised: the note keeps its Hold \
                     whatever its pitch",
                ),
            ] {
                let selected = ev.pitch_mode == variant;
                if ui
                    .selectable_label(selected, label)
                    .on_hover_text(hover)
                    .clicked()
                    && !selected
                {
                    ev.pitch_mode = variant;
                    res.changed = true;
                    res.rebake = true;
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("Volume")
                .on_hover_text("How loud the note is in the mix, and how solid its block is drawn");
            res.merge(ranged_slider(ui, &mut ev.volume, 0.0..=1.0, |s| {
                s.show_value(true)
            }));
        });

        if ui
            .button(if picked > 1 {
                format!("Delete {picked} notes")
            } else {
                "Delete note".to_owned()
            })
            .clicked()
        {
            delete = true;
        }
    }

    if let Some((from, to)) = reassign_all {
        reassign_notes(recipe, &from, &to);
        res.changed = true;
        res.rebake = true;
    }
    if delete {
        // Every picked note, highest index first so the removals do not
        // move each other's indices out from under them.
        let mut picked: Vec<(usize, usize)> = state.selection.iter().copied().collect();
        picked.sort_unstable_by(|a, b| b.cmp(a));
        for (track, event) in picked {
            if let Some(track) = recipe.tracks.get_mut(track)
                && event < track.events.len()
            {
                track.events.remove(event);
            }
        }
        state.clear_selection();
        res.changed = true;
        res.rebake = true;
    }

    res
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::Envelope;
    use crate::ui::CapTone;
    use crate::ui::test_paint::text_painted;

    fn recipe_with(instruments: &[&str], tracks: usize) -> SequenceRecipe {
        SequenceRecipe {
            instruments: instruments
                .iter()
                .map(|id| Instrument {
                    id: (*id).to_string(),
                    patch: AudioPatch::default(),
                })
                .collect(),
            tracks: (0..tracks).map(|_| Track::default()).collect(),
            ..SequenceRecipe::default()
        }
    }

    /// Every division the picker offers, at the value it rounds to, past
    /// the end of the sequence, and below zero.
    ///
    /// Walked from [`Snap::ALL`], so a division added to the picker cannot
    /// go untested: it is the same array the combo is built from.
    #[test]
    fn snapping_rounds_to_every_division_and_never_below_zero() {
        let cases = [
            (Snap::Off, 1.61, 1.61),
            (Snap::Eighth, 1.61, 1.625),
            (Snap::Quarter, 1.61, 1.5),
            (Snap::Beat, 1.61, 2.0),
            (Snap::Bar, 1.61, 0.0),
            // Halfway rounds up on every grid, the way `f32::round` does.
            (Snap::Eighth, 0.0625, 0.125),
            (Snap::Quarter, 0.125, 0.25),
            (Snap::Beat, 0.5, 1.0),
            (Snap::Bar, 2.0, 4.0),
            // Past the end is left alone: the timeline grows to hold it.
            (Snap::Off, 400.4, 400.4),
            (Snap::Bar, 400.4, 400.0),
            (Snap::Beat, 400.4, 400.0),
            // Below zero is not: a note before beat 0 cannot be baked.
            (Snap::Off, -3.0, 0.0),
            (Snap::Eighth, -3.0, 0.0),
            (Snap::Quarter, -3.0, 0.0),
            (Snap::Beat, -3.0, 0.0),
            (Snap::Bar, -3.0, 0.0),
            (Snap::Quarter, -0.01, 0.0),
        ];
        for (snap, beats, want) in cases {
            let got = snap_beat(beats, snap);
            assert!(
                (got - want).abs() < 1e-4,
                "{snap:?} put {beats} at {got}, not {want}"
            );
        }
        // The quarter grid every drag snapped to before there was a choice
        // is still the default, so nothing moved for anyone who never opens
        // the picker.
        assert_eq!(Snap::default(), Snap::Quarter);
        for snap in Snap::ALL {
            assert!(snap_beat(-1.0, snap) >= 0.0, "{snap:?} allowed a negative");
            assert!(snap.step() > 0.0, "{snap:?} has no nudge step");
            assert!(!snap.label().is_empty(), "{snap:?} has no label");
        }
    }

    #[test]
    fn unique_instrument_id_avoids_collisions() {
        let recipe = recipe_with(&["inst1", "inst2", "drum"], 0);
        assert_eq!(unique_instrument_id(&recipe), "inst3");
    }

    #[test]
    fn active_instrument_accessor_reflects_state() {
        let mut state = SequenceEditorState::default();
        assert_eq!(state.active_instrument(), None);
        state.active_instrument = Some(2);
        assert_eq!(state.active_instrument(), Some(2));
    }

    #[test]
    fn editor_renders_headless_without_panicking() {
        let mut recipe = recipe_with(&["wind", "kick"], 2);
        recipe.tracks[0].events.push(Event {
            time_beats: 0.0,
            instrument_id: "wind".into(),
            gate_beats: 4.0,
            ..Event::default()
        });
        recipe.loop_start_beats = Some(2.0);
        recipe.loop_crossfade_beats = 1.0;
        let mut state = SequenceEditorState {
            active_instrument: Some(0),
            ..Default::default()
        };

        let ctx = egui::Context::default();
        for _ in 0..3 {
            let _ = ctx.run_ui(egui::RawInput::default(), |root| {
                egui::Panel::left("seq").show(root, |ui| {
                    sequence_recipe_editor(ui, &mut recipe, &mut state, Id::new("seq"));
                });
                egui::CentralPanel::default().show(root, |ui| {
                    active_instrument_canvas(ui, &mut recipe, &mut state, Id::new("seq_canvas"));
                });
            });
        }
    }

    #[test]
    fn empty_recipe_renders_headless_without_panicking() {
        let mut recipe = SequenceRecipe::default();
        let mut state = SequenceEditorState::default();
        let ctx = egui::Context::default();
        let _ = ctx.run_ui(egui::RawInput::default(), |root| {
            egui::CentralPanel::default().show(root, |ui| {
                sequence_recipe_editor(ui, &mut recipe, &mut state, Id::new("seq"));
            });
        });
    }

    // -----------------------------------------------------------------------
    // Renaming instruments and missing instruments (#55, Overlands #1328)
    // -----------------------------------------------------------------------
    //
    // These drive the editor through its real widgets, the way a user does:
    // a name field is found by the text it shows (AccessKit), focused, and
    // typed into, and what the editor paints is read back from its shapes.
    // Nothing here reaches past the widgets to call the rename directly, so
    // the same tests ran against the field that edited `inst.id` in place.

    use egui::accesskit;

    use crate::ui::style::set_editor_style;
    use crate::ui::style::tests::{AA, distinct_style};
    use crate::ui::test_paint::{button_labels, colours, contrast_on, glyph_labels, shapes};

    /// A note of `instrument` at `time_beats`.
    fn note(instrument: &str, time_beats: f32) -> Event {
        Event {
            time_beats,
            instrument_id: instrument.into(),
            gate_beats: 0.5,
            ..Event::default()
        }
    }

    /// `inst1` and `inst2`, with `inst1` playing on two of three tracks.
    fn two_instrument_recipe() -> SequenceRecipe {
        let mut recipe = recipe_with(&["inst1", "inst2"], 3);
        recipe.tracks[0].events = vec![note("inst1", 0.0), note("inst2", 1.0)];
        recipe.tracks[1].events = vec![note("inst1", 2.0), note("inst1", 3.0)];
        recipe.tracks[2].events = vec![note("inst2", 0.0)];
        recipe
    }

    /// Every event's instrument id, track by track.
    fn note_ids(recipe: &SequenceRecipe) -> Vec<Vec<&str>> {
        recipe
            .tracks
            .iter()
            .map(|t| t.events.iter().map(|e| e.instrument_id.as_str()).collect())
            .collect()
    }

    fn instrument_ids(recipe: &SequenceRecipe) -> Vec<&str> {
        recipe.instruments.iter().map(|i| i.id.as_str()).collect()
    }

    fn key(key: egui::Key) -> egui::Event {
        egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }
    }

    /// `key` with `modifiers` held: what the editor's own shortcuts read.
    fn chord(key: egui::Key, modifiers: egui::Modifiers) -> egui::Event {
        egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers,
        }
    }

    fn typed(text: &str) -> egui::Event {
        egui::Event::Text(text.into())
    }

    fn backspaces(n: usize) -> Vec<egui::Event> {
        (0..n).map(|_| key(egui::Key::Backspace)).collect()
    }

    /// The text of every shape in `shapes`, nested ones included.
    fn shape_texts(shapes: &[egui::epaint::ClippedShape]) -> Vec<String> {
        fn walk(shape: &egui::Shape, out: &mut Vec<String>) {
            match shape {
                egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| walk(s, out)),
                egui::Shape::Text(text) => out.push(text.galley.text().to_owned()),
                _ => {}
            }
        }
        let mut out = Vec::new();
        for clipped in shapes {
            walk(&clipped.shape, &mut out);
        }
        out
    }

    /// The sequence editor on a headless context, laid out the way the
    /// `host_window` example lays it out: the editor in a left panel, the
    /// active instrument's canvas in the central panel.
    struct Driver {
        ctx: egui::Context,
        recipe: SequenceRecipe,
        state: SequenceEditorState,
        out: egui::FullOutput,
        /// The side panel's own rect, as the editor was given it: what the
        /// timeline's fit-to-width has to fit inside.
        panel: Rect,
    }

    impl Driver {
        fn new(recipe: SequenceRecipe) -> Self {
            Self::with_state(recipe, SequenceEditorState::default())
        }

        /// With [`distinct_style`] set, so each role the editor paints has
        /// a colour nothing else does.
        fn with_state(recipe: SequenceRecipe, state: SequenceEditorState) -> Self {
            let ctx = egui::Context::default();
            set_editor_style(&ctx, distinct_style());
            Self::on(ctx, recipe, state)
        }

        /// Under `visuals`, with no style set.
        fn themed(
            recipe: SequenceRecipe,
            state: SequenceEditorState,
            visuals: egui::Visuals,
        ) -> Self {
            let ctx = egui::Context::default();
            ctx.set_visuals(visuals);
            Self::on(ctx, recipe, state)
        }

        fn on(ctx: egui::Context, recipe: SequenceRecipe, state: SequenceEditorState) -> Self {
            ctx.enable_accesskit();
            // A hover's own text is half of what this step draws, and a
            // test that waits out egui's half-second would be a test of the
            // clock.
            ctx.all_styles_mut(|s| {
                s.interaction.tooltip_delay = 0.0;
                s.interaction.tooltip_grace_time = 0.0;
            });
            let mut driver = Self {
                ctx,
                recipe,
                state,
                out: egui::FullOutput::default(),
                panel: Rect::NOTHING,
            };
            // A new area's first pass is an invisible sizing pass; act from
            // the third frame on.
            driver.frame(Vec::new());
            driver.frame(Vec::new());
            driver
        }

        /// One frame with `events` as its input; the editor's response.
        fn frame(&mut self, events: Vec<egui::Event>) -> EditorResponse {
            self.frame_held(egui::Modifiers::NONE, events)
        }

        /// One frame with `modifiers` held down as the platform reports
        /// them.
        ///
        /// `InputState::modifiers` comes from the raw input and not from an
        /// event's own copy of them, so a Shift-arrow needs both: the event
        /// carries the chord that is matched, and the raw modifiers are
        /// what the fine step is read from.
        fn frame_held(
            &mut self,
            modifiers: egui::Modifiers,
            events: Vec<egui::Event>,
        ) -> EditorResponse {
            let Self {
                ctx,
                recipe,
                state,
                out,
                panel,
            } = self;
            let mut res = EditorResponse::NONE;
            let input = egui::RawInput {
                events,
                modifiers,
                ..Default::default()
            };
            *out = ctx.run_ui(input, |root| {
                egui::Panel::left("seq")
                    .default_size(480.0)
                    .show(root, |ui| {
                        *panel = ui.max_rect();
                        res.merge(sequence_recipe_editor(ui, recipe, state, Id::new("seq")));
                    });
                egui::CentralPanel::default().show(root, |ui| {
                    res.merge(active_instrument_canvas(
                        ui,
                        recipe,
                        state,
                        Id::new("seq_canvas"),
                    ));
                });
            });
            res
        }

        fn nodes(&self) -> impl Iterator<Item = (accesskit::NodeId, &accesskit::Node)> {
            self.out
                .platform_output
                .accesskit_update
                .iter()
                .flat_map(|update| update.nodes.iter().map(|(id, node)| (*id, node)))
        }

        /// The values of every single-line text field, top to bottom.
        fn fields(&self) -> Vec<String> {
            let mut fields: Vec<(f64, String)> = self
                .nodes()
                .filter(|(_, n)| n.role() == accesskit::Role::TextInput)
                .map(|(_, n)| {
                    let top = n.bounds().map_or(f64::MAX, |b| b.y0);
                    (top, n.value().unwrap_or_default().to_owned())
                })
                .collect();
            fields.sort_by(|a, b| a.0.total_cmp(&b.0));
            fields.into_iter().map(|(_, v)| v).collect()
        }

        /// The topmost single-line text field showing exactly `text`.
        fn field(&self, text: &str) -> Option<accesskit::NodeId> {
            self.nodes()
                .filter(|(_, n)| n.role() == accesskit::Role::TextInput && n.value() == Some(text))
                .min_by(|a, b| {
                    let top = |n: &accesskit::Node| n.bounds().map_or(f64::MAX, |b| b.y0);
                    top(a.1).total_cmp(&top(b.1))
                })
                .map(|(id, _)| id)
        }

        /// The topmost button whose label starts with `prefix`.
        fn button(&self, prefix: &str) -> Option<accesskit::NodeId> {
            self.nodes()
                .filter(|(_, n)| {
                    n.role() == accesskit::Role::Button
                        && n.label().is_some_and(|l| l.starts_with(prefix))
                })
                .min_by(|a, b| {
                    let top = |n: &accesskit::Node| n.bounds().map_or(f64::MAX, |b| b.y0);
                    top(a.1).total_cmp(&top(b.1))
                })
                .map(|(id, _)| id)
        }

        fn act(&mut self, target: accesskit::NodeId, action: accesskit::Action) -> EditorResponse {
            self.frame(vec![egui::Event::AccessKitActionRequest(
                accesskit::ActionRequest {
                    action,
                    target_tree: accesskit::TreeId::ROOT,
                    target_node: target,
                    data: None,
                },
            )])
        }

        /// Focus the topmost text field showing `text`, as a click into it
        /// does; the cursor lands at the end.
        fn focus(&mut self, text: &str) {
            let field = self
                .field(text)
                .unwrap_or_else(|| panic!("no field shows {text:?}; fields: {:?}", self.fields()));
            self.act(field, accesskit::Action::Focus);
            assert!(
                self.ctx.memory(|m| m.focused()).is_some(),
                "the field showing {text:?} did not take the focus"
            );
        }

        /// Click the first button whose label starts with `prefix`.
        fn click(&mut self, prefix: &str) -> EditorResponse {
            let button = self
                .button(prefix)
                .unwrap_or_else(|| panic!("no button labelled {prefix:?}\u{2026}"));
            self.act(button, accesskit::Action::Click)
        }

        /// Put the pointer on the timeline and leave it there.
        ///
        /// The editor takes the keyboard only while it has the pointer and
        /// nothing holds text focus (#60), so a test of an arrow or a
        /// Ctrl+D has to point at it first — with no pointer anywhere, the
        /// keys belong to the host.
        fn point_at_timeline(&mut self) {
            // The *visible* part of it: zoomed in, the timeline is several
            // times the panel's width and its centre is off the panel
            // altogether, where the pointer is over nothing.
            let at = self.state.timeline_rect().intersect(self.panel).center();
            self.frame(vec![egui::Event::PointerMoved(at)]);
        }

        /// Where a removal was asked from: `instrument(row)` and
        /// `track(index)` into the rects the editor published.
        ///
        /// Read from [`SequenceEditorState::removal_asks`] rather than
        /// worked out here: a gutter cross is painted geometry, and a test
        /// that re-derives it is testing its own arithmetic.
        fn ask_rect(&self, instrument: Option<usize>, track: Option<usize>) -> Rect {
            let asks = self.state.removal_asks();
            let instruments = self.recipe.instruments.len();
            let i = match (instrument, track) {
                (Some(row), _) => row,
                (_, Some(t)) => instruments + t,
                _ => unreachable!(),
            };
            *asks.get(i).unwrap_or_else(|| {
                panic!(
                    "nothing published for {instrument:?}/{track:?}; {} asks",
                    asks.len()
                )
            })
        }

        /// Click at `at` with `button`, as a user does: move, press,
        /// release.
        fn click_at(&mut self, at: Pos2, button: egui::PointerButton) -> EditorResponse {
            self.frame(vec![egui::Event::PointerMoved(at)]);
            let mut res = EditorResponse::NONE;
            for pressed in [true, false] {
                res.merge(self.frame(vec![egui::Event::PointerButton {
                    pos: at,
                    button,
                    pressed,
                    modifiers: egui::Modifiers::NONE,
                }]));
            }
            res.merge(self.frame(Vec::new()));
            res
        }

        /// Press the cross in track `track`'s gutter.
        fn remove_track_by_cross(&mut self, track: usize) -> EditorResponse {
            let at = self.ask_rect(None, Some(track)).center();
            self.click_at(at, egui::PointerButton::Primary)
        }

        /// Press the cross on instrument `row`'s line.
        fn remove_instrument_by_cross(&mut self, row: usize) -> EditorResponse {
            let at = self.ask_rect(Some(row), None).center();
            self.click_at(at, egui::PointerButton::Primary)
        }

        /// Open track `track`'s menu on clear ground and choose an item
        /// whose label starts with `prefix`.
        ///
        /// The pointer is moved *into* the menu first, as anyone choosing
        /// from it must: an item that re-reads the lane under the pointer
        /// greys out the instant it can be reached (#62).
        fn open_track_menu(&mut self, track: usize) {
            // Clear ground: to the right of every note the track holds,
            // found from the published geometry rather than guessed at a
            // beat the recipe might have filled.
            let visible = self.state.timeline_rect().intersect(self.panel);
            let past_notes = self
                .state
                .notes()
                .iter()
                .filter(|n| n.track == track)
                .map(|n| n.extent.right())
                .fold(visible.left() + GUTTER, f32::max);
            let at = Pos2::new(
                (past_notes + 12.0).min(visible.right() - 8.0),
                visible.top() + RULER_H + (track as f32 + 0.5) * LANE_H,
            );
            self.frame(vec![egui::Event::PointerMoved(at)]);
            for pressed in [true, false] {
                self.frame(vec![egui::Event::PointerButton {
                    pos: at,
                    button: egui::PointerButton::Secondary,
                    pressed,
                    modifiers: egui::Modifiers::NONE,
                }]);
            }
            // The menu is an Area of its own, painted the frame after the
            // click that opened it.
            self.frame(Vec::new());
            // Into the menu, as anyone choosing from it must (#62).
            self.frame(vec![egui::Event::PointerMoved(at + Vec2::new(14.0, 18.0))]);
        }

        fn remove_track_by_menu(&mut self, track: usize) -> EditorResponse {
            self.open_track_menu(track);
            let res = self.click("Remove this track");
            self.frame(Vec::new());
            res
        }

        fn add_note_by_menu(&mut self, track: usize) -> EditorResponse {
            self.open_track_menu(track);
            let res = self.click("Add a");
            self.frame(Vec::new());
            res
        }

        /// Rest the pointer on the topmost control whose label starts with
        /// `prefix` until its tooltip opens, and return what was painted.
        ///
        /// A disabled widget never shows `on_hover_text` and only ever
        /// shows `on_disabled_hover_text` (#1289), so the only honest way
        /// to ask whether a refusal explains itself is to hover it.
        fn hover_texts(&mut self, prefix: &str) -> Vec<String> {
            let node = self
                .button(prefix)
                .unwrap_or_else(|| panic!("no control labelled {prefix:?}\u{2026}"));
            let b = self
                .nodes()
                .find(|(id, _)| *id == node)
                .and_then(|(_, n)| n.bounds())
                .expect("the control has no bounds");
            let at = Pos2::new(0.5 * (b.x0 + b.x1) as f32, 0.5 * (b.y0 + b.y1) as f32);
            // egui hides tooltips while a click is more recent than the
            // last pointer movement — "it is common to click a widget and
            // then rest the mouse there" (Tooltip::should_show_tooltip). A
            // menu is opened by a click, so the click has to be let go
            // stale *before* the pointer moves onto the item, or the
            // reason never appears and the test reads that as a missing
            // hover. A dozen frames is a fifth of a second at the harness's
            // predicted frame time; the gate needs a tenth.
            for _ in 0..12 {
                self.frame(Vec::new());
            }
            self.frame(vec![egui::Event::PointerMoved(at)]);
            // And egui opens a tooltip on the frame after the hover is
            // established, not on the frame the pointer arrives.
            self.frame(Vec::new());
            self.frame(Vec::new());
            self.texts()
        }

        /// The text of everything the last frame painted.
        fn texts(&self) -> Vec<String> {
            shape_texts(&self.out.shapes)
        }

        fn paints(&self, needle: &str) -> bool {
            self.texts().iter().any(|t| t.contains(needle))
        }
    }

    #[test]
    fn renaming_an_instrument_moves_its_notes_on_every_track_and_its_canvas_layout() {
        let mut state = SequenceEditorState::default();
        state.set_active_instrument(Some(0));
        let mut driver = Driver::with_state(two_instrument_recipe(), state);
        // The canvas laid inst1's nodes out while it was open; close it so
        // nothing redraws that layout under the rename.
        let layout = format!("{:?}", driver.state.canvas_states["inst1"]);
        driver.state.set_active_instrument(None);
        driver.frame(Vec::new());

        driver.focus("inst1");
        driver.frame(backspaces(5));
        driver.frame(vec![typed("lead")]);
        let res = driver.frame(vec![key(egui::Key::Enter)]);

        assert_eq!(instrument_ids(&driver.recipe), ["lead", "inst2"]);
        assert_eq!(
            note_ids(&driver.recipe),
            [vec!["lead", "inst2"], vec!["lead", "lead"], vec!["inst2"]],
            "every note of the renamed instrument follows it, on every track"
        );
        assert!(res.changed && res.rebake, "the rename commits: {res:?}");
        let keys: Vec<&String> = driver.state.canvas_states.keys().collect();
        assert_eq!(keys, ["lead"], "the canvas layout is re-keyed, not dropped");
        assert_eq!(format!("{:?}", driver.state.canvas_states["lead"]), layout);
    }

    #[test]
    fn a_rename_commits_when_the_field_loses_focus() {
        let mut driver = Driver::new(two_instrument_recipe());
        driver.focus("inst1");
        driver.frame(vec![typed("x")]);
        // Tab moves the focus on at the end of its frame, so the field sees
        // the loss on the next one.
        driver.frame(vec![key(egui::Key::Tab)]);
        driver.frame(Vec::new());

        assert_eq!(instrument_ids(&driver.recipe), ["inst1x", "inst2"]);
        assert_eq!(
            note_ids(&driver.recipe),
            [
                vec!["inst1x", "inst2"],
                vec!["inst1x", "inst1x"],
                vec!["inst2"]
            ]
        );
    }

    #[test]
    fn empty_duplicate_and_over_long_names_are_refused_with_a_reason() {
        // "é" is two bytes: 62 of them after "inst1" is 129 bytes in only 67
        // characters, so the limit is counted in bytes, as the sanitiser
        // counts it.
        let over_long = format!("inst1{}", "\u{E9}".repeat(62));
        let cases: [(&str, Vec<Vec<egui::Event>>, &str, &str); 4] = [
            ("empty", vec![backspaces(5)], "", "empty"),
            (
                "blank",
                vec![backspaces(5), vec![typed("   ")]],
                "   ",
                "empty",
            ),
            (
                "duplicate",
                vec![backspaces(1), vec![typed("2")]],
                "inst2",
                "Another instrument is called 'inst2'",
            ),
            (
                "over-long",
                vec![vec![typed(&"\u{E9}".repeat(62))]],
                &over_long,
                "the limit is 128",
            ),
        ];
        for (case, keystrokes, shown, reason) in cases {
            let mut driver = Driver::new(two_instrument_recipe());
            let before = driver.recipe.clone();
            driver.focus("inst1");
            for events in keystrokes {
                driver.frame(events);
            }
            assert!(
                driver.paints(reason),
                "{case}: the reason shows while typing"
            );
            let res = driver.frame(vec![key(egui::Key::Enter)]);
            driver.frame(Vec::new());

            assert_eq!(
                driver.recipe, before,
                "{case}: a refused name changes nothing"
            );
            assert_eq!(res, EditorResponse::NONE, "{case}: nothing to write back");
            assert!(
                driver.paints(reason),
                "{case}: the reason stays after the refusal; painted: {:?}",
                driver.texts()
            );
            assert_eq!(
                driver.fields()[0],
                shown,
                "{case}: the field keeps what was typed, to fix or to Esc"
            );

            // Esc puts the name back and takes the reason away.
            driver.focus(shown);
            driver.frame(vec![key(egui::Key::Escape)]);
            driver.frame(Vec::new());
            assert_eq!(driver.fields()[0], "inst1", "{case}: Esc restores the name");
            assert!(!driver.paints(reason), "{case}: Esc clears the reason");
            assert_eq!(driver.recipe, before, "{case}: Esc changes nothing");
        }
    }

    #[test]
    fn typing_through_a_colliding_name_moves_no_notes() {
        // "inst1" -> "inst" -> "inst2" -> "inst23", with inst2 in the recipe.
        // Were each keystroke applied, "inst2" would merge inst1's notes into
        // inst2's and nothing could tell them apart again.
        let mut driver = Driver::new(two_instrument_recipe());
        let before = driver.recipe.clone();
        driver.focus("inst1");
        for step in [backspaces(1), vec![typed("2")], vec![typed("3")]] {
            let res = driver.frame(step);
            assert_eq!(driver.recipe, before, "nothing moves while typing");
            assert!(!res.changed, "typing is not an edit of the recipe");
        }
        driver.frame(vec![key(egui::Key::Enter)]);

        assert_eq!(instrument_ids(&driver.recipe), ["inst23", "inst2"]);
        assert_eq!(
            note_ids(&driver.recipe),
            [
                vec!["inst23", "inst2"],
                vec!["inst23", "inst23"],
                vec!["inst2"]
            ],
            "only inst1's notes follow it; inst2 keeps its own"
        );
    }

    #[test]
    fn esc_restores_the_name_and_applies_nothing() {
        let mut driver = Driver::new(two_instrument_recipe());
        let before = driver.recipe.clone();
        driver.focus("inst1");
        driver.frame(vec![typed("zzz")]);
        assert_eq!(driver.recipe, before);
        driver.frame(vec![key(egui::Key::Escape)]);
        for _ in 0..3 {
            driver.frame(Vec::new());
        }
        assert_eq!(driver.recipe, before, "nothing is applied later either");
        assert_eq!(driver.fields()[..2], ["inst1", "inst2"]);
    }

    /// One sine at `freq_hz`: an instrument that is audible in a bake.
    fn sine(freq_hz: f32) -> AudioPatch {
        use crate::{GraphNode, NodeGraph, NodeId, NodeKind, SineOsc};
        AudioPatch {
            seed: 0,
            graph: NodeGraph {
                nodes: vec![GraphNode {
                    id: NodeId(0),
                    kind: NodeKind::Sine(SineOsc {
                        freq_hz,
                        phase_offset: 0.0,
                        amplitude: 0.5,
                    }),
                    inputs: Default::default(),
                }],
                output: NodeId(0),
            },
        }
    }

    #[test]
    fn a_rename_is_a_pure_relabel_of_the_mixdown() {
        use crate::mixdown::bake_sequence;
        let mut recipe = two_instrument_recipe();
        recipe.instruments[0].patch = sine(330.0);
        recipe.instruments[1].patch = sine(495.0);
        recipe.sample_rate = 8_000;
        recipe.bpm = 120.0;
        recipe.duration_beats = 4.0;
        let bits = |samples: Vec<f32>| samples.into_iter().map(f32::to_bits).collect::<Vec<_>>();
        let before = bits(bake_sequence(&recipe));

        // The control: a rename that leaves the notes behind silences them,
        // so this bake can see a rename that is not a pure relabel.
        let mut in_place = recipe.clone();
        in_place.instruments[0].id = "lead".into();
        assert_ne!(bits(bake_sequence(&in_place)), before);

        let mut driver = Driver::new(recipe);
        driver.focus("inst1");
        driver.frame(backspaces(5));
        driver.frame(vec![typed("lead")]);
        driver.frame(vec![key(egui::Key::Enter)]);
        assert_eq!(instrument_ids(&driver.recipe), ["lead", "inst2"]);
        assert_eq!(
            bits(bake_sequence(&driver.recipe)),
            before,
            "the renamed recipe bakes sample for sample the same"
        );
    }

    /// `wind` and `kick`, with notes of `ghost` on two tracks: an instrument
    /// that was renamed or removed while notes still used it.
    fn ghost_recipe() -> SequenceRecipe {
        let mut recipe = recipe_with(&["wind", "kick"], 2);
        recipe.tracks[0].events = vec![note("ghost", 0.0), note("wind", 1.0)];
        recipe.tracks[1].events = vec![note("ghost", 2.0)];
        recipe
    }

    #[test]
    fn a_note_naming_no_instrument_is_marked_and_offers_a_reassign() {
        let mut state = SequenceEditorState::default();
        state.set_selected_event(Some((0, 0)));
        let mut driver = Driver::with_state(ghost_recipe(), state);

        assert!(
            driver.paints("missing: ghost"),
            "the block says whose note it is; painted: {:?}",
            driver.texts()
        );
        assert!(
            !driver.paints("missing: wind"),
            "a note whose instrument exists is not marked"
        );
        assert!(driver.paints("No instrument called 'ghost'"));
        assert!(driver.paints("Reassign all notes of 'ghost' to"));

        driver.click("Reassign all notes of 'ghost' to");
        // The menu is a new area: its first frame is a sizing pass.
        driver.frame(Vec::new());
        driver.frame(Vec::new());
        let res = driver.click("kick");
        assert_eq!(
            note_ids(&driver.recipe),
            [vec!["kick", "wind"], vec!["kick"]],
            "every ghost note, on every track, and nothing else"
        );
        assert!(res.changed && res.rebake, "{res:?}");
        driver.frame(Vec::new());
        assert!(!driver.paints("missing:"), "no note is missing any more");
    }

    #[test]
    fn missing_instrument_notes_render_headless_without_panicking() {
        // No instruments at all, a note with an empty id and one with an id
        // nothing answers to, each open in the inspector in turn.
        let mut recipe = recipe_with(&[], 1);
        recipe.tracks[0].events = vec![note("", 0.0), note("ghost", 1.0)];
        for selected in [(0, 0), (0, 1)] {
            let mut state = SequenceEditorState::default();
            state.set_selected_event(Some(selected));
            let mut driver = Driver::with_state(recipe.clone(), state);
            driver.frame(Vec::new());
            assert!(driver.paints("missing:"), "{selected:?}");
        }
    }

    #[test]
    fn a_refused_name_moves_with_its_row_and_never_applies_itself() {
        let mut driver = Driver::new(two_instrument_recipe());
        // inst2 typed to "inst1": refused while inst1 exists.
        driver.focus("inst2");
        driver.frame(backspaces(1));
        driver.frame(vec![typed("1")]);
        driver.frame(vec![key(egui::Key::Enter)]);
        assert!(driver.paints("Another instrument is called 'inst1'"));

        // Removing inst1 frees the name and moves inst2's row up; the typed
        // name moves with it but waits for the user.
        //
        // Notes play inst1, so the cross asks rather than removes (#63,
        // Overlands #1336 C2) and the removal is the answer to it.
        driver.click("\u{2716}");
        driver.frame(Vec::new());
        driver.click("Remove");
        for _ in 0..3 {
            driver.frame(Vec::new());
        }
        assert_eq!(instrument_ids(&driver.recipe), ["inst2"]);
        assert_eq!(driver.fields()[0], "inst1", "the row keeps what was typed");
        assert!(driver.paints("Not applied yet"));
        assert!(
            driver.paints("missing: inst1"),
            "the removed instrument's notes are marked"
        );

        // Enter in the field applies it: inst2's notes follow, and the notes
        // the removal left behind find an instrument by that name again.
        driver.focus("inst1");
        driver.frame(vec![key(egui::Key::Enter)]);
        driver.frame(Vec::new());
        assert_eq!(instrument_ids(&driver.recipe), ["inst1"]);
        assert_eq!(
            note_ids(&driver.recipe),
            [
                vec!["inst1", "inst1"],
                vec!["inst1", "inst1"],
                vec!["inst1"]
            ]
        );
        assert!(!driver.paints("missing:"));
        assert!(!driver.paints("Not applied yet"));
    }

    #[test]
    fn names_are_trimmed_and_limited_to_symbios_audios_byte_bound() {
        assert_eq!(
            EditorLimits::default().get(Cap::InstrumentIdBytes),
            128,
            "the bound Overlands' sanitiser pins"
        );
        let recipe = recipe_with(&["inst1", "inst2"], 0);
        let at_limit = "a".repeat(128);
        let padded = format!("  {at_limit}\t");
        assert_eq!(
            check_name(&recipe, 0, &at_limit, 128),
            Ok(at_limit.as_str())
        );
        assert_eq!(check_name(&recipe, 0, &padded, 128), Ok(at_limit.as_str()));
        assert_eq!(
            check_name(&recipe, 0, &"a".repeat(129), 128),
            Err(NameRefusal::TooLong {
                bytes: 129,
                limit: 128
            })
        );
        // A row's own name is not a collision with itself.
        assert_eq!(check_name(&recipe, 0, "inst1", 128), Ok("inst1"));
        assert_eq!(
            check_name(&recipe, 0, " inst2 ", 128),
            Err(NameRefusal::Taken("inst2".into()))
        );
    }

    #[test]
    fn a_rename_in_a_recipe_with_a_duplicate_id_leaves_the_notes_with_the_other() {
        // Only a recipe from outside the editor has two instruments with one
        // id. Which of them a note meant cannot be known, so the notes stay
        // with the name, which the other still has.
        let mut recipe = recipe_with(&["a", "a"], 1);
        recipe.tracks[0].events = vec![note("a", 0.0)];
        let mut layouts = HashMap::from([("a".to_owned(), PatchEditorState::default())]);
        rename_instrument(&mut recipe, &mut layouts, 0, "b");
        assert_eq!(instrument_ids(&recipe), ["b", "a"]);
        assert_eq!(note_ids(&recipe), [vec!["a"]]);
        assert!(layouts.contains_key("a") && !layouts.contains_key("b"));
    }

    #[test]
    fn a_rename_drops_a_layout_a_removed_instrument_left_under_the_new_name() {
        let mut recipe = recipe_with(&["inst1"], 0);
        let mut layouts = HashMap::from([("gone".to_owned(), PatchEditorState::default())]);
        rename_instrument(&mut recipe, &mut layouts, 0, "gone");
        assert!(
            layouts.is_empty(),
            "a never-opened instrument starts from a fresh layout, not a stranger's"
        );
    }

    // -----------------------------------------------------------------------
    // Colours from the editor style, words on the buttons (#58, Overlands
    // #1331 E1 E2)
    // -----------------------------------------------------------------------

    /// `wind` plays one long note with a release tail on lane 0, selected;
    /// `kick` one short note on lane 1. The recipe loops from beat 2 with a
    /// one-beat crossfade.
    fn two_lane_recipe() -> (SequenceRecipe, SequenceEditorState) {
        let mut recipe = recipe_with(&["wind", "kick"], 2);
        recipe.tracks[0].events = vec![Event {
            gate_beats: 4.0,
            release_beats: 1.0,
            ..note("wind", 0.0)
        }];
        recipe.tracks[1].events = vec![note("kick", 1.0)];
        recipe.loop_start_beats = Some(2.0);
        recipe.loop_crossfade_beats = 1.0;
        // At the fitted zoom a half-beat note is a few points wide and
        // holds no label, so these read the timeline zoomed in, which is
        // where a note's name is something to check the contrast of.
        let mut state = view(48.0, Snap::default());
        state.set_selected_event(Some((0, 0)));
        (recipe, state)
    }

    /// The block a note label is painted on: the last opaque fill of a
    /// note's height under the label's centre. An outline is painted as a
    /// rect too, with a transparent fill, after the block it outlines.
    fn block_under(out: &egui::FullOutput, label: &egui::Rect) -> Option<egui::epaint::RectShape> {
        let block_h = LANE_H - 8.0;
        shapes(out)
            .into_iter()
            .filter_map(|s| match s {
                egui::Shape::Rect(r)
                    if (r.rect.height() - block_h).abs() < 1.0
                        && r.fill.is_opaque()
                        && r.rect.contains(label.center()) =>
                {
                    Some(r)
                }
                _ => None,
            })
            .last()
    }

    /// Every note label reading `name` on the timeline, with its colour and
    /// the fill of the block under it.
    fn note_labels(out: &egui::FullOutput, name: &str) -> Vec<(Color32, Color32)> {
        shapes(out)
            .into_iter()
            .filter_map(|s| match s {
                egui::Shape::Text(t) if t.galley.text() == name => {
                    let rect = t.visual_bounding_rect();
                    let block = block_under(out, &rect)?;
                    Some((crate::ui::test_paint::text_colour(&t), block.fill))
                }
                _ => None,
            })
            .collect()
    }

    /// The acceptance of #1331 for notes: in egui's dark and light themes, a
    /// note's name reads on its block, selected or not.
    #[test]
    fn a_notes_name_reads_on_its_block_in_dark_and_light() {
        for (theme, visuals) in [
            ("dark", egui::Visuals::dark()),
            ("light", egui::Visuals::light()),
        ] {
            let (recipe, state) = two_lane_recipe();
            let driver = Driver::themed(recipe, state, visuals);
            for name in ["wind", "kick"] {
                let found = note_labels(&driver.out, name);
                assert_eq!(found.len(), 1, "{theme}: one {name:?} note label");
                let (text, fill) = found[0];
                let ratio = contrast_on(text, fill);
                assert!(
                    ratio >= AA,
                    "{theme}: {name:?} is {ratio:.2}:1 on its block ({text:?} on {fill:?})"
                );
            }
        }
    }

    /// A style the host set is what the timeline paints, role by role.
    #[test]
    fn a_set_style_is_what_the_timeline_paints() {
        let s = distinct_style();
        let (recipe, state) = two_lane_recipe();
        let driver = Driver::with_state(recipe, state);
        let painted = colours(&driver.out);
        for (role, colour) in [
            ("timeline_ground", s.timeline_ground),
            ("timeline_grid", s.timeline_grid),
            ("ground_text", s.ground_text),
            ("lane", s.lane),
            ("lane_alt", s.lane_alt),
            ("loop_start", s.loop_start),
            ("loop_end", s.loop_end),
            ("crossfade_band", s.crossfade_band),
            ("note_selected", s.note_selected),
        ] {
            assert!(painted.contains(&colour), "{role} is not painted");
        }
        // `note_fill` is no longer painted as itself: since #62 it is what
        // each instrument's tint is derived *from* — its saturation and its
        // luminance, with a hue per id — so what the blocks paint is the
        // tint, and the label still goes on it in `note_text`.
        let tints = instrument_tints(["wind", "kick"], &s);
        for name in ["wind", "kick"] {
            assert_eq!(
                note_labels(&driver.out, name),
                [(s.note_text, tints[name])],
                "{name}: the label in note_text on its own instrument's tint"
            );
        }
        assert!(
            !painted.contains(&s.note_fill),
            "note_fill is painted as itself, so some block skipped the tint"
        );
    }

    /// A note with no instrument and a refused name are in the style's
    /// error colour.
    #[test]
    fn missing_notes_and_refused_names_are_in_the_styles_error_colour() {
        let s = distinct_style();
        let mut state = SequenceEditorState::default();
        state.set_selected_event(Some((0, 0)));
        let mut driver = Driver::with_state(ghost_recipe(), state);
        assert!(colours(&driver.out).contains(&s.error), "the ghost note");
        let (_, missing) = crate::ui::test_paint::text_painted(&driver.out, "missing: ghost")
            .expect("the ghost note's label");
        assert_eq!(missing, s.error);

        driver.focus("wind");
        driver.frame(backspaces(4));
        driver.frame(vec![typed("kick")]);
        let (_, refusal) =
            crate::ui::test_paint::text_painted(&driver.out, "Another instrument is called 'kick'")
                .expect("the reason");
        assert_eq!(refusal, s.error);
    }

    /// E2: the sequence editor's buttons say what they do in words. The
    /// remove cross stays (Overlands' `affordances::CROSS`), and so does the
    /// pencil of the instrument's Edit toggle, which Overlands draws for the
    /// same act ("✏ Edit audio…").
    #[test]
    fn the_sequence_buttons_say_what_they_do_in_words() {
        let (recipe, state) = two_lane_recipe();
        let mut driver = Driver::with_state(recipe, state);
        let front = button_labels(&driver.out);
        for word in [
            "Add instrument",
            "Add track",
            "Fit",
            "More",
            "Delete note",
            "\u{270F} Edit",
        ] {
            assert!(
                front.iter().any(|l| l == word),
                "no {word:?} button; buttons: {front:?}"
            );
        }
        // Mutate recipe moved under More with the JSON box (#62, Overlands
        // #1335 C9), so its wording is checked where it now lives.
        driver.click("More");
        let under = button_labels(&driver.out);
        assert!(
            under.iter().any(|l| l == "Mutate recipe"),
            "no Mutate recipe under More: {under:?}"
        );
        for labels in [&front, &under] {
            assert_eq!(
                glyph_labels(labels, &['\u{2716}', '\u{270F}']),
                Vec::<String>::new(),
                "buttons labelled with a glyph where the vocabulary says a word"
            );
        }
    }

    // ---- step 7a (#60, Overlands #1333): one history per recipe ---------

    /// An instrument's patch lives inside the recipe, so an edit to it in
    /// the embedded canvas is an edit to the recipe — and the recipe's
    /// history is the one that takes the step. Were the canvas's own
    /// history on as well, one Ctrl+Z would walk two stacks at once.
    #[test]
    fn the_sequence_editor_has_exactly_one_history() {
        let state = SequenceEditorState {
            active_instrument: Some(0),
            ..Default::default()
        };
        let mut driver = Driver::with_state(two_instrument_recipe(), state);
        let before = driver.recipe.clone();
        assert!(!driver.state.can_undo(), "nothing edited yet");

        // Add a node to the open instrument's patch, through its canvas.
        driver.click("Add node");
        driver.click("Reverb");
        assert_ne!(driver.recipe, before, "the canvas edited the recipe");

        // The embedded canvas keeps no history of its own.
        let canvas = driver
            .state
            .canvas_states
            .values()
            .next()
            .expect("the open instrument has a canvas state");
        assert!(
            !canvas.history_is_enabled(),
            "the embedded patch canvas is keeping a second history of the recipe"
        );
        assert!(!canvas.can_undo(), "and it offers no undo of its own");

        // The recipe's history has it, and one undo takes it all back.
        assert!(driver.state.can_undo());
        assert!(driver.state.undo(&mut driver.recipe));
        assert_eq!(driver.recipe, before);
        assert!(driver.state.redo(&mut driver.recipe));
        assert_ne!(driver.recipe, before);
    }

    /// An edit made in the recipe's own widgets goes on the same one
    /// history as an edit made in the embedded canvas.
    #[test]
    fn the_recipes_own_edits_and_its_patches_share_the_one_history() {
        let state = SequenceEditorState {
            active_instrument: Some(0),
            ..Default::default()
        };
        let mut driver = Driver::with_state(two_instrument_recipe(), state);
        let start = driver.recipe.clone();

        driver.click("Add track");
        let after_track = driver.recipe.clone();
        assert_ne!(after_track, start);

        driver.click("Add node");
        driver.click("Reverb");
        let after_node = driver.recipe.clone();
        assert_ne!(after_node, after_track);

        assert!(driver.state.undo(&mut driver.recipe));
        assert_eq!(driver.recipe, after_track, "the node edit came off first");
        assert!(driver.state.undo(&mut driver.recipe));
        assert_eq!(driver.recipe, start, "then the track edit");
        assert!(!driver.state.can_undo());
    }

    // -----------------------------------------------------------------------
    // A readable timeline (#62, Overlands #1335 C3, C4, C5, C9)
    // -----------------------------------------------------------------------
    //
    // These read what the timeline actually painted, the way the rename
    // tests do: the ground it drew, the rects on it, and the text over
    // them. Nothing asks the drawing code what it meant to draw.

    /// The recipe `host_window` seeds its sequence slot with, cut to what
    /// the timeline cares about: five lanes, thirty-four beats, a loop that
    /// starts at two and blends for two, and notes off the native pitch.
    fn wide_recipe() -> SequenceRecipe {
        let note = |id: &str, time_beats: f32, pitch_multiplier: f32, gate_beats: f32| Event {
            time_beats,
            instrument_id: id.into(),
            pitch_multiplier,
            volume: 0.5,
            gate_beats,
            release_beats: 0.5,
            ..Event::default()
        };
        let mut recipe = recipe_with(&["bed", "pluck", "melody"], 3);
        recipe.bpm = 60.0;
        recipe.duration_beats = 34.0;
        recipe.loop_start_beats = Some(2.0);
        recipe.loop_crossfade_beats = 2.0;
        recipe.tracks[0].events = vec![note("bed", 0.0, 1.0, 34.0)];
        recipe.tracks[1].events = (0..6)
            .map(|i| note("pluck", 2.0 + 4.0 * i as f32, 1.5, 1.0))
            .collect();
        recipe.tracks[2].events = vec![
            note("melody", 2.0, 1.0, 4.0),
            note("melody", 8.0, 1.25, 4.0),
            note("pluck", 16.0, 0.75, 4.0),
        ];
        recipe
    }

    /// Every rect the last frame painted, with its fill.
    fn rects(out: &egui::FullOutput) -> Vec<(Rect, Color32)> {
        shapes(out)
            .into_iter()
            .filter_map(|s| match s {
                egui::Shape::Rect(r) => Some((r.rect, r.fill)),
                _ => None,
            })
            .collect()
    }

    /// The timeline's own ground: the one rect painted in the style's
    /// `timeline_ground`, which under [`distinct_style`] nothing else uses.
    fn timeline_ground(driver: &Driver) -> Rect {
        let ground = distinct_style().timeline_ground;
        rects(&driver.out)
            .into_iter()
            .find(|(_, fill)| *fill == ground)
            .map(|(rect, _)| rect)
            .expect("the timeline painted its ground")
    }

    /// The bounding rect of the text reading exactly `text`.
    fn text_rect(driver: &Driver, text: &str) -> Option<Rect> {
        shapes(&driver.out).into_iter().find_map(|s| match s {
            egui::Shape::Text(t) if t.galley.text() == text => Some(t.visual_bounding_rect()),
            _ => None,
        })
    }

    /// C4: the timeline opens fitted to the width it was given, so the loop
    /// end and the crossfade are on screen without a scroll.
    ///
    /// At 48 points a beat — what every sequence editor opened with — the
    /// seeded recipe's thirty-four beats are 1 632 points of content in a
    /// 480-point panel, and everything past beat ten is off screen with no
    /// scrollbar at rest to say so.
    #[test]
    fn the_timeline_opens_fitted_to_the_width_it_is_given() {
        let mut driver = Driver::new(wide_recipe());
        driver.frame(Vec::new());
        let ground = timeline_ground(&driver);
        assert!(
            ground.width() <= driver.panel.width(),
            "the timeline drew {} points of content inside a {}-point panel",
            ground.width(),
            driver.panel.width()
        );
    }

    /// C4: every marker is labelled, and every label is inside the
    /// timeline. A label pushed off the end by a marker at the last beat is
    /// a label nobody reads.
    #[test]
    fn every_marker_label_lies_inside_the_timeline() {
        let mut driver = Driver::new(wide_recipe());
        driver.frame(Vec::new());
        let ground = timeline_ground(&driver);
        for label in ["loop start", "blend", "end"] {
            let rect = text_rect(&driver, label)
                .unwrap_or_else(|| panic!("no marker is labelled {label:?}: {:?}", driver.texts()));
            assert!(
                ground.contains_rect(rect),
                "the {label:?} label is at {rect:?}, outside the timeline {ground:?}"
            );
        }
    }

    /// C4: the marker lines are drawn over the notes, not under them.
    ///
    /// The seeded recipe has two notes as long as the whole sequence, and
    /// they covered the loop and end lines completely — so the markers C4
    /// is about were invisible at exactly the zoom that had just brought
    /// them on screen. The band stays underneath: a wash over a block
    /// makes it look faded, which is what a quiet note looks like.
    #[test]
    fn the_marker_lines_are_drawn_over_the_notes_and_the_band_under_them() {
        let mut recipe = wide_recipe();
        // One note across the whole sequence, over every marker there is.
        recipe.tracks[0].events = vec![Event {
            gate_beats: 34.0,
            ..note("bed", 0.0)
        }];
        let driver = Driver::with_state(recipe, view(48.0, Snap::default()));
        let s = distinct_style();
        let painted = shapes(&driver.out);
        let last_block = painted
            .iter()
            .rposition(|shape| matches!(shape, egui::Shape::Rect(r) if r.fill == instrument_tints(["bed"], &s)["bed"]))
            .expect("the bed note's block");
        let marker = painted
            .iter()
            .rposition(|shape| {
                matches!(shape, egui::Shape::LineSegment { stroke, .. } if stroke.color == s.loop_start)
            })
            .expect("the loop start line");
        assert!(
            marker > last_block,
            "the loop marker is painted at {marker}, under the note at {last_block}"
        );
        let band = painted
            .iter()
            .position(|shape| matches!(shape, egui::Shape::Rect(r) if r.fill == s.crossfade_band))
            .expect("the crossfade band");
        assert!(
            band < last_block,
            "the crossfade band is painted at {band}, over the note at {last_block}"
        );
    }

    /// C3: a lane says which instrument it is for, in its gutter, where a
    /// note block can never reach — so the name is the lane's and not some
    /// note's that happens to read the same.
    ///
    /// Track 2 holds two `melody` notes and one `pluck`, so its name is
    /// the instrument most of it plays and not the last one stored.
    #[test]
    fn a_lane_takes_its_name_from_its_dominant_instrument() {
        let mut driver = Driver::new(wide_recipe());
        driver.frame(Vec::new());
        let ground = timeline_ground(&driver);
        let gutter =
            Rect::from_x_y_ranges(ground.left()..=ground.left() + GUTTER, ground.y_range());
        for name in ["bed", "pluck", "melody"] {
            let in_gutter = shapes(&driver.out).into_iter().any(|s| match s {
                egui::Shape::Text(t) => {
                    t.galley.text() == name && gutter.contains_rect(t.visual_bounding_rect())
                }
                _ => false,
            });
            assert!(
                in_gutter,
                "no lane gutter is named {name:?}: {:?}",
                driver.texts()
            );
        }
    }

    /// C3: two instruments are two colours, and a note's colour is its
    /// instrument's wherever the note is.
    #[test]
    fn notes_of_different_instruments_are_painted_different_colours() {
        let mut driver = Driver::new(wide_recipe());
        driver.frame(Vec::new());
        let ground = timeline_ground(&driver);
        let style = distinct_style();
        // The blocks: rects on the timeline, shorter than a lane, that are
        // none of the grounds the timeline paints under them.
        let grounds = [
            style.timeline_ground,
            style.lane,
            style.lane_alt,
            style.crossfade_band,
            style.release_tail,
        ];
        let fills: HashSet<Color32> = rects(&driver.out)
            .into_iter()
            .filter(|(r, fill)| {
                ground.contains_rect(*r)
                    && r.height() > 0.0
                    && r.height() < LANE_H
                    && !grounds.contains(fill)
                    && fill.a() > 0
            })
            .map(|(_, fill)| fill)
            .collect();
        assert!(
            fills.len() >= 3,
            "three instruments' notes were painted {} colour(s): {fills:?}",
            fills.len()
        );
    }

    /// C3: a note off the native pitch says so on its block, and a block
    /// says as much as it has room for and no more.
    ///
    /// At the fitted zoom a one-beat note is ten points wide, so the widest
    /// label it can hold is nothing: a label was drawn at the block's left
    /// edge whatever the width, and wrote its instrument's name across the
    /// four notes after it. What fits at which zoom is the test.
    #[test]
    fn a_note_says_its_pitch_when_its_block_has_room_and_nothing_when_it_does_not() {
        let mut driver = Driver::new(wide_recipe());
        driver.frame(Vec::new());
        // Fitted: the 34-beat note is wide enough for its name, the
        // 4-beat ones for a pitch, the 1-beat ones for neither.
        let ground = timeline_ground(&driver);
        for (label, want) in [
            ("bed", true),
            ("\u{00D7}1.25", true),
            ("pluck \u{00D7}1.5", false),
        ] {
            let on_a_block = shapes(&driver.out).into_iter().any(|s| match s {
                egui::Shape::Text(t) => {
                    t.galley.text() == label
                        && t.visual_bounding_rect().left() > ground.left() + GUTTER
                }
                _ => false,
            });
            assert_eq!(
                on_a_block,
                want,
                "{label:?} on a block at the fitted zoom: {:?}",
                driver.texts()
            );
        }

        // Zoomed in to edit, every note has room for both.
        driver.state.zoom = Some(96.0);
        driver.frame(Vec::new());
        for label in ["pluck \u{00D7}1.5", "melody \u{00D7}1.25", "bed"] {
            assert!(
                driver.paints(label),
                "no note says {label:?} at 96 points a beat: {:?}",
                driver.texts()
            );
        }

        // And a label never leaves the block it belongs to.
        for note in driver.state.notes() {
            let room = note.body.width() - LABEL_PAD;
            for text in shapes(&driver.out) {
                if let egui::Shape::Text(t) = text
                    && t.galley.rect.left() >= 0.0
                {
                    let at = t.visual_bounding_rect();
                    if at.center().y > note.body.top()
                        && at.center().y < note.body.bottom()
                        && (at.left() - note.body.left() - LABEL_PAD).abs() < 1.0
                    {
                        assert!(
                            at.width() <= room + 1.0,
                            "{:?} is {} points wide on a {}-point block",
                            t.galley.text(),
                            at.width(),
                            note.body.width()
                        );
                    }
                }
            }
        }
    }

    /// C5: the timeline and the inspector say what a field does, not what
    /// the schema calls it.
    #[test]
    fn the_labels_are_words_and_not_schema_names() {
        let mut state = view(48.0, Snap::default());
        state.set_selected_event(Some((2, 1)));
        let mut driver = Driver::with_state(wide_recipe(), state);
        driver.frame(Vec::new());
        let texts = driver.texts();
        for schema in [
            "px/beat",
            "duration (beats)",
            "gate (beats)",
            "release (beats)",
            "seamless loop",
            "crossfade (beats)",
            "loop start (beats)",
            "time (beats)",
            "Varispeed",
            "Time-preserving",
        ] {
            assert!(
                !texts.iter().any(|t| t == schema),
                "the editor still says {schema:?}"
            );
        }
        for word in [
            "Zoom",
            "Length",
            "Hold",
            "Tail",
            "Pitch",
            "Loop",
            "Blend into loop",
        ] {
            assert!(
                texts.iter().any(|t| t.starts_with(word)),
                "nothing says {word:?}: {texts:?}"
            );
        }
    }

    /// C5: the inspector's header counts tracks and notes from one, the
    /// way its lanes are numbered on the ruler and the way anyone counts.
    #[test]
    fn the_inspector_header_counts_from_one() {
        let mut state = view(48.0, Snap::default());
        state.set_selected_event(Some((2, 1)));
        let mut driver = Driver::with_state(wide_recipe(), state);
        driver.frame(Vec::new());
        assert!(
            driver.paints("Track 3 \u{00B7} note 2"),
            "the header is not 1-based: {:?}",
            driver.texts()
        );
    }

    /// C9: the genetics controls and the JSON box are off the front of the
    /// editor and under More, the way #59 put the canvas's away.
    #[test]
    fn the_genetics_and_json_controls_live_under_more() {
        let mut driver = Driver::new(wide_recipe());
        driver.frame(Vec::new());
        let labels = button_labels(&driver.out);
        assert!(
            labels.iter().any(|l| l == "More"),
            "the timeline has no More menu: {labels:?}"
        );
        assert!(
            !labels.iter().any(|l| l.starts_with("Mutate")),
            "Mutate recipe is still on the front row: {labels:?}"
        );
        assert!(
            !driver.paints("Import / Export JSON"),
            "the JSON fold is still on the front row"
        );
        driver.click("More");
        let under = button_labels(&driver.out);
        assert!(
            under.iter().any(|l| l.starts_with("Mutate")),
            "Mutate recipe is not under More either: {under:?}"
        );
    }

    /// C4: the fit maths, at the sizes it is actually asked for.
    ///
    /// The gutter and the tail pad are not beats, so a fit that divided the
    /// whole width by the beats would leave the last beat and the end
    /// marker's label off the edge — the very thing the fit is for.
    #[test]
    fn the_fit_puts_every_beat_inside_the_width_it_is_given() {
        // The seeded recipe in the host window's sequence slot: 40 beats of
        // timeline (34 of sequence and the 4 spare beats, plus the held
        // note's tail, rounded up) in a 900-point slot.
        let ppb = fit_px_per_beat(900.0, 40.0);
        assert!((ppb - (900.0 - GUTTER - TAIL_PAD) / 40.0).abs() < 1e-4);
        // Every size where the fit is not up against the slider's floor: a
        // sequence too long for MIN_PPB cannot fit and is scrolled instead,
        // which the clamps below hold.
        for (width, beats) in [(900.0, 40.0), (480.0, 40.0), (240.0, 8.0), (1836.0, 200.0)] {
            let ppb = fit_px_per_beat(width, beats);
            let drawn = GUTTER + beats * ppb + TAIL_PAD;
            assert!(
                drawn <= width + 1e-3,
                "{beats} beats fitted to {width} points drew {drawn}"
            );
        }
        // Held to the range the slider offers at both ends: a sequence too
        // long to fit is scrolled, and a two-beat one is not magnified to
        // the width of the panel.
        assert_eq!(fit_px_per_beat(200.0, 4096.0), MIN_PPB);
        assert_eq!(fit_px_per_beat(1836.0, 2.0), MAX_PPB);
        // Nothing to divide by, and nothing to divide: no panic, no NaN.
        assert!(fit_px_per_beat(0.0, 40.0).is_finite());
        assert_eq!(fit_px_per_beat(-50.0, 40.0), MIN_PPB);
        assert!(fit_px_per_beat(900.0, 0.0).is_finite());
        assert!(fit_px_per_beat(900.0, -3.0).is_finite());
    }

    /// C4: a marker's label is inside the timeline wherever the marker is —
    /// including at the very last beat, which is where the end marker
    /// always is.
    #[test]
    fn a_marker_label_stays_inside_the_timeline_wherever_the_marker_is() {
        let bounds = Rect::from_min_size(Pos2::new(30.0, 10.0), Vec2::new(400.0, 200.0));
        let size = Vec2::new(52.0, 12.0);
        for x in [-100.0, 30.0, 31.0, 200.0, 375.0, 380.0, 430.0, 900.0] {
            let at = marker_label_rect(x, size, 24.0, bounds);
            assert!(
                bounds.contains_rect(at),
                "a marker at {x} put its label at {at:?}, outside {bounds:?}"
            );
        }
        // Right of the marker when there is room, left of it when there is
        // not, so the label never covers the line it is about.
        assert!(marker_label_rect(200.0, size, 24.0, bounds).left() > 200.0);
        assert!(marker_label_rect(430.0, size, 24.0, bounds).right() < 430.0);
        // A label wider than the timeline has nowhere to go and is clamped
        // to the left edge rather than moved outside it.
        let huge = marker_label_rect(200.0, Vec2::new(600.0, 12.0), 24.0, bounds);
        assert_eq!(huge.left(), bounds.left() + LABEL_PAD);
    }

    /// C3: an instrument's colour is its own, the same every frame, and it
    /// does not depend on which lane the notes are on or what order the
    /// instruments are in.
    /// The smallest gap between any two of `tints`, going round the hue
    /// wheel whichever way is shorter.
    fn closest_hues(tints: &[Color32]) -> f32 {
        let hues: Vec<f32> = tints
            .iter()
            .map(|c| egui::ecolor::Hsva::from(*c).h)
            .collect();
        let mut closest = 1.0_f32;
        for (i, a) in hues.iter().enumerate() {
            for b in &hues[i + 1..] {
                let d = (a - b).abs();
                closest = closest.min(d.min(1.0 - d));
            }
        }
        closest
    }

    #[test]
    fn an_instruments_tint_follows_its_id_and_not_its_place() {
        let style = distinct_style();
        let names = ["bed", "gust", "pluck", "melody", "bass"];
        let map = instrument_tints(names, &style);
        let tints: Vec<Color32> = names.iter().map(|n| map[*n]).collect();
        // Stable: the same recipe gives the same colours, asked again, and
        // in any order.
        assert_eq!(map, instrument_tints(names, &style));
        let mut backwards = names;
        backwards.reverse();
        assert_eq!(map, instrument_tints(backwards, &style));
        // Distinct, and distinct enough to tell apart. `gust` and `pluck`
        // hash to within four degrees of each other, which is what made
        // the seeded recipe's lanes one magenta in the picture that caught
        // this — being different Color32s is not the bar.
        let unique: HashSet<Color32> = tints.iter().copied().collect();
        assert_eq!(unique.len(), names.len(), "two instruments share a tint");
        let closest = closest_hues(&tints);
        assert!(
            closest >= 1.0 / HUE_BUCKETS as f32 - 1e-3,
            "the two closest tints are {:.1} degrees apart",
            closest * 360.0
        );
        // The control: their raw hashes are the collision this spreads.
        let raw = [
            instrument_tint("gust", &style),
            instrument_tint("pluck", &style),
        ];
        assert_eq!(
            closest_hues(&raw),
            0.0,
            "gust and pluck no longer ask for the same hue, so this test \
             has stopped covering what it was written for"
        );
        // An id keeps the hue it asked for when nothing contests it.
        assert_eq!(map["bed"], instrument_tint("bed", &style));
        // More instruments than hues: they share rather than panic.
        let many: Vec<String> = (0..HUE_BUCKETS + 4).map(|i| format!("inst{i}")).collect();
        let crowd = instrument_tints(many.iter().map(String::as_str), &style);
        assert_eq!(crowd.len(), many.len());

        // And what the timeline paints for an id is the same whichever lane
        // the note is on and whatever order the recipe lists it in.
        let mut a = recipe_with(&["bed", "pluck"], 2);
        a.tracks[0].events = vec![Event {
            gate_beats: 8.0,
            ..note("bed", 0.0)
        }];
        a.tracks[1].events = vec![Event {
            gate_beats: 8.0,
            ..note("pluck", 0.0)
        }];
        let mut b = recipe_with(&["pluck", "bed"], 2);
        b.tracks[0].events = a.tracks[1].events.clone();
        b.tracks[1].events = a.tracks[0].events.clone();

        let painted = |recipe: SequenceRecipe| -> Vec<(String, Color32)> {
            let driver = Driver::with_state(recipe, view(48.0, Snap::default()));
            let mut found: Vec<(String, Color32)> = shapes(&driver.out)
                .into_iter()
                .filter_map(|s| match s {
                    egui::Shape::Text(t) => {
                        let name = t.galley.text().to_owned();
                        let at = t.visual_bounding_rect();
                        let block = block_under(&driver.out, &at)?;
                        Some((name, block.fill))
                    }
                    _ => None,
                })
                .collect();
            found.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.to_array().cmp(&b.1.to_array())));
            found
        };
        let first = painted(a);
        assert_eq!(
            first,
            painted(b),
            "an instrument's notes changed colour when the lanes were swapped"
        );
        assert_eq!(first.len(), 2, "both notes were painted and labelled");
        let map = instrument_tints(["bed", "pluck"], &style);
        for (name, fill) in &first {
            assert_eq!(*fill, map[name.as_str()], "{name}'s tint");
        }
    }

    /// C3: every tint has the note fill's luminance, which is why one
    /// contrast test covers all of them.
    ///
    /// Contrast is a function of luminance alone, so holding the luminance
    /// makes `note_text` on any instrument's tint exactly as readable as
    /// `note_text` on the style's own `note_fill` — the pair #58's test
    /// already holds to AA. Rotating the hue at a fixed *value* instead
    /// would not: a yellow and a blue of the same HSV value are four stops
    /// of luminance apart.
    #[test]
    fn every_instrument_tint_reads_as_well_as_the_note_fill_it_came_from() {
        for (theme, visuals) in [
            ("dark", egui::Visuals::dark()),
            ("light", egui::Visuals::light()),
        ] {
            let style = EditorStyle::from_visuals(&visuals);
            let base = contrast_on(style.note_text, style.note_fill);
            let ids = ["bed", "gust", "pluck", "melody", "bass", "", "inst1"];
            let tints = instrument_tints(ids, &style);
            for id in ids {
                let tint = tints[id];
                let ratio = contrast_on(style.note_text, tint);
                assert!(
                    (ratio - base).abs() < 0.25,
                    "{theme}: {id:?} is {ratio:.2}:1 against the note fill's {base:.2}:1"
                );
                assert!(ratio >= AA, "{theme}: {id:?} is {ratio:.2}:1");
                // A silent note is faint, and its name still reads on it.
                let quiet = tint.gamma_multiply(volume_alpha(0.0));
                let over_lane = contrast_on(style.note_text, style.lane.blend(quiet));
                assert!(
                    over_lane >= AA,
                    "{theme}: {id:?} at volume 0 is {over_lane:.2}:1 on the lane"
                );
            }
        }
    }

    /// C3: a lane's name is the instrument most of it plays, and a tie does
    /// not depend on the order the events happen to be stored in.
    #[test]
    fn the_dominant_instrument_is_the_commonest_and_ties_break_on_time() {
        let of = |events: Vec<Event>| {
            let track = Track { events };
            dominant_instrument(&track).map(|(id, n)| (id.to_owned(), n))
        };
        assert_eq!(of(Vec::new()), None, "an empty lane has no name");
        assert_eq!(
            of(vec![note("a", 0.0), note("b", 1.0), note("b", 2.0)]),
            Some(("b".to_owned(), 2))
        );
        // A tie goes to whichever starts first, whichever order they are in.
        assert_eq!(
            of(vec![note("late", 8.0), note("early", 1.0)]),
            Some(("early".to_owned(), 1))
        );
        assert_eq!(
            of(vec![note("early", 1.0), note("late", 8.0)]),
            Some(("early".to_owned(), 1))
        );
        // A tie on both counts and both times goes to the lower id, so it
        // is an answer and not an accident.
        assert_eq!(
            of(vec![note("b", 1.0), note("a", 1.0)]),
            Some(("a".to_owned(), 1))
        );
    }

    /// C5: beats become a time anyone can hear, at the BPM in force.
    #[test]
    fn beats_are_written_as_a_time_at_the_recipes_bpm() {
        assert_eq!(beats_as_time(34.0, 60.0), "34 s");
        assert_eq!(beats_as_time(34.0, 120.0), "17 s");
        assert_eq!(beats_as_time(34.0, 300.0), "6.8 s");
        assert_eq!(beats_as_time(2.0, 60.0), "2.0 s");
        assert_eq!(beats_as_time(0.25, 60.0), "0.2 s");
        // Past a minute it is minutes and seconds, not "126 s".
        assert_eq!(beats_as_time(126.0, 60.0), "2:06");
        // A BPM of nothing is not a division by nothing.
        assert_eq!(beats_as_time(34.0, 0.0), "0.0 s");
    }

    /// C3: a pitch is written the short way, and the native pitch — which
    /// is every note's default — is not written at all.
    #[test]
    fn a_pitch_is_written_only_when_it_is_not_the_native_one() {
        assert_eq!(pitch_label(1.0), None);
        assert_eq!(pitch_label(1.001), None);
        assert_eq!(pitch_label(1.5).as_deref(), Some("\u{00D7}1.5"));
        assert_eq!(pitch_label(2.0).as_deref(), Some("\u{00D7}2"));
        assert_eq!(pitch_label(0.75).as_deref(), Some("\u{00D7}0.75"));
        // Two decimals, rounded the way Rust rounds them.
        assert_eq!(pitch_label(1.125).as_deref(), Some("\u{00D7}1.12"));
        assert_eq!(pitch_label(1.126).as_deref(), Some("\u{00D7}1.13"));
    }

    // ---- C6: adding, copying and moving notes --------------------------

    /// A view state zoomed to `zoom` points a beat, on `snap`'s grid.
    ///
    /// Most of these tests set a zoom: at the fitted one a half-beat note is
    /// a few points wide and holds no label, which is right for the picture
    /// and useless for reading one back.
    fn view(zoom: f32, snap: Snap) -> SequenceEditorState {
        SequenceEditorState {
            zoom: Some(zoom),
            snap,
            ..SequenceEditorState::default()
        }
    }

    /// The `(track, event)` of every note whose block is picked, read off
    /// the state rather than off a click count.
    fn picked(driver: &Driver) -> Vec<(usize, usize)> {
        let mut at: Vec<(usize, usize)> = driver.state.selected_events().collect();
        at.sort_unstable();
        at
    }

    /// Where every note on `track` starts.
    fn starts(recipe: &SequenceRecipe, track: usize) -> Vec<f32> {
        recipe.tracks[track]
            .events
            .iter()
            .map(|e| e.time_beats)
            .collect()
    }

    /// C6: a note added to a lane is one of that lane's instrument, not the
    /// recipe's first — a note added to the bass lane used to arrive as a
    /// `bed`, silent-looking in the wrong colour on the wrong row.
    #[test]
    fn a_note_added_to_a_lane_takes_that_lanes_instrument() {
        let mut recipe = wide_recipe();
        let mut state = view(48.0, Snap::Beat);
        // The lane whose dominant instrument is not the recipe's first.
        let track = 1;
        assert_eq!(
            dominant_instrument(&recipe.tracks[track]).map(|(id, _)| id),
            Some("pluck")
        );
        let lane_names: Vec<Option<(String, usize)>> = recipe
            .tracks
            .iter()
            .map(|t| dominant_instrument(t).map(|(id, n)| (id.to_owned(), n)))
            .collect();
        let res = apply_actions(
            &mut recipe,
            &mut state,
            vec![Action::AddNote { track, beat: 7.4 }],
            &lane_names,
        );
        assert!(res.changed && res.rebake, "adding a note commits: {res:?}");
        let added = recipe.tracks[track].events.last().expect("the new note");
        assert_eq!(added.instrument_id, "pluck");
        assert_eq!(added.time_beats, 7.0, "and it lands on the grid");
        assert_eq!(
            state.selected_event,
            Some((track, recipe.tracks[track].events.len() - 1)),
            "the new note is the one the inspector opens on"
        );

        // An empty lane has no instrument of its own, so it falls back to
        // the recipe's first rather than to no instrument at all, which
        // would bake to silence.
        let mut recipe = wide_recipe();
        recipe.tracks.push(Track::default());
        let empty = recipe.tracks.len() - 1;
        let mut lane_names = lane_names.clone();
        lane_names.push(None);
        apply_actions(
            &mut recipe,
            &mut state,
            vec![Action::AddNote {
                track: empty,
                beat: 0.0,
            }],
            &lane_names,
        );
        assert_eq!(recipe.tracks[empty].events[0].instrument_id, "bed");
    }

    /// C6: a duplicate copies every picked note, leaves the copies picked,
    /// and is one step of the history.
    #[test]
    fn a_duplicate_copies_every_picked_note_and_leaves_the_copies_picked() {
        let mut driver = Driver::with_state(wide_recipe(), view(48.0, Snap::default()));
        let before = driver.recipe.clone();

        driver.point_at_timeline();
        driver.state.select_only((2, 0));
        driver.state.select_also((2, 1));
        driver.frame(Vec::new());
        assert_eq!(picked(&driver), [(2, 0), (2, 1)]);

        driver.frame(vec![chord(egui::Key::D, egui::Modifiers::COMMAND)]);
        assert_eq!(
            driver.recipe.tracks[2].events.len(),
            before.tracks[2].events.len() + 2,
            "two notes picked, two copies"
        );
        assert_eq!(
            picked(&driver),
            [(2, 3), (2, 4)],
            "the copies are what a drag or a nudge now moves"
        );
        // A copy is the note it came from, in the same place: the drag it
        // is about to be carried by decides where it goes.
        for (copy, original) in [(3, 0), (4, 1)] {
            assert_eq!(
                driver.recipe.tracks[2].events[copy],
                before.tracks[2].events[original]
            );
        }

        // One undo takes the whole duplicate back.
        assert!(driver.state.undo(&mut driver.recipe));
        assert_eq!(driver.recipe, before);
    }

    /// C6: an arrow moves every picked note by the grid, and one undo takes
    /// the whole nudge back.
    #[test]
    fn an_arrow_nudges_every_picked_note_by_the_grid() {
        let mut driver = Driver::with_state(wide_recipe(), view(48.0, Snap::Beat));
        let before = starts(&driver.recipe, 1);

        driver.point_at_timeline();
        driver.state.select_only((1, 0));
        driver.state.select_also((1, 2));
        driver.frame(Vec::new());

        driver.frame(vec![key(egui::Key::ArrowRight)]);
        let after = starts(&driver.recipe, 1);
        assert_eq!(after[0], before[0] + 1.0);
        assert_eq!(after[2], before[2] + 1.0);
        assert_eq!(after[1], before[1], "a note nobody picked did not move");

        // Shift is the fine step, an eighth of the grid.
        driver.frame_held(
            egui::Modifiers::SHIFT,
            vec![chord(egui::Key::ArrowLeft, egui::Modifiers::SHIFT)],
        );
        let fine = starts(&driver.recipe, 1);
        assert!((fine[0] - (after[0] - 0.125)).abs() < 1e-4, "{fine:?}");

        // A note is never nudged before beat 0, where it cannot be baked.
        driver.state.select_only((1, 0));
        for _ in 0..40 {
            driver.frame(vec![key(egui::Key::ArrowLeft)]);
        }
        assert_eq!(starts(&driver.recipe, 1)[0], 0.0);

        // And every nudge was a step of its own on the one history.
        assert!(driver.state.can_undo());
        while driver.state.undo(&mut driver.recipe) {}
        assert_eq!(starts(&driver.recipe, 1), before);
    }

    /// C6: the snap picker is what a drag lands on, division by division.
    #[test]
    fn the_snap_picker_is_the_grid_a_drag_lands_on() {
        for snap in Snap::ALL {
            let mut driver = Driver::with_state(wide_recipe(), view(48.0, snap));
            driver.state.select_only((1, 0));
            driver.frame(Vec::new());
            // Put the note somewhere off every grid, and stop the drag.
            driver.recipe.tracks[1].events[0].time_beats = 5.31;
            driver.state.drag_mode = DragMode::Move;
            driver.frame(Vec::new());
            let at = driver.recipe.tracks[1].events[0].time_beats;
            assert_eq!(at, 5.31, "{snap:?}: a note moved with no drag in progress");
            assert_eq!(
                snap_beat(at, snap),
                match snap {
                    Snap::Off => 5.31,
                    Snap::Eighth => 5.25,
                    Snap::Quarter => 5.25,
                    Snap::Beat => 5.0,
                    Snap::Bar => 4.0,
                },
                "{snap:?}"
            );
        }
        // The picker offers exactly the divisions the maths knows.
        let mut driver = Driver::with_state(wide_recipe(), view(48.0, Snap::Bar));
        driver.frame(Vec::new());
        assert!(driver.paints("bar"), "the picker shows the grid in force");
        for snap in Snap::ALL {
            driver.state.set_snap(snap);
            driver.frame(Vec::new());
            assert_eq!(driver.state.snap(), snap);
            assert!(
                driver.paints(snap.label()),
                "the picker does not show {snap:?}"
            );
        }
    }

    /// C6: a box dragged out over the lanes picks the notes it covers, and
    /// Escape lets them go one press at a time.
    #[test]
    fn a_box_dragged_over_the_lanes_picks_the_notes_it_covers() {
        // A zoom that puts the three notes the box will cover inside the
        // panel: the editor only takes the keyboard while it has the
        // pointer, so a box dragged off the panel cannot then be Escaped.
        let mut driver = Driver::with_state(wide_recipe(), view(24.0, Snap::default()));
        driver.frame(Vec::new());

        // The notes are where the timeline says they are, not where a
        // second reckoning of the zoom and the scroll would put them.
        let notes = driver.state.notes().to_vec();
        assert!(!notes.is_empty(), "the timeline published no geometry");
        let over = |track: usize, event: usize| {
            notes
                .iter()
                .find(|n| n.track == track && n.event == event)
                .map(|n| n.body)
                .expect("the note was drawn")
        };
        let first = over(1, 0);
        let third = over(1, 2);
        let from = Pos2::new(first.left() - 2.0, first.top() - 1.0);
        let to = Pos2::new(third.right() + 2.0, third.bottom() + 1.0);

        driver.frame(vec![
            egui::Event::PointerMoved(from),
            egui::Event::PointerButton {
                pos: from,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ]);
        driver.frame(vec![egui::Event::PointerMoved(to)]);
        driver.frame(vec![egui::Event::PointerMoved(to)]);
        assert!(
            driver.state.marquee.is_some(),
            "no box was dragged out: {:?}",
            driver.state.marquee
        );
        assert_eq!(
            picked(&driver),
            [(1, 0), (1, 1), (1, 2)],
            "the box picked what it covers"
        );

        driver.frame(vec![egui::Event::PointerButton {
            pos: to,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }]);
        assert!(driver.state.marquee.is_none(), "the box stayed up");
        assert_eq!(
            picked(&driver),
            [(1, 0), (1, 1), (1, 2)],
            "and the notes stay picked"
        );

        // Escape spends one press per step: the selection, and nothing
        // after it, so the host's Esc ladder keeps its rung.
        let res = driver.frame(vec![key(egui::Key::Escape)]);
        assert!(picked(&driver).is_empty());
        assert!(
            driver.state.took_escape(),
            "the editor did not claim the Escape"
        );
        assert!(!res.changed, "clearing a selection is not an edit");
        driver.frame(vec![key(egui::Key::Escape)]);
        assert!(
            !driver.state.took_escape(),
            "a second Escape with nothing picked took a rung off the host's ladder"
        );
    }

    /// C6: the inspector edits the last note picked and says how many are.
    #[test]
    fn the_inspector_edits_one_of_the_picked_notes_and_says_how_many_there_are() {
        let mut driver = Driver::with_state(wide_recipe(), view(48.0, Snap::default()));
        driver.state.select_only((2, 0));
        driver.state.select_also((2, 2));
        driver.frame(Vec::new());
        assert_eq!(driver.state.selected_event(), Some((2, 2)));
        assert!(
            driver.paints("2 notes picked"),
            "the inspector does not say how many: {:?}",
            driver.texts()
        );
        assert!(
            driver.paints("Track 3 \u{00B7} note 3"),
            "and which one it edits"
        );

        // Delete takes all of them, highest index first so the removals do
        // not move each other out from under themselves.
        let before = driver.recipe.tracks[2].events.len();
        driver.click("Delete 2 notes");
        assert_eq!(driver.recipe.tracks[2].events.len(), before - 2);
        assert_eq!(
            driver.recipe.tracks[2].events[0].pitch_multiplier, 1.25,
            "the note between them is the one left"
        );
        assert!(picked(&driver).is_empty());
    }

    /// A shift-click adds a note to the picked set and takes it out again.
    #[test]
    fn a_shift_click_adds_a_note_to_the_selection_and_takes_it_out_again() {
        let mut driver = Driver::with_state(wide_recipe(), view(48.0, Snap::default()));
        driver.state.select_only((0, 0));
        driver.state.select_also((1, 0));
        assert_eq!(picked(&driver), [(0, 0), (1, 0)]);
        driver.state.select_also((1, 0));
        assert_eq!(picked(&driver), [(0, 0)]);
        assert_eq!(
            driver.state.selected_event(),
            Some((0, 0)),
            "the inspector moved to what is left, not to nothing"
        );
        // And a plain click is only ever the one note.
        driver.state.select_only((1, 1));
        assert_eq!(picked(&driver), [(1, 1)]);
    }

    /// A removal moves the indices after it down, so a selection kept
    /// across one is pruned rather than left meaning different notes.
    #[test]
    fn a_selection_never_outlives_the_notes_it_names() {
        let mut driver = Driver::with_state(wide_recipe(), view(48.0, Snap::default()));
        driver.state.select_only((1, 5));
        driver.state.select_also((1, 4));
        driver.recipe.tracks[1].events.truncate(2);
        driver.frame(Vec::new());
        assert!(
            picked(&driver).is_empty(),
            "notes that are gone are still picked: {:?}",
            picked(&driver)
        );
        assert_eq!(driver.state.selected_event(), None);

        // A track that goes altogether takes its notes' selection with it.
        driver.state.select_only((2, 0));
        driver.recipe.tracks.truncate(2);
        driver.frame(Vec::new());
        assert!(picked(&driver).is_empty());
    }

    /// C6: a drag carries every picked note, and Alt makes it carry copies
    /// and leave the originals where they were.
    ///
    /// Driven through the real pointer, because the mechanism is egui's:
    /// a drag belongs to the id of the widget the press landed on, and an
    /// Alt-drag's copies have ids nothing has pressed — so the drag has to
    /// keep reading the original's response and carry the copies by its
    /// delta, which is what `drag_anchor` is for.
    #[test]
    fn a_drag_carries_every_picked_note_and_alt_carries_copies_instead() {
        for alt in [false, true] {
            let mut driver = Driver::with_state(wide_recipe(), view(24.0, Snap::Beat));
            driver.frame(Vec::new());
            let before = starts(&driver.recipe, 1);
            let held = if alt {
                egui::Modifiers::ALT
            } else {
                egui::Modifiers::NONE
            };

            // Pick two of the six plucks, and press the first of them well
            // clear of its right edge so the drag moves rather than resizes.
            driver.state.select_only((1, 0));
            driver.state.select_also((1, 2));
            driver.frame(Vec::new());
            let body = driver
                .state
                .notes()
                .iter()
                .find(|n| (n.track, n.event) == (1, 0))
                .map(|n| n.body)
                .expect("the note was drawn");
            let from = Pos2::new(body.left() + 2.0, body.center().y);
            let to = Pos2::new(from.x + 2.0 * 24.0, from.y);

            driver.frame_held(held, vec![egui::Event::PointerMoved(from)]);
            driver.frame_held(
                held,
                vec![egui::Event::PointerButton {
                    pos: from,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: held,
                }],
            );
            driver.frame_held(held, vec![egui::Event::PointerMoved(to)]);
            driver.frame_held(held, vec![egui::Event::PointerMoved(to)]);
            driver.frame_held(
                held,
                vec![egui::Event::PointerButton {
                    pos: to,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: held,
                }],
            );
            driver.frame(Vec::new());

            let after = starts(&driver.recipe, 1);
            if alt {
                assert_eq!(
                    after.len(),
                    before.len() + 2,
                    "an Alt-drag left no copies: {after:?}"
                );
                assert_eq!(
                    &after[..before.len()],
                    &before[..],
                    "an Alt-drag moved the originals"
                );
                assert_eq!(
                    &after[before.len()..],
                    &[before[0] + 2.0, before[2] + 2.0],
                    "the copies did not follow the drag"
                );
                assert_eq!(
                    picked(&driver),
                    [(1, 6), (1, 7)],
                    "the copies are what is picked at the end"
                );
            } else {
                assert_eq!(after.len(), before.len(), "a plain drag copied something");
                assert_eq!(after[0], before[0] + 2.0, "the note under the pointer");
                assert_eq!(after[2], before[2] + 2.0, "and the other picked one");
                assert_eq!(after[1], before[1], "a note nobody picked did not move");
            }
            // Either way it is one step of the history.
            assert!(driver.state.undo(&mut driver.recipe));
            assert_eq!(starts(&driver.recipe, 1), before, "alt={alt}");
        }
    }

    /// C6: a right-click on a track opens its menu at the beat pressed, and
    /// the note it adds is of that track's instrument.
    ///
    /// The menu is where the add control lives because nothing that acts on
    /// a press may be drawn where that press lands: 7b's remove cross at a
    /// wire's midpoint took the click that was picking the wire, and a
    /// floating add button on a lane would take the click that picks a note
    /// (crate #67, third coordinate fact).
    #[test]
    fn a_right_click_on_a_track_offers_a_note_of_that_tracks_instrument() {
        let mut driver = Driver::with_state(wide_recipe(), view(24.0, Snap::Beat));
        driver.frame(Vec::new());

        // Clear ground on track 1, between two plucks.
        let tl = driver.state.timeline_rect();
        let at = Pos2::new(
            tl.left() + GUTTER + 4.2 * 24.0,
            tl.top() + RULER_H + 1.5 * LANE_H,
        );
        driver.frame(vec![egui::Event::PointerMoved(at)]);
        for pressed in [true, false] {
            driver.frame(vec![egui::Event::PointerButton {
                pos: at,
                button: egui::PointerButton::Secondary,
                pressed,
                modifiers: egui::Modifiers::NONE,
            }]);
        }
        // The menu is an Area of its own, so it is painted the frame after
        // the click that opened it.
        driver.frame(Vec::new());
        assert!(
            driver.paints("Add a pluck note here"),
            "no track menu opened at the pointer: {:?}",
            driver.texts()
        );
        assert!(driver.paints("Track 2"), "and it says which track it is");

        // Into the menu, as anyone choosing from it has to: the lane is no
        // longer under the pointer there, which is what greyed the item out
        // until the beat was remembered instead of re-read.
        driver.frame(vec![egui::Event::PointerMoved(at + Vec2::new(14.0, 18.0))]);
        let enabled = driver
            .nodes()
            .any(|(_, n)| n.label() == Some("Add a pluck note here") && !n.is_disabled());
        assert!(
            enabled,
            "the menu item greyed out once the pointer reached it"
        );

        let before = driver.recipe.tracks[1].events.len();
        driver.click("Add a pluck note");
        driver.frame(Vec::new());
        assert_eq!(driver.recipe.tracks[1].events.len(), before + 1);
        let added = driver.recipe.tracks[1].events.last().expect("the new note");
        assert_eq!(added.instrument_id, "pluck");
        assert_eq!(added.time_beats, 4.0, "on the grid the picker is set to");
    }

    /// The timeline hands out the geometry it painted, which is what a host
    /// overlay and a scripted harness press against.
    #[test]
    fn the_timeline_publishes_the_geometry_it_painted() {
        let mut driver = Driver::with_state(wide_recipe(), view(48.0, Snap::default()));
        driver.frame(Vec::new());

        let tl = driver.state.timeline_rect();
        assert!(tl.is_positive(), "the timeline published no rect");
        let notes = driver.state.notes();
        assert_eq!(
            notes.len(),
            driver
                .recipe
                .tracks
                .iter()
                .map(|t| t.events.len())
                .sum::<usize>(),
            "one rect per note"
        );
        for note in notes {
            assert!(
                tl.contains_rect(note.body.intersect(tl)),
                "note {:?} is not on the timeline",
                (note.track, note.event)
            );
            let event = &driver.recipe.tracks[note.track].events[note.event];
            let want = tl.left() + GUTTER + event.time_beats * 48.0;
            assert!(
                (note.body.left() - want).abs() < 0.01,
                "note {:?} at beat {} was published at {}, not {want}",
                (note.track, note.event),
                event.time_beats,
                note.body.left()
            );
            assert!(
                note.extent.contains_rect(note.body),
                "a note's extent does not contain its block"
            );
        }
        // Empty before it has been drawn, so nobody reads a stale rect.
        let fresh = SequenceEditorState::default();
        assert_eq!(fresh.timeline_rect(), Rect::NOTHING);
        assert!(fresh.notes().is_empty());
    }

    // -----------------------------------------------------------------
    // Limits and validation (#63, Overlands #1336 C2, C8)
    // -----------------------------------------------------------------

    /// A recipe whose `cap` is already used up, and limits small enough to
    /// reach without building a 4 096-note track in a test.
    fn tiny_limits() -> EditorLimits {
        EditorLimits::from_envelope(Envelope {
            max_nodes: 3,
            max_connections_per_port: 2,
            max_track_events: 3,
            max_instruments: 2,
            max_tracks: 2,
            max_instrument_id_bytes: 4,
        })
    }

    fn driver_with_limits(recipe: SequenceRecipe, limits: EditorLimits) -> Driver {
        let mut state = SequenceEditorState::default();
        state.set_limits(limits);
        Driver::with_state(recipe, state)
    }

    /// Whether a button whose label starts with `prefix` is offered *and*
    /// enabled, read from the AccessKit tree the way a screen reader does.
    fn offered(driver: &Driver, prefix: &str) -> bool {
        driver.nodes().any(|(_, n)| {
            n.role() == accesskit::Role::Button
                && n.label().is_some_and(|l| l.starts_with(prefix))
                && !n.is_disabled()
        })
    }

    /// **C8's acceptance, as one walk of the roster.** Every field of
    /// [`EditorLimits`] both refuses something at its cap and says so.
    ///
    /// The match is exhaustive, so a seventh cap cannot be added without an
    /// arm here that names the surface it guards — the shape
    /// `the_distinct_style_gives_every_role_its_own_colour` has for style
    /// roles and `Snap::ALL` has for the grid.
    #[test]
    fn every_cap_refuses_at_its_number_and_says_why() {
        let limits = tiny_limits();
        for cap in Cap::ALL {
            let n = limits.get(cap);
            let reason = limits.at(cap, n).full_reason();
            match cap {
                Cap::Instruments => {
                    let full = recipe_with(&["a", "b"], 1);
                    let mut driver = driver_with_limits(full, limits);
                    driver.frame(Vec::new());
                    assert!(
                        !offered(&driver, "Add instrument"),
                        "a {n}-instrument recipe still offers another"
                    );
                    let said = driver.hover_texts("Add instrument");
                    assert!(
                        said.iter().any(|t| t == &reason),
                        "the disabled Add does not say why: {said:?}"
                    );
                    let mut driver = driver_with_limits(recipe_with(&["a"], 1), limits);
                    driver.frame(Vec::new());
                    assert!(offered(&driver, "Add instrument"), "one below the cap");
                    driver.click("Add instrument");
                    assert_eq!(driver.recipe.instruments.len(), 2);
                }
                Cap::Tracks => {
                    let mut driver = driver_with_limits(recipe_with(&["a"], 2), limits);
                    driver.frame(Vec::new());
                    assert!(!offered(&driver, "Add track"), "a {n}-track recipe");
                    let said = driver.hover_texts("Add track");
                    assert!(
                        said.iter().any(|t| t == &reason),
                        "the disabled Add does not say why: {said:?}"
                    );
                    let mut driver = driver_with_limits(recipe_with(&["a"], 1), limits);
                    driver.frame(Vec::new());
                    assert!(offered(&driver, "Add track"));
                    driver.click("Add track");
                    assert_eq!(driver.recipe.tracks.len(), 2);
                }
                Cap::TrackEvents => {
                    // Both doors: the menu's item, and the double-click.
                    let mut recipe = recipe_with(&["a"], 1);
                    recipe.tracks[0].events = (0..3).map(|i| note("a", i as f32)).collect();
                    let mut driver = driver_with_limits(recipe, limits);
                    driver.frame(Vec::new());
                    let before = driver.recipe.tracks[0].events.len();
                    driver.open_track_menu(0);
                    assert!(
                        !offered(&driver, "Add a"),
                        "a full track still offers a note"
                    );
                    // In the menu itself: egui shows no tooltip at all
                    // while a menu is open, so a reason left to
                    // `on_disabled_hover_text` here would never be read.
                    assert!(
                        driver.texts().iter().any(|t| t == &reason),
                        "the menu's refused item does not say why: {:?}",
                        driver.texts()
                    );
                    driver.click("Add a");
                    driver.frame(Vec::new());
                    assert_eq!(
                        driver.recipe.tracks[0].events.len(),
                        before,
                        "a full track took another note from its menu"
                    );
                    let mut recipe = recipe_with(&["a"], 1);
                    recipe.tracks[0].events = vec![note("a", 0.0)];
                    let mut driver = driver_with_limits(recipe, limits);
                    driver.frame(Vec::new());
                    driver.add_note_by_menu(0);
                    assert_eq!(
                        driver.recipe.tracks[0].events.len(),
                        2,
                        "one below the cap the menu still adds"
                    );
                }
                Cap::InstrumentIdBytes => {
                    let recipe = recipe_with(&["ab"], 1);
                    let mut driver = driver_with_limits(recipe, limits);
                    driver.focus("ab");
                    driver.frame(vec![typed("cdefg")]);
                    driver.frame(vec![key(egui::Key::Enter)]);
                    assert_eq!(
                        instrument_ids(&driver.recipe),
                        ["ab"],
                        "a name past {n} bytes was applied anyway"
                    );
                    assert!(
                        driver.paints(&format!("the limit is {n}")),
                        "the field does not say the limit: {:?}",
                        driver.texts()
                    );
                }
                // The two the canvas owns; they are driven in graph.rs,
                // where the canvas's own harness is. Named here so the
                // match stays exhaustive and the roster stays one list.
                Cap::Nodes | Cap::ConnectionsPerPort => {
                    assert!(!reason.is_empty());
                }
            }
        }
    }

    /// The count is on screen, and it changes colour as it fills.
    #[test]
    fn the_readout_says_how_many_of_how_many_and_warns_near_the_cap() {
        let limits = EditorLimits::from_envelope(Envelope {
            max_tracks: 10,
            ..Envelope::default()
        });
        for (tracks, want, tone) in [
            (1_usize, "1 / 10", CapTone::Quiet),
            (8, "8 / 10", CapTone::Warn),
            (10, "10 / 10", CapTone::Full),
        ] {
            let mut driver = driver_with_limits(recipe_with(&["a"], tracks), limits);
            driver.frame(Vec::new());
            assert!(
                driver.texts().iter().any(|t| t == want),
                "no {want} readout with {tracks} tracks: {:?}",
                driver.texts()
            );
            let style = distinct_style();
            let painted = text_painted(&driver.out, want)
                .unwrap_or_else(|| panic!("{want} was not painted"))
                .1;
            match tone {
                CapTone::Warn => assert_eq!(painted, style.warn, "{want} is not in the warn tone"),
                CapTone::Full => assert_eq!(painted, style.error, "{want} is not in the full tone"),
                CapTone::Quiet => {
                    assert_ne!(painted, style.warn);
                    assert_ne!(painted, style.error);
                }
            }
        }
    }

    /// **C2's acceptance.** A track with notes asks and does not go until
    /// Remove; an empty one goes at once.
    #[test]
    fn a_track_with_notes_asks_before_it_goes_and_an_empty_one_does_not() {
        let mut recipe = recipe_with(&["a"], 2);
        recipe.tracks[0].events = vec![note("a", 0.0), note("a", 1.0)];
        let mut driver = Driver::new(recipe);

        driver.remove_track_by_cross(0);
        assert_eq!(driver.recipe.tracks.len(), 2, "it went without asking");
        assert!(driver.state.awaiting_confirmation());
        assert!(
            driver.paints("Remove track 1?") && driver.paints("2 notes"),
            "the question does not say what is lost: {:?}",
            driver.texts()
        );
        let res = driver.click("Remove");
        assert_eq!(driver.recipe.tracks.len(), 1);
        assert!(!driver.state.awaiting_confirmation());
        assert!(res.changed && res.rebake, "a confirmed removal commits");

        // The one left is empty, so it goes on the click.
        driver.frame(Vec::new());
        driver.remove_track_by_cross(0);
        assert!(
            driver.recipe.tracks.is_empty(),
            "an empty track still asked"
        );
        assert!(!driver.state.awaiting_confirmation());
    }

    /// Both doors ask. Step 8 gave a track a second way out — "Remove this
    /// track" in its own menu — so a confirmation wired to the gutter's
    /// cross alone would have left the menu unguarded.
    #[test]
    fn both_ways_of_removing_a_track_ask_the_same_question() {
        let mut recipe = recipe_with(&["a"], 2);
        recipe.tracks[0].events = vec![note("a", 0.0)];
        for by_menu in [false, true] {
            let mut driver = Driver::new(recipe.clone());
            if by_menu {
                driver.remove_track_by_menu(0);
            } else {
                driver.remove_track_by_cross(0);
            }
            assert_eq!(
                driver.recipe.tracks.len(),
                2,
                "removed without asking (by_menu = {by_menu})"
            );
            assert!(driver.state.awaiting_confirmation(), "by_menu = {by_menu}");
            driver.click("Remove");
            assert_eq!(driver.recipe.tracks.len(), 1, "by_menu = {by_menu}");
        }
    }

    /// An instrument other notes name asks and says how many; an unused one
    /// goes at once.
    #[test]
    fn an_instrument_notes_use_asks_and_says_how_many() {
        let mut recipe = recipe_with(&["used", "spare"], 1);
        recipe.tracks[0].events = vec![note("used", 0.0), note("used", 1.0), note("used", 2.0)];
        let mut driver = Driver::new(recipe);
        driver.frame(Vec::new());

        // The spare is named by nothing, so its cross removes it outright.
        driver.remove_instrument_by_cross(1);
        assert_eq!(instrument_ids(&driver.recipe), ["used"]);
        assert!(!driver.state.awaiting_confirmation());

        driver.remove_instrument_by_cross(0);
        assert_eq!(instrument_ids(&driver.recipe), ["used"], "it went anyway");
        assert!(
            driver.paints("Remove \u{201C}used\u{201D}?") && driver.paints("3 notes"),
            "the question does not name it or count them: {:?}",
            driver.texts()
        );
        driver.click("Remove");
        assert!(driver.recipe.instruments.is_empty());
    }

    /// Keep and Escape both put the question away and leave everything
    /// where it was, and Escape spends exactly one press and reports it.
    #[test]
    fn keep_and_escape_both_cancel_a_removal() {
        let mut recipe = recipe_with(&["a"], 2);
        recipe.tracks[0].events = vec![note("a", 0.0)];

        let mut driver = Driver::new(recipe.clone());
        driver.remove_track_by_cross(0);
        driver.click("Keep");
        assert_eq!(driver.recipe.tracks.len(), 2, "Keep removed it");
        assert!(!driver.state.awaiting_confirmation());

        let mut driver = Driver::new(recipe);
        driver.remove_track_by_cross(0);
        driver.point_at_timeline();
        assert!(driver.state.awaiting_confirmation());
        driver.frame(vec![key(egui::Key::Escape)]);
        assert_eq!(driver.recipe.tracks.len(), 2, "Escape removed it");
        assert!(!driver.state.awaiting_confirmation());
        assert!(
            driver.state.took_escape(),
            "the press was spent without saying so, and the host's Esc \
             ladder would close the pop-out under the question"
        );
        // And it was exactly one press: the next one is the selection's.
        driver.frame(Vec::new());
        assert!(!driver.state.took_escape());
    }

    /// A confirmed removal is one committed edit, so one undo brings it
    /// back — the confirmation is a second look, not a replacement for the
    /// history step 7a gave the recipe.
    #[test]
    fn one_undo_brings_back_a_confirmed_removal() {
        let mut recipe = recipe_with(&["a"], 2);
        recipe.tracks[0].events = vec![note("a", 0.0), note("a", 1.5)];
        let mut driver = Driver::new(recipe);
        driver.frame(Vec::new());

        driver.remove_track_by_cross(0);
        driver.click("Remove");
        driver.frame(Vec::new());
        assert_eq!(driver.recipe.tracks.len(), 1);

        driver.point_at_timeline();
        driver.frame(vec![chord(egui::Key::Z, egui::Modifiers::COMMAND)]);
        assert_eq!(
            driver.recipe.tracks.len(),
            2,
            "one undo did not bring it back"
        );
        assert_eq!(
            note_ids(&driver.recipe)[0],
            ["a", "a"],
            "the notes came back with it"
        );
    }

    /// The buttons are nowhere near the cross that summoned them.
    ///
    /// An affordance at the point the user clicks is hovered by that same
    /// click, and a question whose Remove lands under a pointer that has
    /// just pressed ✖ is a double-click away from removing something
    /// nobody looked at. Measured against the *published* rects rather than
    /// worked out again here.
    #[test]
    fn the_question_is_drawn_clear_of_every_cross_that_could_have_asked_it() {
        let mut recipe = recipe_with(&["used", "spare"], 3);
        recipe.tracks[0].events = vec![note("used", 0.0)];
        recipe.tracks[1].events = vec![note("used", 2.0)];
        for by_menu in [false, true] {
            let mut driver = Driver::new(recipe.clone());
            if by_menu {
                driver.remove_track_by_menu(1);
            } else {
                driver.remove_track_by_cross(1);
            }
            driver.frame(Vec::new());
            let bar = driver
                .state
                .confirmation_rect()
                .expect("no question is up (by_menu = {by_menu})");
            let asks = driver.state.removal_asks().to_vec();
            assert!(!asks.is_empty(), "no removal control was published");
            for cross in &asks {
                assert!(
                    !bar.intersects(*cross),
                    "the question at {bar:?} covers a removal cross at {cross:?} \
                     (by_menu = {by_menu})"
                );
            }
            // And the buttons themselves, not just the frame around them.
            for button in ["Keep", "Remove"] {
                let node = driver.button(button).expect("button");
                let b = driver
                    .nodes()
                    .find(|(id, _)| *id == node)
                    .and_then(|(_, n)| n.bounds())
                    .expect("bounds");
                let rect = Rect::from_min_max(
                    Pos2::new(b.x0 as f32, b.y0 as f32),
                    Pos2::new(b.x1 as f32, b.y1 as f32),
                );
                for cross in &asks {
                    assert!(
                        !rect.intersects(*cross),
                        "{button} at {rect:?} sits on a cross at {cross:?}"
                    );
                }
            }
        }
    }

    /// While a question is up, nothing else offers a removal: the bar
    /// pushed every row below it down, so a pointer left where it clicked
    /// is resting on some *other* row's cross.
    #[test]
    fn no_other_removal_is_offered_while_a_question_is_up() {
        let mut recipe = recipe_with(&["used", "spare"], 2);
        recipe.tracks[0].events = vec![note("used", 0.0)];
        let mut driver = Driver::new(recipe);
        driver.frame(Vec::new());
        assert!(offered(&driver, "\u{2716}"), "the crosses start live");

        driver.remove_track_by_cross(0);
        driver.frame(Vec::new());
        assert!(
            !offered(&driver, "\u{2716}"),
            "an instrument can still be removed while a question is up"
        );
        driver.click("Keep");
        driver.frame(Vec::new());
        assert!(offered(&driver, "\u{2716}"), "they never came back");
    }
}
