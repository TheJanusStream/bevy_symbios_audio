//! The caps the editors hold a document to, and the one place each is named
//! (#63, Overlands #1336 C8).
//!
//! Every cap here is enforced at the record boundary by the host — Overlands'
//! sanitiser deletes the excess after the edit — so until this module the
//! editor's side of the bargain was to not know. The 65th instrument was
//! added, and a quarter of a second later the sanitiser dropped one; the
//! 4 097th note was pushed and truncated away; a name past 128 bytes was cut
//! back to a UTF-8 boundary. All of it silently, with no count anywhere on
//! screen. That is the #1197 theme "caps enforced by deletion instead of
//! refusal at the point of edit", and Overlands' own `ui::room::caps` is the
//! shape of the fix: refuse at every insert, disable the control with the
//! reason, show `N / cap`.
//!
//! The numbers are not this crate's to invent. They are symbios-audio's
//! [`Envelope`] — the same six counts Overlands pins in
//! `the_caps_match_the_upstream_envelope` — so an editor built on
//! [`EditorLimits::default`] refuses exactly what the record boundary would
//! have deleted. A host that wants tighter ones passes its own
//! [`Envelope`]; nothing here invents a seventh number.
//!
//! # One roster
//!
//! [`Cap::ALL`] is the only list. A readout, a disabled reason and the
//! arithmetic that refuses an add all read the same variant through
//! [`EditorLimits::get`], so a readout cannot name a cap the maths does not
//! enforce — the lesson of `Snap::ALL` and `KIND_LABELS` applied to the one
//! table that comes from upstream. [`EditorLimits::from_envelope`]
//! destructures [`Envelope`] exhaustively, so a field added upstream is a
//! compile error here rather than a cap the editor quietly does not hold.

use bevy_egui::egui;

use crate::Envelope;

use super::style::EditorStyle;

/// One bounded thing in an audio document.
///
/// The six counts of symbios-audio's [`Envelope`], as things the editor can
/// name, count and refuse.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Cap {
    /// Nodes in one patch's graph ([`Envelope::max_nodes`]).
    Nodes,
    /// Connections wired into one input port
    /// ([`Envelope::max_connections_per_port`]).
    ConnectionsPerPort,
    /// Notes on one track ([`Envelope::max_track_events`]).
    TrackEvents,
    /// Instruments in one recipe ([`Envelope::max_instruments`]).
    Instruments,
    /// Tracks in one recipe ([`Envelope::max_tracks`]).
    Tracks,
    /// Length of an instrument's id, in bytes
    /// ([`Envelope::max_instrument_id_bytes`]).
    ///
    /// The one cap that is a *size* rather than a count of things: nothing
    /// adds a byte, so it is read through [`EditorLimits::get`] by the name
    /// field rather than gating an Add button.
    InstrumentIdBytes,
}

impl Cap {
    /// Every cap, in [`Envelope`]'s own field order.
    ///
    /// The one roster: a readout that walks this cannot name a cap the
    /// arithmetic does not enforce.
    pub const ALL: [Self; 6] = [
        Self::Nodes,
        Self::ConnectionsPerPort,
        Self::TrackEvents,
        Self::Instruments,
        Self::Tracks,
        Self::InstrumentIdBytes,
    ];

    /// What the cap counts, plural — the word a readout and a reason use.
    #[must_use]
    pub fn plural(self) -> &'static str {
        match self {
            Self::Nodes => "nodes",
            Self::ConnectionsPerPort => "connections",
            Self::TrackEvents => "notes",
            Self::Instruments => "instruments",
            Self::Tracks => "tracks",
            Self::InstrumentIdBytes => "bytes",
        }
    }

    /// What holds them, as a sentence's subject.
    #[must_use]
    pub fn holder(self) -> &'static str {
        match self {
            Self::Nodes => "This patch",
            Self::ConnectionsPerPort => "This port",
            Self::TrackEvents => "This track",
            Self::Instruments | Self::Tracks => "This recipe",
            Self::InstrumentIdBytes => "This name",
        }
    }

    /// What to do about it, as the second half of the refusal.
    #[must_use]
    pub fn remedy(self) -> &'static str {
        match self {
            Self::Nodes => "Delete a node first",
            Self::ConnectionsPerPort => "Remove a connection first",
            Self::TrackEvents => "Delete a note first",
            Self::Instruments => "Remove an instrument first",
            Self::Tracks => "Remove a track first",
            Self::InstrumentIdBytes => "Use a shorter name",
        }
    }
}

/// How close a count sits to its cap — the readout's colour, and whether the
/// next add is refused.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CapTone {
    /// Room to spare.
    Quiet,
    /// At or past 80 % of the cap: the readout is worth reading.
    Warn,
    /// At the cap: the next add is refused.
    Full,
}

/// The fraction of a cap at which its readout stops being background.
///
/// Overlands' `ui::room::caps` warns at the same point, so a user who has
/// learned what an amber count means in the room reads the same thing here.
const WARN_FRACTION: f32 = 0.8;

impl CapTone {
    /// The colour a readout in this tone is drawn in.
    #[must_use]
    pub fn colour(self, ui: &egui::Ui, style: &EditorStyle) -> egui::Color32 {
        match self {
            Self::Quiet => ui.visuals().weak_text_color(),
            Self::Warn => style.warn,
            Self::Full => style.error,
        }
    }
}

/// The caps an editor holds its document to.
///
/// [`Default`] is symbios-audio's [`Envelope::default`] — the bounds the
/// record boundary already enforces — so an editor that is given no limits
/// refuses exactly what the sanitiser would otherwise have deleted.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct EditorLimits {
    /// The counts themselves. Public so a host can read what it set, and so
    /// the only numbers in play are upstream's.
    pub envelope: Envelope,
}

impl Default for EditorLimits {
    fn default() -> Self {
        Self::from_envelope(Envelope::default())
    }
}

impl EditorLimits {
    /// Hold an editor to `envelope`'s counts.
    #[must_use]
    pub fn from_envelope(envelope: Envelope) -> Self {
        // Destructured exhaustively and reassembled rather than stored
        // whole: a count added upstream is a compile error here — a cap
        // with no `Cap` variant, no readout and no refusal — instead of a
        // bound the editor silently does not hold.
        let Envelope {
            max_nodes,
            max_connections_per_port,
            max_track_events,
            max_instruments,
            max_tracks,
            max_instrument_id_bytes,
        } = envelope;
        Self {
            envelope: Envelope {
                max_nodes,
                max_connections_per_port,
                max_track_events,
                max_instruments,
                max_tracks,
                max_instrument_id_bytes,
            },
        }
    }

    /// The number `cap` is held to.
    #[must_use]
    pub fn get(&self, cap: Cap) -> usize {
        match cap {
            Cap::Nodes => self.envelope.max_nodes,
            Cap::ConnectionsPerPort => self.envelope.max_connections_per_port,
            Cap::TrackEvents => self.envelope.max_track_events,
            Cap::Instruments => self.envelope.max_instruments,
            Cap::Tracks => self.envelope.max_tracks,
            Cap::InstrumentIdBytes => self.envelope.max_instrument_id_bytes,
        }
    }

    /// `used` against `cap`: what a control asks before it offers an add,
    /// and what a readout draws.
    #[must_use]
    pub fn at(&self, cap: Cap, used: usize) -> CapState {
        CapState {
            cap,
            used,
            limit: self.get(cap),
        }
    }

    /// Every cap paired with its number, in [`Cap::ALL`]'s order.
    #[must_use]
    pub fn caps(&self) -> [(Cap, usize); Cap::ALL.len()] {
        Cap::ALL.map(|cap| (cap, self.get(cap)))
    }
}

/// One cap, and how much of it is used.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CapState {
    /// Which cap.
    pub cap: Cap,
    /// How many there are now.
    pub used: usize,
    /// How many there may be.
    pub limit: usize,
}

impl CapState {
    /// Whether one more fits.
    #[must_use]
    pub fn has_room(&self) -> bool {
        self.used < self.limit
    }

    /// How close this is to the cap.
    #[must_use]
    pub fn tone(&self) -> CapTone {
        if !self.has_room() {
            CapTone::Full
        } else if self.used as f32 >= self.limit as f32 * WARN_FRACTION {
            CapTone::Warn
        } else {
            CapTone::Quiet
        }
    }

    /// The readout: `"5 / 64"`.
    #[must_use]
    pub fn readout(&self) -> String {
        format!("{} / {}", self.used, self.limit)
    }

    /// Why an add is refused — a disabled control's
    /// `on_disabled_hover_text`, which is the only hover a disabled widget
    /// ever shows (#1289).
    #[must_use]
    pub fn full_reason(&self) -> String {
        format!(
            "{} is full: {} of {} {}. {}",
            self.cap.holder(),
            self.used,
            self.limit,
            self.cap.plural(),
            self.cap.remedy()
        )
    }

    /// What the readout's own hover says, whether or not there is room.
    #[must_use]
    pub fn hover(&self) -> String {
        if self.has_room() {
            format!(
                "{} of {} {} — the most a saved recipe can hold",
                self.used,
                self.limit,
                self.cap.plural()
            )
        } else {
            self.full_reason()
        }
    }
}

/// Draw `state`'s `"5 / 64"` readout, coloured by how close it is.
///
/// The count beside the control that adds one, so the number a user needs
/// before the refusal is on screen rather than behind a hover.
pub fn cap_readout(ui: &mut egui::Ui, state: CapState, style: &EditorStyle) -> egui::Response {
    let tone = state.tone();
    ui.label(
        egui::RichText::new(state.readout())
            .small()
            .color(tone.colour(ui, style)),
    )
    .on_hover_text(state.hover())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The defaults are upstream's, not a copy of them.
    #[test]
    fn the_default_limits_are_the_upstream_envelope() {
        assert_eq!(EditorLimits::default().envelope, Envelope::default());
        let limits = EditorLimits::default();
        assert_eq!(limits.get(Cap::Nodes), 256);
        assert_eq!(limits.get(Cap::ConnectionsPerPort), 64);
        assert_eq!(limits.get(Cap::TrackEvents), 4096);
        assert_eq!(limits.get(Cap::Instruments), 64);
        assert_eq!(limits.get(Cap::Tracks), 64);
        assert_eq!(limits.get(Cap::InstrumentIdBytes), 128);
    }

    /// **The roster guard.** Every [`Cap`] reads a *different* field of the
    /// envelope, and between them they read all six.
    ///
    /// Six distinct numbers go in; if two variants read one field, or one
    /// reads a field another already has, the answers collide. The
    /// exhaustive destructure in [`EditorLimits::from_envelope`] is the
    /// other half: it catches a seventh field arriving upstream, which no
    /// walk of `Cap::ALL` could.
    #[test]
    fn every_cap_reads_its_own_field_of_the_envelope() {
        let limits = EditorLimits::from_envelope(Envelope {
            max_nodes: 1,
            max_connections_per_port: 2,
            max_track_events: 3,
            max_instruments: 4,
            max_tracks: 5,
            max_instrument_id_bytes: 6,
        });
        let mut seen: Vec<usize> = limits.caps().iter().map(|(_, n)| *n).collect();
        assert_eq!(seen, [1, 2, 3, 4, 5, 6], "in Envelope's own field order");
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), Cap::ALL.len(), "no two caps read one field");
    }

    /// Every cap can say what it is and why it refused, in its own words.
    #[test]
    fn every_cap_has_a_reason_naming_its_own_number() {
        let limits = EditorLimits::default();
        for cap in Cap::ALL {
            let limit = limits.get(cap);
            let state = limits.at(cap, limit);
            let reason = state.full_reason();
            assert!(
                reason.contains(&limit.to_string()),
                "{cap:?}'s reason does not name its number: {reason}"
            );
            assert!(
                reason.contains(cap.plural()),
                "{cap:?}'s reason does not say what it counts: {reason}"
            );
            assert!(
                reason.contains(cap.remedy()),
                "{cap:?}'s reason does not say what to do: {reason}"
            );
            assert_eq!(state.readout(), format!("{limit} / {limit}"));
        }
    }

    /// Quiet, then warn, then full — and full is exactly where the next add
    /// is refused, not a point either side of it.
    #[test]
    fn the_tone_turns_at_four_fifths_and_full_is_where_the_add_is_refused() {
        let limits = EditorLimits::from_envelope(Envelope {
            max_tracks: 10,
            ..Envelope::default()
        });
        let tone = |used| limits.at(Cap::Tracks, used).tone();
        assert_eq!(tone(0), CapTone::Quiet);
        assert_eq!(tone(7), CapTone::Quiet);
        assert_eq!(tone(8), CapTone::Warn, "80 % of ten");
        assert_eq!(tone(9), CapTone::Warn);
        assert_eq!(tone(10), CapTone::Full);
        assert!(limits.at(Cap::Tracks, 9).has_room());
        assert!(!limits.at(Cap::Tracks, 10).has_room());
        assert!(
            !limits.at(Cap::Tracks, 11).has_room(),
            "a document already past the cap offers no add either"
        );
    }

    /// A readout's hover says the number whether or not there is room —
    /// a disabled control shows only `on_disabled_hover_text` (#1289), so
    /// the readout is where the count is always readable.
    #[test]
    fn the_readout_hover_names_the_count_either_way() {
        let limits = EditorLimits::default();
        let room = limits.at(Cap::Instruments, 5);
        assert!(room.hover().contains("5 of 64 instruments"));
        let full = limits.at(Cap::Instruments, 64);
        assert_eq!(full.hover(), full.full_reason());
    }
}
