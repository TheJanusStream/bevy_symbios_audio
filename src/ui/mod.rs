//! Egui editor widgets for the audio patch schema (Cargo feature `egui`).
//!
//! Reusable, composable controls so any `bevy_egui` host can embed audio
//! parameter editing without re-deriving editor code.  The design mirrors the
//! sibling crate `bevy_symbios_texture`'s `ui` module so downstream consumers
//! (Overlands) get a consistent surface across both crates.
//!
//! # Layering
//!
//! Where texture configs are *flat* (one struct per texture), an audio patch
//! is a *graph*, so this module is built in tiers:
//!
//! - [`node`] — per-node config editors (one widget group per
//!   [`crate::node::NodeKind`] inner config) plus [`node::node_kind_editor`], a
//!   kind picker + body for a single node.
//! - [`graph`] — a pannable/zoomable visual node-graph canvas
//!   ([`graph::audio_patch_canvas`]) that edits a whole
//!   [`crate::patch::AudioPatch`]: drag nodes, wire ports, set the output.
//! - [`preview`] — a pure-egui [`preview::waveform`] widget plus a Bevy
//!   bake-and-play monitor ([`preview::AudioEditorPlugin`]) for auditioning
//!   edits.
//! - [`audition`] — [`audition::audition_strip`], the row a host puts above
//!   an editor to hear it: Audition, Stop, an Auto re-bake, a status chip, a
//!   caption saying what is played, and the waveform. It returns the
//!   [`preview::MonitorRequest`] to write rather than writing it.
//! - [`sequence`] — a [`sequence::sequence_recipe_editor`] timeline (transport,
//!   instruments, draggable track/event lanes) for a whole
//!   [`crate::sequence::SequenceRecipe`], with [`sequence::active_instrument_canvas`]
//!   opening any instrument's patch in the [`graph`] canvas.
//! - [`evolve`] / [`io`] — cross-cutting polish wired into the editors above:
//!   `symbios_genetics`-backed Mutate / Reroll seed helpers ([`evolve`]) and
//!   a reusable JSON copy/paste section ([`io::json_io`]).
//! - [`style`] — [`EditorStyle`], the named colour roles every widget above
//!   paints with. A host sets its own with [`set_editor_style`] when its
//!   theme changes; with none set, the widgets derive one from the `Visuals`
//!   they are drawn with ([`EditorStyle::from_visuals`]).
//!
//! Every editor composes the same [`EditorResponse`] contract.
//!
//! Buttons say what they do in words ("Add node", "Fit view", "Delete
//! note"). The glyphs that stay are the remove cross `✖` and the valid
//! check `✔`, the code points Overlands' affordances use for the same acts,
//! the audition's `▶` and `⏹`, which Overlands draws on its own Play and
//! Stop, and the pencil of an instrument's `✏ Edit` toggle, which Overlands
//! puts on its own Edit audio button.
//!
//! The two canvases, [`graph::audio_patch_canvas`] and
//! [`sequence::active_instrument_canvas`], claim all the space left in the
//! `Ui` they are given, so a host draws them **last**. Controls that must
//! stay visible go above them, and a sequence editor goes beside them in a
//! panel. In an `egui::Window`, anything drawn after a canvas is never seen,
//! and the window grows every frame until it reaches its constraint. Each
//! function's docs explain why, and the `host_window` example is the layout
//! to copy.
//!
//! # The change/commit contract
//!
//! Every editor returns an [`EditorResponse`] with two flags:
//!
//! - [`EditorResponse::changed`] — a value moved this frame, *including
//!   mid-drag*.  Write the edited config back to your resource so the widget
//!   doesn't visually snap back.
//! - [`EditorResponse::rebake`] — a value was *committed* (a drag ended, or a
//!   non-drag widget changed).  Trigger an expensive re-bake / re-play only
//!   when this is `true`, so dragging a slider doesn't re-render audio every
//!   frame.
//!
//! This matches texture's `(writeback, regen)` tuple, recast as a named struct
//! so the many sub-editors of a graph compose cleanly via [`EditorResponse::merge`].
//!
//! All egui access goes through `bevy_egui::egui` (never a direct `egui`
//! dependency) so the widgets stay pinned to the host's `bevy_egui` version.

use bevy_egui::egui;

pub mod audition;
pub mod evolve;
pub mod graph;
mod history;
pub mod io;
pub mod node;
pub mod preview;
pub mod sequence;
pub mod style;
#[cfg(test)]
mod test_paint;

pub use audition::{AUTO_QUIET_SECS, AuditionSource, AuditionState, audition_strip};
pub use evolve::{mutate_node_kind, mutate_patch, randomize_seed};
pub use graph::{PatchEditorState, WireGeom, audio_patch_canvas};
pub use io::{JsonIoState, json_io};
pub use node::{
    adsr_envelope_editor, biquad_bandpass_editor, biquad_highpass_editor, biquad_lowpass_editor,
    brown_noise_editor, gain_editor, gate_editor, lfo_editor, mix_editor, node_kind_body,
    node_kind_editor, node_kind_label, pink_noise_editor, sawtooth_osc_editor, sine_osc_editor,
    square_osc_editor, triangle_osc_editor, white_noise_editor,
};
pub use preview::{
    AudioEditorPlugin, AudioMonitor, MonitorRequest, MonitorStatus, waveform, waveform_sized,
};
pub use sequence::{
    NoteGeom, SequenceEditorState, Snap, active_instrument_canvas, sequence_recipe_editor,
};
pub use style::{EditorStyle, clear_editor_style, editor_style, set_editor_style};

/// Outcome of running an editor widget for one frame.
///
/// See the [module docs](crate::ui#the-changecommit-contract) for the
/// semantics of the two flags and how a host should react to them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EditorResponse {
    /// A value moved this frame, *including mid-drag*.  Persist the edited
    /// config so the widget doesn't snap back on the next frame.
    pub changed: bool,
    /// A value was *committed* — a drag ended or a non-drag widget changed.
    /// Use this to gate expensive work (re-baking / re-playing audio).
    pub rebake: bool,
}

impl EditorResponse {
    /// The neutral response — nothing changed and nothing to re-bake.
    pub const NONE: Self = Self {
        changed: false,
        rebake: false,
    };

    /// Fold another response into this one (logical OR of both flags).
    #[inline]
    pub fn merge(&mut self, other: EditorResponse) {
        self.changed |= other.changed;
        self.rebake |= other.rebake;
    }

    /// Combine two responses, returning the merged result.  Handy for
    /// one-line composition: `a.or(b).or(c)`.
    #[inline]
    #[must_use]
    pub fn or(mut self, other: EditorResponse) -> Self {
        self.merge(other);
        self
    }
}

// ---------------------------------------------------------------------------
// Shared widget helpers
// ---------------------------------------------------------------------------
//
// These translate raw egui `Response`s into the [`EditorResponse`] contract.
// `node.rs`'s `impl_node_editor!` macro calls them; they're `pub` so later
// phases (graph canvas, sequence timeline) and external hosts can reuse the
// exact same debouncing.

/// Add a slider and report drag-aware change/commit flags.
///
/// `changed` fires on any movement (including mid-drag) so the caller can
/// write the value back; `rebake` fires only when the drag stops or a
/// non-drag edit (typed value) lands — never every frame of a continuous drag.
pub fn slider_debounced(ui: &mut egui::Ui, slider: impl egui::Widget) -> EditorResponse {
    let r = ui.add(slider);
    EditorResponse {
        changed: r.changed(),
        rebake: r.drag_stopped() || (r.changed() && !r.dragged()),
    }
}

/// Horizontal labelled [`egui::DragValue`] for an `f32`, clamped to `range`.
///
/// Used for wide-range "amount" fields (e.g. an LFO's depth/offset) where a
/// slider would be unwieldy.  Same drag-aware debouncing as
/// [`slider_debounced`].  The value is clamped after editing rather than via
/// `DragValue`'s own range API so the helper stays version-agnostic.
pub fn drag_debounced(
    ui: &mut egui::Ui,
    label: &str,
    val: &mut f32,
    speed: f32,
    range: std::ops::RangeInclusive<f32>,
) -> EditorResponse {
    ui.horizontal(|ui| {
        ui.label(label);
        drag_value_debounced(ui, val, speed, range, "")
    })
    .inner
}

/// The [`egui::DragValue`] of [`drag_debounced`] without its label, with
/// `suffix` written into the value ("500 Hz").
///
/// This is the control half of a node body's grid row (#59): the label is
/// the cell to its left, so every label in the box shares one edge.
pub fn drag_value_debounced(
    ui: &mut egui::Ui,
    val: &mut f32,
    speed: f32,
    range: std::ops::RangeInclusive<f32>,
    suffix: &str,
) -> EditorResponse {
    let r = ui.add(egui::DragValue::new(val).speed(speed).suffix(suffix));
    let res = EditorResponse {
        changed: r.changed(),
        rebake: r.drag_stopped() || (r.changed() && !r.dragged()),
    };
    if res.changed {
        *val = val.clamp(*range.start(), *range.end());
    }
    res
}

/// Checkbox that treats every toggle as both a change and a commit (booleans
/// have no drag phase, so there's nothing to debounce).
pub fn bool_instant(ui: &mut egui::Ui, val: &mut bool, label: &str) -> EditorResponse {
    let r = ui.checkbox(val, label);
    EditorResponse {
        changed: r.changed(),
        rebake: r.changed(),
    }
}

/// Every non-ASCII glyph a string literal under `src/ui` draws must be in
/// egui's own default faces (#52).
///
/// This crate ships no font; the host does. The floor every host has is
/// egui's embedded tail — Ubuntu-Light, Noto Emoji and a small icon face —
/// and a host's own body face (Overlands puts Noto Sans in front) adds
/// Latin, Greek and Cyrillic and little else. So a symbol outside egui's
/// tail is an empty box everywhere, it looks like a styled button until
/// someone renders it, and nothing at build time can see it: five of them
/// shipped in 0.4.0. Coverage is not guessable from a glyph's looks — `✔`
/// (U+2714) is in Noto Emoji and `✓` (U+2713) is not — so this asks the
/// charmaps, which is the question epaint asks per character.
#[cfg(test)]
mod glyph_guard {
    use bevy_egui::egui;

    /// The contents of every `"…"` literal in `source`, `\u{…}` escapes
    /// decoded (an escaped code point is a glyph on screen like any other),
    /// every other escape consumed opaquely, `//` comments skipped, and a
    /// char literal recognised so `'"'` cannot open a string. A mis-lexed
    /// literal costs coverage, never a false failure.
    fn string_literals(source: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut chars = source.chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                '/' if chars.peek() == Some(&'/') => {
                    for c in chars.by_ref() {
                        if c == '\n' {
                            break;
                        }
                    }
                }
                '\'' => {
                    let mut probe = chars.clone();
                    let is_char = match probe.next() {
                        Some('\\') => {
                            probe.next();
                            probe.next() == Some('\'')
                        }
                        Some(_) => probe.next() == Some('\''),
                        None => false,
                    };
                    if is_char {
                        if chars.peek() == Some(&'\\') {
                            chars.next();
                        }
                        chars.next();
                        chars.next();
                    }
                }
                '"' => {
                    let mut literal = String::new();
                    loop {
                        match chars.next() {
                            None | Some('"') => break,
                            Some('\\') => match chars.next() {
                                Some('u') if chars.peek() == Some(&'{') => {
                                    chars.next();
                                    let mut hex = String::new();
                                    for h in chars.by_ref() {
                                        if h == '}' {
                                            break;
                                        }
                                        hex.push(h);
                                    }
                                    if let Some(c) =
                                        u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32)
                                    {
                                        literal.push(c);
                                    }
                                }
                                _ => {}
                            },
                            Some(other) => literal.push(other),
                        }
                    }
                    out.push(literal);
                }
                _ => {}
            }
        }
        out
    }

    /// Whether one of egui's default proportional faces owns a glyph for `c`.
    fn egui_default_faces_draw(defs: &egui::FontDefinitions, c: char) -> bool {
        use skrifa::MetadataProvider;
        defs.families[&egui::FontFamily::Proportional]
            .iter()
            .map(|name| &defs.font_data[name])
            .any(|face| {
                let font = skrifa::FontRef::from_index(&face.font, face.index)
                    .expect("an egui default face parses");
                font.charmap().map(c).is_some()
            })
    }

    #[test]
    fn every_editor_glyph_is_in_eguis_default_faces() {
        let defs = egui::FontDefinitions::default();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/ui");
        let mut sources: Vec<_> = std::fs::read_dir(&dir)
            .expect("src/ui is readable")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "rs"))
            .collect();
        sources.sort();
        assert!(sources.len() >= 5, "the walk found no editor sources");

        let mut missing = Vec::new();
        let mut seen = 0usize;
        for path in &sources {
            let source = std::fs::read_to_string(path).expect("editor source is readable");
            for literal in string_literals(&source) {
                for c in literal.chars().filter(|c| !c.is_ascii()) {
                    seen += 1;
                    if !egui_default_faces_draw(&defs, c) {
                        missing.push(format!(
                            "{c} U+{:04X} in {}",
                            u32::from(c),
                            path.file_name().unwrap_or_default().to_string_lossy()
                        ));
                    }
                }
            }
        }
        missing.sort();
        missing.dedup();
        assert!(
            missing.is_empty(),
            "editor glyphs egui's default faces cannot draw (tofu in every host):\n  {}",
            missing.join("\n  ")
        );
        // A floor, not a count: the scan's failure mode is reading nothing.
        // #58 took the emoji off the buttons (the plus, wastebasket, die,
        // clipboard and arrow circle, 12 literals) and the scan still saw 37:
        // 20 in drawn code (the remove cross, the valid check, the audition's
        // play and stop, the Edit pencil, the wire arrows, dashes, quotes,
        // ellipses, the pitch times sign) and 17 in tests. The floor was
        // expected to trip and did not, so it stays at 15, well under what is
        // drawn and well over what a blind scan reads.
        assert!(
            seen >= 15,
            "the scan saw only {seen} non-ASCII glyphs and has gone blind"
        );
    }

    /// The guard can tell a drawn glyph from tofu, and it sees through an
    /// escape: the shipped pencil, `\u{270E}`, is reported missing while its
    /// emoji-presentation sibling U+270F draws.
    #[test]
    fn the_glyph_guard_sees_the_shipped_tofu() {
        let defs = egui::FontDefinitions::default();
        assert!(!egui_default_faces_draw(&defs, '\u{270E}'));
        assert!(egui_default_faces_draw(&defs, '\u{270F}'));
        assert!(egui_default_faces_draw(&defs, 'e'));
        // The expected value is spelled with char literals: a `\u{…}` in a
        // string literal here would be the very tofu the walk above flags.
        let lexed = string_literals("ui.label(\"(\\u{270E})\")");
        assert_eq!(lexed.len(), 1);
        let chars: Vec<char> = lexed[0].chars().collect();
        assert_eq!(chars, vec!['(', '\u{270E}', ')']);
    }
}
