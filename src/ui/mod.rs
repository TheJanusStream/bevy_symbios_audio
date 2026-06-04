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
//! - [`sequence`] — a [`sequence::sequence_recipe_editor`] timeline (transport,
//!   instruments, draggable track/event lanes) for a whole
//!   [`crate::sequence::SequenceRecipe`], with [`sequence::active_instrument_canvas`]
//!   opening any instrument's patch in the [`graph`] canvas.
//!
//! Every editor composes the same [`EditorResponse`] contract.
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

pub mod graph;
pub mod node;
pub mod preview;
pub mod sequence;

pub use graph::{PatchEditorState, audio_patch_canvas};
pub use node::{
    adsr_envelope_editor, biquad_bandpass_editor, biquad_highpass_editor, biquad_lowpass_editor,
    brown_noise_editor, gain_editor, gate_editor, lfo_editor, mix_editor, node_kind_body,
    node_kind_editor, node_kind_label, pink_noise_editor, sawtooth_osc_editor, sine_osc_editor,
    square_osc_editor, triangle_osc_editor, white_noise_editor,
};
pub use preview::{
    AudioEditorPlugin, AudioMonitor, MonitorRequest, MonitorStatus, waveform, waveform_sized,
};
pub use sequence::{SequenceEditorState, active_instrument_canvas, sequence_recipe_editor};

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
        let r = ui.add(egui::DragValue::new(val).speed(speed));
        let res = EditorResponse {
            changed: r.changed(),
            rebake: r.drag_stopped() || (r.changed() && !r.dragged()),
        };
        if res.changed {
            *val = val.clamp(*range.start(), *range.end());
        }
        res
    })
    .inner
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
