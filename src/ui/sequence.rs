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
//! - **Timeline** — one lane per track; events are blocks whose x is
//!   `time_beats`, width is `gate_beats`, with a translucent tail for
//!   `release_beats`.  Drag a block to move it, drag its right edge to resize
//!   the gate, double-click an empty lane to add an event.
//! - **Inspector** — full numeric editing of the selected event, plus delete.
//!
//! Like the rest of [`crate::ui`] this is pure egui returning an
//! [`EditorResponse`]; the host drives the bake-and-play monitor (Phase 3) off
//! `rebake`, baking the whole recipe with [`crate::mixdown::bake_sequence`].
//!
//! Editor-only view state (active instrument, per-instrument canvas layout,
//! selection, zoom) lives in [`SequenceEditorState`], never in the serialized
//! recipe.

use std::collections::HashMap;

use bevy_egui::egui::{self, Align2, Color32, Id, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};

use crate::patch::AudioPatch;
use crate::sequence::{Event, Instrument, SequenceRecipe, Track};

use super::{
    EditorResponse, PatchEditorState, audio_patch_canvas, drag_debounced, slider_debounced,
};

const RULER_H: f32 = 22.0;
const LANE_H: f32 = 34.0;
/// Left margin inside the timeline reserved for the per-lane remove button.
const GUTTER: f32 = 22.0;
const DEFAULT_PPB: f32 = 48.0;
/// Beat grid that drags snap to on release.
const SNAP: f32 = 0.25;
/// Smallest gate a block can be resized to (beats).
const MIN_GATE: f32 = 0.1;

/// Whether an in-progress block drag is moving the event or resizing its gate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum DragMode {
    #[default]
    Move,
    Resize,
}

/// Editor-side view state for the sequence editor — kept out of the
/// serialized [`SequenceRecipe`].  Construct with [`Default`].
#[derive(Clone, Debug)]
pub struct SequenceEditorState {
    /// Index of the instrument whose patch is open in the canvas, if any.
    active_instrument: Option<usize>,
    /// Per-instrument patch-canvas layout, keyed by instrument id so each
    /// instrument keeps its node positions when you switch away and back.
    canvas_states: HashMap<String, PatchEditorState>,
    /// Selected `(track, event)` for the inspector.
    selected_event: Option<(usize, usize)>,
    /// Timeline zoom, pixels per beat.
    px_per_beat: f32,
    /// Move-vs-resize for the active block drag.
    drag_mode: DragMode,
}

impl Default for SequenceEditorState {
    fn default() -> Self {
        Self {
            active_instrument: None,
            canvas_states: HashMap::new(),
            selected_event: None,
            px_per_beat: DEFAULT_PPB,
            drag_mode: DragMode::Move,
        }
    }
}

impl SequenceEditorState {
    /// Index of the instrument currently open for patch editing, if any.
    pub fn active_instrument(&self) -> Option<usize> {
        self.active_instrument
    }
}

/// Snap a beat value to the [`SNAP`] grid, clamped to `>= 0`.
fn snap_beat(beats: f32) -> f32 {
    ((beats / SNAP).round() * SNAP).max(0.0)
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

    egui::CollapsingHeader::new("Transport")
        .default_open(true)
        .show(ui, |ui| res.merge(transport(ui, recipe)));

    ui.separator();
    res.merge(instruments_panel(ui, recipe, state));

    ui.separator();
    res.merge(timeline(ui, recipe, state, id));

    ui.separator();
    egui::CollapsingHeader::new("Event inspector")
        .default_open(true)
        .show(ui, |ui| res.merge(event_inspector(ui, recipe, state)));

    res
}

/// Draw the active instrument's patch in the Phase-2 node canvas, or a hint if
/// none is selected.  Each instrument keeps its own canvas layout (keyed by
/// id) in `state`.
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
        ui.label("Select an instrument (\u{270E}) to edit its patch here.");
        return EditorResponse::NONE;
    };
    let inst_id = recipe.instruments[i].id.clone();
    ui.label(format!("Patch for instrument \u{201C}{inst_id}\u{201D}"));
    let canvas_state = state.canvas_states.entry(inst_id).or_default();
    audio_patch_canvas(
        ui,
        &mut recipe.instruments[i].patch,
        canvas_state,
        id.with("inst_canvas"),
    )
}

// ---------------------------------------------------------------------------
// Transport
// ---------------------------------------------------------------------------

fn transport(ui: &mut egui::Ui, recipe: &mut SequenceRecipe) -> EditorResponse {
    let mut res = EditorResponse::NONE;

    res.merge(slider_debounced(
        ui,
        egui::Slider::new(&mut recipe.bpm, 20.0..=300.0)
            .logarithmic(true)
            .text("BPM"),
    ));

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

    res.merge(drag_debounced(
        ui,
        "duration (beats)",
        &mut recipe.duration_beats,
        0.25,
        0.25..=512.0,
    ));

    let dur = recipe.duration_beats.max(0.0);
    let mut looping = recipe.loop_start_beats.is_some();
    if ui.checkbox(&mut looping, "seamless loop").changed() {
        recipe.loop_start_beats = looping.then_some(0.0);
        res.changed = true;
        res.rebake = true;
    }
    if let Some(loop_start) = recipe.loop_start_beats.as_mut() {
        res.merge(drag_debounced(
            ui,
            "loop start (beats)",
            loop_start,
            0.25,
            0.0..=dur,
        ));
    }
    if recipe.loop_start_beats.is_some() {
        res.merge(drag_debounced(
            ui,
            "crossfade (beats)",
            &mut recipe.loop_crossfade_beats,
            0.25,
            0.0..=dur.max(0.25),
        ));
    }

    res
}

// ---------------------------------------------------------------------------
// Instruments
// ---------------------------------------------------------------------------

fn instruments_panel(
    ui: &mut egui::Ui,
    recipe: &mut SequenceRecipe,
    state: &mut SequenceEditorState,
) -> EditorResponse {
    let mut res = EditorResponse::NONE;
    let mut add = false;
    let mut remove: Option<usize> = None;

    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Instruments").strong());
        if ui.button("\u{2795} Add").clicked() {
            add = true;
        }
    });

    for (i, inst) in recipe.instruments.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            let active = state.active_instrument == Some(i);
            if ui
                .selectable_label(active, "\u{270E}")
                .on_hover_text("Edit this instrument's patch")
                .clicked()
            {
                state.active_instrument = if active { None } else { Some(i) };
            }
            let r = ui.add(egui::TextEdit::singleline(&mut inst.id).desired_width(140.0));
            res.changed |= r.changed();
            // A rename changes which events resolve to this instrument, so it
            // affects the bake — but only commit a re-bake once editing ends.
            res.rebake |= r.lost_focus() && r.changed();
            let nodes = inst.patch.graph.nodes.len();
            ui.label(egui::RichText::new(format!("{nodes} node(s)")).weak());
            if ui
                .button("\u{2716}")
                .on_hover_text("Remove instrument")
                .clicked()
            {
                remove = Some(i);
            }
        });
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
        recipe.instruments.remove(i);
        match state.active_instrument {
            Some(a) if a == i => state.active_instrument = None,
            Some(a) if a > i => state.active_instrument = Some(a - 1),
            _ => {}
        }
        res.changed = true;
        res.rebake = true;
    }

    res
}

// ---------------------------------------------------------------------------
// Timeline
// ---------------------------------------------------------------------------

fn timeline(
    ui: &mut egui::Ui,
    recipe: &mut SequenceRecipe,
    state: &mut SequenceEditorState,
    id: Id,
) -> EditorResponse {
    let mut res = EditorResponse::NONE;

    ui.horizontal(|ui| {
        if ui.button("\u{2795} Track").clicked() {
            recipe.tracks.push(Track::default());
            res.changed = true;
            res.rebake = true;
        }
        res.merge(slider_debounced(
            ui,
            egui::Slider::new(&mut state.px_per_beat, 12.0..=160.0).text("px/beat"),
        ));
        ui.label(format!("{} track(s)", recipe.tracks.len()));
    });

    let ppb = state.px_per_beat.max(4.0);
    let dur = recipe.duration_beats.max(1.0);
    let loop_start = recipe.loop_start_beats;
    let crossfade = recipe.loop_crossfade_beats;
    let default_inst = recipe
        .instruments
        .first()
        .map(|i| i.id.clone())
        .unwrap_or_default();

    let mut total = dur;
    for t in &recipe.tracks {
        for e in &t.events {
            total = total.max(e.time_beats + e.gate_beats + e.release_beats);
        }
    }
    total = (total + 4.0).ceil();
    let lanes = recipe.tracks.len();

    let mut add_event: Option<(usize, f32)> = None;
    let mut remove_track: Option<usize> = None;

    egui::ScrollArea::horizontal()
        .id_salt(id.with("timeline_scroll"))
        .show(ui, |ui| {
            let width = GUTTER + total * ppb + 16.0;
            let height = RULER_H + (lanes.max(1) as f32) * LANE_H;
            let (rect, _) = ui.allocate_exact_size(Vec2::new(width, height), Sense::hover());
            let painter = ui.painter_at(rect);
            let bx = |beat: f32| rect.left() + GUTTER + beat * ppb;

            painter.rect_filled(rect, 0.0, Color32::from_gray(22));

            // Beat ruler + grid lines.
            let step = if ppb < 20.0 {
                4
            } else if ppb < 40.0 {
                2
            } else {
                1
            };
            let mut beat = 0i32;
            while (beat as f32) <= total {
                let x = bx(beat as f32);
                painter.line_segment(
                    [Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())],
                    Stroke::new(1.0, Color32::from_gray(44)),
                );
                if beat % step == 0 {
                    painter.text(
                        Pos2::new(x + 2.0, rect.top() + 1.0),
                        Align2::LEFT_TOP,
                        beat.to_string(),
                        egui::FontId::proportional(10.0),
                        Color32::from_gray(130),
                    );
                }
                beat += 1;
            }

            // Loop markers + crossfade shade.
            let lanes_top = rect.top() + RULER_H;
            if let Some(ls) = loop_start {
                let x = bx(ls);
                painter.line_segment(
                    [Pos2::new(x, lanes_top), Pos2::new(x, rect.bottom())],
                    Stroke::new(2.0, Color32::from_rgb(120, 200, 140)),
                );
            }
            let x_end = bx(dur);
            painter.line_segment(
                [Pos2::new(x_end, lanes_top), Pos2::new(x_end, rect.bottom())],
                Stroke::new(2.0, Color32::from_rgb(230, 190, 90)),
            );
            if crossfade > 0.0 {
                let x0 = bx((dur - crossfade).max(0.0));
                painter.rect_filled(
                    Rect::from_min_max(Pos2::new(x0, lanes_top), Pos2::new(x_end, rect.bottom())),
                    0.0,
                    Color32::from_rgba_unmultiplied(230, 190, 90, 28),
                );
            }

            // Lanes + events.
            for (ti, track) in recipe.tracks.iter_mut().enumerate() {
                let lane_top = lanes_top + ti as f32 * LANE_H;
                painter.rect_filled(
                    Rect::from_min_max(
                        Pos2::new(rect.left(), lane_top),
                        Pos2::new(rect.right(), lane_top + LANE_H),
                    ),
                    0.0,
                    if ti % 2 == 0 {
                        Color32::from_gray(30)
                    } else {
                        Color32::from_gray(34)
                    },
                );

                // Per-lane remove button.
                let x_rect = Rect::from_min_size(
                    Pos2::new(rect.left() + 3.0, lane_top + (LANE_H - 14.0) * 0.5),
                    Vec2::splat(14.0),
                );
                let x_resp = ui.interact(x_rect, id.with(("rm_track", ti)), Sense::click());
                painter.text(
                    x_rect.center(),
                    Align2::CENTER_CENTER,
                    "\u{2716}",
                    egui::FontId::proportional(13.0),
                    if x_resp.hovered() {
                        Color32::from_rgb(220, 120, 120)
                    } else {
                        Color32::from_gray(90)
                    },
                );
                if x_resp.clicked() {
                    remove_track = Some(ti);
                }

                // Empty-lane double-click adds an event at that beat.
                let bg = Rect::from_min_max(
                    Pos2::new(rect.left() + GUTTER, lane_top),
                    Pos2::new(rect.right(), lane_top + LANE_H),
                );
                let bg_resp = ui.interact(bg, id.with(("lane_bg", ti)), Sense::click());
                if bg_resp.double_clicked()
                    && let Some(p) = bg_resp.interact_pointer_pos()
                {
                    add_event = Some((ti, snap_beat((p.x - bx(0.0)) / ppb)));
                }

                for (ei, ev) in track.events.iter_mut().enumerate() {
                    let ex = bx(ev.time_beats);
                    let gate_w = (ev.gate_beats * ppb).max(6.0);
                    let etop = lane_top + 4.0;
                    let eh = LANE_H - 8.0;
                    let body = Rect::from_min_size(Pos2::new(ex, etop), Vec2::new(gate_w, eh));
                    let selected = state.selected_event == Some((ti, ei));

                    let tail_w = ev.release_beats * ppb;
                    if tail_w > 0.5 {
                        painter.rect_filled(
                            Rect::from_min_size(
                                Pos2::new(body.right(), etop),
                                Vec2::new(tail_w, eh),
                            ),
                            2.0,
                            Color32::from_rgba_unmultiplied(120, 160, 230, 60),
                        );
                    }
                    painter.rect_filled(
                        body,
                        3.0,
                        if selected {
                            Color32::from_rgb(90, 160, 250)
                        } else {
                            Color32::from_rgb(70, 110, 170)
                        },
                    );
                    painter.rect_stroke(
                        body,
                        3.0,
                        Stroke::new(1.0, Color32::from_gray(15)),
                        StrokeKind::Inside,
                    );
                    painter.text(
                        Pos2::new(body.left() + 4.0, body.center().y),
                        Align2::LEFT_CENTER,
                        &ev.instrument_id,
                        egui::FontId::proportional(11.0),
                        Color32::WHITE,
                    );

                    let resp = ui.interact(body, id.with(("ev", ti, ei)), Sense::click_and_drag());
                    if resp.clicked() {
                        state.selected_event = Some((ti, ei));
                    }
                    if resp.drag_started() {
                        let near_right = resp
                            .interact_pointer_pos()
                            .is_some_and(|p| p.x >= body.right() - 8.0);
                        state.drag_mode = if near_right {
                            DragMode::Resize
                        } else {
                            DragMode::Move
                        };
                        state.selected_event = Some((ti, ei));
                    }
                    if resp.dragged() {
                        let dx = resp.drag_delta().x / ppb;
                        match state.drag_mode {
                            DragMode::Move => ev.time_beats = (ev.time_beats + dx).max(0.0),
                            DragMode::Resize => ev.gate_beats = (ev.gate_beats + dx).max(MIN_GATE),
                        }
                        res.changed = true;
                    }
                    if resp.drag_stopped() {
                        match state.drag_mode {
                            DragMode::Move => ev.time_beats = snap_beat(ev.time_beats),
                            DragMode::Resize => {
                                ev.gate_beats = snap_beat(ev.gate_beats).max(MIN_GATE)
                            }
                        }
                        res.rebake = true;
                    }
                }
            }
        });

    if let Some(ti) = remove_track {
        if ti < recipe.tracks.len() {
            recipe.tracks.remove(ti);
        }
        state.selected_event = None;
        res.changed = true;
        res.rebake = true;
    }
    if let Some((ti, beat)) = add_event
        && ti < recipe.tracks.len()
    {
        recipe.tracks[ti].events.push(Event {
            time_beats: beat,
            instrument_id: default_inst,
            volume: 0.8,
            ..Event::default()
        });
        state.selected_event = Some((ti, recipe.tracks[ti].events.len() - 1));
        res.changed = true;
        res.rebake = true;
    }

    res
}

// ---------------------------------------------------------------------------
// Event inspector
// ---------------------------------------------------------------------------

fn event_inspector(
    ui: &mut egui::Ui,
    recipe: &mut SequenceRecipe,
    state: &mut SequenceEditorState,
) -> EditorResponse {
    let mut res = EditorResponse::NONE;

    let Some((ti, ei)) = state.selected_event else {
        ui.label("No event selected — click an event, or double-click a lane to add one.");
        return res;
    };
    if ti >= recipe.tracks.len() || ei >= recipe.tracks[ti].events.len() {
        state.selected_event = None;
        ui.label("No event selected.");
        return res;
    }

    let inst_ids: Vec<String> = recipe.instruments.iter().map(|i| i.id.clone()).collect();
    let dur = recipe.duration_beats.max(1.0);
    let mut delete = false;

    {
        let ev = &mut recipe.tracks[ti].events[ei];
        ui.label(format!("Track {ti}, event #{ei}"));

        egui::ComboBox::from_id_salt("evt_instrument")
            .selected_text(if ev.instrument_id.is_empty() {
                "(none)"
            } else {
                ev.instrument_id.as_str()
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
            });

        res.merge(drag_debounced(
            ui,
            "time (beats)",
            &mut ev.time_beats,
            0.05,
            0.0..=dur * 4.0,
        ));
        res.merge(drag_debounced(
            ui,
            "gate (beats)",
            &mut ev.gate_beats,
            0.05,
            MIN_GATE..=dur * 4.0,
        ));
        res.merge(drag_debounced(
            ui,
            "release (beats)",
            &mut ev.release_beats,
            0.05,
            0.0..=32.0,
        ));
        res.merge(slider_debounced(
            ui,
            egui::Slider::new(&mut ev.pitch_multiplier, 0.25..=4.0)
                .logarithmic(true)
                .text("pitch \u{00D7}"),
        ));
        res.merge(slider_debounced(
            ui,
            egui::Slider::new(&mut ev.volume, 0.0..=1.0).text("volume"),
        ));

        if ui.button("\u{1F5D1} Delete event").clicked() {
            delete = true;
        }
    }

    if delete {
        recipe.tracks[ti].events.remove(ei);
        state.selected_event = None;
        res.changed = true;
        res.rebake = true;
    }

    res
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn snap_beat_rounds_to_quarter_grid_and_clamps() {
        assert_eq!(snap_beat(0.12), 0.0);
        assert_eq!(snap_beat(0.13), 0.25);
        assert_eq!(snap_beat(1.6), 1.5);
        assert_eq!(snap_beat(-3.0), 0.0);
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
            let _ = ctx.run(egui::RawInput::default(), |ctx| {
                egui::SidePanel::left("seq").show(ctx, |ui| {
                    sequence_recipe_editor(ui, &mut recipe, &mut state, Id::new("seq"));
                });
                egui::CentralPanel::default().show(ctx, |ui| {
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
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                sequence_recipe_editor(ui, &mut recipe, &mut state, Id::new("seq"));
            });
        });
    }
}
