//! `patch_editor` — interactive visual editor for an [`AudioPatch`].
//!
//! A pannable / zoomable node-graph canvas (Phase 2): drag node title bars to
//! move them, drag from a node's output dot (right edge) onto another node's
//! input dot (left edge) to wire them, edit each node's parameters in place,
//! and use the toolbar to add / delete nodes and choose the graph output. The
//! status line turns red if the graph stops being a valid DAG.
//!
//! Pan by dragging empty canvas; zoom with the scroll wheel.
//!
//! A bake-and-play monitor sits across the top: "▶ Bake & Play" bakes the
//! edited patch and loops it through a waveform display, "⏹ Stop" halts it.
//!
//! Run with:
//!   cargo run --example patch_editor --features egui

use std::collections::BTreeMap;

use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};

use bevy_symbios_audio::{
    AudioPatch, BiquadLowpass, Connection, GraphNode, Lfo, LfoShape, NodeGraph, NodeId, NodeKind,
    SineOsc,
    ui::{
        AudioEditorPlugin, AudioMonitor, MonitorRequest, MonitorStatus, PatchEditorState,
        audio_patch_canvas, waveform,
    },
};

/// Monitor preview length for a one-shot patch, in seconds.
const PREVIEW_SECS: f32 = 2.0;
const PREVIEW_SR: u32 = 44_100;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "bevy_symbios_audio — patch editor".into(),
                resolution: (900u32, 700u32).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins((EguiPlugin::default(), AudioEditorPlugin))
        .init_resource::<Editor>()
        .add_systems(Startup, setup_camera)
        .add_systems(EguiPrimaryContextPass, render_ui)
        .run();
}

/// The patch being edited plus its canvas view/layout state.
#[derive(Resource)]
struct Editor {
    patch: AudioPatch,
    state: PatchEditorState,
}

impl Default for Editor {
    fn default() -> Self {
        Self {
            patch: starter_patch(),
            state: PatchEditorState::default(),
        }
    }
}

fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}

/// A three-node patch (sine → lowpass, LFO sweeping the cutoff) so the canvas
/// opens on a wired graph rather than a blank sheet.
fn starter_patch() -> AudioPatch {
    let sine = NodeId(0);
    let lfo = NodeId(1);
    let filter = NodeId(2);

    let mut filter_inputs: BTreeMap<String, Vec<Connection>> = BTreeMap::new();
    filter_inputs.insert("in".into(), vec![Connection::from_node(sine)]);
    filter_inputs.insert("cutoff_hz".into(), vec![Connection::modulation(lfo, 500.0)]);

    AudioPatch {
        seed: 0,
        graph: NodeGraph {
            nodes: vec![
                GraphNode {
                    id: sine,
                    kind: NodeKind::Sine(SineOsc {
                        freq_hz: 110.0,
                        phase_offset: 0.0,
                        amplitude: 0.6,
                    }),
                    inputs: BTreeMap::new(),
                },
                GraphNode {
                    id: lfo,
                    kind: NodeKind::Lfo(Lfo {
                        rate_hz: 0.4,
                        shape: LfoShape::Sine,
                        depth: 0.5,
                        offset: 0.5,
                    }),
                    inputs: BTreeMap::new(),
                },
                GraphNode {
                    id: filter,
                    kind: NodeKind::BiquadLowpass(BiquadLowpass {
                        cutoff_hz: 150.0,
                        q: 1.5,
                    }),
                    inputs: filter_inputs,
                },
            ],
            output: filter,
        },
    }
}

fn render_ui(
    mut contexts: EguiContexts,
    mut editor: ResMut<Editor>,
    monitor: Res<AudioMonitor>,
    mut requests: MessageWriter<MonitorRequest>,
) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };

    // Monitor controls + waveform (top).
    egui::TopBottomPanel::top("monitor").show(ctx, |ui| {
        ui.horizontal(|ui| {
            if ui.button("\u{25B6} Bake & Play").clicked() {
                requests.write(MonitorRequest::PlayPatch {
                    patch: editor.patch.clone(),
                    sample_rate: PREVIEW_SR,
                    duration_secs: PREVIEW_SECS,
                });
            }
            if ui.button("\u{23F9} Stop").clicked() {
                requests.write(MonitorRequest::Stop);
            }
            match &monitor.status {
                MonitorStatus::Idle => {
                    ui.label("idle");
                }
                MonitorStatus::Baking => {
                    ui.spinner();
                    ui.label("baking\u{2026}");
                }
                MonitorStatus::Playing => {
                    ui.label("playing (loop)");
                }
                MonitorStatus::Error(e) => {
                    ui.colored_label(egui::Color32::from_rgb(220, 120, 120), e);
                }
            }
        });
        waveform(ui, &monitor.last_samples);
    });

    // Node-graph canvas (fills the rest).
    let editor = editor.as_mut();
    egui::CentralPanel::default().show(ctx, |ui| {
        audio_patch_canvas(
            ui,
            &mut editor.patch,
            &mut editor.state,
            egui::Id::new("patch_canvas"),
        );
    });
}
