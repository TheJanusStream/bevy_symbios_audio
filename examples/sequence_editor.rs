//! `sequence_editor` — interactive editor for a [`SequenceRecipe`].
//!
//! A small DAW-style layout:
//! - **top:** the bake-and-play monitor (▶ Bake & Play bakes the *whole* recipe
//!   via `bake_sequence`, ⏹ stops) with a live waveform;
//! - **left:** the sequence editor — transport, instruments, and the
//!   track/event timeline (drag blocks to move, drag a block's right edge to
//!   resize its gate, double-click an empty lane to add an event);
//! - **center:** the active instrument's patch in the node-graph canvas
//!   (click ✎ next to an instrument to open it).
//!
//! Run with:
//!   cargo run --example sequence_editor --features egui

use std::collections::BTreeMap;

use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};

use bevy_symbios_audio::{
    AdsrEnvelope, AudioPatch, Connection, Gate, GraphNode, Instrument, NodeGraph, NodeId, NodeKind,
    SequenceRecipe, SineOsc, Track,
    sequence::Event,
    ui::{
        AudioEditorPlugin, AudioMonitor, MonitorRequest, MonitorStatus, SequenceEditorState,
        active_instrument_canvas, sequence_recipe_editor, waveform,
    },
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "bevy_symbios_audio — sequence editor".into(),
                resolution: (1200u32, 760u32).into(),
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

#[derive(Resource)]
struct Editor {
    recipe: SequenceRecipe,
    state: SequenceEditorState,
}

impl Default for Editor {
    fn default() -> Self {
        Self {
            recipe: starter_recipe(),
            state: SequenceEditorState::default(),
        }
    }
}

fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}

/// A drone instrument (held sine) and a plucked instrument (sine through a
/// gate-driven ADSR), arranged on two tracks with a seamless loop — so the
/// editor opens on something that actually plays.
fn starter_recipe() -> SequenceRecipe {
    SequenceRecipe {
        bpm: 96.0,
        sample_rate: 44_100,
        duration_beats: 8.0,
        loop_start_beats: Some(0.0),
        loop_crossfade_beats: 1.0,
        instruments: vec![
            Instrument {
                id: "drone".into(),
                patch: drone_patch(),
            },
            Instrument {
                id: "pluck".into(),
                patch: pluck_patch(),
            },
        ],
        tracks: vec![
            Track {
                events: vec![Event {
                    time_beats: 0.0,
                    instrument_id: "drone".into(),
                    pitch_multiplier: 1.0,
                    volume: 0.5,
                    gate_beats: 8.0,
                    release_beats: 0.0,
                    ..Default::default()
                }],
            },
            Track {
                events: vec![
                    pluck_event(0.0, 1.0),
                    pluck_event(1.5, 1.5),
                    pluck_event(3.0, 1.25),
                    pluck_event(5.0, 2.0),
                ],
            },
        ],
    }
}

fn pluck_event(time_beats: f32, pitch_multiplier: f32) -> Event {
    Event {
        time_beats,
        instrument_id: "pluck".into(),
        pitch_multiplier,
        volume: 0.8,
        gate_beats: 0.5,
        release_beats: 0.6,
        ..Default::default()
    }
}

/// A plain ~110 Hz sine, held for the event's gate.
fn drone_patch() -> AudioPatch {
    AudioPatch {
        seed: 0,
        graph: NodeGraph {
            nodes: vec![GraphNode {
                id: NodeId(0),
                kind: NodeKind::Sine(SineOsc {
                    freq_hz: 110.0,
                    phase_offset: 0.0,
                    amplitude: 0.8,
                }),
                inputs: BTreeMap::new(),
            }],
            output: NodeId(0),
        },
    }
}

/// Sine → amplitude shaped by an ADSR driven off the note gate
/// (`Gate → AdsrEnvelope.gate`, `Adsr → Sine.amplitude`).
fn pluck_patch() -> AudioPatch {
    let sine = NodeId(0);
    let gate = NodeId(1);
    let env = NodeId(2);

    let mut env_inputs = BTreeMap::new();
    env_inputs.insert("gate".to_string(), vec![Connection::from_node(gate)]);

    let mut sine_inputs = BTreeMap::new();
    // ADSR (0..1) drives amplitude; base amplitude 0 so the envelope is the
    // whole signal level (a clean pluck).
    sine_inputs.insert("amplitude".to_string(), vec![Connection::from_node(env)]);

    AudioPatch {
        seed: 0,
        graph: NodeGraph {
            nodes: vec![
                GraphNode {
                    id: sine,
                    kind: NodeKind::Sine(SineOsc {
                        freq_hz: 440.0,
                        phase_offset: 0.0,
                        amplitude: 0.0,
                    }),
                    inputs: sine_inputs,
                },
                GraphNode {
                    id: gate,
                    kind: NodeKind::Gate(Gate::default()),
                    inputs: BTreeMap::new(),
                },
                GraphNode {
                    id: env,
                    kind: NodeKind::Adsr(AdsrEnvelope {
                        attack_s: 0.005,
                        decay_s: 0.12,
                        sustain_level: 0.6,
                        release_s: 0.3,
                        ..AdsrEnvelope::default()
                    }),
                    inputs: env_inputs,
                },
            ],
            output: sine,
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

    egui::TopBottomPanel::top("monitor").show(ctx, |ui| {
        ui.horizontal(|ui| {
            if ui.button("\u{25B6} Bake & Play").clicked() {
                requests.write(MonitorRequest::PlaySequence {
                    recipe: editor.recipe.clone(),
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

    let editor = editor.as_mut();
    egui::SidePanel::left("sequence")
        .default_width(420.0)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                sequence_recipe_editor(
                    ui,
                    &mut editor.recipe,
                    &mut editor.state,
                    egui::Id::new("sequence_editor"),
                );
            });
        });

    egui::CentralPanel::default().show(ctx, |ui| {
        active_instrument_canvas(
            ui,
            &mut editor.recipe,
            &mut editor.state,
            egui::Id::new("instrument_canvas"),
        );
    });
}
