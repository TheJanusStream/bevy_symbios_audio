//! `patch_editor` — interactive visual editor for an [`AudioPatch`].
//!
//! A pannable / zoomable node-graph canvas: drag node numbers to move them,
//! drag from a node's output dot (right edge) onto another node's input —
//! its dot, or anywhere on its named row in the "Inputs" list — to wire
//! them, and let go over nothing to open the Add menu there and drive what
//! you choose. Hover a wire to see what it does, click it to pick it, and
//! edit its amount on the wire itself. Edit each node's parameters in
//! place, and use the toolbar to add / delete nodes and choose the graph
//! output. A graph that cannot bake says which nodes are at fault, by name,
//! and outlines them.
//!
//! Pan by dragging empty canvas; zoom with the scroll wheel.
//!
//! The crate's audition strip sits across the top: "▶ Audition" bakes the
//! edited patch and loops it, "⏹ Stop" halts it, and with Auto on every
//! committed edit re-bakes what plays. The chip says what the monitor is
//! doing, the caption what is played, and the waveform shows the last bake.
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
        AudioEditorPlugin, AudioMonitor, AuditionSource, AuditionState, MonitorRequest,
        PatchEditorState, audio_patch_canvas, audition_strip,
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

/// The patch being edited, its canvas view/layout state, and its audition
/// strip, with whether the canvas committed an edit last frame (the strip is
/// drawn above the canvas, so it hears of a commit a frame later).
#[derive(Resource)]
struct Editor {
    patch: AudioPatch,
    state: PatchEditorState,
    audition: AuditionState,
    committed: bool,
}

impl Default for Editor {
    fn default() -> Self {
        Self {
            patch: starter_patch(),
            state: PatchEditorState::default(),
            audition: AuditionState::default(),
            committed: false,
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

    // egui 0.35 shows panels into a `Ui`, not a `Context`. A top-level layout
    // draws into a screen-sized background layer (bevy_egui 0.41's side_panel
    // example), adding panels outermost first and `CentralPanel` last.
    let mut viewport_ui = egui::Ui::new(
        ctx.clone(),
        "viewport".into(),
        egui::UiBuilder::new()
            .layer_id(egui::LayerId::background())
            .max_rect(ctx.viewport_rect()),
    );

    // The audition strip (top).
    let editor = editor.as_mut();
    egui::Panel::top("monitor").show(&mut viewport_ui, |ui| {
        let source = AuditionSource::patch(&editor.patch, PREVIEW_SR, PREVIEW_SECS);
        if let Some(request) = audition_strip(
            ui,
            &monitor,
            &mut editor.audition,
            source,
            editor.committed,
            false,
        ) {
            requests.write(request);
        }
    });

    // Node-graph canvas (fills the rest).
    egui::CentralPanel::default().show(&mut viewport_ui, |ui| {
        editor.committed = audio_patch_canvas(
            ui,
            &mut editor.patch,
            &mut editor.state,
            egui::Id::new("patch_canvas"),
        )
        .rebake;
    });
}
