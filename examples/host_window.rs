//! `host_window` — both editors embedded the way a host app embeds them, each
//! in its own resizable `egui::Window`, and the permanent screenshot harness
//! for their layout (Overlands #1327, #54 here).
//!
//! The two windows are laid out like Overlands' audio pop-out:
//!
//! - **Patch slot:** the audition strip (Audition, Stop, the monitor's status,
//!   and the waveform once something has been baked), a separator, then
//!   [`audio_patch_canvas`] **last**.
//! - **Sequence slot:** the same strip, then [`sequence_recipe_editor`] in a
//!   resizable, scrolling left panel, with [`active_instrument_canvas`]
//!   filling the rest in a central panel.
//!
//! The order matters. A canvas is an `egui::Scene` and takes all the space
//! left in its `Ui`. Anything drawn after it lands below the window's
//! content, and an egui window never gives height back, so the window grows
//! every frame until it reaches the screen edge while the audition strip
//! stays out of sight. The docs on `audio_patch_canvas` explain the
//! mechanism. This example logs each window's size every 30 frames, so a
//! layout that grows shows up in the log as well as on screen.
//!
//! The sequence is the size of a real room's ambient bed: five instruments on
//! five lanes, 34 beats at 60 BPM, 22 050 Hz. It opens with the pluck showing
//! in the canvas.
//!
//! Run with:
//!   cargo run --example host_window --features egui
//!   cargo run --example host_window --features egui -- --light
//!
//! `--shot <path>` waits for the layout to settle, saves a picture of the app
//! window and quits. It plays nothing: the waveform is filled from a
//! synchronous bake, so the strip looks as it does after an audition:
//!   cargo run --example host_window --features egui -- --shot dark.png
//!   cargo run --example host_window --features egui -- --light --shot light.png

use std::collections::BTreeMap;

use bevy::diagnostic::FrameCount;
use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};

use bevy_symbios_audio::{
    AdsrEnvelope, AudioPatch, BiquadLowpass, BrownNoise, Connection, Gate, GraphNode, Instrument,
    Lfo, LfoShape, NodeGraph, NodeId, NodeKind, PinkNoise, SawtoothOsc, SequenceRecipe, SineOsc,
    Track, TriangleOsc, bake,
    sequence::Event,
    ui::{
        AudioEditorPlugin, AudioMonitor, MonitorRequest, MonitorStatus, PatchEditorState,
        SequenceEditorState, active_instrument_canvas, audio_patch_canvas, sequence_recipe_editor,
        waveform,
    },
};

/// The size Overlands gives its audio pop-out (its `src/ui/layout.rs`).
const SLOT: egui::Vec2 = egui::vec2(900.0, 640.0);
/// Space between the windows and around them.
const GAP: f32 = 12.0;
/// What the patch slot's Audition bakes.
const PATCH_SAMPLE_RATE: u32 = 44_100;
const PATCH_SECS: f32 = 4.0;
/// The instrument the sequence slot opens on (the pluck: three wired nodes).
const OPEN_INSTRUMENT: usize = 2;
/// Frames with both windows up before `--shot` takes its picture.
const SETTLE_FRAMES: u32 = 60;
/// Frames to wait for a capture's file before giving up on it.
const GIVE_UP: u32 = 600;
/// How often each window's size is logged, in frames.
const LOG_EVERY: u32 = 30;

fn main() {
    let args = Args::from_env();
    let width = 2.0 * SLOT.x + 3.0 * GAP;
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "bevy_symbios_audio — host window".into(),
                resolution: (width as u32, 720u32).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins((EguiPlugin::default(), AudioEditorPlugin))
        .insert_resource(Editor::new(!args.light))
        .insert_resource(args)
        .init_resource::<Shot>()
        .add_systems(Startup, (setup_camera, fill_waveform_for_the_shot))
        .add_systems(EguiPrimaryContextPass, render_ui)
        .add_systems(Update, (log_window_sizes, shoot))
        .run();
}

/// The command line: `--light` and `--shot <path>`.
#[derive(Resource)]
struct Args {
    light: bool,
    shot: Option<String>,
}

impl Args {
    fn from_env() -> Self {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let shot = args.iter().position(|a| a == "--shot").map(|at| {
            args.get(at + 1).cloned().unwrap_or_else(|| {
                eprintln!("--shot needs a path: --shot host_window.png");
                std::process::exit(2);
            })
        });
        Self {
            light: args.iter().any(|a| a == "--light"),
            shot,
        }
    }
}

/// Both slots' working copies and view state, the theme, and what the
/// windows measured this frame.
#[derive(Resource)]
struct Editor {
    patch: AudioPatch,
    patch_state: PatchEditorState,
    recipe: SequenceRecipe,
    sequence_state: SequenceEditorState,
    dark: bool,
    /// The theme last handed to egui, so it is set once per change.
    applied: Option<bool>,
    /// Each window's rect as shown this frame: patch slot, sequence slot.
    shown: [Option<egui::Rect>; 2],
    /// Frames on which both windows were shown.
    frames_shown: u32,
}

impl Editor {
    fn new(dark: bool) -> Self {
        let mut sequence_state = SequenceEditorState::default();
        sequence_state.set_active_instrument(Some(OPEN_INSTRUMENT));
        Self {
            patch: filtered_drone_patch(),
            patch_state: PatchEditorState::default(),
            recipe: seeded_size_recipe(),
            sequence_state,
            dark,
            applied: None,
            shown: [None; 2],
            frames_shown: 0,
        }
    }
}

fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}

/// Under `--shot`, bake the patch slot here and hand the buffer to the
/// monitor, so the strips draw their waveform the way they do after an
/// audition, without anything playing through the speakers.
fn fill_waveform_for_the_shot(
    args: Res<Args>,
    editor: Res<Editor>,
    mut monitor: ResMut<AudioMonitor>,
) {
    if args.shot.is_some() {
        monitor.last_samples = bake(&editor.patch, PATCH_SAMPLE_RATE, PATCH_SECS);
        monitor.sample_rate = PATCH_SAMPLE_RATE;
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
    let editor = editor.as_mut();
    if editor.applied != Some(editor.dark) {
        ctx.set_theme(if editor.dark {
            egui::Theme::Dark
        } else {
            egui::Theme::Light
        });
        editor.applied = Some(editor.dark);
    }

    // egui 0.35 shows panels into a `Ui`, not a `Context`: a screen-sized
    // background layer holds the host's own bar and backdrop.
    let mut viewport_ui = egui::Ui::new(
        ctx.clone(),
        "viewport".into(),
        egui::UiBuilder::new()
            .layer_id(egui::LayerId::background())
            .max_rect(ctx.viewport_rect()),
    );
    egui::Panel::top("host_bar").show(&mut viewport_ui, |ui| {
        ui.horizontal(|ui| {
            ui.label("Two audio slots, each in its own window, as a host shows them.");
            ui.separator();
            ui.selectable_value(&mut editor.dark, true, "Dark");
            ui.selectable_value(&mut editor.dark, false, "Light");
        });
    });
    // What the bar leaves: the windows open in it and are constrained to
    // it, as a host's toolbar leaves room for its windows.
    let free = viewport_ui.available_rect_before_wrap();
    egui::CentralPanel::default().show(&mut viewport_ui, |_| {});

    let patch_at = free.left_top() + egui::vec2(GAP, GAP);
    let sequence_at = patch_at + egui::vec2(SLOT.x + GAP, 0.0);
    editor.shown = [
        patch_slot(ctx, editor, &monitor, &mut requests, patch_at, free),
        sequence_slot(ctx, editor, &monitor, &mut requests, sequence_at, free),
    ];
    if editor.shown.iter().all(Option::is_some) {
        editor.frames_shown += 1;
    }
}

/// A slot's window: the size and constraint a host gives a pop-out editor.
fn slot_window(title: &str, pos: egui::Pos2, free: egui::Rect) -> egui::Window<'static> {
    egui::Window::new(title)
        .default_pos(pos)
        .default_size(SLOT)
        .resizable(true)
        .constrain_to(free)
}

/// The patch slot: the strip first, then the canvas LAST, so the canvas
/// takes whatever height is left and the window keeps its size.
fn patch_slot(
    ctx: &egui::Context,
    editor: &mut Editor,
    monitor: &AudioMonitor,
    requests: &mut MessageWriter<MonitorRequest>,
    pos: egui::Pos2,
    free: egui::Rect,
) -> Option<egui::Rect> {
    let id = egui::Id::new("patch_slot");
    slot_window("Patch slot", pos, free)
        .show(ctx, |ui| {
            audition_strip(ui, monitor, requests, || MonitorRequest::PlayPatch {
                patch: editor.patch.clone(),
                sample_rate: PATCH_SAMPLE_RATE,
                duration_secs: PATCH_SECS,
            });
            ui.separator();
            audio_patch_canvas(
                ui,
                &mut editor.patch,
                &mut editor.patch_state,
                id.with("canvas"),
            );
        })
        .map(|shown| shown.response.rect)
}

/// The sequence slot: the strip first, then the sequence editor in a
/// resizable, scrolling left panel and the instrument canvas in the central
/// panel, which a window's `Ui` can hold as well as a screen can.
fn sequence_slot(
    ctx: &egui::Context,
    editor: &mut Editor,
    monitor: &AudioMonitor,
    requests: &mut MessageWriter<MonitorRequest>,
    pos: egui::Pos2,
    free: egui::Rect,
) -> Option<egui::Rect> {
    let id = egui::Id::new("sequence_slot");
    slot_window("Sequence slot", pos, free)
        .show(ctx, |ui| {
            audition_strip(ui, monitor, requests, || MonitorRequest::PlaySequence {
                recipe: editor.recipe.clone(),
            });
            ui.separator();
            // Panel ids are global, so they are salted with the slot: two
            // windows must never share one panel's width.
            egui::Panel::left(id.with("sequence_panel"))
                .resizable(true)
                .default_size(420.0)
                .min_size(300.0)
                .frame(egui::Frame::NONE.inner_margin(egui::Margin {
                    right: 6,
                    ..egui::Margin::ZERO
                }))
                .show(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            sequence_recipe_editor(
                                ui,
                                &mut editor.recipe,
                                &mut editor.sequence_state,
                                id.with("sequence"),
                            );
                        });
                });
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE.inner_margin(egui::Margin {
                    left: 6,
                    ..egui::Margin::ZERO
                }))
                .show(ui, |ui| {
                    active_instrument_canvas(
                        ui,
                        &mut editor.recipe,
                        &mut editor.sequence_state,
                        id.with("instrument_canvas"),
                    );
                });
        })
        .map(|shown| shown.response.rect)
}

/// Audition, Stop, the monitor's status, and the waveform of the last bake.
/// `make_request` builds the play message only when Audition is pressed, so
/// the working copy is cloned only then.
fn audition_strip(
    ui: &mut egui::Ui,
    monitor: &AudioMonitor,
    requests: &mut MessageWriter<MonitorRequest>,
    make_request: impl FnOnce() -> MonitorRequest,
) {
    ui.horizontal(|ui| {
        if ui
            .add_enabled(!monitor.is_baking(), egui::Button::new("\u{25B6} Audition"))
            .clicked()
        {
            requests.write(make_request());
        }
        if ui.button("\u{23F9} Stop").clicked() {
            requests.write(MonitorRequest::Stop);
        }
        match &monitor.status {
            MonitorStatus::Idle => ui.weak("idle"),
            MonitorStatus::Baking => ui.weak("baking\u{2026}"),
            MonitorStatus::Playing => ui.weak("playing (loop)"),
            MonitorStatus::Error(e) => ui.colored_label(ui.visuals().error_fg_color, e),
        };
    });
    if !monitor.last_samples.is_empty() {
        waveform(ui, &monitor.last_samples);
    }
}

/// Each window's size every [`LOG_EVERY`] frames. A window whose content is
/// taller than it grows on every frame, and this is where that shows.
fn log_window_sizes(frame: Res<FrameCount>, editor: Res<Editor>) {
    if frame.0 == 0 || !frame.0.is_multiple_of(LOG_EVERY) {
        return;
    }
    let size = |rect: Option<egui::Rect>| {
        rect.map_or_else(
            || "not shown".to_string(),
            |r| format!("{:.0}x{:.0}", r.width(), r.height()),
        )
    };
    info!(
        "frame {}: patch slot {}, sequence slot {}",
        frame.0,
        size(editor.shown[0]),
        size(editor.shown[1])
    );
}

/// Where `--shot` is: not asked for, waiting for the layout, or waiting for
/// the file.
#[derive(Resource, Default)]
struct Shot {
    taken: bool,
    waited: u32,
    /// The file's size last frame: a capture is written on a task-pool
    /// thread, so "exists" alone can mean "half written".
    last_len: Option<u64>,
}

/// Save a picture of the app window and quit, for `--shot <path>`.
fn shoot(
    mut commands: Commands,
    args: Res<Args>,
    editor: Res<Editor>,
    mut shot: ResMut<Shot>,
    mut exit: MessageWriter<AppExit>,
) {
    let Some(path) = &args.shot else {
        return;
    };
    if !shot.taken {
        // Wait for the layout, not for a fixed number of frames from
        // startup: the count starts once both windows are actually up.
        if editor.frames_shown < SETTLE_FRAMES {
            return;
        }
        // A stale picture from an earlier run would read as this one.
        let _ = std::fs::remove_file(path);
        info!(
            "--shot: capturing {path} with the patch slot at {:?} and the sequence slot at {:?}",
            editor.shown[0].map(|r| r.size()),
            editor.shown[1].map(|r| r.size())
        );
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path.clone()));
        shot.taken = true;
        return;
    }
    // Quit when the file is written, not a frame count after asking: the
    // capture is read back across frames and saved on another thread.
    shot.waited += 1;
    let len = std::fs::metadata(path).map(|m| m.len()).ok();
    if len.is_some_and(|n| n > 0) && len == shot.last_len {
        exit.write(AppExit::Success);
    } else if shot.waited > GIVE_UP {
        error!("--shot: {path} was not written after {GIVE_UP} frames");
        exit.write(AppExit::error());
    }
    shot.last_len = len;
}

// ---------------------------------------------------------------------------
// What the slots hold
// ---------------------------------------------------------------------------

/// Sawtooth into a lowpass with an LFO on the cutoff: a wired patch for the
/// patch slot.
fn filtered_drone_patch() -> AudioPatch {
    let saw = NodeId(0);
    let lfo = NodeId(1);
    let filter = NodeId(2);
    let mut filter_inputs = BTreeMap::new();
    filter_inputs.insert("in".to_string(), vec![Connection::from_node(saw)]);
    filter_inputs.insert(
        "cutoff_hz".to_string(),
        vec![Connection::modulation(lfo, 400.0)],
    );
    AudioPatch {
        seed: 0,
        graph: NodeGraph {
            nodes: vec![
                GraphNode {
                    id: saw,
                    kind: NodeKind::Sawtooth(SawtoothOsc {
                        freq_hz: 82.5,
                        amplitude: 0.5,
                        ..SawtoothOsc::default()
                    }),
                    inputs: BTreeMap::new(),
                },
                GraphNode {
                    id: lfo,
                    kind: NodeKind::Lfo(Lfo {
                        rate_hz: 0.25,
                        shape: LfoShape::Sine,
                        depth: 0.5,
                        offset: 0.5,
                    }),
                    inputs: BTreeMap::new(),
                },
                GraphNode {
                    id: filter,
                    kind: NodeKind::BiquadLowpass(BiquadLowpass {
                        cutoff_hz: 300.0,
                        q: 1.2,
                    }),
                    inputs: filter_inputs,
                },
            ],
            output: filter,
        },
    }
}

/// Five instruments on five lanes, 34 beats at 60 BPM and 22 050 Hz with a
/// two-beat run-up and crossfade: the shape of a seeded Overlands room's
/// ambient bed (bed, gust, pluck, melody, bass), which is what the sequence
/// slot has to fit.
fn seeded_size_recipe() -> SequenceRecipe {
    const RUN_UP: f32 = 2.0;
    const LOOP: f32 = 32.0;
    let held = |id: &str, volume: f32| Event {
        time_beats: 0.0,
        instrument_id: id.into(),
        pitch_multiplier: 1.0,
        volume,
        gate_beats: RUN_UP + LOOP,
        release_beats: 2.0,
        ..Default::default()
    };
    let note = |id: &str, time_beats: f32, pitch_multiplier: f32, gate_beats: f32| Event {
        time_beats,
        instrument_id: id.into(),
        pitch_multiplier,
        volume: 0.5,
        gate_beats,
        release_beats: 0.5,
        ..Default::default()
    };
    let plucks = [2.0, 6.0, 9.5, 13.0, 18.0, 21.5, 25.0, 29.0]
        .iter()
        .zip([1.0, 1.5, 1.25, 2.0, 1.0, 1.125, 1.5, 0.75])
        .map(|(&t, p)| note("pluck", t, p, 0.25))
        .collect();
    let melody = [
        (2.0, 1.0),
        (3.0, 1.125),
        (4.0, 1.25),
        (6.0, 1.5),
        (10.0, 1.25),
        (11.0, 1.125),
        (12.0, 1.0),
        (18.0, 1.5),
        (20.0, 1.25),
        (26.0, 1.125),
        (28.0, 1.0),
    ]
    .into_iter()
    .map(|(t, p)| note("melody", t, p, 0.75))
    .collect();
    let bass = (0..4)
        .map(|i| {
            note(
                "bass",
                RUN_UP + 8.0 * i as f32,
                [1.0, 0.75, 0.875, 1.0][i],
                7.5,
            )
        })
        .collect();

    SequenceRecipe {
        bpm: 60.0,
        sample_rate: 22_050,
        duration_beats: RUN_UP + LOOP,
        loop_start_beats: Some(RUN_UP),
        loop_crossfade_beats: 2.0,
        instruments: vec![
            instrument(
                "bed",
                noise_bed(
                    NodeKind::PinkNoise(PinkNoise { amplitude: 0.4 }),
                    700.0,
                    None,
                ),
            ),
            instrument(
                "gust",
                noise_bed(
                    NodeKind::BrownNoise(BrownNoise { amplitude: 0.6 }),
                    400.0,
                    Some(1.0 / 16.0),
                ),
            ),
            instrument(
                "pluck",
                enveloped(NodeKind::Sine(SineOsc {
                    freq_hz: 660.0,
                    phase_offset: 0.0,
                    amplitude: 0.0,
                })),
            ),
            instrument(
                "melody",
                enveloped(NodeKind::Triangle(TriangleOsc {
                    freq_hz: 440.0,
                    amplitude: 0.0,
                    ..TriangleOsc::default()
                })),
            ),
            instrument(
                "bass",
                noise_bed(
                    NodeKind::Sawtooth(SawtoothOsc {
                        freq_hz: 55.0,
                        amplitude: 0.4,
                        ..SawtoothOsc::default()
                    }),
                    250.0,
                    None,
                ),
            ),
        ],
        tracks: vec![
            Track {
                events: vec![held("bed", 0.5)],
            },
            Track {
                events: vec![held("gust", 0.35)],
            },
            Track { events: plucks },
            Track { events: melody },
            Track { events: bass },
        ],
    }
}

fn instrument(id: &str, patch: AudioPatch) -> Instrument {
    Instrument {
        id: id.into(),
        patch,
    }
}

/// `source` into a lowpass at `cutoff_hz`, with an LFO sweeping the cutoff
/// at `sweep_hz` when given.
fn noise_bed(source: NodeKind, cutoff_hz: f32, sweep_hz: Option<f32>) -> AudioPatch {
    let src = NodeId(0);
    let filter = NodeId(1);
    let mut filter_inputs = BTreeMap::new();
    filter_inputs.insert("in".to_string(), vec![Connection::from_node(src)]);
    let mut nodes = vec![GraphNode {
        id: src,
        kind: source,
        inputs: BTreeMap::new(),
    }];
    if let Some(rate_hz) = sweep_hz {
        let lfo = NodeId(2);
        filter_inputs.insert(
            "cutoff_hz".to_string(),
            vec![Connection::modulation(lfo, cutoff_hz * 0.5)],
        );
        nodes.push(GraphNode {
            id: lfo,
            kind: NodeKind::Lfo(Lfo {
                rate_hz,
                shape: LfoShape::Sine,
                depth: 0.5,
                offset: 0.5,
            }),
            inputs: BTreeMap::new(),
        });
    }
    nodes.push(GraphNode {
        id: filter,
        kind: NodeKind::BiquadLowpass(BiquadLowpass { cutoff_hz, q: 0.7 }),
        inputs: filter_inputs,
    });
    AudioPatch {
        seed: 0,
        graph: NodeGraph {
            nodes,
            output: filter,
        },
    }
}

/// An oscillator whose amplitude is a gate-driven ADSR, so each event plays
/// as a note (`Gate -> AdsrEnvelope.gate`, `Adsr -> osc.amplitude`).
fn enveloped(osc: NodeKind) -> AudioPatch {
    let voice = NodeId(0);
    let gate = NodeId(1);
    let env = NodeId(2);
    let mut env_inputs = BTreeMap::new();
    env_inputs.insert("gate".to_string(), vec![Connection::from_node(gate)]);
    let mut voice_inputs = BTreeMap::new();
    voice_inputs.insert("amplitude".to_string(), vec![Connection::from_node(env)]);
    AudioPatch {
        seed: 0,
        graph: NodeGraph {
            nodes: vec![
                GraphNode {
                    id: voice,
                    kind: osc,
                    inputs: voice_inputs,
                },
                GraphNode {
                    id: gate,
                    kind: NodeKind::Gate(Gate::default()),
                    inputs: BTreeMap::new(),
                },
                GraphNode {
                    id: env,
                    kind: NodeKind::Adsr(AdsrEnvelope {
                        attack_s: 0.01,
                        decay_s: 0.2,
                        sustain_level: 0.5,
                        release_s: 0.4,
                        ..AdsrEnvelope::default()
                    }),
                    inputs: env_inputs,
                },
            ],
            output: voice,
        },
    }
}
