//! `host_window` — both editors embedded the way a host app embeds them, each
//! in its own resizable `egui::Window`, and the permanent screenshot harness
//! for their layout (Overlands #1327, #54 here).
//!
//! The two windows are laid out like Overlands' audio pop-out:
//!
//! - **Patch slot:** the crate's [`audition_strip`] (Audition, Stop, Auto, the
//!   status chip, the caption and the waveform of the slot's last bake), a
//!   separator, then [`audio_patch_canvas`] **last**. It auditions the way
//!   Overlands' world plays a construct's patch: one second at 22 050 Hz,
//!   looped.
//! - **Sequence slot:** the same strip, then [`sequence_recipe_editor`] in a
//!   resizable, scrolling left panel, with [`active_instrument_canvas`]
//!   filling the rest in a central panel.
//!
//! Each editor's commit (`EditorResponse::rebake`) goes to its strip on the
//! next frame, so with Auto on a playing audition re-bakes after an edit. The
//! host bar's Mute stands in for a host's own sound switch: it silences every
//! sink and tells the strips, whose chip then says Muted instead of Playing.
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
//! window and quits. Under `--shot` every sink is muted, so a picture never
//! makes a sound, whatever the strips say:
//!   cargo run --example host_window --features egui -- --shot dark.png
//!   cargo run --example host_window --features egui -- --light --shot light.png
//!
//! `--status <state>` puts the strips in one of the chip's states before the
//! picture, through real requests to the monitor: `idle` (nothing asked),
//! `playing` (the patch slot auditioned; the default under `--shot`), `muted`
//! (the same with the host bar's Mute on), `baking` (the sequence slot
//! auditioned, pictured while its bake runs) and `error` (the patch slot
//! auditioned with `--broken`'s patch). `--broken` opens the patch slot with
//! a loop in its graph, so the canvas outlines the two nodes of the loop and
//! names them:
//!   cargo run --example host_window --features egui -- --status baking --shot baking.png
//!   cargo run --example host_window --features egui -- --status error --shot error.png
//!
//! A `baking` picture needs a bake slower than a few frames: the dev profile
//! bakes the seeded-size sequence in about a second, a release build in well
//! under a tenth of one, too fast to picture.
//!
//! `--orphan` opens the sequence with notes that name no instrument: the
//! pluck is renamed "harp" and its eight notes still say "pluck", which is
//! what the instrument name field did before #55. The first of them is
//! selected and the side panel starts scrolled to the bottom, so the picture
//! shows the missing notes on the timeline and the inspector's reassign
//! offer:
//!   cargo run --example host_window --features egui -- --orphan --shot orphan.png
//!
//! `--rename <name>` types `<name>` over the open instrument's name and
//! presses Enter, as a user would: the field is found in egui's AccessKit
//! tree by the name it shows, focused by an AccessKit request, and the keys
//! go in through `EguiInput`. A valid name renames the instrument and its
//! notes follow; a taken, empty or over-long one is refused, and the picture
//! shows the field in the error colour with the reason under it:
//!   cargo run --example host_window --features egui -- --rename bass --shot refused.png
//!
//! `--drag` starts a wire from the patch slot's Sawtooth output and holds it
//! over the Lowpass's `q` row, well to the right of the port's dot, without
//! letting go, so a picture shows what the canvas says during a drag. The
//! output port and the row label are found in the AccessKit tree, whose
//! bounds are in canvas units; the canvas layer's transform takes them to
//! the screen, and the pointer events go in through `EguiInput`:
//!   cargo run --example host_window --features egui -- --drag --shot drag.png

use std::collections::BTreeMap;

use bevy::audio::{AudioSink, AudioSinkPlayback};
use bevy::diagnostic::FrameCount;
use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy_egui::{
    EguiContext, EguiContexts, EguiInput, EguiOutput, EguiPlugin, EguiPreUpdateSet,
    EguiPrimaryContextPass, PrimaryEguiContext, egui, egui::accesskit,
};

use bevy_symbios_audio::{
    AdsrEnvelope, AudioPatch, BiquadLowpass, BrownNoise, Connection, Gate, GraphNode, Instrument,
    Lfo, LfoShape, NodeGraph, NodeId, NodeKind, PinkNoise, SawtoothOsc, SequenceRecipe, SineOsc,
    Track, TriangleOsc,
    sequence::Event,
    ui::{
        AudioEditorPlugin, AudioMonitor, AuditionSource, AuditionState, MonitorRequest,
        MonitorStatus, PatchEditorState, SequenceEditorState, active_instrument_canvas,
        audio_patch_canvas, audition_strip, sequence_recipe_editor,
    },
};

/// The size Overlands gives its audio pop-out (its `src/ui/layout.rs`).
const SLOT: egui::Vec2 = egui::vec2(900.0, 640.0);
/// Space between the windows and around them.
const GAP: f32 = 12.0;
/// What the patch slot's Audition bakes: what Overlands' world bakes for a
/// construct's patch, and loops.
const PATCH_SAMPLE_RATE: u32 = 22_050;
const PATCH_SECS: f32 = 1.0;
/// The patch slot's caption says whose numbers those are.
const PATCH_NOTE: &str = "as a construct plays it";
/// The instrument the sequence slot opens on (the pluck: three wired nodes).
const OPEN_INSTRUMENT: usize = 2;
/// Frames with both windows up before `--shot` takes its picture.
const SETTLE_FRAMES: u32 = 60;
/// Frames to wait for a capture's file before giving up on it.
const GIVE_UP: u32 = 600;
/// How often each window's size is logged, in frames.
const LOG_EVERY: u32 = 30;
/// The lane of the instrument `--orphan` renames, and its new name.
const ORPHAN_TRACK: usize = 2;
const ORPHAN_NAME: &str = "harp";
/// Frames for which `--orphan` pins the side panel to its bottom: long
/// enough for the layout to settle, and then the scroll bar is the user's.
const ORPHAN_SCROLL_FRAMES: u32 = 30;
/// Frames with both windows up before `--rename` starts typing: past the
/// windows' sizing passes, and well before `--shot` takes its picture.
const TYPE_AFTER_FRAMES: u32 = 10;
/// Frames `--drag` takes to carry the wire from the output to the row.
const DRAG_FRAMES: u32 = 12;
/// How far right of the `q` label `--drag` holds the pointer, in canvas
/// units: on the row, and farther from the port's dot than a drop used to
/// snap from.
const DRAG_PAST_LABEL: f32 = 70.0;
/// Frames with both windows up when `--status` asks the monitor for its
/// state: after the windows' sizing passes, and in time for a bake to land
/// before `--shot` takes its picture. `baking` asks only once the layout has
/// settled, and the picture follows [`BAKING_FOR`] later, inside the bake.
const STATUS_AT_FRAMES: u32 = 20;
/// How long `--status baking` lets the bake run before the picture, so the
/// chip shows a count.
const BAKING_FOR: std::time::Duration = std::time::Duration::from_millis(100);

fn main() -> AppExit {
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
        .insert_resource(Editor::new(&args))
        .insert_resource(args)
        .init_resource::<Shot>()
        .init_resource::<Typist>()
        .init_resource::<Dragger>()
        .add_systems(Startup, setup_camera)
        .add_systems(
            PreUpdate,
            (type_the_rename, drag_a_wire)
                .after(EguiPreUpdateSet::ProcessInput)
                .before(EguiPreUpdateSet::BeginPass),
        )
        .add_systems(EguiPrimaryContextPass, render_ui)
        .add_systems(
            Update,
            (
                log_window_sizes,
                ask_for_the_status,
                silence_the_sinks,
                shoot,
            ),
        )
        .run()
}

/// The command line: `--light`, `--orphan`, `--rename <name>`, `--drag`,
/// `--broken`, `--status <state>` and `--shot <path>`.
#[derive(Resource)]
struct Args {
    light: bool,
    orphan: bool,
    rename: Option<String>,
    drag: bool,
    broken: bool,
    status: Option<Status>,
    shot: Option<String>,
}

/// The chip state `--status` asks for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Status {
    Idle,
    Baking,
    Playing,
    Muted,
    Error,
}

impl Status {
    fn parse(name: &str) -> Self {
        match name {
            "idle" => Self::Idle,
            "baking" => Self::Baking,
            "playing" => Self::Playing,
            "muted" => Self::Muted,
            "error" => Self::Error,
            other => {
                eprintln!("--status {other}: expected idle, baking, playing, muted or error");
                std::process::exit(2);
            }
        }
    }
}

impl Args {
    fn from_env() -> Self {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let value = |flag: &str, example: &str| {
            args.iter().position(|a| a == flag).map(|at| {
                args.get(at + 1).cloned().unwrap_or_else(|| {
                    eprintln!("{flag} needs a value: {flag} {example}");
                    std::process::exit(2);
                })
            })
        };
        let shot = value("--shot", "host_window.png");
        let status = value("--status", "baking")
            .map(|name| Status::parse(&name))
            // A picture shows the strip after an audition unless it asks
            // for another state.
            .or(shot.as_ref().map(|_| Status::Playing));
        Self {
            light: args.iter().any(|a| a == "--light"),
            orphan: args.iter().any(|a| a == "--orphan"),
            rename: value("--rename", "lead"),
            drag: args.iter().any(|a| a == "--drag"),
            broken: args.iter().any(|a| a == "--broken") || status == Some(Status::Error),
            status,
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
    /// The patch slot's strip, and whether its canvas committed an edit on
    /// the last frame (the strip is drawn above the canvas).
    patch_audition: AuditionState,
    patch_committed: bool,
    recipe: SequenceRecipe,
    sequence_state: SequenceEditorState,
    sequence_audition: AuditionState,
    sequence_committed: bool,
    /// The host bar's Mute: every sink silent, and the strips told.
    muted: bool,
    dark: bool,
    /// The theme last handed to egui, so it is set once per change.
    applied: Option<bool>,
    /// Each window's rect as shown this frame: patch slot, sequence slot.
    shown: [Option<egui::Rect>; 2],
    /// Frames on which both windows were shown.
    frames_shown: u32,
    /// `--orphan`: pin the side panel to its bottom while the layout settles.
    scroll_to_bottom: bool,
}

impl Editor {
    fn new(args: &Args) -> Self {
        let mut recipe = seeded_size_recipe();
        let mut sequence_state = SequenceEditorState::default();
        sequence_state.set_active_instrument(Some(OPEN_INSTRUMENT));
        if args.orphan {
            orphan_the_pluck(&mut recipe);
            sequence_state.set_selected_event(Some((ORPHAN_TRACK, 0)));
        }
        Self {
            patch: if args.broken {
                looped_drone_patch()
            } else {
                filtered_drone_patch()
            },
            patch_state: PatchEditorState::default(),
            patch_audition: AuditionState::default(),
            patch_committed: false,
            recipe,
            sequence_state,
            sequence_audition: AuditionState::default(),
            sequence_committed: false,
            muted: args.status == Some(Status::Muted),
            dark: !args.light,
            applied: None,
            shown: [None; 2],
            frames_shown: 0,
            scroll_to_bottom: args.orphan,
        }
    }
}

/// Rename the pluck in place and leave its notes naming the old id, the way
/// the instrument name field did before #55.
fn orphan_the_pluck(recipe: &mut SequenceRecipe) {
    let pluck = &mut recipe.instruments[OPEN_INSTRUMENT];
    debug_assert!(
        recipe.tracks[ORPHAN_TRACK]
            .events
            .iter()
            .all(|e| e.instrument_id == pluck.id),
        "the orphaned lane is the pluck's"
    );
    pluck.id = ORPHAN_NAME.into();
}

fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}

/// `--status`: ask the monitor for the state the picture should show, the
/// way pressing Audition does, once the windows have settled.
fn ask_for_the_status(
    args: Res<Args>,
    mut editor: ResMut<Editor>,
    mut requests: MessageWriter<MonitorRequest>,
    mut asked: Local<bool>,
) {
    let Some(status) = args.status else {
        return;
    };
    let at = if status == Status::Baking {
        SETTLE_FRAMES
    } else {
        STATUS_AT_FRAMES
    };
    if *asked || editor.frames_shown < at {
        return;
    }
    *asked = true;
    let editor = editor.as_mut();
    let request = match status {
        Status::Idle => return,
        Status::Playing | Status::Muted | Status::Error => {
            let source = AuditionSource::patch(&editor.patch, PATCH_SAMPLE_RATE, PATCH_SECS);
            editor.patch_audition.play(&source)
        }
        Status::Baking => editor
            .sequence_audition
            .play(&AuditionSource::sequence(&editor.recipe)),
    };
    info!("--status {status:?}: asking the monitor to play a slot");
    requests.write(request);
}

/// Hold every sink to the host bar's Mute, and silence them all under
/// `--shot`, the way a host's own sound switch mutes the monitor's voice.
fn silence_the_sinks(args: Res<Args>, editor: Res<Editor>, mut sinks: Query<&mut AudioSink>) {
    let silent = editor.muted || args.shot.is_some();
    for mut sink in &mut sinks {
        if silent && !sink.is_muted() {
            sink.mute();
        } else if !silent && sink.is_muted() {
            sink.unmute();
        }
    }
}

fn render_ui(
    mut contexts: EguiContexts,
    args: Res<Args>,
    mut editor: ResMut<Editor>,
    monitor: Res<AudioMonitor>,
    mut requests: MessageWriter<MonitorRequest>,
) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    if args.rename.is_some() || args.drag {
        // `type_the_rename` and `drag_a_wire` find their widgets in this tree.
        ctx.enable_accesskit();
    }
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
            ui.separator();
            ui.checkbox(&mut editor.muted, "Mute")
                .on_hover_text("The host's own sound switch: every sink silent");
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
            let source = AuditionSource::patch(&editor.patch, PATCH_SAMPLE_RATE, PATCH_SECS)
                .with_note(PATCH_NOTE);
            if let Some(request) = audition_strip(
                ui,
                monitor,
                &mut editor.patch_audition,
                source,
                editor.patch_committed,
                editor.muted,
            ) {
                requests.write(request);
            }
            ui.separator();
            let res = audio_patch_canvas(
                ui,
                &mut editor.patch,
                &mut editor.patch_state,
                id.with("canvas"),
            );
            editor.patch_committed = res.rebake;
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
    let pin_to_bottom = editor.scroll_to_bottom && editor.frames_shown < ORPHAN_SCROLL_FRAMES;
    slot_window("Sequence slot", pos, free)
        .show(ctx, |ui| {
            if let Some(request) = audition_strip(
                ui,
                monitor,
                &mut editor.sequence_audition,
                AuditionSource::sequence(&editor.recipe),
                editor.sequence_committed,
                editor.muted,
            ) {
                requests.write(request);
            }
            ui.separator();
            let mut committed = false;
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
                    let mut scroll = egui::ScrollArea::vertical().auto_shrink([false, false]);
                    if pin_to_bottom {
                        // Past the end; the scroll area clamps it to the
                        // bottom. Finite: `f32::MAX` overflows the layout to
                        // -inf and trips egui's NaN assertion.
                        scroll = scroll.vertical_scroll_offset(1.0e6);
                    }
                    scroll.show(ui, |ui| {
                        committed |= sequence_recipe_editor(
                            ui,
                            &mut editor.recipe,
                            &mut editor.sequence_state,
                            id.with("sequence"),
                        )
                        .rebake;
                    });
                });
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE.inner_margin(egui::Margin {
                    left: 6,
                    ..egui::Margin::ZERO
                }))
                .show(ui, |ui| {
                    committed |= active_instrument_canvas(
                        ui,
                        &mut editor.recipe,
                        &mut editor.sequence_state,
                        id.with("instrument_canvas"),
                    )
                    .rebake;
                });
            editor.sequence_committed = committed;
        })
        .map(|shown| shown.response.rect)
}

/// Where `--rename` is in its script: one step per frame, because a field
/// focused on one frame reads keys from the next.
#[derive(Resource, Default)]
enum Typist {
    #[default]
    Finding,
    Typing {
        /// Characters in the name being typed over.
        chars: usize,
    },
    Committing,
    Done,
}

/// `--rename <name>`: focus the open instrument's name field, type `<name>`
/// over it and press Enter. It runs between bevy_egui reading this frame's
/// input and beginning the pass, and it finds the field in the AccessKit
/// tree of the last pass, so it needs none of the editor's private ids.
fn type_the_rename(
    args: Res<Args>,
    editor: Res<Editor>,
    mut typist: ResMut<Typist>,
    mut contexts: Query<(&mut EguiInput, &EguiOutput), With<PrimaryEguiContext>>,
) {
    let Some(name) = &args.rename else {
        return;
    };
    let Ok((mut input, output)) = contexts.single_mut() else {
        return;
    };
    let key = |key| egui::Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    };
    match *typist {
        Typist::Finding => {
            if editor.frames_shown < TYPE_AFTER_FRAMES {
                return;
            }
            let current = editor.recipe.instruments[OPEN_INSTRUMENT].id.as_str();
            let Some(field) = output
                .platform_output
                .accesskit_update
                .iter()
                .flat_map(|update| &update.nodes)
                .find(|(_, node)| {
                    node.role() == accesskit::Role::TextInput && node.value() == Some(current)
                })
                .map(|(id, _)| *id)
            else {
                return;
            };
            input.0.events.push(egui::Event::AccessKitActionRequest(
                accesskit::ActionRequest {
                    action: accesskit::Action::Focus,
                    target_tree: accesskit::TreeId::ROOT,
                    target_node: field,
                    data: None,
                },
            ));
            *typist = Typist::Typing {
                chars: current.chars().count(),
            };
        }
        Typist::Typing { chars } => {
            input
                .0
                .events
                .extend((0..chars).map(|_| key(egui::Key::Backspace)));
            input.0.events.push(egui::Event::Text(name.clone()));
            *typist = Typist::Committing;
        }
        Typist::Committing => {
            input.0.events.push(key(egui::Key::Enter));
            info!("--rename: typed {name:?} over the open instrument's name and pressed Enter");
            *typist = Typist::Done;
        }
        Typist::Done => {}
    }
}

/// Where `--drag` is: looking for the port and the row, carrying the wire
/// (from, to, frames done), or holding it for the picture.
#[derive(Resource, Default)]
enum Dragger {
    #[default]
    Finding,
    Carrying(egui::Pos2, egui::Pos2, u32),
    Holding,
}

/// `--drag`: press on the Sawtooth's output port in the patch slot, carry
/// the wire to the Lowpass's `q` row over [`DRAG_FRAMES`] frames, and hold
/// it there without releasing. Both widgets are found in the last pass's
/// AccessKit tree, in canvas units, and taken to the screen by whichever
/// layer transform puts the port inside the patch slot's window.
fn drag_a_wire(
    args: Res<Args>,
    editor: Res<Editor>,
    mut dragger: ResMut<Dragger>,
    mut contexts: Query<(&mut EguiContext, &mut EguiInput, &EguiOutput), With<PrimaryEguiContext>>,
) {
    if !args.drag {
        return;
    }
    let Ok((mut context, mut input, output)) = contexts.single_mut() else {
        return;
    };
    match *dragger {
        Dragger::Finding => {
            let Some(window) = editor.shown[0] else {
                return;
            };
            if editor.frames_shown < TYPE_AFTER_FRAMES {
                return;
            }
            let bounds = |wanted: &dyn Fn(&accesskit::Node) -> bool| {
                output
                    .platform_output
                    .accesskit_update
                    .iter()
                    .flat_map(|update| &update.nodes)
                    .find(|(_, node)| wanted(node))
                    .and_then(|(_, node)| node.bounds())
                    .map(|b| {
                        egui::Rect::from_min_max(
                            egui::pos2(b.x0 as f32, b.y0 as f32),
                            egui::pos2(b.x1 as f32, b.y1 as f32),
                        )
                    })
            };
            let Some(port) = bounds(&|n| {
                n.label()
                    .is_some_and(|l| l.starts_with("Output of #0 Sawtooth"))
            }) else {
                return;
            };
            let Some(row_label) =
                bounds(&|n| n.role() == accesskit::Role::Label && n.value() == Some("q:"))
            else {
                return;
            };
            let transforms = context.get_mut().memory(|m| m.to_global.clone());
            let Some(to_screen) = transforms
                .values()
                .find(|t| window.contains(**t * port.center()))
            else {
                return;
            };
            let from = *to_screen * port.center();
            let to =
                *to_screen * egui::pos2(row_label.right() + DRAG_PAST_LABEL, row_label.center().y);
            info!("--drag: pressing at {from:?} and carrying the wire to {to:?}");
            input.0.events.push(egui::Event::PointerMoved(from));
            input.0.events.push(egui::Event::PointerButton {
                pos: from,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            });
            *dragger = Dragger::Carrying(from, to, 0);
        }
        Dragger::Carrying(from, to, done) => {
            let done = done + 1;
            let t = done as f32 / DRAG_FRAMES as f32;
            input
                .0
                .events
                .push(egui::Event::PointerMoved(from.lerp(to, t)));
            *dragger = if done >= DRAG_FRAMES {
                info!("--drag: holding a wire from the Sawtooth over the Lowpass's q row");
                Dragger::Holding
            } else {
                Dragger::Carrying(from, to, done)
            };
        }
        Dragger::Holding => {}
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

/// Whether the monitor shows what `--status` asked for, so the picture can
/// be taken. `Err` when it never will: a `baking` picture whose bake ended
/// before it had run [`BAKING_FOR`].
fn status_reached(status: Option<Status>, monitor: &AudioMonitor) -> Result<bool, String> {
    Ok(match status {
        None | Some(Status::Idle) => true,
        Some(Status::Playing | Status::Muted) => monitor.status == MonitorStatus::Playing,
        Some(Status::Error) => matches!(monitor.status, MonitorStatus::Error(_)),
        Some(Status::Baking) => match monitor.bake_elapsed() {
            Some(elapsed) => elapsed >= BAKING_FOR,
            None if monitor.status == MonitorStatus::Playing => {
                return Err(format!(
                    "the bake ended in under {BAKING_FOR:?}, before the picture; \
                     a baking picture needs the dev profile"
                ));
            }
            None => false,
        },
    })
}

/// Save a picture of the app window and quit, for `--shot <path>`.
fn shoot(
    mut commands: Commands,
    args: Res<Args>,
    editor: Res<Editor>,
    monitor: Res<AudioMonitor>,
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
        match status_reached(args.status, &monitor) {
            Ok(true) => {}
            Ok(false) => {
                shot.waited += 1;
                if shot.waited > GIVE_UP {
                    error!("--shot: the monitor never reached {:?}", args.status);
                    exit.write(AppExit::error());
                }
                return;
            }
            Err(why) => {
                error!("--shot: {why}");
                exit.write(AppExit::error());
                return;
            }
        }
        shot.waited = 0;
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

/// `--broken`: the drone with a gain after the lowpass wired back into the
/// lowpass's input, so `#2 Lowpass` and `#3 Gain` feed each other in a loop
/// and the patch cannot bake.
fn looped_drone_patch() -> AudioPatch {
    let mut patch = filtered_drone_patch();
    let (filter, gain) = (NodeId(2), NodeId(3));
    let mut gain_inputs = BTreeMap::new();
    gain_inputs.insert("in".to_string(), vec![Connection::from_node(filter)]);
    patch.graph.nodes.push(GraphNode {
        id: gain,
        kind: NodeKind::Gain(Default::default()),
        inputs: gain_inputs,
    });
    let lowpass = patch
        .graph
        .nodes
        .iter_mut()
        .find(|n| n.id == filter)
        .expect("the drone has its lowpass");
    lowpass
        .inputs
        .entry("in".to_string())
        .or_default()
        .push(Connection::from_node(gain));
    patch
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
