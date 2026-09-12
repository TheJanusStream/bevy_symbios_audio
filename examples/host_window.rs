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
//! The host sets no editor style, so the editors derive their colours from
//! egui's theme (`EditorStyle::from_visuals`) and follow the bar's Dark and
//! Light. A host with its own palette would call `set_editor_style` when its
//! theme changes, as Overlands does.
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
//!
//! `--playhead <secs>` seeks the audition to `secs` once its bake has
//! landed, so the cursor on the waveform and on the timeline is in the same
//! place in every run. Through a real seek, the path a click on the
//! waveform takes — under `--shot` every sink is muted, and muting is
//! `set_volume(0)` rather than a pause, so a voice does go on advancing,
//! but how far it has got when the picture is taken depends on the
//! machine. Use it with `--status playing` for the patch waveform's cursor
//! and `--status baking` for the timeline's:
//!
//!   cargo run --example host_window --features egui -- --status playing --playhead 0.6 --shot cursor.png
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
//!
//! `--wire` parks the pointer on the wire that drives the Lowpass's cutoff,
//! so a picture shows a hovered wire and its tooltip; `--pick` clicks it as
//! well, so the picture shows the wire picked — its cross and the amount
//! editor that opens on it:
//!   cargo run --example host_window --features egui -- --wire --shot wire.png
//!   cargo run --example host_window --features egui -- --pick --shot picked.png
//!
//! `--unheard` points the patch slot's output at its Sawtooth, so the LFO
//! and the filter are in the patch and nothing plays them:
//!   cargo run --example host_window --features egui -- --unheard --shot unheard.png
//!
//! `--menu add` opens the toolbar's Add node menu and `--menu node` the
//! context menu on node #0's grip, each held open for the picture:
//!   cargo run --example host_window --features egui -- --menu add --shot menu.png
//!
//! `--notice <text>` draws `<text>` as the host's own first line above the
//! editors, and `--notice-live <text>` draws it in the accent a host uses
//! when somebody is actually listening. It is how the audience notice
//! Overlands puts at the top of its pop-out gets into a picture of this
//! layout (Overlands #1337 A4). The sentence is passed in rather than
//! written here on purpose: its wording lives in the host, in one tested
//! function, and a copy in this example would be a second answer:
//!   cargo run --example host_window --features egui -- \
//!       --notice "Nobody else is here — but anyone who arrives sees these unsaved edits, not your last save." \
//!       --shot alone.png
//!   cargo run --example host_window --features egui -- \
//!       --notice-live "3 people here see these edits as you make them." \
//!       --shot heard.png
//!
//! `--notes <what>` drives the sequence slot's timeline: `picked` picks two
//! notes on one track, so the picture shows their outlines and the
//! inspector's count; `box` holds a marquee out over three of them; `snap`
//! opens the snap picker; and `track` opens a track's own menu at a beat on
//! a clear stretch of it. Each finds its notes through
//! [`SequenceEditorState::notes`], the geometry the timeline publishes,
//! rather than through the AccessKit tree, where a painted block has no
//! bounds of its own:
//!   cargo run --example host_window --features egui -- --notes picked --shot picked.png
//!   cargo run --example host_window --features egui -- --notes box --shot box.png
//!   cargo run --example host_window --features egui -- --notes snap --shot snap.png
//!   cargo run --example host_window --features egui -- --notes track --shot track.png
//!
//! # Why a scripted gesture waits
//!
//! Every one of those presses where a widget *is*, and for the opening
//! frames a widget is not where it will end up. Two things move it. Both
//! windows size themselves, the canvas's [`egui::Scene`] auto-fits, and a
//! node box that has never been drawn is placed at a guessed size and laid
//! out again once it has been measured — that settles in the first handful
//! of frames. And then the audition the picture wants starts playing, the
//! strip above the canvas grows a waveform, and everything below it moves
//! down by the height of one: about 54 points, or two rows of a node's
//! Inputs list. A gesture fired on a fixed frame count grabs the right port
//! and then watches the row it was aimed at slide away — the `--drag` wire
//! landed on the Lowpass's `in` row rather than its `q` (crate #67).
//!
//! So a gesture waits for what `--shot` waits for (the monitor has reached
//! the state `--status` asked for) and then for its own target points to
//! hold still for [`STILL_FRAMES`] frames; and `--shot` in turn waits for
//! the gesture to be in place, rather than for a frame count of its own.

use std::collections::BTreeMap;

use bevy::audio::{AudioSink, AudioSinkPlayback};
use bevy::diagnostic::FrameCount;
use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy_egui::{
    EguiContexts, EguiInput, EguiOutput, EguiPlugin, EguiPreUpdateSet, EguiPrimaryContextPass,
    PrimaryEguiContext, egui, egui::accesskit,
};

use bevy_symbios_audio::{
    AdsrEnvelope, AudioPatch, BiquadLowpass, BrownNoise, Connection, Envelope, Gate, GraphNode,
    Instrument, Lfo, LfoShape, NodeGraph, NodeId, NodeKind, PinkNoise, SawtoothOsc, SequenceRecipe,
    SineOsc, Track, TriangleOsc,
    sequence::Event,
    ui::{
        AudioEditorPlugin, AudioMonitor, AuditionSource, AuditionState, EditorLimits,
        MonitorControl, MonitorRequest, MonitorStatus, PatchEditorState, SequenceEditorState,
        active_instrument_canvas, audio_patch_canvas, audition_strip, patch_hearing,
        sequence_recipe_editor,
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
/// Typing needs no coordinates — the field is focused by an AccessKit
/// request — so a frame count is enough for it, and not for a gesture that
/// presses somewhere ([`Settle`]).
const TYPE_AFTER_FRAMES: u32 = 10;
/// Frames running for which a gesture's target points must not have moved
/// before it acts. See "Why a scripted gesture waits" above.
const STILL_FRAMES: u32 = 6;
/// How far a target point may move between frames and still count as
/// still, in points: a rounding wobble is not the layout moving.
const STILL_SLACK: f32 = 0.5;
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
        .init_resource::<Hoverer>()
        .init_resource::<Menuer>()
        .init_resource::<Noter>()
        .init_resource::<Asker>()
        .init_resource::<Parker>()
        .add_systems(Startup, setup_camera)
        .add_systems(
            PreUpdate,
            (
                type_the_rename,
                drag_a_wire,
                hover_a_wire,
                open_a_menu,
                work_the_timeline,
                ask_to_remove_a_track,
                park_on_a_widget,
            )
                .after(EguiPreUpdateSet::ProcessInput)
                .before(EguiPreUpdateSet::BeginPass),
        )
        .add_systems(EguiPrimaryContextPass, render_ui)
        .add_systems(
            Update,
            (
                log_window_sizes,
                ask_for_the_status,
                move_the_playhead,
                silence_the_sinks,
                shoot,
            ),
        )
        .run()
}

/// The command line: `--light`, `--orphan`, `--rename <name>`, `--drag`,
/// `--wire`, `--menu <which>`, `--notes <what>`, `--limits`, `--confirm`,
/// `--beyond`, `--hover <label>`, `--broken`, `--status <state>`,
/// `--playhead <secs>` and `--shot <path>`.
#[derive(Resource)]
struct Args {
    light: bool,
    orphan: bool,
    rename: Option<String>,
    drag: bool,
    wire: bool,
    /// Pick the wire `--wire` parks on, as well as hovering it.
    pick: bool,
    unheard: bool,
    menu: Option<Menu>,
    notes: Option<Notes>,
    /// `--limits`: hold both slots to caps the seeded content already
    /// fills, so every Add is disabled and every readout is at its number.
    limits: bool,
    /// `--confirm`: press a track's gutter cross and leave the question up.
    confirm: bool,
    /// `--beyond`: put a value outside its slider's track into the patch
    /// slot, which is what an editor that never rewrites one looks like.
    beyond: bool,
    /// `--hover <label>`: park the pointer on the widget whose label starts
    /// with this, so its tooltip is in the picture.
    hover: Option<String>,
    broken: bool,
    /// `--notice <text>` / `--notice-live <text>`: the host's own line
    /// above the editors, and whether anybody is listening.
    notice: Option<String>,
    notice_live: bool,
    status: Option<Status>,
    /// `--playhead <secs>`: seek the audition to `secs` once it is
    /// playing, so a picture of the cursor is the same picture every run.
    playhead: Option<f32>,
    shot: Option<String>,
}

/// Which menu `--menu` opens and holds open for the picture.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Menu {
    /// The toolbar's Add node menu.
    Add,
    /// The context menu on node #0's grip.
    Node,
}

impl Menu {
    fn parse(name: &str) -> Self {
        match name {
            "add" => Self::Add,
            "node" => Self::Node,
            other => {
                eprintln!("--menu {other}: expected add or node");
                std::process::exit(2);
            }
        }
    }
}

/// What `--notes` does to the sequence slot's timeline.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Notes {
    /// Pick two notes on one track with a click and a shift-click.
    Picked,
    /// Hold a marquee out over three of them.
    Box,
    /// Open the toolbar's snap picker.
    Snap,
    /// Open a track's own menu at a beat on a clear stretch of it.
    Track,
}

impl Notes {
    fn parse(name: &str) -> Self {
        match name {
            "picked" => Self::Picked,
            "box" => Self::Box,
            "snap" => Self::Snap,
            "track" => Self::Track,
            other => {
                eprintln!("--notes {other}: expected picked, box, snap or track");
                std::process::exit(2);
            }
        }
    }
}

/// The chip state `--status` asks for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Status {
    Idle,
    Baking,
    Playing,
    /// The SEQUENCE slot playing, asked for early enough that its bake has
    /// landed and its timeline is showing a cursor when the picture is
    /// taken. `baking` asks for the same slot at the shot frame itself, to
    /// picture the Baking chip; this one asks with time to spare.
    PlayingSequence,
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
            "playing-sequence" => Self::PlayingSequence,
            "error" => Self::Error,
            other => {
                eprintln!(
                    "--status {other}: expected idle, baking, playing, \
                     playing-sequence, muted or error"
                );
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
            wire: args.iter().any(|a| a == "--wire" || a == "--pick"),
            pick: args.iter().any(|a| a == "--pick"),
            unheard: args.iter().any(|a| a == "--unheard"),
            menu: value("--menu", "add").as_deref().map(Menu::parse),
            notes: value("--notes", "picked").as_deref().map(Notes::parse),
            limits: args.iter().any(|a| a == "--limits"),
            confirm: args.iter().any(|a| a == "--confirm"),
            beyond: args.iter().any(|a| a == "--beyond"),
            hover: value("--hover", "Add track"),
            broken: args.iter().any(|a| a == "--broken") || status == Some(Status::Error),
            notice: value("--notice", "").or_else(|| value("--notice-live", "")),
            notice_live: args.iter().any(|a| a == "--notice-live"),
            status,
            playhead: value("--playhead", "1.0").and_then(|v| v.parse().ok()),
            shot,
        }
    }

    /// Whether any flag drives the app through a gesture of its own. Those
    /// find their widgets in the AccessKit tree, which is off unless one
    /// does.
    fn scripted(&self) -> bool {
        self.rename.is_some()
            || self.drag
            || self.wire
            || self.menu.is_some()
            || self.notes.is_some()
            || self.confirm
            || self.hover.is_some()
    }

    /// Whether the gesture this run asked for is in place, so `--shot` can
    /// take its picture. A gesture that never gets there is caught by
    /// `--shot`'s own [`GIVE_UP`].
    fn gesture_in_place(&self, gestures: &Gestures) -> bool {
        (!self.drag || matches!(*gestures.dragger, Dragger::Holding(_)))
            && (!self.wire || matches!(*gestures.hoverer, Hoverer::Parked(_)))
            && (self.menu.is_none() || matches!(*gestures.menuer, Menuer::Open(_)))
            && (self.notes.is_none() || matches!(*gestures.noter, Noter::Done(_)))
            && (!self.confirm || matches!(*gestures.asker, Asker::Asked))
            && (self.hover.is_none() || matches!(*gestures.parker, Parker::Parked(_)))
    }
}

/// Where each scripted gesture has got to, as one system parameter: three
/// resources that are always read together.
#[derive(SystemParam)]
struct Gestures<'w> {
    dragger: Res<'w, Dragger>,
    hoverer: Res<'w, Hoverer>,
    menuer: Res<'w, Menuer>,
    noter: Res<'w, Noter>,
    asker: Res<'w, Asker>,
    parker: Res<'w, Parker>,
}

/// `--confirm`: where the gesture that asks to remove a track has got to.
#[derive(Resource, Default, Clone, Copy, PartialEq, Eq, Debug)]
enum Asker {
    #[default]
    Finding,
    /// Pressed the cross; the question comes up on the next frame.
    Pressed,
    /// The question is up and the pointer is parked off every control.
    Asked,
}

/// `--hover`: where the gesture that parks the pointer on a widget has got
/// to.
#[derive(Resource, Default, Clone, Copy, PartialEq, Debug)]
enum Parker {
    #[default]
    Finding,
    /// On the widget, holding still while egui's tooltip delay runs out.
    Waiting(egui::Pos2, u32),
    /// Held still long enough that the tooltip is up.
    Parked(egui::Pos2),
}

/// Frames the pointer is held on a widget before its hover is counted as
/// shown.
///
/// egui's `interaction.tooltip_delay` is half a second, so a picture taken
/// on the frame the pointer arrives is a picture of no tooltip — which is
/// what the first run of `--hover` produced.
const HOVER_FRAMES: u32 = 48;

/// Frames the parked position is re-sent before the pointer is left alone,
/// so it lands even if the real cursor is still being fed in.
const SETTLE_MOVES: u32 = 4;

/// The toolbar button that adds a node, by the label it wears: the harness
/// finds it in the AccessKit tree, and the canvas's own tests look for the
/// same string.
const ADD_NODE: &str = "Add node";

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
    /// `--notice` / `--notice-live`: the host's own line above the editors.
    notice: Option<String>,
    notice_live: bool,
}

impl Editor {
    fn new(args: &Args) -> Self {
        let mut recipe = seeded_size_recipe();
        let mut sequence_state = SequenceEditorState::default();
        sequence_state.set_active_instrument(Some(OPEN_INSTRUMENT));
        // A host that knows what its world plays beds at says so, the way
        // Overlands does: 22 050 and 44 100, and not the ladder up to 96
        // (#64, Overlands #1337 C7). The seeded recipe here IS a room's
        // ambient bed, at 22 050.
        sequence_state.set_sample_rates(&[22_050, 44_100]);
        if args.orphan {
            orphan_the_pluck(&mut recipe);
            sequence_state.set_selected_event(Some((ORPHAN_TRACK, 0)));
        }
        let mut patch_state = PatchEditorState::default();
        if args.limits {
            // The seeded recipe's own size, so every cap is reached by
            // what the picture already shows rather than by a wall of
            // filler: five instruments, five tracks, and the open
            // instrument's three nodes.
            let caps = EditorLimits::from_envelope(Envelope {
                max_instruments: recipe.instruments.len(),
                max_tracks: recipe.tracks.len(),
                max_nodes: 3,
                max_track_events: 4,
                ..Envelope::default()
            });
            sequence_state.set_limits(caps);
            patch_state.set_limits(caps);
        }
        Self {
            patch: match () {
                _ if args.broken => looped_drone_patch(),
                // Valid, and two of its three nodes bake into nothing: the
                // output is the Sawtooth, so the filter the LFO drives is
                // not on the way to it.
                _ if args.unheard => {
                    let mut patch = filtered_drone_patch();
                    patch.graph.output = NodeId(0);
                    patch
                }
                _ if args.beyond => beyond_the_track_patch(),
                _ => filtered_drone_patch(),
            },
            patch_state,
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
            notice: args.notice.clone(),
            notice_live: args.notice_live,
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
        Status::Baking | Status::PlayingSequence => editor
            .sequence_audition
            .play(&AuditionSource::sequence(&editor.recipe)),
    };
    info!("--status {status:?}: asking the monitor to play a slot");
    requests.write(request);
}

/// `--playhead <secs>`: put the audition's cursor at `secs`, once there is
/// a bake for it to be inside.
///
/// Through a real [`MonitorControl::Seek`], not by writing a position: it
/// is the same path a click on the waveform takes, so the picture proves
/// the mechanism rather than a value poked past it. Under `--shot` every
/// sink is muted — muting is `set_volume(0)`, not a pause, so a voice does
/// go on advancing — but how far it has got by the time the picture is
/// taken depends on the machine, and a cursor that lands somewhere
/// different every run is a picture nothing can be checked against. This
/// pins it.
fn move_the_playhead(
    args: Res<Args>,
    editor: Res<Editor>,
    monitor: Res<AudioMonitor>,
    mut controls: MessageWriter<MonitorControl>,
    mut asked: Local<bool>,
) {
    let Some(secs) = args.playhead else {
        return;
    };
    // Once the bake has landed: a seek before there is a buffer has
    // nowhere to land and is dropped.
    if *asked || editor.frames_shown < STATUS_AT_FRAMES || monitor.loop_secs().is_none() {
        return;
    }
    *asked = true;
    info!("--playhead: seeking the audition to {secs} s");
    controls.write(MonitorControl::Seek(secs));
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
    mut controls: MessageWriter<MonitorControl>,
) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    if args.scripted() {
        // Every scripted gesture finds its widgets in this tree.
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
        patch_slot(
            ctx,
            editor,
            &monitor,
            &mut requests,
            &mut controls,
            patch_at,
            free,
        ),
        sequence_slot(
            ctx,
            editor,
            &monitor,
            &mut requests,
            &mut controls,
            sequence_at,
            free,
        ),
    ];
    if editor.shown.iter().all(Option::is_some) {
        editor.frames_shown += 1;
    }
}

/// The host's own first line above an editor: what Overlands' pop-out
/// draws there, so a picture of this layout is a picture of that one
/// (Overlands #1337 A4).
///
/// The sentence arrives on the command line. Its wording is the host's and
/// lives in one tested function there; a copy here would be a second
/// answer to the question "who is hearing this", with nothing keeping the
/// two honest. All this example owns is where it goes and what it looks
/// like when somebody is listening.
fn host_notice(ui: &mut egui::Ui, notice: Option<&str>, live: bool) {
    let Some(text) = notice else {
        return;
    };
    let colour = if live {
        ui.visuals().hyperlink_color
    } else {
        ui.visuals().weak_text_color()
    };
    ui.label(egui::RichText::new(text).small().color(colour));
    ui.add_space(4.0);
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
    controls: &mut MessageWriter<MonitorControl>,
    pos: egui::Pos2,
    free: egui::Rect,
) -> Option<egui::Rect> {
    let id = egui::Id::new("patch_slot");
    slot_window("Patch slot", pos, free)
        .show(ctx, |ui| {
            host_notice(ui, editor.notice.as_deref(), editor.notice_live);
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
            for control in editor.patch_audition.take_controls() {
                controls.write(control);
            }
            ui.separator();
            let res = audio_patch_canvas(
                ui,
                &mut editor.patch,
                &mut editor.patch_state,
                id.with("canvas"),
            );
            editor.patch_committed = res.rebake;
            // "Hear this node": a COPY of the patch, output moved, played
            // in place of the whole thing. The patch itself is untouched
            // (#65, Overlands #1338 D3).
            if let Some(node) = editor.patch_state.take_hear_node() {
                requests.write(MonitorRequest::PlayPatch {
                    patch: patch_hearing(&editor.patch, node),
                    sample_rate: PATCH_SAMPLE_RATE,
                    duration_secs: PATCH_SECS,
                });
            }
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
    controls: &mut MessageWriter<MonitorControl>,
    pos: egui::Pos2,
    free: egui::Rect,
) -> Option<egui::Rect> {
    let id = egui::Id::new("sequence_slot");
    let pin_to_bottom = editor.scroll_to_bottom && editor.frames_shown < ORPHAN_SCROLL_FRAMES;
    slot_window("Sequence slot", pos, free)
        .show(ctx, |ui| {
            host_notice(ui, editor.notice.as_deref(), editor.notice_live);
            // Solo and mute decide what the audition plays: a copy with
            // the silenced tracks left out, or the recipe itself when
            // every track is heard.
            let heard = editor.sequence_state.heard_recipe(&editor.recipe);
            let playing = heard.as_ref().unwrap_or(&editor.recipe);
            if let Some(request) = audition_strip(
                ui,
                monitor,
                &mut editor.sequence_audition,
                AuditionSource::sequence(playing),
                editor.sequence_committed,
                editor.muted,
            ) {
                requests.write(request);
            }
            for control in editor.sequence_audition.take_controls() {
                controls.write(control);
            }
            // The timeline's cursor, but ONLY while the monitor is playing
            // this slot's own audition: a cursor running over a timeline
            // whose sound is not the one in the room says this recipe is
            // sounding when it is not.
            editor.sequence_state.set_playhead(
                editor
                    .sequence_audition
                    .is_playing(monitor)
                    .then(|| monitor.position_secs())
                    .flatten(),
            );
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

/// Whether a scripted gesture may read the layout yet: the windows are up,
/// and the monitor has reached the state this run's picture is of.
///
/// The second half is what a frame count cannot express — an audition
/// starting is a layout change (the strip grows a waveform), and it happens
/// whenever the bake happens to finish. See "Why a scripted gesture waits".
fn layout_is_final(args: &Args, editor: &Editor, monitor: &AudioMonitor) -> bool {
    editor.frames_shown >= TYPE_AFTER_FRAMES
        && matches!(status_reached(args.status, monitor), Ok(true))
}

/// A gesture's target points, and how many frames running they have been
/// where they are.
///
/// The opening frames of the app are not a settled layout: the windows size
/// themselves, the canvas's `Scene` auto-fits, and a node box placed at a
/// guessed size is laid out again once it has been measured. A press taken
/// then lands on the right widget and is left holding stale geometry a
/// moment later (crate #67), so a gesture recomputes its points every frame
/// and waits for them to stop moving.
#[derive(Default)]
struct Settle {
    was: Option<Vec<egui::Pos2>>,
    still: u32,
}

impl Settle {
    /// `now`, once it has been the same for [`STILL_FRAMES`] frames running.
    fn settled(&mut self, now: Vec<egui::Pos2>) -> Option<Vec<egui::Pos2>> {
        let moved = self.was.as_ref().is_none_or(|was| {
            was.len() != now.len()
                || was
                    .iter()
                    .zip(&now)
                    .any(|(a, b)| a.distance(*b) > STILL_SLACK)
        });
        if moved {
            self.still = 0;
            self.was = Some(now);
            return None;
        }
        self.still += 1;
        (self.still >= STILL_FRAMES).then_some(now)
    }
}

/// Every widget in the last pass's AccessKit tree that `wanted` accepts,
/// as the rect the tree gives for it.
///
/// Bounds are in the coordinates of the layer the widget was drawn on: the
/// screen for the toolbar and the window chrome, scene units for anything
/// inside a canvas.
fn accesskit_bounds(
    output: &EguiOutput,
    wanted: &dyn Fn(&accesskit::Node) -> bool,
) -> Vec<egui::Rect> {
    output
        .platform_output
        .accesskit_update
        .iter()
        .flat_map(|update| &update.nodes)
        .filter(|(_, node)| wanted(node))
        .filter_map(|(_, node)| node.bounds())
        .map(|b| {
            egui::Rect::from_min_max(
                egui::pos2(b.x0 as f32, b.y0 as f32),
                egui::pos2(b.x1 as f32, b.y1 as f32),
            )
        })
        .collect()
}

/// The one widget the patch slot's canvas drew that `wanted` accepts, in
/// screen points.
///
/// The AccessKit tree covers the whole app, and the sequence slot draws a
/// canvas of its own, so a label like `"q:"` is not unique: `to_screen` is
/// the patch canvas's own transform, and a widget of some other canvas's
/// lands outside the patch window once it has been applied. `None` when
/// nothing matches, and when more than one does — a gesture that cannot
/// tell which widget it means must not guess.
fn in_the_patch_canvas(
    output: &EguiOutput,
    window: egui::Rect,
    to_screen: egui::emath::TSTransform,
    what: &str,
    wanted: &dyn Fn(&accesskit::Node) -> bool,
) -> Option<egui::Rect> {
    let found: Vec<egui::Rect> = accesskit_bounds(output, wanted)
        .into_iter()
        .map(|rect| to_screen * rect)
        .filter(|rect| window.contains(rect.center()))
        .collect();
    match found.as_slice() {
        [one] => Some(*one),
        [] => None,
        many => {
            warn!(
                "{what}: {} of them in the patch canvas, so none",
                many.len()
            );
            None
        }
    }
}

/// The one widget the patch slot drew *outside* its canvas — its toolbar,
/// its strip — that `wanted` accepts. Those are drawn on the window's own
/// layer, which carries no transform, so the tree's bounds are already
/// screen points.
fn in_the_patch_window(
    output: &EguiOutput,
    window: egui::Rect,
    what: &str,
    wanted: &dyn Fn(&accesskit::Node) -> bool,
) -> Option<egui::Rect> {
    let found: Vec<egui::Rect> = accesskit_bounds(output, wanted)
        .into_iter()
        .filter(|rect| window.contains(rect.center()))
        .collect();
    match found.as_slice() {
        [one] => Some(*one),
        [] => None,
        many => {
            warn!(
                "{what}: {} of them in the patch window, so none",
                many.len()
            );
            None
        }
    }
}

/// Node #0's grip in the patch slot, in screen points.
///
/// `"#0"` is not a name that says which canvas it is in: the sequence slot
/// draws an instrument's patch, whose nodes are numbered from zero too, and
/// both windows are in the one AccessKit tree. The Sawtooth's output port
/// *is* unique, and the grip is the leftmost thing on the same title row as
/// it, so the port is what picks which `"#0"` is meant.
fn node_zero_grip(
    output: &EguiOutput,
    window: egui::Rect,
    to_screen: egui::emath::TSTransform,
) -> Option<egui::Rect> {
    let port = in_the_patch_canvas(output, window, to_screen, "the Sawtooth's output", &|n| {
        n.label()
            .is_some_and(|l| l.starts_with("Output of #0 Sawtooth"))
    })?;
    accesskit_bounds(output, &label_is("#0"))
        .into_iter()
        .map(|rect| to_screen * rect)
        .filter(|rect| window.contains(rect.center()))
        // On the port's row, and left of it: the grip is where the title
        // starts and the output dot is where it ends.
        .filter(|rect| {
            (rect.center().y - port.center().y).abs() <= port.height()
                && rect.center().x < port.center().x
        })
        .min_by(|a, b| a.center().x.total_cmp(&b.center().x))
}

/// A label the canvas painted, by the text it shows.
fn label_is(text: &'static str) -> impl Fn(&accesskit::Node) -> bool {
    move |n: &accesskit::Node| n.role() == accesskit::Role::Label && n.value() == Some(text)
}

/// Press the primary button at `at`, having moved there first.
fn press_at(input: &mut EguiInput, at: egui::Pos2, button: egui::PointerButton) {
    input.0.events.push(egui::Event::PointerMoved(at));
    input.0.events.push(egui::Event::PointerButton {
        pos: at,
        button,
        pressed: true,
        modifiers: egui::Modifiers::NONE,
    });
}

/// Let the button go at `at`.
fn release_at(input: &mut EguiInput, at: egui::Pos2, button: egui::PointerButton) {
    input.0.events.push(egui::Event::PointerButton {
        pos: at,
        button,
        pressed: false,
        modifiers: egui::Modifiers::NONE,
    });
}

/// Where `--drag` is: looking for the port and the row, carrying the wire
/// (from, to, frames done), or holding it for the picture.
#[derive(Resource, Default)]
enum Dragger {
    #[default]
    Finding,
    Carrying(egui::Pos2, egui::Pos2, u32),
    /// Holding the wire at this point, which the pointer is put back to on
    /// every frame after.
    Holding(egui::Pos2),
}

/// `--drag`: press on the Sawtooth's output port in the patch slot, carry
/// the wire to the Lowpass's `q` row over [`DRAG_FRAMES`] frames, and hold
/// it there without releasing. Both widgets are found in the last pass's
/// AccessKit tree, in canvas units, and taken to the screen by the canvas's
/// own transform — and the press waits until neither has moved for
/// [`STILL_FRAMES`] frames.
fn drag_a_wire(
    args: Res<Args>,
    editor: Res<Editor>,
    monitor: Res<AudioMonitor>,
    mut dragger: ResMut<Dragger>,
    mut settle: Local<Settle>,
    mut contexts: Query<(&mut EguiInput, &EguiOutput), With<PrimaryEguiContext>>,
) {
    if !args.drag {
        return;
    }
    let Ok((mut input, output)) = contexts.single_mut() else {
        return;
    };
    match *dragger {
        Dragger::Finding => {
            let Some(window) = editor.shown[0] else {
                return;
            };
            if !layout_is_final(&args, &editor, &monitor) {
                return;
            }
            let to_screen = editor.patch_state.canvas_to_screen();
            let port = in_the_patch_canvas(
                output,
                window,
                to_screen,
                "the Sawtooth's output port",
                &|n| {
                    n.label()
                        .is_some_and(|l| l.starts_with("Output of #0 Sawtooth"))
                },
            );
            let row_label = in_the_patch_canvas(
                output,
                window,
                to_screen,
                "the Lowpass's q row",
                &label_is("q:"),
            );
            let (Some(port), Some(row_label)) = (port, row_label) else {
                return;
            };
            // Well to the right of the dot, so the picture shows a drop
            // that the row took rather than one that snapped to the dot.
            let over_the_row = egui::pos2(
                row_label.right() + DRAG_PAST_LABEL * to_screen.scaling,
                row_label.center().y,
            );
            let Some(points) = settle.settled(vec![port.center(), over_the_row]) else {
                return;
            };
            let (from, to) = (points[0], points[1]);
            info!("--drag: pressing at {from:?} and carrying the wire to {to:?}");
            press_at(&mut input, from, egui::PointerButton::Primary);
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
                Dragger::Holding(to)
            } else {
                Dragger::Carrying(from, to, done)
            };
        }
        // The pointer is put back where it was left: bevy_egui feeds the
        // real one every frame, and a picture taken after it had wandered
        // off would show no drag at all.
        Dragger::Holding(at) => input.0.events.push(egui::Event::PointerMoved(at)),
    }
}

/// Where `--wire` is: looking for the wire, or parked on it.
#[derive(Resource, Default)]
enum Hoverer {
    #[default]
    Finding,
    /// `--pick`: parked on the wire, with the click still to come.
    Picking(egui::Pos2),
    Parked(egui::Pos2),
}

/// `--wire`: park the pointer on the wire that drives the Lowpass's
/// `cutoff_hz`, so the picture shows a hovered wire and its tooltip.
///
/// The wire's own geometry comes from the canvas rather than from the
/// AccessKit tree: a wire is painted, not laid out, so the tree has no
/// bounds for one until it is hovered, which is what this is for.
fn hover_a_wire(
    args: Res<Args>,
    editor: Res<Editor>,
    monitor: Res<AudioMonitor>,
    mut hoverer: ResMut<Hoverer>,
    mut settle: Local<Settle>,
    mut contexts: Query<&mut EguiInput, With<PrimaryEguiContext>>,
) {
    if !args.wire {
        return;
    }
    let Ok(mut input) = contexts.single_mut() else {
        return;
    };
    if let Hoverer::Parked(at) = *hoverer {
        input.0.events.push(egui::Event::PointerMoved(at));
        return;
    }
    if let Hoverer::Picking(at) = *hoverer {
        // The click goes in on the frame after the park, so the canvas has
        // already seen the pointer on the wire and knows which one it is.
        press_at(&mut input, at, egui::PointerButton::Primary);
        release_at(&mut input, at, egui::PointerButton::Primary);
        info!("--pick: clicking the wire at {at:?}");
        *hoverer = Hoverer::Parked(at);
        return;
    }
    if editor.shown[0].is_none() || !layout_is_final(&args, &editor, &monitor) {
        return;
    }
    let to_screen = editor.patch_state.canvas_to_screen();
    let Some(wire) = editor
        .patch_state
        .wires()
        .iter()
        .find(|w| w.port == "cutoff_hz")
    else {
        return;
    };
    let on_the_wire = to_screen * wire.midpoint();
    let Some(points) = settle.settled(vec![on_the_wire]) else {
        return;
    };
    info!(
        "--wire: parking on the wire from #{} to #{} · {} at {:?}",
        wire.from.0, wire.to.0, wire.port, points[0]
    );
    input.0.events.push(egui::Event::PointerMoved(points[0]));
    *hoverer = if args.pick {
        Hoverer::Picking(points[0])
    } else {
        Hoverer::Parked(points[0])
    };
}

/// Where `--menu` is: looking for what to click, or holding the menu open
/// with the pointer parked inside it.
#[derive(Resource, Default)]
enum Menuer {
    #[default]
    Finding,
    /// Clicked at this point on the last frame; the menu opens under it.
    Clicked(egui::Pos2),
    Open(egui::Pos2),
}

/// How far inside an opened menu `--menu` parks the pointer, in points:
/// far enough to be over the menu rather than the widget that opened it,
/// and not so far as to be over an item it might look chosen.
const MENU_PARK: egui::Vec2 = egui::vec2(12.0, 14.0);

/// `--confirm`: press a track's gutter cross so the question it now asks is
/// in the picture, and park the pointer clear of everything.
///
/// The cross is painted geometry, not a widget: its rect comes from
/// [`SequenceEditorState::removal_asks`], the same list the geometry test
/// measures the question against. Working it out again from the lane pitch
/// and the gutter width would be a second answer waiting to disagree
/// (crate #67).
fn ask_to_remove_a_track(
    args: Res<Args>,
    editor: Res<Editor>,
    monitor: Res<AudioMonitor>,
    mut asker: ResMut<Asker>,
    mut settle: Local<Settle>,
    mut contexts: Query<&mut EguiInput, With<PrimaryEguiContext>>,
) {
    if !args.confirm {
        return;
    }
    let Ok(mut input) = contexts.single_mut() else {
        return;
    };
    match *asker {
        Asker::Asked => return,
        Asker::Pressed => {
            // Off every control, so nothing in the picture is drawn hovered
            // and no hover text covers the question.
            if let Some(window) = editor.shown[1] {
                input
                    .0
                    .events
                    .push(egui::Event::PointerMoved(window.right_bottom()));
            }
            info!("--confirm: the question is up");
            *asker = Asker::Asked;
            return;
        }
        Asker::Finding => {}
    }
    if editor.shown[1].is_none() || !layout_is_final(&args, &editor, &monitor) {
        return;
    }
    // The first track's cross. The instrument rows are published first, so
    // the tracks start after them.
    let rows = editor.recipe.instruments.len();
    let Some(cross) = editor.sequence_state.removal_asks().get(rows).copied() else {
        return;
    };
    let Some(points) = settle.settled(vec![cross.center()]) else {
        return;
    };
    info!("--confirm: pressing track 1's cross at {:?}", points[0]);
    input.0.events.push(egui::Event::PointerMoved(points[0]));
    press_at(&mut input, points[0], egui::PointerButton::Primary);
    release_at(&mut input, points[0], egui::PointerButton::Primary);
    *asker = Asker::Pressed;
}

/// `--hover <label>`: park the pointer on the widget whose label starts
/// with that, so egui's tooltip delay runs out and the hover is in the
/// picture.
///
/// The reason a disabled control gives is only ever shown on hover (#1289),
/// so a picture of a refusal is a picture of a hover or it is a picture of
/// a grey button and nothing else.
fn park_on_a_widget(
    args: Res<Args>,
    editor: Res<Editor>,
    monitor: Res<AudioMonitor>,
    mut parker: ResMut<Parker>,
    mut settle: Local<Settle>,
    mut contexts: Query<(&mut EguiInput, &EguiOutput), With<PrimaryEguiContext>>,
) {
    let Some(label) = args.hover.as_deref() else {
        return;
    };
    let Ok((mut input, output)) = contexts.single_mut() else {
        return;
    };
    match *parker {
        // Nothing more is sent: the pointer is where it was put, and
        // saying so again would restart egui's movement clock and close
        // the tooltip this gesture exists to photograph.
        Parker::Parked(_) => return,
        Parker::Waiting(at, held) => {
            // Sent for the first few frames only, and then the pointer is
            // left alone. egui stamps its movement clock on *every*
            // `PointerMoved`, whether or not the position changed
            // (`InputState`), and holds a tooltip back until the pointer
            // has been still for `tooltip_delay` — so a position re-sent
            // every frame is a pointer that never rests and a hover that
            // never opens. The first run of this parked perfectly and
            // photographed no tooltip at all.
            if held < SETTLE_MOVES {
                input.0.events.push(egui::Event::PointerMoved(at));
            }
            *parker = if held >= HOVER_FRAMES {
                info!("--hover {label:?}: its hover has had time to open");
                Parker::Parked(at)
            } else {
                Parker::Waiting(at, held + 1)
            };
            return;
        }
        Parker::Finding => {}
    }
    if !layout_is_final(&args, &editor, &monitor) {
        return;
    }
    // Whatever role it wears: the marker beside an out-of-track slider is a
    // label, not a button.
    let found = accesskit_bounds(output, &|n| n.label().is_some_and(|l| l.starts_with(label)));
    let Some(rect) = found.first().copied() else {
        return;
    };
    let Some(points) = settle.settled(vec![rect.center()]) else {
        return;
    };
    info!("--hover {label:?}: parking on it at {:?}", points[0]);
    input.0.events.push(egui::Event::PointerMoved(points[0]));
    *parker = Parker::Waiting(points[0], 0);
}

/// `--menu`: open a menu and hold it open for the picture.
///
/// egui closes a menu on a click outside it, and bevy_egui feeds the real
/// pointer every frame, so the pointer is parked just inside the menu and
/// put back there on every frame after. That is what a fixed frame count
/// could not do: the click landed and the menu was gone long before
/// `--shot` fired (crate #67).
fn open_a_menu(
    args: Res<Args>,
    editor: Res<Editor>,
    monitor: Res<AudioMonitor>,
    mut menuer: ResMut<Menuer>,
    mut settle: Local<Settle>,
    mut contexts: Query<(&mut EguiInput, &EguiOutput), With<PrimaryEguiContext>>,
) {
    let Some(which) = args.menu else {
        return;
    };
    let Ok((mut input, output)) = contexts.single_mut() else {
        return;
    };
    match *menuer {
        Menuer::Finding => {
            let Some(window) = editor.shown[0] else {
                return;
            };
            if !layout_is_final(&args, &editor, &monitor) {
                return;
            }
            let (at, button) = match which {
                // The toolbar is drawn on the window's own layer, so its
                // bounds need no transform.
                Menu::Add => (
                    in_the_patch_window(output, window, "the Add node button", &|n| {
                        n.role() == accesskit::Role::Button && n.label() == Some(ADD_NODE)
                    }),
                    egui::PointerButton::Primary,
                ),
                Menu::Node => (
                    node_zero_grip(output, window, editor.patch_state.canvas_to_screen()),
                    egui::PointerButton::Secondary,
                ),
            };
            let Some(at) = at else {
                return;
            };
            let Some(points) = settle.settled(vec![at.center()]) else {
                return;
            };
            info!("--menu {which:?}: clicking at {:?}", points[0]);
            press_at(&mut input, points[0], button);
            release_at(&mut input, points[0], button);
            *menuer = Menuer::Clicked(points[0]);
        }
        // The menu opens on the frame after the click, so the pointer moves
        // into it only once there is something to be inside.
        Menuer::Clicked(at) => {
            let park = at + MENU_PARK;
            input.0.events.push(egui::Event::PointerMoved(park));
            info!("--menu {which:?}: holding it open with the pointer at {park:?}");
            *menuer = Menuer::Open(park);
        }
        Menuer::Open(park) => input.0.events.push(egui::Event::PointerMoved(park)),
    }
}

/// Which track `--notes picked` and `--notes box` work on.
///
/// The seeded recipe's `bass` track: four notes of seven and a half beats
/// each, so at the zoom the timeline opens at they are the only blocks wide
/// enough to see an outline on, let alone a label.
const NOTES_TRACK: usize = 4;
/// Which track `--notes track` opens its menu on: the `pluck` track, whose
/// notes are a quarter beat each and whose gaps are four, so there is a
/// clear stretch to right-click that no block is under.
const MENU_TRACK: usize = 2;

/// Where `--notes` is: looking for its notes, in the middle of its gesture,
/// or done and holding whatever it opened.
#[derive(Resource, Default)]
enum Noter {
    #[default]
    Finding,
    /// Clicked the first note; the shift-click at this point comes next.
    Clicked(egui::Pos2),
    /// Pressed at the first point; the drag to the second comes next.
    Pressed(egui::Pos2, egui::Pos2, u32),
    /// Clicked what opens a menu; the pointer moves into it next.
    Opened(egui::Pos2),
    /// In place, with the pointer parked where it has to stay.
    Done(egui::Pos2),
}

/// `--notes`: drive the sequence slot's timeline into the state a picture
/// wants — notes picked, a box held out over them, the snap picker open, or
/// a track's own menu open.
///
/// The notes come from [`SequenceEditorState::notes`], which is the
/// timeline's own record of what it painted. The AccessKit tree cannot
/// answer this: a note block is painted and interacted, not laid out, so
/// the tree has no bounds for one, and working the rects out again from the
/// zoom, the gutter and the scroll offset would be a second answer waiting
/// to disagree with the first (crate #67).
///
/// Like every gesture here it waits for [`layout_is_final`] and then for its
/// own points to hold still, because the audition starting grows the strip
/// above it and moves everything below by the height of a waveform.
fn work_the_timeline(
    args: Res<Args>,
    editor: Res<Editor>,
    monitor: Res<AudioMonitor>,
    mut noter: ResMut<Noter>,
    mut settle: Local<Settle>,
    mut contexts: Query<(&mut EguiInput, &EguiOutput), With<PrimaryEguiContext>>,
) {
    let Some(what) = args.notes else {
        return;
    };
    let Ok((mut input, output)) = contexts.single_mut() else {
        return;
    };
    match *noter {
        Noter::Done(at) => {
            input.0.events.push(egui::Event::PointerMoved(at));
            return;
        }
        // A shift-click adds the second note to the picked set. The
        // modifier goes on the frame's raw input as well as on the event:
        // egui reads a chord from the event and `modifiers.shift` from the
        // frame, and the editor needs the first.
        Noter::Clicked(second) => {
            input.0.modifiers = egui::Modifiers::SHIFT;
            input.0.events.push(egui::Event::PointerMoved(second));
            input.0.events.push(egui::Event::PointerButton {
                pos: second,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::SHIFT,
            });
            input.0.events.push(egui::Event::PointerButton {
                pos: second,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::SHIFT,
            });
            info!("--notes picked: shift-clicking the second note at {second:?}");
            *noter = Noter::Done(second);
            return;
        }
        // The box is held *down*: a marquee is gone the moment the button
        // comes up, so the press stays down for the picture and the
        // pointer is put back at the far corner every frame.
        Noter::Pressed(from, to, held) => {
            input.0.events.push(egui::Event::PointerMoved(to));
            if held == 0 {
                info!("--notes box: dragging from {from:?} to {to:?}");
            }
            *noter = if held >= DRAG_FRAMES {
                Noter::Done(to)
            } else {
                Noter::Pressed(from, to, held + 1)
            };
            return;
        }
        Noter::Opened(at) => {
            let park = at + MENU_PARK;
            input.0.events.push(egui::Event::PointerMoved(park));
            info!("--notes {what:?}: holding the menu open at {park:?}");
            *noter = Noter::Done(park);
            return;
        }
        Noter::Finding => {}
    }

    let Some(window) = editor.shown[1] else {
        return;
    };
    if !layout_is_final(&args, &editor, &monitor) {
        return;
    }

    match what {
        Notes::Picked | Notes::Box => {
            let state = &editor.sequence_state;
            let at = |event: usize| {
                state
                    .notes()
                    .iter()
                    .find(|n| n.track == NOTES_TRACK && n.event == event)
                    .map(|n| n.body)
            };
            // Well clear of a block's right edge, which is its resize grip.
            let inside = |body: egui::Rect| egui::pos2(body.left() + 2.0, body.center().y);
            let (Some(first), Some(last)) = (at(0), at(2)) else {
                return;
            };
            if !window.contains(first.center()) || !window.contains(last.center()) {
                warn!("--notes: the notes are not inside the sequence slot");
                return;
            }
            let (from, to) = if what == Notes::Box {
                // A corner outside the three blocks, so the box is drawn
                // round them rather than starting on one of them — a press
                // on a block is a move, not a marquee.
                (
                    egui::pos2(first.left() - 6.0, first.top() - 3.0),
                    egui::pos2(last.right() + 6.0, last.bottom() + 3.0),
                )
            } else {
                (inside(first), inside(at(1).unwrap_or(last)))
            };
            let Some(points) = settle.settled(vec![from, to]) else {
                return;
            };
            let (from, to) = (points[0], points[1]);
            if what == Notes::Box {
                press_at(&mut input, from, egui::PointerButton::Primary);
                *noter = Noter::Pressed(from, to, 0);
            } else {
                info!("--notes picked: clicking the first note at {from:?}");
                press_at(&mut input, from, egui::PointerButton::Primary);
                release_at(&mut input, from, egui::PointerButton::Primary);
                *noter = Noter::Clicked(to);
            }
        }
        Notes::Snap => {
            // By the grid it shows, not by being a combo box: the slot has
            // a Sample rate combo too, and picking "the leftmost combo in
            // the window" opened that one instead.
            let showing = editor.sequence_state.snap().label();
            // A combo box reports its selected text as its *value*, not as
            // its label — egui's `WidgetInfo::current_text_value` — so
            // asking for the label found nothing and the gesture timed out.
            let Some(rect) = in_the_patch_window(output, window, "the snap picker", &|n| {
                n.role() == accesskit::Role::ComboBox
                    && (n.value() == Some(showing) || n.label() == Some(showing))
            }) else {
                return;
            };
            let Some(points) = settle.settled(vec![rect.center()]) else {
                return;
            };
            info!(
                "--notes snap: opening the {showing} picker at {:?}",
                points[0]
            );
            press_at(&mut input, points[0], egui::PointerButton::Primary);
            release_at(&mut input, points[0], egui::PointerButton::Primary);
            *noter = Noter::Opened(points[0]);
        }
        Notes::Track => {
            // Halfway between the tail of one note and the start of the
            // next: clear ground on the track, found from the rects the
            // timeline published rather than from a beat and a gutter
            // width this example would have to know.
            let on = |event: usize| {
                editor
                    .sequence_state
                    .notes()
                    .iter()
                    .find(|n| n.track == MENU_TRACK && n.event == event)
                    .copied()
            };
            let (Some(first), Some(second)) = (on(0), on(1)) else {
                return;
            };
            let at = egui::pos2(
                0.5 * (first.extent.right() + second.body.left()),
                first.body.center().y,
            );
            if !window.contains(at) {
                warn!("--notes track: the gap between the first two notes is off the slot");
                return;
            }
            let Some(points) = settle.settled(vec![at]) else {
                return;
            };
            info!(
                "--notes track: right-clicking clear ground at {:?}",
                points[0]
            );
            press_at(&mut input, points[0], egui::PointerButton::Secondary);
            release_at(&mut input, points[0], egui::PointerButton::Secondary);
            *noter = Noter::Opened(points[0]);
        }
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
        // Playing AND far enough in for the cursor to be off the very
        // start: a playhead pinned to beat 0 in every picture is a picture
        // that proves nothing.
        Some(Status::PlayingSequence) => {
            monitor.status == MonitorStatus::Playing
                && monitor.position_secs().is_some_and(|at| at > 0.2)
        }
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
    gestures: Gestures,
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
        // And wait for the gesture, not for a count at all: a gesture waits
        // for a layout that has stopped moving, which may be later than
        // this (crate #67).
        if !args.gesture_in_place(&gestures) {
            shot.waited += 1;
            if shot.waited > GIVE_UP {
                error!("--shot: the scripted gesture never got into place");
                exit.write(AppExit::error());
            }
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
/// The drone holding values the record allows and the sliders' tracks do
/// not: an amplitude of 2.5 against a 0-1 track, a Q of 40 against 0.1-20.
///
/// What #1336 B11 is about. egui's default clamping rewrote both by drawing
/// them, with no `changed` flag, and the next commit wrote the loss into the
/// record; the editor now shows them as they are and marks the ones that
/// are off their track.
fn beyond_the_track_patch() -> AudioPatch {
    let mut patch = filtered_drone_patch();
    for node in &mut patch.graph.nodes {
        match &mut node.kind {
            NodeKind::Sawtooth(c) => c.amplitude = 2.5,
            NodeKind::BiquadLowpass(c) => c.q = 40.0,
            _ => {}
        }
    }
    patch
}

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
