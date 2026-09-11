//! Waveform preview and a bake-and-play audio *monitor*.
//!
//! Two independent pieces:
//!
//! - [`waveform`] — a pure-egui widget that draws a sample buffer as a
//!   min/max envelope.  No audio device, no Bevy; safe on wasm and reusable on
//!   its own.
//! - The **monitor** — the first Bevy-touching part of [`crate::ui`]: a
//!   [`MonitorRequest`] message, the [`AudioMonitor`] resource, the bake/poll
//!   systems, and [`AudioEditorPlugin`] that wires them up.  This is a 2-D,
//!   non-spatial *author* monitor for auditioning edits — deliberately
//!   distinct from a game's spatial-audio pipeline (e.g. Overlands attaches
//!   spatial `AudioPlayer`s to world entities; this just plays the buffer flat
//!   so you can hear what you're editing).
//!
//! # Why a background bake (not the crate's rayon pool)
//!
//! Baking runs on [`AsyncComputeTaskPool`], **not** the crate's private rayon
//! pool ([`crate::async_gen`]).  The rayon pool isn't wasm-friendly, whereas
//! `AsyncComputeTaskPool` runs cooperatively on wasm — the same choice the
//! Overlands spatial pipeline makes.  Invalid patches are surfaced as
//! [`MonitorStatus::Error`] rather than panicking, since the graph canvas can
//! easily produce a cyclic or dangling graph.
//!
//! # A replaced bake stops
//!
//! A patch bake runs through [`try_bake_cancellable`] with a flag the monitor
//! sets when the bake is replaced or stopped, so on native a superseded bake
//! gives its pool thread back within a few thousand samples instead of
//! baking to the end for nobody (#57). A sequence bake cannot be cancelled
//! upstream ([`bake_sequence`] has no cancellable form), so a superseded one
//! runs on; its task is dropped, and its result can never be played.
//!
//! On wasm the pool is the main thread, and a bake runs to its end in the one
//! poll that starts it: the flag cannot interrupt it there, and a bake
//! replaced before it starts never runs at all. A sequence bake is a
//! 0.1–0.35 s stall of the page there, which is why the audition strip's Auto
//! re-bake starts off for sequences on the web (see [`crate::ui::audition`]).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use bevy::audio::{AudioPlayer, AudioSource, PlaybackSettings};
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};
use bevy_egui::egui;
// `std::time::Instant::now` panics on wasm32-unknown-unknown.
use web_time::Instant;

use crate::bake::try_bake_cancellable;
use crate::{AudioPatch, SequenceRecipe, bake_sequence, samples_to_audio_source};

// ---------------------------------------------------------------------------
// Waveform widget (pure egui)
// ---------------------------------------------------------------------------

/// Draw `samples` as a waveform filling the available width at a default
/// height.  See [`waveform_sized`].
pub fn waveform(ui: &mut egui::Ui, samples: &[f32]) -> egui::Response {
    let size = egui::vec2(ui.available_width(), 72.0);
    waveform_sized(ui, samples, size)
}

/// Draw `samples` as a min/max envelope inside an allocated rect of `size`.
///
/// Each horizontal pixel column spans a slice of the buffer and is drawn as a
/// vertical line from that slice's minimum to its maximum sample — the
/// standard cheap audio overview that reads well at any zoom and needs no
/// audio device (wasm-safe).  Samples are clamped to `[-1, 1]` for display.
pub fn waveform_sized(ui: &mut egui::Ui, samples: &[f32], size: egui::Vec2) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, egui::Color32::from_gray(18));

    let mid = rect.center().y;
    painter.line_segment(
        [egui::pos2(rect.left(), mid), egui::pos2(rect.right(), mid)],
        egui::Stroke::new(1.0, egui::Color32::from_gray(70)),
    );

    if samples.is_empty() {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "no signal",
            egui::FontId::proportional(12.0),
            egui::Color32::from_gray(110),
        );
        return resp;
    }

    let half = rect.height() * 0.5 * 0.94;
    let cols = rect.width().max(1.0) as usize;
    let n = samples.len();
    let color = egui::Color32::from_rgb(120, 200, 140);
    for x in 0..cols {
        let start = x * n / cols;
        let end = ((x + 1) * n / cols).clamp(start + 1, n);
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &s in &samples[start..end] {
            lo = lo.min(s);
            hi = hi.max(s);
        }
        if !lo.is_finite() {
            continue;
        }
        let px = rect.left() + x as f32;
        let y_hi = mid - hi.clamp(-1.0, 1.0) * half;
        let y_lo = mid - lo.clamp(-1.0, 1.0) * half;
        painter.line_segment(
            [egui::pos2(px, y_hi), egui::pos2(px, y_lo)],
            egui::Stroke::new(1.0, color),
        );
    }
    resp
}

// ---------------------------------------------------------------------------
// Bake-and-play monitor (Bevy)
// ---------------------------------------------------------------------------

/// What a [`MonitorRequest`] bake produced: `(samples, sample_rate)` or an
/// error string (an invalid patch, surfaced rather than panicked).
type BakeResult = Result<(Vec<f32>, u32), String>;

/// Ask the [`AudioMonitor`] to (re)bake and play, or to stop.
///
/// Write one of these from your UI (the host owns the egui context); the
/// [`AudioEditorPlugin`] systems do the baking and playback. The audition
/// strip, [`crate::ui::audition_strip`], returns them for the host to write.
///
/// A new `Play*` replaces whatever is currently playing: a bake in flight is
/// cancelled at once, and the voice already playing keeps playing until the
/// new bake lands, so a re-bake after an edit does not cut the sound out.
#[derive(Message, Clone)]
pub enum MonitorRequest {
    /// Bake `patch` for `duration_secs` at `sample_rate`, then loop it.
    PlayPatch {
        /// The patch to bake.
        patch: AudioPatch,
        /// Samples per second of the bake.
        sample_rate: u32,
        /// Length of the bake, and so of the loop, in seconds.
        duration_secs: f32,
    },
    /// Bake `recipe` (at its own sample rate) and loop it.
    PlaySequence {
        /// The recipe to bake with [`crate::mixdown::bake_sequence`].
        recipe: SequenceRecipe,
    },
    /// Stop playback and cancel any in-flight bake.
    Stop,
}

/// Current state of the monitor, for the UI to display.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum MonitorStatus {
    /// Nothing playing.
    #[default]
    Idle,
    /// A bake is running on the task pool.
    Baking,
    /// Playing (looping) the last baked buffer.
    Playing,
    /// The last bake failed (e.g. the graph didn't topo-sort).
    Error(String),
}

/// Resource holding monitor state: the in-flight bake task, the playing voice
/// entity, the current [`MonitorStatus`], and the last baked buffer (so a
/// [`waveform`] can be drawn).  Added by [`AudioEditorPlugin`].
#[derive(Resource, Default)]
pub struct AudioMonitor {
    task: Option<Task<BakeResult>>,
    /// The currently-playing voice entity, despawned when replaced or stopped.
    current: Option<Entity>,
    /// The flag the patch bake in flight polls. Set, and dropped, when that
    /// bake is replaced or stopped; `None` for a sequence bake, which
    /// cannot be cancelled. Only `cancel_bake` sets it, and it drops the
    /// task in the same step, so a cancelled bake is never polled again and
    /// its partial buffer is never played.
    cancel: Option<Arc<AtomicBool>>,
    /// When the bake in flight started, for [`Self::bake_elapsed`].
    bake_started: Option<Instant>,
    /// Which request [`Self::status`] is about: the fingerprint of the one
    /// baking, playing or failed, or `None` after a stop. Several audition
    /// strips can share this monitor, and each shows only its own audition.
    auditioning: Option<u64>,
    /// The fingerprint of the request that baked [`Self::last_samples`].
    samples_of: Option<u64>,
    /// Status, for UI display.
    pub status: MonitorStatus,
    /// The most recent baked buffer — feed this to [`waveform`].
    pub last_samples: Vec<f32>,
    /// Sample rate of [`Self::last_samples`].
    pub sample_rate: u32,
}

impl AudioMonitor {
    /// `true` while a bake is in flight.
    pub fn is_baking(&self) -> bool {
        self.status == MonitorStatus::Baking
    }

    /// How long the bake in flight has been running, or `None` when nothing
    /// is baking.
    pub fn bake_elapsed(&self) -> Option<Duration> {
        self.bake_started.map(|started| started.elapsed())
    }

    /// The fingerprint of the request [`Self::status`] is about, or `None`
    /// when it is about nothing in particular (after a stop, or when a host
    /// set the public fields itself).
    pub(crate) fn auditioning(&self) -> Option<u64> {
        self.auditioning
    }

    /// The fingerprint of the request that baked [`Self::last_samples`].
    pub(crate) fn samples_of(&self) -> Option<u64> {
        self.samples_of
    }

    /// Cancel the bake in flight, if any: set its flag and drop its task.
    fn cancel_bake(&mut self) {
        if let Some(flag) = self.cancel.take() {
            flag.store(true, Ordering::Relaxed);
        }
        // Dropping a task cancels it unless it is already running, and a
        // running sequence bake can only finish into a channel nobody reads.
        self.task = None;
        self.bake_started = None;
    }

    /// Start showing a bake of `request` in flight on `task`.
    fn begin(&mut self, request: &MonitorRequest, task: Task<BakeResult>) {
        self.task = Some(task);
        self.bake_started = Some(Instant::now());
        self.auditioning = fingerprint(request);
        self.status = MonitorStatus::Baking;
    }

    /// Put the monitor in `status` for `request` without baking anything, as
    /// if `request` had been handled and had got that far.
    #[cfg(test)]
    pub(crate) fn stage(&mut self, request: &MonitorRequest, status: MonitorStatus) {
        self.auditioning = fingerprint(request);
        match status {
            MonitorStatus::Baking => self.bake_started = Some(Instant::now()),
            MonitorStatus::Playing => {
                self.last_samples = (0..2048).map(|i| (i as f32 * 0.05).sin()).collect();
                self.sample_rate = 22_050;
                self.samples_of = self.auditioning;
            }
            MonitorStatus::Idle | MonitorStatus::Error(_) => {}
        }
        self.status = status;
    }

    /// Move the start of the bake in flight `by` into the past.
    #[cfg(test)]
    pub(crate) fn backdate_bake(&mut self, by: Duration) {
        if let Some(started) = &mut self.bake_started {
            *started -= by;
        }
    }
}

/// A fingerprint of what `request` plays: equal for equal requests, and
/// different, in practice, for different ones. `None` for
/// [`MonitorRequest::Stop`], and for a request that will not serialise (a
/// NaN parameter), which then belongs to no strip in particular.
pub(crate) fn fingerprint(request: &MonitorRequest) -> Option<u64> {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::hash::DefaultHasher::new();
    match request {
        MonitorRequest::PlayPatch {
            patch,
            sample_rate,
            duration_secs,
        } => {
            0u8.hash(&mut hasher);
            serde_json::to_vec(patch).ok()?.hash(&mut hasher);
            sample_rate.hash(&mut hasher);
            duration_secs.to_bits().hash(&mut hasher);
        }
        MonitorRequest::PlaySequence { recipe } => {
            1u8.hash(&mut hasher);
            serde_json::to_vec(recipe).ok()?.hash(&mut hasher);
        }
        MonitorRequest::Stop => return None,
    }
    Some(hasher.finish())
}

/// Registers the monitor: the [`MonitorRequest`] message, the [`AudioMonitor`]
/// resource, and the bake/poll systems.  Add alongside `bevy_egui`'s plugin.
///
/// Playback uses Bevy's `AudioPlayer`, so the app needs Bevy's audio plugin
/// (present in `DefaultPlugins`).
pub struct AudioEditorPlugin;

impl Plugin for AudioEditorPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<MonitorRequest>()
            .init_resource::<AudioMonitor>()
            .add_systems(
                Update,
                (handle_monitor_requests, poll_monitor_bakes).chain(),
            );
    }
}

/// Stop playback and cancel any in-flight bake.
fn stop_monitor(monitor: &mut AudioMonitor, commands: &mut Commands) {
    if let Some(entity) = monitor.current.take() {
        commands.entity(entity).despawn();
    }
    monitor.cancel_bake();
    monitor.auditioning = None;
    monitor.status = MonitorStatus::Idle;
}

/// Read [`MonitorRequest`]s and dispatch background bakes (or stop).
///
/// A play request cancels the bake in flight but leaves the voice playing:
/// it is replaced when the new bake lands (see [`MonitorRequest`]).
fn handle_monitor_requests(
    mut requests: MessageReader<MonitorRequest>,
    mut monitor: ResMut<AudioMonitor>,
    mut commands: Commands,
) {
    for req in requests.read() {
        match req {
            MonitorRequest::Stop => stop_monitor(&mut monitor, &mut commands),
            MonitorRequest::PlayPatch {
                patch,
                sample_rate,
                duration_secs,
            } => {
                monitor.cancel_bake();
                let cancel = Arc::new(AtomicBool::new(false));
                let flag = Arc::clone(&cancel);
                let patch = patch.clone();
                let (sr, dur) = (*sample_rate, *duration_secs);
                let task = AsyncComputeTaskPool::get().spawn(async move {
                    try_bake_cancellable(&patch, sr, dur, &flag)
                        .map(|samples| (samples, sr))
                        .map_err(|e| e.to_string())
                });
                monitor.begin(req, task);
                monitor.cancel = Some(cancel);
            }
            MonitorRequest::PlaySequence { recipe } => {
                monitor.cancel_bake();
                let recipe = recipe.clone();
                let task = AsyncComputeTaskPool::get().spawn(async move {
                    let sr = recipe.sample_rate;
                    Ok((bake_sequence(&recipe), sr))
                });
                monitor.begin(req, task);
            }
        }
    }
}

/// Poll the in-flight bake; when it finishes, play the buffer (looping,
/// non-spatial) in place of the voice before it, and stash it for the
/// waveform. A failed bake silences the voice before it too: an error chip
/// over the last version's sound would say one thing and play another.
fn poll_monitor_bakes(
    mut monitor: ResMut<AudioMonitor>,
    mut commands: Commands,
    mut sources: ResMut<Assets<AudioSource>>,
) {
    let Some(mut task) = monitor.task.take() else {
        return;
    };
    let Some(result) = block_on(poll_once(&mut task)) else {
        // Still baking — put the task back for next frame.
        monitor.task = Some(task);
        return;
    };
    monitor.cancel = None;
    monitor.bake_started = None;
    if let Some(prev) = monitor.current.take() {
        commands.entity(prev).despawn();
    }
    match result {
        Ok((samples, sample_rate)) => {
            let handle = sources.add(samples_to_audio_source(&samples, sample_rate));
            let entity = commands
                .spawn((AudioPlayer::new(handle), PlaybackSettings::LOOP))
                .id();
            monitor.current = Some(entity);
            monitor.sample_rate = sample_rate;
            monitor.last_samples = samples;
            monitor.samples_of = monitor.auditioning;
            monitor.status = MonitorStatus::Playing;
        }
        Err(e) => {
            monitor.status = MonitorStatus::Error(e);
            monitor.last_samples.clear();
            monitor.samples_of = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AudioEditorPlugin, AudioMonitor, MonitorStatus, waveform};
    use bevy::prelude::*;
    use bevy_egui::egui;

    #[test]
    fn waveform_renders_headless_for_empty_and_full_buffers() {
        let ctx = egui::Context::default();
        let sine: Vec<f32> = (0..2000).map(|i| (i as f32 * 0.05).sin()).collect();
        for buf in [Vec::new(), sine] {
            let _ = ctx.run_ui(egui::RawInput::default(), |root| {
                egui::CentralPanel::default().show(root, |ui| {
                    waveform(ui, &buf);
                });
            });
        }
    }

    #[test]
    fn plugin_registers_resource_and_starts_idle() {
        let mut app = App::new();
        app.add_plugins(AudioEditorPlugin);
        let monitor = app
            .world()
            .get_resource::<AudioMonitor>()
            .expect("AudioEditorPlugin must insert AudioMonitor");
        assert_eq!(monitor.status, MonitorStatus::Idle);
        assert!(!monitor.is_baking());
        assert!(monitor.last_samples.is_empty());
    }

    // -----------------------------------------------------------------------
    // Superseded bakes (#57, Overlands #1330 D2)
    // -----------------------------------------------------------------------

    use super::MonitorRequest;
    use crate::{AudioPatch, GraphNode, NodeGraph, NodeId, NodeKind, SineOsc};
    use std::collections::BTreeMap;
    use std::sync::atomic::Ordering;

    /// The monitor on a bare app: the task pools, the plugin and the audio
    /// assets its poll system writes into. No audio device.
    fn monitor_app() -> App {
        let mut app = App::new();
        app.add_plugins((bevy::app::TaskPoolPlugin::default(), AudioEditorPlugin));
        app.init_resource::<Assets<bevy::audio::AudioSource>>();
        app
    }

    fn sine() -> AudioPatch {
        AudioPatch {
            seed: 0,
            graph: NodeGraph {
                nodes: vec![GraphNode {
                    id: NodeId(0),
                    kind: NodeKind::Sine(SineOsc::default()),
                    inputs: BTreeMap::new(),
                }],
                output: NodeId(0),
            },
        }
    }

    /// A bake long enough to still be running many frames after it starts:
    /// ten minutes at 48 kHz, which only a cancel ends early.
    fn endless() -> MonitorRequest {
        MonitorRequest::PlayPatch {
            patch: sine(),
            sample_rate: 48_000,
            duration_secs: 600.0,
        }
    }

    /// Update until the monitor is no longer baking, or give up.
    fn until_settled(app: &mut App) {
        for _ in 0..2000 {
            app.update();
            if !app.world().resource::<AudioMonitor>().is_baking() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        panic!("the bake never finished");
    }

    fn voices(app: &mut App) -> usize {
        app.world_mut()
            .query::<&bevy::audio::AudioPlayer>()
            .iter(app.world())
            .count()
    }

    /// A second play request while a bake runs: the first bake's cancel flag
    /// is set, so its worker stops within a few thousand samples instead of
    /// baking ten minutes of audio for nobody, and its result is never
    /// played. Only the second is heard, once.
    #[test]
    fn a_second_play_request_while_baking_cancels_the_first_and_never_plays_it() {
        let mut app = monitor_app();
        app.world_mut().write_message(endless());
        app.update();
        let first = {
            let monitor = app.world().resource::<AudioMonitor>();
            assert_eq!(monitor.status, MonitorStatus::Baking);
            assert!(
                monitor.bake_elapsed().is_some(),
                "a bake in flight is timed"
            );
            monitor
                .cancel
                .clone()
                .expect("a patch bake carries a cancel flag")
        };
        assert!(!first.load(Ordering::Relaxed));

        app.world_mut().write_message(MonitorRequest::PlayPatch {
            patch: sine(),
            sample_rate: 22_050,
            duration_secs: 0.5,
        });
        app.update();
        assert!(
            first.load(Ordering::Relaxed),
            "replacing a bake flips the cancel flag of the one it replaces"
        );
        until_settled(&mut app);

        let monitor = app.world().resource::<AudioMonitor>();
        assert_eq!(monitor.status, MonitorStatus::Playing);
        assert_eq!(
            monitor.last_samples.len(),
            11_025,
            "what plays is the second request's half second at 22.05 kHz"
        );
        assert!(monitor.bake_elapsed().is_none(), "nothing is baking now");
        assert_eq!(
            voices(&mut app),
            1,
            "one voice: the first result never played"
        );
    }

    /// Stop cancels the bake in flight the same way, and plays nothing.
    #[test]
    fn stop_while_baking_cancels_the_bake_and_plays_nothing() {
        let mut app = monitor_app();
        app.world_mut().write_message(endless());
        app.update();
        let flag = app
            .world()
            .resource::<AudioMonitor>()
            .cancel
            .clone()
            .expect("a patch bake carries a cancel flag");
        app.world_mut().write_message(MonitorRequest::Stop);
        app.update();
        assert!(flag.load(Ordering::Relaxed), "Stop flips the cancel flag");
        for _ in 0..20 {
            app.update();
        }
        let monitor = app.world().resource::<AudioMonitor>();
        assert_eq!(monitor.status, MonitorStatus::Idle);
        assert!(monitor.bake_elapsed().is_none());
        assert_eq!(voices(&mut app), 0);
    }

    /// A sequence bake cannot be cancelled upstream, so a request that
    /// replaces it must still keep its result from ever playing.
    #[test]
    fn a_superseded_sequence_bake_is_never_played() {
        let mut app = monitor_app();
        let long_sequence = crate::SequenceRecipe {
            duration_beats: 400.0,
            ..crate::SequenceRecipe::default()
        };
        app.world_mut().write_message(MonitorRequest::PlaySequence {
            recipe: long_sequence,
        });
        app.update();
        app.world_mut().write_message(MonitorRequest::PlayPatch {
            patch: sine(),
            sample_rate: 22_050,
            duration_secs: 0.25,
        });
        until_settled(&mut app);
        // Give a still-running sequence bake time to land, if it could.
        for _ in 0..50 {
            app.update();
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let monitor = app.world().resource::<AudioMonitor>();
        assert_eq!(monitor.last_samples.len(), 5_513, "0.25 s at 22.05 kHz");
        assert_eq!(voices(&mut app), 1);
    }
}
