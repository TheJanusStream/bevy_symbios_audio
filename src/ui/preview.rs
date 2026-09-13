//! Waveform preview and the audio *monitor* an audition plays through.
//!
//! Two independent pieces:
//!
//! - [`waveform`] — a pure-egui widget that draws a sample buffer as a
//!   min/max envelope.  No audio device, no Bevy; safe on wasm and reusable on
//!   its own. [`waveform_with_cursor`] is the same picture with a playhead
//!   on it and a seconds axis along the bottom, and it reports where a
//!   click landed ([`WaveformResponse`]).
//! - The **monitor** — the first Bevy-touching part of [`crate::ui`]: a
//!   [`MonitorRequest`] message, the [`AudioMonitor`] resource, the bake/poll
//!   systems, and [`AudioEditorPlugin`] that wires them up.  This is a 2-D,
//!   non-spatial *author* monitor for auditioning edits — deliberately
//!   distinct from a game's spatial-audio pipeline (e.g. Overlands attaches
//!   spatial `AudioPlayer`s to world entities; this just plays the buffer flat
//!   so you can hear what you're editing).
//!
//!   Its voice is a [`LoopedSamples`]: it plays from the bake's loop start —
//!   a sequence's `loop_start_beats`, a patch's first sample — to the end and
//!   back to the loop start, which is the loop a world plays the same bake
//!   as when its player starts at [`sequence_loop_start`] (see
//!   [`crate::looping`]), and it can be moved while it plays.
//!
//!   What it publishes back, once a voice is playing:
//!   [`AudioMonitor::position_secs`] (where that voice is in its buffer),
//!   [`AudioMonitor::loop_secs`] (how long the buffer is) and
//!   [`AudioMonitor::volume`]. What it takes besides a `MonitorRequest` is a
//!   [`MonitorControl`] — a seek and a level — which is a separate message
//!   type because a request bakes and a control does not.
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

use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use bevy::audio::{
    AddAudioSource, AudioPlayer, AudioPlugin, AudioSink, AudioSinkPlayback, PlaybackSettings,
};
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};
use bevy_egui::egui;
// `std::time::Instant::now` panics on wasm32-unknown-unknown.
use web_time::Instant;

use crate::bake::try_bake_cancellable;
use crate::looping::{LoopPlayhead, LoopedSamples, sequence_loop_start};
use crate::{AudioPatch, SequenceRecipe, bake_sequence};

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
///
/// Its ground, zero line and trace are the [`EditorStyle`](super::EditorStyle)'s
/// `waveform_*` roles, and an empty buffer's "no signal" is its
/// `ground_text`. An empty buffer draws no zero line: there is nothing for
/// it to be the zero of, and it struck through the words.
pub fn waveform_sized(ui: &mut egui::Ui, samples: &[f32], size: egui::Vec2) -> egui::Response {
    let style = super::style::editor_style(ui);
    let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, style.waveform_ground);

    if samples.is_empty() {
        // No zero line: there is no signal for it to be the zero of, and it
        // ran straight through the words saying so (#59, Overlands #1332
        // B16).
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "no signal",
            egui::FontId::proportional(12.0),
            style.ground_text,
        );
        return resp;
    }

    let mid = rect.center().y;
    painter.line_segment(
        [egui::pos2(rect.left(), mid), egui::pos2(rect.right(), mid)],
        egui::Stroke::new(1.0, style.waveform_zero),
    );

    let half = rect.height() * 0.5 * 0.94;
    let cols = rect.width().max(1.0) as usize;
    let n = samples.len();
    let color = style.waveform_trace;
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

/// What a [`waveform_with_cursor`] was asked for: its response, and where a
/// click on it asked to move the playhead.
#[derive(Clone, Debug)]
pub struct WaveformResponse {
    /// The widget's own response, as [`waveform_sized`] returns.
    pub response: egui::Response,
    /// Where a click landed, in seconds from the buffer's start, or `None`
    /// on a frame with no click. Already inside `0.0..secs`.
    pub seek: Option<f32>,
}

/// Draw `samples` as [`waveform_sized`] does, with a playhead at
/// `position` and a seconds axis along the bottom, and report where a click
/// asked to seek to.
///
/// `secs` is what the whole buffer is worth in seconds —
/// [`AudioMonitor::loop_secs`], not the visible width — and `position` is a
/// place inside it (see [`AudioMonitor::position_secs`]). A `position`
/// outside `0.0..secs` draws no cursor rather than one clamped to an end it
/// is not at: a cursor parked on the last pixel says "playing the final
/// sample", which would be a lie told every frame after a stop.
///
/// [`WaveformResponse::seek`] reports where a click landed. While a cursor
/// is drawn, the pointer over the waveform is a pointing hand: a cursor is
/// drawn for a voice that is playing, and the audition monitor moves a
/// playing voice to where it is asked ([`MonitorControl::Seek`]). With no
/// cursor there is no hand, because there is nothing playing to move, and
/// [`audition_strip`](super::audition_strip) asks for no seek then.
///
/// The axis thins its own labels out so they never crowd, so a 0.2 s patch
/// audition and a 34 s bed are both readable at the same width.
pub fn waveform_with_cursor(
    ui: &mut egui::Ui,
    samples: &[f32],
    position: Option<f32>,
    secs: f32,
    size: egui::Vec2,
) -> WaveformResponse {
    let style = super::style::editor_style(ui);
    let response = waveform_sized(ui, samples, size);
    let rect = response.rect;
    if samples.is_empty() || !secs.is_finite() || secs <= 0.0 {
        return WaveformResponse {
            response,
            seek: None,
        };
    }
    let painter = ui.painter_at(rect);
    let x_of = |t: f32| rect.left() + (t / secs).clamp(0.0, 1.0) * rect.width();

    paint_seconds_axis(&painter, rect, secs, &style, ui);

    let cursor = position.filter(|t| (0.0..secs).contains(t));
    if let Some(t) = cursor {
        let x = x_of(t);
        painter.line_segment(
            [
                egui::pos2(x, rect.top() + 1.0),
                egui::pos2(x, rect.bottom() - 1.0),
            ],
            egui::Stroke::new(1.5, style.playhead),
        );
    }

    // A click reports where it landed. The hand only while a cursor is
    // drawn: that is a voice playing, which a click can move. Over the last
    // bake with nothing playing, a hand would promise a move with nothing
    // to make.
    let mut response = response.interact(egui::Sense::click());
    if cursor.is_some() {
        response = response.on_hover_cursor(egui::CursorIcon::PointingHand);
    }
    let seek = response
        .clicked()
        .then(|| response.interact_pointer_pos())
        .flatten()
        .map(|p| ((p.x - rect.left()) / rect.width().max(1.0) * secs).clamp(0.0, secs));
    WaveformResponse { response, seek }
}

/// The narrowest a seconds-axis label may sit from its neighbour before the
/// axis draws fewer of them, in points.
const AXIS_LABEL_GAP: f32 = 56.0;

/// Tick marks and second labels along the bottom of a waveform.
///
/// The step is chosen so labels never come closer than [`AXIS_LABEL_GAP`]:
/// a 0.2 s patch audition and a 34 s bed are drawn by the same code at the
/// same width, and a fixed "one tick a second" would put 34 overlapping
/// labels on the second of them.
fn paint_seconds_axis(
    painter: &egui::Painter,
    rect: egui::Rect,
    secs: f32,
    style: &super::EditorStyle,
    ui: &egui::Ui,
) {
    let most = (rect.width() / AXIS_LABEL_GAP).floor().max(1.0);
    let rough = secs / most;
    // A step from the 1-2-5 ladder at or above `rough`, so the labels are
    // numbers a reader can hold (0.5, 1, 2, 5, 10…) rather than 0.37.
    let magnitude = 10.0_f32.powf(rough.max(f32::MIN_POSITIVE).log10().floor());
    let step = [1.0, 2.0, 5.0, 10.0]
        .into_iter()
        .map(|m| m * magnitude)
        .find(|s| *s >= rough)
        .unwrap_or(magnitude * 10.0);
    let decimals = if step < 1.0 { 1 } else { 0 };
    let font = egui::FontId::proportional(9.0);
    let mut t = step;
    while t < secs {
        let x = rect.left() + (t / secs) * rect.width();
        painter.line_segment(
            [
                egui::pos2(x, rect.bottom() - 5.0),
                egui::pos2(x, rect.bottom() - 1.0),
            ],
            egui::Stroke::new(1.0, style.waveform_zero),
        );
        let label =
            painter.layout_no_wrap(format!("{t:.decimals$} s"), font.clone(), style.ground_text);
        // The loop runs while `t < secs`, so the last tick can land within
        // a label's own width of the right edge: at 120 points and 2.05 s
        // the step is 2 s and "2 s" hung 10 points off the waveform. A
        // label that would not fit is not drawn — its tick still is, and
        // the end of the buffer is the rect's own edge (Overlands #1339).
        if x + 2.0 + label.size().x <= rect.right() - 1.0 {
            painter.galley(
                egui::pos2(x + 2.0, rect.bottom() - 1.0 - label.size().y),
                label,
                style.ground_text,
            );
        }
        t += step;
    }
    let _ = ui;
}

// ---------------------------------------------------------------------------
// Bake-and-play monitor (Bevy)
// ---------------------------------------------------------------------------

/// What a [`MonitorRequest`] bake produced, or an error string (an invalid
/// patch, surfaced rather than panicked).
type BakeResult = Result<Baked, String>;

/// A finished bake: its samples, their rate, and where its loop starts.
struct Baked {
    samples: Vec<f32>,
    sample_rate: u32,
    /// Where the voice starts and loops back to, from the first sample: a
    /// sequence's loop start as [`sequence_loop_start`] finds it in the
    /// recipe that was baked, and a patch's first sample.
    loop_start: Duration,
}

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
    /// Bake `recipe` (at its own sample rate) and loop it from its loop
    /// start, where [`sequence_loop_start`] says the bake's loop begins.
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

/// Ask the monitor to do something to the voice that is already playing,
/// rather than to bake a different one.
///
/// A separate message from [`MonitorRequest`] on purpose. `MonitorRequest`
/// is not `#[non_exhaustive]`, so growing *it* would break every exhaustive
/// match a host has written — the breaking change the epic collects for a
/// single 0.5 rather than dribbling out (Overlands #1326). This type is new,
/// so adding it breaks nothing, and it is `#[non_exhaustive]` from birth so
/// the next thing that wants to steer a playing voice needs no second type.
///
/// Write these the way you write a [`MonitorRequest`]; [`AudioEditorPlugin`]
/// registers both. The audition strip returns them through
/// [`AuditionState::take_controls`](super::AuditionState::take_controls).
#[derive(Message, Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum MonitorControl {
    /// Move the playing voice to `secs` into its buffer, and the published
    /// cursor with it.
    ///
    /// Clamped into the buffer, so a click past the end of a waveform asks
    /// for the end rather than for an error. Ignored when nothing plays. A
    /// seek into a sequence's run-up, before its loop start, plays the
    /// run-up from there once, and the voice loops from the loop start after
    /// that.
    ///
    /// [`AudioMonitor::position_secs`] moves on the frame the seek is
    /// written, and the sound at the voice's next sample: the monitor moves
    /// its voice through the voice's own [`LoopPlayhead`], which the source
    /// takes between two samples.
    ///
    /// # Not through the sink
    ///
    /// `AudioSinkPlayback::try_seek` is not used, for two reasons. Until
    /// 0.4.11 the monitor looped an `AudioSource` under
    /// `PlaybackMode::Loop`, which appends `decoder.repeat_infinite()`
    /// (bevy_audio 0.19 `src/audio_output.rs:165`), and `repeat_infinite`
    /// wraps its source in rodio's `Buffered`, whose `try_seek` is an
    /// unconditional `NotSupported` (rodio 0.22 `src/source/buffered.rs:240`):
    /// every seek was refused. And rodio's `Player::try_seek` waits on a
    /// `std::sync::mpsc` channel for the audio callback to answer. On wasm32
    /// that wait never ends — std's thread parker there does nothing, so
    /// `Receiver::recv` spins (measured in node: no return, a whole core,
    /// killed after eight seconds), and the web's audio callback runs on the
    /// very thread that is spinning. 0.4.10 made that call on every click on
    /// a playing waveform (Overlands #1341).
    Seek(f32),
    /// Set the monitor's own output level, `0.0..=1.0`.
    ///
    /// The monitor's, not the app's: it scales this author monitor's voice
    /// and nothing else the host plays. It survives a re-bake, because it
    /// is a property of the monitor rather than of the buffer in it.
    Volume(f32),
}

/// Resource holding monitor state: the in-flight bake task, the playing voice
/// entity and its playhead, the current [`MonitorStatus`], and the last baked
/// buffer (so a [`waveform`] can be drawn).  Added by [`AudioEditorPlugin`].
#[derive(Resource)]
pub struct AudioMonitor {
    task: Option<Task<BakeResult>>,
    /// The currently-playing voice entity, despawned when replaced or stopped.
    current: Option<Entity>,
    /// The playhead of [`Self::current`]'s voice, which the voice's own
    /// source keeps: where it is, and how a seek moves it. `Some` exactly
    /// while there is a voice.
    playhead: Option<LoopPlayhead>,
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
    /// Where the playing voice is in its buffer, in seconds from its first
    /// sample. `None` when nothing is playing.
    ///
    /// Private and assign-on-change: see [`publish_monitor_position`].
    position: Option<f32>,
    /// The monitor's own output level, from [`MonitorControl::Volume`].
    volume: f32,
}

impl Default for AudioMonitor {
    fn default() -> Self {
        Self {
            task: None,
            current: None,
            playhead: None,
            cancel: None,
            bake_started: None,
            auditioning: None,
            samples_of: None,
            status: MonitorStatus::default(),
            last_samples: Vec::new(),
            sample_rate: 0,
            position: None,
            // Full: a monitor nobody has turned down is not turned down.
            volume: 1.0,
        }
    }
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

    /// How long [`Self::last_samples`] is, in seconds: the length a waveform
    /// or a timeline maps the playing voice's position onto, from the bake's
    /// first sample. The voice loops inside it, from its loop start to this.
    /// `None` when there is no buffer, or when its sample rate is zero
    /// (nothing has baked yet).
    pub fn loop_secs(&self) -> Option<f32> {
        if self.last_samples.is_empty() || self.sample_rate == 0 {
            return None;
        }
        Some(self.last_samples.len() as f32 / self.sample_rate as f32)
    }

    /// Where the playing voice is in its buffer, in seconds from the bake's
    /// first sample, or `None` when nothing is playing.
    ///
    /// From the loop start up to [`Self::loop_secs`] while the voice loops:
    /// a sequence's run-up, before its loop start, is never played, and a
    /// patch loops from its first sample. A [`MonitorControl::Seek`] can put
    /// it anywhere, the run-up included, which then plays once.
    ///
    /// Read from the voice's own source rather than from its sink. A sink's
    /// position starts at zero wherever its voice starts, counts up for ever
    /// across the passes of a loop, and arrives a frame after the voice; the
    /// source knows the sample it plays next from the frame the voice is
    /// spawned, and moves on the frame a seek is asked for.
    pub fn position_secs(&self) -> Option<f32> {
        self.position
    }

    /// The monitor's own output level, `0.0..=1.0`. Set with
    /// [`MonitorControl::Volume`].
    pub fn volume(&self) -> f32 {
        self.volume
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
    /// if `request` had been handled and had got that far — for `Playing`, a
    /// voice at its first sample, as a patch's is from the frame it spawns.
    #[cfg(test)]
    pub(crate) fn stage(&mut self, request: &MonitorRequest, status: MonitorStatus) {
        self.auditioning = fingerprint(request);
        match status {
            MonitorStatus::Baking => self.bake_started = Some(Instant::now()),
            MonitorStatus::Playing => {
                self.last_samples = (0..2048).map(|i| (i as f32 * 0.05).sin()).collect();
                self.sample_rate = 22_050;
                self.samples_of = self.auditioning;
                self.position = Some(0.0);
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

/// Registers the monitor: the [`MonitorRequest`] and [`MonitorControl`]
/// messages, the [`AudioMonitor`] resource, the bake/poll systems, and the
/// [`LoopedSamples`] audio source its voice plays.  Add alongside
/// `bevy_egui`'s plugin.
///
/// Playback uses Bevy's `AudioPlayer`, so the app needs Bevy's audio plugin
/// (present in `DefaultPlugins`). The voice's source is registered once the
/// app has finished adding plugins, and only when that plugin is among them,
/// so the order the plugins are added in does not matter. An app without it
/// has nowhere to make the voice: its auditions still bake and show, and play
/// nothing.
pub struct AudioEditorPlugin;

impl Plugin for AudioEditorPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<MonitorRequest>()
            .add_message::<MonitorControl>()
            .init_resource::<AudioMonitor>()
            .add_systems(
                Update,
                (
                    handle_monitor_requests,
                    poll_monitor_bakes,
                    apply_monitor_controls,
                    publish_monitor_position,
                )
                    .chain(),
            );
    }

    fn finish(&self, app: &mut App) {
        // Here rather than in `build`: `add_audio_source` needs Bevy's asset
        // and audio plugins, which a host may add after this one. Skipped
        // when the source is already registered, since registering it again
        // would add its playback systems a second time.
        if app.is_plugin_added::<AudioPlugin>()
            && !app.world().contains_resource::<Assets<LoopedSamples>>()
        {
            app.add_audio_source::<LoopedSamples>();
        }
    }
}

/// Stop playback and cancel any in-flight bake.
fn stop_monitor(monitor: &mut AudioMonitor, commands: &mut Commands) {
    if let Some(entity) = monitor.current.take() {
        commands.entity(entity).despawn();
    }
    monitor.playhead = None;
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
                        .map(|samples| Baked {
                            samples,
                            sample_rate: sr,
                            loop_start: Duration::ZERO,
                        })
                        .map_err(|e| e.to_string())
                });
                monitor.begin(req, task);
                monitor.cancel = Some(cancel);
            }
            MonitorRequest::PlaySequence { recipe } => {
                monitor.cancel_bake();
                let recipe = recipe.clone();
                let task = AsyncComputeTaskPool::get().spawn(async move {
                    Ok(Baked {
                        samples: bake_sequence(&recipe),
                        sample_rate: recipe.sample_rate,
                        // From the recipe exactly as it was baked, so the
                        // loop start is the sample the mixdown folded its
                        // tail into.
                        loop_start: sequence_loop_start(&recipe).unwrap_or_default(),
                    })
                });
                monitor.begin(req, task);
            }
        }
    }
}

/// Poll the in-flight bake; when it finishes, play the buffer (looping from
/// its loop start, non-spatial) in place of the voice before it, and stash
/// it for the waveform. A failed bake silences the voice before it too: an
/// error chip over the last version's sound would say one thing and play
/// another.
fn poll_monitor_bakes(
    mut monitor: ResMut<AudioMonitor>,
    mut commands: Commands,
    // Optional: the voice's source is registered only alongside Bevy's audio
    // plugin (see `AudioEditorPlugin`), and an app without one has nowhere to
    // make the voice. Its bake still lands and shows.
    sources: Option<ResMut<Assets<LoopedSamples>>>,
) {
    // `is_some` through the immutable deref first: `take()` is a
    // `deref_mut`, and a `deref_mut` stamps a change tick whether or not
    // anything is written. Unguarded, this system stamped `AudioMonitor`
    // on every frame of every session — nearly all of which have no bake
    // in flight at all — which is the defect Overlands #1340 is filed for
    // (#65, Overlands #1338 D1). Found by the guard test on the position
    // publisher next door, which could not pass while this one stamped.
    if monitor.task.is_none() {
        return;
    }
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
    monitor.playhead = None;
    // A rate of zero cannot be played. Refused here, where the `u32` becomes
    // the voice's `NonZeroU32`, rather than handed to the backend.
    let playable = result.and_then(|baked| match NonZeroU32::new(baked.sample_rate) {
        Some(rate) => Ok((baked, rate)),
        None => Err("the bake came back at a sample rate of zero, which cannot play".to_string()),
    });
    match playable {
        Ok((baked, rate)) => {
            if let Some(mut sources) = sources {
                let voice = LoopedSamples::new(&baked.samples[..], rate, baked.loop_start);
                monitor.playhead = Some(voice.playhead());
                let handle = sources.add(voice);
                // Under `Once`, never `Loop`: bevy_audio appends a decoder
                // bare under `Once`, and under `Loop` it wraps it in the rodio
                // `Buffered` that refuses every seek. The voice loops by
                // itself. At the monitor's own level, not full: the level
                // belongs to the monitor rather than to the buffer in it, so
                // a re-bake after an edit must not undo a turn-down.
                let entity = commands
                    .spawn((
                        AudioPlayer::<LoopedSamples>(handle),
                        PlaybackSettings::ONCE
                            .with_volume(bevy::audio::Volume::Linear(monitor.volume)),
                    ))
                    .id();
                monitor.current = Some(entity);
            } else {
                bevy::log::warn_once!(
                    "the audition monitor has no Assets<LoopedSamples> to make its voice in, \
                     so auditions bake and play nothing: add Bevy's AudioPlugin, alongside \
                     which AudioEditorPlugin registers the voice's source"
                );
            }
            monitor.sample_rate = baked.sample_rate;
            monitor.last_samples = baked.samples;
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

/// Where a seek to `secs` actually lands in a buffer `loop_secs` long, or
/// `None` when there is no buffer to land in.
///
/// Clamped rather than refused: a click a pixel past the right-hand end of
/// a waveform means "the end", and a click a pixel before its left means
/// "the start". Neither is a mistake worth an error, and a seek that
/// silently did nothing at the very edges would read as a dead zone.
fn seek_target(secs: f32, loop_secs: Option<f32>) -> Option<f32> {
    let loop_secs = loop_secs?;
    if !secs.is_finite() {
        return None;
    }
    Some(secs.clamp(0.0, loop_secs))
}

/// Publish the playing voice's position in its buffer, for a waveform or a
/// timeline to draw a cursor at.
///
/// Copied from the voice's [`LoopPlayhead`]: the sample the voice plays
/// next, or where a seek it has not reached yet will put it. See
/// [`AudioMonitor::position_secs`] for why the source and not the sink.
///
/// # Why this reads through `ResMut` before it writes
///
/// `ResMut::deref_mut` stamps a change tick whether or not anything is
/// actually written, and this system runs every frame of every session —
/// including the overwhelming majority in which no audition is playing at
/// all, and every frame of one whose voice is not moving (a host with no
/// audio device). Taking the `&mut` unconditionally would stamp
/// [`AudioMonitor`] on every one of those frames, which is the defect
/// Overlands #1340 is filed for, arriving here instead of there. So: read
/// through the immutable deref, compare, and assign only on a real
/// difference. A moving playhead does stamp a tick — it genuinely changed —
/// and nothing in this crate or in Overlands reads this resource's change
/// tick, so that cost stops at the resource.
fn publish_monitor_position(mut monitor: ResMut<AudioMonitor>) {
    // Immutable deref: reading through a `ResMut` stamps nothing.
    let next = monitor
        .playhead
        .as_ref()
        .map(|playhead| playhead.position().as_secs_f32());
    if monitor.position != next {
        monitor.position = next;
    }
}

/// Apply [`MonitorControl`]s to the voice that is playing.
///
/// A seek with nothing playing is dropped rather than surfaced: it is not a
/// fault in the patch being edited, and [`MonitorStatus::Error`] is reserved
/// for a bake that failed.
fn apply_monitor_controls(
    mut controls: MessageReader<MonitorControl>,
    mut monitor: ResMut<AudioMonitor>,
    mut voices: Query<(Option<&mut AudioSink>, &mut PlaybackSettings)>,
) {
    // Nothing to apply: do not take the `&mut` at all (see
    // `publish_monitor_position` on why that matters).
    if controls.is_empty() {
        return;
    }
    for control in controls.read() {
        match *control {
            MonitorControl::Seek(secs) => {
                // Through the voice's own playhead, which takes `&self` and
                // returns at once — never the sink's `try_seek`, see
                // [`MonitorControl::Seek`]. The publisher reads the new
                // place later this same frame.
                if let (Some(target), Some(playhead)) =
                    (seek_target(secs, monitor.loop_secs()), &monitor.playhead)
                {
                    playhead.seek(Duration::from_secs_f32(target));
                }
            }
            MonitorControl::Volume(level) => {
                let level = level.clamp(0.0, 1.0);
                if monitor.volume != level {
                    monitor.volume = level;
                }
                if let Some(entity) = monitor.current
                    && let Ok((sink, mut settings)) = voices.get_mut(entity)
                {
                    match sink {
                        Some(mut sink) => sink.set_volume(bevy::audio::Volume::Linear(level)),
                        // Its sink has not arrived: Bevy builds it a frame
                        // after the voice is spawned, from these settings. So
                        // the level goes where it will be read — otherwise a
                        // level set in the frame a bake lands was lost to the
                        // voice that bake made.
                        None => settings.volume = bevy::audio::Volume::Linear(level),
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AudioEditorPlugin, AudioMonitor, MonitorControl, MonitorStatus, waveform};
    use crate::looping::LoopedSamples;
    use bevy::audio::{AudioPlayer, AudioSink, Decodable, PlaybackSettings};
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

    /// One frame of a waveform of `samples`; what it painted.
    fn painted_waveform(ctx: &egui::Context, samples: &[f32]) -> egui::FullOutput {
        ctx.run_ui(egui::RawInput::default(), |root| {
            egui::CentralPanel::default().show(root, |ui| {
                waveform(ui, samples);
            });
        })
    }

    fn a_sine() -> Vec<f32> {
        (0..2000).map(|i| (i as f32 * 0.05).sin()).collect()
    }

    /// E3 (Overlands #1339): the cursor and the seconds axis stay on the
    /// waveform.
    ///
    /// Nothing measured where either landed. The axis places a label at
    /// `x + 2` from its tick with `LEFT_BOTTOM`, and the tick loop runs
    /// while `t < secs`, so a step that lands just under the end puts a
    /// label most of its own width past the right edge — a waveform is
    /// drawn at whatever width its host has, and the step is chosen from
    /// that width, so the pair that does it is a matrix and not a guess.
    ///
    /// Measured against [`WaveformResponse::response`]'s own rect, which is
    /// what the widget told its host it took.
    #[test]
    fn the_cursor_and_the_seconds_axis_lie_inside_the_waveform() {
        use crate::ui::test_paint::shapes;
        let samples = a_sine();
        let ctx = egui::Context::default();
        let input = || egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1000.0, 400.0),
            )),
            ..Default::default()
        };
        for width in [120.0_f32, 260.0, 890.0] {
            for secs in [0.2_f32, 1.0, 2.05, 7.5, 34.0] {
                for at in [0.0_f32, 0.5, 0.999] {
                    let position = secs * at;
                    let mut rect = None;
                    let out = ctx.run_ui(input(), |root| {
                        egui::CentralPanel::default().show(root, |ui| {
                            let drawn = super::waveform_with_cursor(
                                ui,
                                &samples,
                                Some(position),
                                secs,
                                egui::vec2(width, 72.0),
                            );
                            rect = Some(drawn.response.rect);
                        });
                    });
                    let rect = rect.expect("the waveform drew").expand(0.5);
                    let case = format!("{width} points, {secs} s, cursor at {position:.3} s");
                    for shape in shapes(&out) {
                        match shape {
                            egui::Shape::Text(t) => {
                                let at = t.visual_bounding_rect();
                                assert!(
                                    rect.contains_rect(at),
                                    "{case}: the axis label {:?} at {at:?} is outside \
                                     the waveform {rect:?}",
                                    t.galley.text()
                                );
                            }
                            egui::Shape::LineSegment { points, .. } => {
                                for p in points {
                                    assert!(
                                        rect.contains(p),
                                        "{case}: a line reaches {p:?}, outside the \
                                         waveform {rect:?}"
                                    );
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    /// The waveform offers its click only while a cursor is drawn: a
    /// pointing hand over a waveform with a playhead on it, which is a voice
    /// a click can move, and a plain pointer over one without — no position,
    /// or a position off the buffer, which draws no cursor.
    #[test]
    fn the_waveform_is_a_pointing_hand_only_while_a_cursor_is_drawn() {
        let samples = a_sine();
        let ctx = egui::Context::default();
        for (position, hand) in [(Some(0.5_f32), true), (None, false), (Some(9.0), false)] {
            let frame = |events: Vec<egui::Event>| {
                let mut rect = egui::Rect::NOTHING;
                let out = ctx.run_ui(
                    egui::RawInput {
                        events,
                        ..Default::default()
                    },
                    |root| {
                        egui::CentralPanel::default().show(root, |ui| {
                            rect = super::waveform_with_cursor(
                                ui,
                                &samples,
                                position,
                                2.0,
                                egui::vec2(300.0, 72.0),
                            )
                            .response
                            .rect;
                        });
                    },
                );
                (out, rect)
            };
            let (_, rect) = frame(Vec::new());
            let (out, _) = frame(vec![egui::Event::PointerMoved(rect.center())]);
            let icon = out.platform_output.cursor_icon;
            assert_eq!(
                icon == egui::CursorIcon::PointingHand,
                hand,
                "a cursor at {position:?}: the pointer over the waveform is {icon:?}"
            );
        }
    }

    /// The acceptance of Overlands #1331 for the waveform: in egui's dark
    /// and light themes the trace reads on its ground. The ground is the
    /// rect the trace's lines lie in, and the trace is the vertical lines.
    #[test]
    fn the_waveform_trace_reads_on_its_ground_in_dark_and_light() {
        use crate::ui::style::tests::AA;
        use crate::ui::test_paint::{contrast_on, shapes};
        for (theme, visuals) in [
            ("dark", egui::Visuals::dark()),
            ("light", egui::Visuals::light()),
        ] {
            let ctx = egui::Context::default();
            ctx.set_visuals(visuals);
            let out = painted_waveform(&ctx, &a_sine());
            let painted = shapes(&out);
            let ground = painted
                .iter()
                .find_map(|s| match s {
                    egui::Shape::Rect(r) if r.rect.height() == 72.0 => Some(r.fill),
                    _ => None,
                })
                .expect("the waveform's ground");
            let traces: Vec<egui::Color32> = painted
                .iter()
                .filter_map(|s| match s {
                    egui::Shape::LineSegment { points, stroke } if points[0].x == points[1].x => {
                        Some(stroke.color)
                    }
                    _ => None,
                })
                .collect();
            assert!(traces.len() > 100, "{theme}: a column per pixel");
            for trace in traces {
                let ratio = contrast_on(trace, ground);
                assert!(
                    ratio >= AA,
                    "{theme}: the trace is {ratio:.2}:1 on {ground:?}"
                );
            }
        }
    }

    /// B16 (#59, Overlands #1332): an empty waveform draws no zero line, so
    /// "no signal" is not struck through by the one thing beside it.
    #[test]
    fn an_empty_waveform_draws_no_line_through_no_signal() {
        use crate::ui::test_paint::text_painted;
        let ctx = egui::Context::default();
        let out = painted_waveform(&ctx, &[]);
        let (label, _) = text_painted(&out, "no signal").expect("an empty waveform says so");
        for shape in crate::ui::test_paint::shapes(&out) {
            if let egui::Shape::LineSegment { points, .. } = shape {
                let line = egui::Rect::from_two_pos(points[0], points[1]);
                assert!(
                    !line.intersects(label),
                    "a line from {:?} to {:?} crosses {label:?}",
                    points[0],
                    points[1]
                );
            }
        }
        // The control: with samples the zero line is there to be crossed.
        let out = painted_waveform(&ctx, &a_sine());
        let horizontal = crate::ui::test_paint::shapes(&out).into_iter().any(
            |s| matches!(s, egui::Shape::LineSegment { points, .. } if points[0].y == points[1].y),
        );
        assert!(horizontal, "a waveform with samples still has a zero line");
    }

    /// A style the host set is what the waveform paints: its ground, zero
    /// line and trace, and an empty one's "no signal".
    #[test]
    fn a_set_style_is_what_the_waveform_paints() {
        use crate::ui::style::set_editor_style;
        use crate::ui::style::tests::distinct_style;
        use crate::ui::test_paint::{colours, text_painted};
        let s = distinct_style();
        let ctx = egui::Context::default();
        set_editor_style(&ctx, s.clone());
        let painted = colours(&painted_waveform(&ctx, &a_sine()));
        for (role, colour) in [
            ("waveform_ground", s.waveform_ground),
            ("waveform_zero", s.waveform_zero),
            ("waveform_trace", s.waveform_trace),
        ] {
            assert!(painted.contains(&colour), "{role} is not painted");
        }
        let out = painted_waveform(&ctx, &[]);
        assert_eq!(
            text_painted(&out, "no signal").map(|(_, c)| c),
            Some(s.ground_text)
        );
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

    /// The voice's source is registered once every plugin is in, whatever
    /// order a host added them in — here the editor's plugin goes in before
    /// Bevy's audio — and an app with no audio plugin is left without it.
    #[test]
    fn the_voices_source_is_registered_after_bevys_audio_whatever_the_order() {
        let mut app = App::new();
        app.add_plugins((
            AudioEditorPlugin,
            bevy::app::TaskPoolPlugin::default(),
            bevy::asset::AssetPlugin::default(),
            bevy::audio::AudioPlugin::default(),
        ));
        app.finish();
        assert!(
            app.world().contains_resource::<Assets<LoopedSamples>>(),
            "with Bevy's audio, the voice's source is registered"
        );

        let mut bare = App::new();
        bare.add_plugins(AudioEditorPlugin);
        bare.finish();
        assert!(
            !bare.world().contains_resource::<Assets<LoopedSamples>>(),
            "with no audio plugin there is nothing to register it with"
        );
    }

    // -----------------------------------------------------------------------
    // Superseded bakes (#57, Overlands #1330 D2)
    // -----------------------------------------------------------------------

    use super::MonitorRequest;
    use crate::{AudioPatch, GraphNode, NodeGraph, NodeId, NodeKind, SequenceRecipe, SineOsc};
    use std::collections::BTreeMap;
    use std::sync::atomic::Ordering;

    /// The monitor on a bare app: the task pools, the plugin and the assets
    /// its poll system writes the voice into. No audio device.
    fn monitor_app() -> App {
        let mut app = App::new();
        app.add_plugins((bevy::app::TaskPoolPlugin::default(), AudioEditorPlugin));
        app.init_resource::<Assets<LoopedSamples>>();
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
            .query::<&AudioPlayer<LoopedSamples>>()
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

    // -- The voice (#68, Overlands #1341) --------------------------------

    /// The voice is a `LoopedSamples` under `PlaybackMode::Once`, at the
    /// monitor's level. `Once`, because bevy_audio appends a decoder bare
    /// under `Once` and wraps it in rodio's seek-refusing `Buffered` under
    /// `Loop`: the voice loops by itself.
    ///
    /// The level is written in the same frame as the request, so this pins
    /// the frame a bake lands in as well: a bake quick enough to land on the
    /// first frame spawned its voice at the old level before the control was
    /// applied, and this failed once in four runs until the level went into
    /// the voice's settings while its sink is on the way (#68).
    #[test]
    fn the_voice_loops_by_itself_under_once() {
        let mut app = monitor_app();
        app.world_mut().write_message(MonitorControl::Volume(0.5));
        app.world_mut().write_message(MonitorRequest::PlayPatch {
            patch: sine(),
            sample_rate: 22_050,
            duration_secs: 0.25,
        });
        until_settled(&mut app);
        let entity = app
            .world()
            .resource::<AudioMonitor>()
            .current
            .expect("a voice");
        assert!(
            app.world()
                .get::<AudioPlayer<LoopedSamples>>(entity)
                .is_some(),
            "the voice plays a LoopedSamples"
        );
        let settings = app
            .world()
            .get::<PlaybackSettings>(entity)
            .expect("its settings");
        assert!(
            matches!(settings.mode, bevy::audio::PlaybackMode::Once),
            "{:?}",
            settings.mode
        );
        assert_eq!(settings.volume.to_linear(), 0.5);
    }

    /// A bake that comes back at a sample rate of zero is an error, not a
    /// voice: a zero rate cannot play, and the monitor refuses it where it
    /// makes the voice rather than handing it to the backend.
    #[test]
    fn a_bake_at_a_sample_rate_of_zero_is_an_error_not_a_voice() {
        let mut app = monitor_app();
        app.world_mut().write_message(MonitorRequest::PlaySequence {
            recipe: SequenceRecipe {
                sample_rate: 0,
                ..a_bed()
            },
        });
        until_settled(&mut app);
        {
            let monitor = app.world().resource::<AudioMonitor>();
            assert!(
                matches!(&monitor.status, MonitorStatus::Error(why) if why.contains("zero")),
                "{:?}",
                monitor.status
            );
            assert_eq!(monitor.position_secs(), None);
        }
        assert_eq!(voices(&mut app), 0);
    }

    /// An app with no voice assets — no Bevy audio plugin, so the source was
    /// never registered — still bakes and shows each audition, and plays
    /// nothing, rather than stopping on a missing resource. The harness is
    /// the shape a 0.4.10 host's test was written in, `Assets<AudioSource>`
    /// for that version's voice, and Overlands' gate caught this version
    /// panicking in exactly it (Overlands #1341).
    #[test]
    fn a_monitor_with_nothing_to_play_through_still_shows_its_bake() {
        let mut app = App::new();
        app.add_plugins((bevy::app::TaskPoolPlugin::default(), AudioEditorPlugin));
        app.init_resource::<Assets<bevy::audio::AudioSource>>();
        app.world_mut().write_message(MonitorRequest::PlayPatch {
            patch: sine(),
            sample_rate: 22_050,
            duration_secs: 0.25,
        });
        until_settled(&mut app);
        {
            let monitor = app.world().resource::<AudioMonitor>();
            assert_eq!(monitor.status, MonitorStatus::Playing, "the bake landed");
            assert_eq!(monitor.last_samples.len(), 5_513, "and shows");
            assert_eq!(monitor.position_secs(), None, "no voice, so no cursor");
        }
        assert_eq!(voices(&mut app), 0, "and nothing to hear");
    }

    /// What an audio device does for the monitor's voice, done by the test.
    ///
    /// These tests run on a bare app with no device, so Bevy never builds a
    /// sink. This builds the one it would — bevy_audio's
    /// `play_queued_audio_system` makes a rodio `Player`, appends the voice's
    /// decoder to it bare under `PlaybackMode::Once`, and inserts
    /// `AudioSink::new(player)` — and then pulls samples out of the player
    /// the way a device's callback does, as many as the test says.
    struct Device {
        out: rodio::queue::SourcesQueueOutput,
    }

    impl Device {
        /// Connect the monitor's voice, as Bevy does a frame after spawning it.
        fn connect(app: &mut App) -> Self {
            let entity = app
                .world()
                .resource::<AudioMonitor>()
                .current
                .expect("a voice is playing");
            let handle = app
                .world()
                .get::<AudioPlayer<LoopedSamples>>(entity)
                .expect("the voice plays a LoopedSamples")
                .0
                .clone();
            let decoder = app
                .world()
                .resource::<Assets<LoopedSamples>>()
                .get(&handle)
                .expect("the voice's asset")
                .decoder();
            let (player, out) = rodio::Player::new();
            player.append(decoder);
            app.world_mut()
                .entity_mut(entity)
                .insert(AudioSink::new(player));
            Self { out }
        }

        /// Play `samples` samples, and hear them.
        fn play(&mut self, samples: usize) -> Vec<f32> {
            self.out.by_ref().take(samples).collect()
        }
    }

    /// Where the monitor says its voice is.
    fn position(app: &App) -> f32 {
        app.world()
            .resource::<AudioMonitor>()
            .position_secs()
            .expect("a voice is playing")
    }

    /// A bed the seeded one's shape, small enough to bake quickly: four
    /// seconds at 60 BPM and 8 kHz, two beats of run-up before its loop start,
    /// and a sine held through its tail.
    fn a_bed() -> SequenceRecipe {
        SequenceRecipe {
            bpm: 60.0,
            sample_rate: 8_000,
            duration_beats: 4.0,
            loop_start_beats: Some(2.0),
            loop_crossfade_beats: 1.0,
            instruments: vec![crate::Instrument {
                id: "tone".into(),
                patch: sine(),
            }],
            tracks: vec![crate::Track {
                events: vec![crate::Event {
                    instrument_id: "tone".into(),
                    gate_beats: 5.0,
                    volume: 0.5,
                    ..crate::Event::default()
                }],
            }],
        }
    }

    /// MEASURED THROUGH THE BACKEND: a looping bed's published cursor never
    /// enters its run-up, pass after pass — and a seek puts it there, once.
    ///
    /// The control is 0.4.10's: its voice looped from sample 0, and its
    /// cursor, wrapped into the buffer, went 34.006 s into a seeded bed and
    /// came out at 0.006 s (Overlands #1341).
    #[test]
    fn the_cursor_never_enters_the_run_up_until_a_seek_puts_it_there_once() {
        let mut app = monitor_app();
        app.world_mut()
            .write_message(MonitorRequest::PlaySequence { recipe: a_bed() });
        until_settled(&mut app);
        let (run_up, end) = (2.0_f32, 4.0_f32);
        assert_eq!(
            app.world().resource::<AudioMonitor>().loop_secs(),
            Some(end)
        );
        assert_eq!(
            position(&app),
            run_up,
            "the voice starts at its loop start, on the frame it is spawned"
        );

        // Three passes and a half, a quarter of a second a frame.
        let mut device = Device::connect(&mut app);
        let (mut passes, mut last) = (0, run_up);
        for frame in 0..28 {
            device.play(2_000);
            app.update();
            let at = position(&app);
            assert!(
                (run_up..end).contains(&at),
                "frame {frame}: the cursor is at {at} s, outside the loop {run_up}..{end}"
            );
            if at < last {
                passes += 1;
            }
            last = at;
        }
        assert!(passes >= 3, "the bed looped {passes} times, not three");

        // A seek into the run-up plays it once.
        app.world_mut().write_message(MonitorControl::Seek(1.0));
        app.update();
        assert_eq!(
            position(&app),
            1.0,
            "the cursor is where the seek put it on the frame it was asked for"
        );
        let (mut in_run_up, mut left) = (0, false);
        for frame in 0..28 {
            device.play(2_000);
            app.update();
            let at = position(&app);
            if at < run_up {
                assert!(
                    !left,
                    "frame {frame}: the cursor went back into the run-up, at {at} s"
                );
                in_run_up += 1;
            } else {
                left = true;
            }
        }
        assert_eq!(
            in_run_up, 3,
            "a quarter-second a frame from 1.0 s is three frames in the run-up"
        );
    }

    /// The loop length is the buffer's own, and is `None` before anything
    /// has baked — a monitor with no samples has nothing to be a length of.
    #[test]
    fn the_loop_length_is_the_buffers_own() {
        let mut monitor = AudioMonitor::default();
        assert_eq!(monitor.loop_secs(), None, "nothing baked yet");
        monitor.last_samples = vec![0.0; 11_025];
        assert_eq!(monitor.sample_rate, 0);
        assert_eq!(
            monitor.loop_secs(),
            None,
            "samples but no rate is no length"
        );
        monitor.sample_rate = 22_050;
        assert_eq!(monitor.loop_secs(), Some(0.5));
    }

    /// #1340's shape, guarded: the position publisher runs every frame of
    /// every session, and `ResMut::deref_mut` stamps a change tick whether
    /// or not anything is written. With nothing playing it must touch
    /// nothing at all.
    ///
    /// The control is below: this fails if the system takes its `&mut`
    /// before knowing there is a change to make.
    #[test]
    fn publishing_a_position_with_nothing_playing_stamps_nothing() {
        let mut app = monitor_app();
        app.update();
        // The resource's own last-changed tick, not `is_changed()` read
        // from outside a system: `app.update()` moves the world's
        // last-change tick past the stamp before a caller out here can see
        // it, so `is_changed()` reads false either way and the test would
        // pass on the very code it is meant to refuse.
        let tick_of = |app: &App| {
            app.world()
                .get_resource_change_ticks::<AudioMonitor>()
                .expect("the monitor")
                .changed
        };
        let before = tick_of(&app);
        for frame in 0..8 {
            app.update();
            assert_eq!(
                tick_of(&app),
                before,
                "frame {frame}: the monitor was stamped with nothing playing"
            );
        }
    }

    /// The same guard with a voice. Its playhead is read every frame there
    /// is one, through the resource's immutable deref, and a voice nothing is
    /// pulling samples from — a host with no audio device — reads the same
    /// place every frame, which must stamp nothing either.
    #[test]
    fn publishing_the_position_of_a_voice_that_is_not_moving_stamps_nothing() {
        let mut app = monitor_app();
        app.world_mut().write_message(MonitorRequest::PlayPatch {
            patch: sine(),
            sample_rate: 22_050,
            duration_secs: 0.25,
        });
        until_settled(&mut app);
        app.update();
        let tick_of = |app: &App| {
            app.world()
                .get_resource_change_ticks::<AudioMonitor>()
                .expect("the monitor")
                .changed
        };
        let before = tick_of(&app);
        for frame in 0..8 {
            app.update();
            assert_eq!(
                tick_of(&app),
                before,
                "frame {frame}: a still voice stamped the monitor"
            );
        }
        assert_eq!(position(&app), 0.0, "a patch's voice starts at 0");
    }

    /// The other half of the guard: a real change IS published. Without
    /// this the test above passes on a system that never writes anything.
    #[test]
    fn a_position_that_really_changed_is_published() {
        let mut app = monitor_app();
        app.update();
        app.world_mut().resource_mut::<AudioMonitor>().position = Some(0.25);
        assert_eq!(
            app.world().resource::<AudioMonitor>().position_secs(),
            Some(0.25),
            "the accessor reads the published position"
        );
        // And the publisher clears it again once there is no voice: a
        // cursor left behind after a stop is a cursor that lies.
        app.update();
        assert_eq!(
            app.world().resource::<AudioMonitor>().position_secs(),
            None,
            "with no voice the position goes away rather than freezing"
        );
    }

    /// A monitor nobody has turned down is at full level, and
    /// `MonitorControl::Volume` moves it and holds it across a re-bake —
    /// the level belongs to the monitor, not to the buffer in it.
    #[test]
    fn the_monitor_volume_is_the_monitors_and_survives_a_re_bake() {
        let mut app = monitor_app();
        assert_eq!(app.world().resource::<AudioMonitor>().volume(), 1.0);
        app.world_mut().write_message(MonitorControl::Volume(0.25));
        app.update();
        assert_eq!(app.world().resource::<AudioMonitor>().volume(), 0.25);

        app.world_mut().write_message(MonitorRequest::PlayPatch {
            patch: sine(),
            sample_rate: 22_050,
            duration_secs: 0.25,
        });
        until_settled(&mut app);
        assert_eq!(
            app.world().resource::<AudioMonitor>().volume(),
            0.25,
            "a re-bake after an edit must not undo a turn-down"
        );
    }

    /// Out-of-range levels are clamped rather than honoured: a negative
    /// gain inverts a waveform and a host that sends one has a bug.
    #[test]
    fn a_monitor_volume_is_clamped_into_range() {
        let mut app = monitor_app();
        for (sent, want) in [(-1.0_f32, 0.0_f32), (4.0, 1.0), (0.5, 0.5)] {
            app.world_mut().write_message(MonitorControl::Volume(sent));
            app.update();
            assert_eq!(app.world().resource::<AudioMonitor>().volume(), want);
        }
    }

    /// A seek with nothing baked moves nothing and raises nothing: it is
    /// not a fault in the patch, and `MonitorStatus::Error` is for a bake
    /// that failed.
    #[test]
    fn a_seek_with_nothing_baked_is_dropped_quietly() {
        let mut app = monitor_app();
        app.world_mut().write_message(MonitorControl::Seek(1.0));
        app.update();
        let monitor = app.world().resource::<AudioMonitor>();
        assert_eq!(monitor.position_secs(), None);
        assert_eq!(monitor.status, MonitorStatus::Idle);
    }

    /// A seek is clamped into the buffer, so a click past the end of a
    /// waveform lands at its end rather than in an error.
    ///
    /// Asked of `seek_target` directly: where a seek lands is arithmetic,
    /// and the tests that play a voice through rodio see where the voice
    /// then plays from.
    #[test]
    fn a_seek_is_clamped_into_the_buffer() {
        use super::seek_target;
        let half = Some(0.5_f32);
        assert_eq!(
            seek_target(9.0, half),
            Some(0.5),
            "past the end lands at the end"
        );
        assert_eq!(
            seek_target(-3.0, half),
            Some(0.0),
            "before the start lands at the start"
        );
        assert_eq!(seek_target(0.2, half), Some(0.2), "inside is left alone");
        assert_eq!(
            seek_target(0.5, half),
            Some(0.5),
            "the end itself is reachable"
        );
        // No buffer, nowhere to land.
        assert_eq!(seek_target(0.2, None), None);
        // A non-finite ask is a bug in the caller, not a place.
        assert_eq!(seek_target(f32::NAN, half), None);
        assert_eq!(seek_target(f32::INFINITY, half), None);
    }

    /// A seek moves the voice AND the published cursor, together — measured
    /// through the backend: rodio's `Player`, built as bevy_audio builds one
    /// for the voice, with the test pulling the samples a device would.
    ///
    /// Until 0.4.11 this was `a_seek_the_backend_refuses_moves_nothing`, and
    /// it passed by asserting the refusal. Run as written against the new
    /// voice it failed — `left: Some(0.3), right: Some(0.0)`, the cursor
    /// moved — and it is this now (#68, Overlands #1341). Its old harness
    /// could not have seen a backend either way: with no audio device no
    /// sink ever existed, so no seek was ever tried.
    #[test]
    fn a_seek_moves_the_voice_and_the_published_cursor_together() {
        let mut app = monitor_app();
        app.world_mut().write_message(MonitorRequest::PlayPatch {
            patch: sine(),
            sample_rate: 22_050,
            duration_secs: 0.5,
        });
        until_settled(&mut app);
        assert_eq!(
            app.world().resource::<AudioMonitor>().loop_secs(),
            Some(0.5)
        );
        let buffer = app.world().resource::<AudioMonitor>().last_samples.clone();
        let one_sample = 1.0 / 22_050.0;

        let mut device = Device::connect(&mut app);
        let heard = device.play(1_000);
        app.update();
        assert_eq!(
            heard,
            buffer[..1_000],
            "the voice plays its bake from the start"
        );
        assert!(
            (position(&app) - 1_000.0 * one_sample).abs() < one_sample / 2.0,
            "the cursor is where the voice is, at {}",
            position(&app)
        );

        app.world_mut().write_message(MonitorControl::Seek(0.3));
        app.update();
        assert!(
            (position(&app) - 0.3).abs() < one_sample,
            "the cursor moved to the seek on the frame it was asked for; it is at {}",
            position(&app)
        );

        // 0.3 s at 22.05 kHz is sample 6 615.
        let heard = device.play(64);
        assert_eq!(
            heard,
            buffer[6_615..6_679],
            "the voice plays from where the seek put it"
        );
        app.update();
        assert!(
            (position(&app) - 6_679.0 * one_sample).abs() < one_sample / 2.0,
            "and the cursor goes on from there with it, at {}",
            position(&app)
        );
        assert_eq!(
            app.world().resource::<AudioMonitor>().status,
            MonitorStatus::Playing,
            "a seek is not a re-bake"
        );
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
