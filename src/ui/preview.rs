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
//! [`MonitorStatus::Error`] (via [`try_bake`]) rather than panicking, since
//! the graph canvas can easily produce a cyclic or dangling graph.

use bevy::audio::{AudioPlayer, AudioSource, PlaybackSettings};
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};
use bevy_egui::egui;

use crate::{AudioPatch, SequenceRecipe, bake_sequence, samples_to_audio_source, try_bake};

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
/// [`AudioEditorPlugin`] systems do the baking and playback.  A new
/// `Play*` replaces whatever is currently playing.
#[derive(Message, Clone)]
pub enum MonitorRequest {
    /// Bake `patch` for `duration_secs` at `sample_rate`, then loop it.
    PlayPatch {
        patch: AudioPatch,
        sample_rate: u32,
        duration_secs: f32,
    },
    /// Bake `recipe` (at its own sample rate) and loop it.
    PlaySequence { recipe: SequenceRecipe },
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
    // Dropping the task cancels it.
    monitor.task = None;
    monitor.status = MonitorStatus::Idle;
}

/// Read [`MonitorRequest`]s and dispatch background bakes (or stop).
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
                stop_monitor(&mut monitor, &mut commands);
                let pool = AsyncComputeTaskPool::get();
                let patch = patch.clone();
                let (sr, dur) = (*sample_rate, *duration_secs);
                monitor.task = Some(pool.spawn(async move {
                    try_bake(&patch, sr, dur)
                        .map(|samples| (samples, sr))
                        .map_err(|e| e.to_string())
                }));
                monitor.status = MonitorStatus::Baking;
            }
            MonitorRequest::PlaySequence { recipe } => {
                stop_monitor(&mut monitor, &mut commands);
                let pool = AsyncComputeTaskPool::get();
                let recipe = recipe.clone();
                monitor.task = Some(pool.spawn(async move {
                    let sr = recipe.sample_rate;
                    Ok((bake_sequence(&recipe), sr))
                }));
                monitor.status = MonitorStatus::Baking;
            }
        }
    }
}

/// Poll the in-flight bake; when it finishes, play the buffer (looping,
/// non-spatial) and stash it for the waveform.
fn poll_monitor_bakes(
    mut monitor: ResMut<AudioMonitor>,
    mut commands: Commands,
    mut sources: ResMut<Assets<AudioSource>>,
) {
    let Some(mut task) = monitor.task.take() else {
        return;
    };
    match block_on(poll_once(&mut task)) {
        // Still baking — put the task back for next frame.
        None => monitor.task = Some(task),
        Some(Ok((samples, sample_rate))) => {
            let handle = sources.add(samples_to_audio_source(&samples, sample_rate));
            if let Some(prev) = monitor.current.take() {
                commands.entity(prev).despawn();
            }
            let entity = commands
                .spawn((AudioPlayer::new(handle), PlaybackSettings::LOOP))
                .id();
            monitor.current = Some(entity);
            monitor.sample_rate = sample_rate;
            monitor.last_samples = samples;
            monitor.status = MonitorStatus::Playing;
        }
        Some(Err(e)) => {
            monitor.status = MonitorStatus::Error(e);
            monitor.last_samples.clear();
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
            let _ = ctx.run(egui::RawInput::default(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
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
}
