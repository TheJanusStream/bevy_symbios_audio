//! Loop a baked buffer from where its bake says the loop starts.
//!
//! [`bake_sequence`](crate::bake_sequence) does not make a buffer that loops
//! from its first sample. For a recipe with `loop_start_beats`, it folds the
//! crossfade tail into the loop region from the loop start on, and cuts the
//! buffer at `duration_beats`: the seam it smooths runs from the last sample
//! back to the **loop start**. Looped from sample 0 instead, a baked bed
//! replays its run-up on every pass and jumps across a seam nothing smoothed
//! (Overlands #1341).
//!
//! Two ways to play it from the loop start, and they land on the same sample:
//!
//! - **Bevy's own looping player.** `PlaybackSettings::LOOP` with its
//!   [`start_position`](bevy::audio::PlaybackSettings::start_position) at
//!   [`sequence_loop_start`]: bevy_audio 0.19 appends
//!   `decoder.skip_duration(start).repeat_infinite()` for that, a loop from
//!   the loop start to the end in which the run-up is never played. One
//!   field and no new asset, which is what a world's player wants. It cannot
//!   be moved while it plays, because `repeat_infinite` buffers its source
//!   and rodio's buffer refuses every seek.
//! - **[`LoopedSamples`]**, an asset whose voice plays from its loop start to
//!   its end and back to its loop start for ever, and moves while it plays
//!   through its [`LoopPlayhead`]. What an editor's monitor wants. Register
//!   it with `App::add_audio_source::<LoopedSamples>()` (the editor's
//!   `AudioEditorPlugin` does) and play it under `PlaybackSettings::ONCE`:
//!   bevy_audio appends a decoder bare under `Once`, and under `Loop` it
//!   would wrap this one in the very buffer that refuses a seek. The voice
//!   loops by itself.
//!
//! A patch bake has no run-up: loop it from its first sample, a loop start of
//! [`Duration::ZERO`].

use std::fmt;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use bevy::asset::Asset;
use bevy::audio::Decodable;
use bevy::reflect::TypePath;
use rodio::source::SeekError;
use rodio::{ChannelCount, Sample, SampleRate, Source};

use crate::{MAX_WAV_SAMPLES, SequenceRecipe};

const NANOS_PER_SEC: u128 = 1_000_000_000;

/// The playhead's "no seek asked for". A buffer index can never be this:
/// a buffer holds at most `isize::MAX` samples.
const NO_SEEK: usize = usize::MAX;

/// Where the buffer [`bake_sequence`](crate::bake_sequence) makes of `recipe`
/// loops from, as a time from its first sample, or `None` when that buffer
/// has no loop point.
///
/// Give it to `PlaybackSettings::start_position`, or to
/// [`LoopedSamples::new`], and the voice loops across the seam the mixdown
/// smoothed.
///
/// `Some` exactly when `loop_start_beats` is set and lands before the end of
/// the buffer — the mixdown's own rule, in its own arithmetic. A hard-cut
/// loop, one with no `loop_crossfade_beats`, is still `Some`: the mixdown has
/// no tail to fold there, but the loop still runs from the point the author
/// set.
///
/// Ask about the recipe exactly as it is baked. A host that clamps a recipe
/// to an [`Envelope`](crate::Envelope) before baking it asks about a clamped
/// copy, or the two answers disagree.
///
/// The time lands on the loop start's own sample under rodio's arithmetic —
/// `Source::skip_duration` floors nanoseconds times the rate over a billion —
/// because it is that sample's nanosecond, rounded up.
pub fn sequence_loop_start(recipe: &SequenceRecipe) -> Option<Duration> {
    let rate = NonZeroU32::new(recipe.sample_rate)?;
    loop_start_sample(recipe).map(|sample| time_of(sample, rate))
}

/// The sample `bake_sequence` folds its tail into for `recipe`, or `None`
/// when its buffer has no loop point.
///
/// The mixdown's own arithmetic, step for step — symbios-audio 0.2
/// `mixdown.rs`: `beat_seconds` (in `f32`), `duration_to_samples` and the
/// head of `apply_loop_crossfade` — because those are private there.
/// `the_loop_start_is_where_the_mixdown_folds_its_tail` holds the two
/// together by baking.
fn loop_start_sample(recipe: &SequenceRecipe) -> Option<usize> {
    let beats = recipe.loop_start_beats?;
    let beat_secs = if recipe.bpm <= 0.0 {
        0.0
    } else {
        60.0 / recipe.bpm
    };
    let main = duration_to_samples(recipe.duration_beats, beat_secs, recipe.sample_rate);
    let start =
        (f64::from(beats) * f64::from(beat_secs) * f64::from(recipe.sample_rate)).round() as usize;
    // `apply_loop_crossfade`'s `loop_start >= main_samples` return, which
    // covers its `main_samples == 0` too: there is nothing past the end to
    // loop into. NOT its `tail_samples == 0` or `crossfade == 0` returns: a
    // hard cut has no tail to fold, and still loops from where it was set.
    (start < main).then_some(start)
}

/// `mixdown.rs`'s `duration_to_samples`: beats to samples, rounded, capped at
/// what a WAV can hold, and zero for a non-positive length or beat.
fn duration_to_samples(beats: f32, beat_secs: f32, sample_rate: u32) -> usize {
    if beats <= 0.0 || beat_secs <= 0.0 {
        return 0;
    }
    let samples = (f64::from(beats) * f64::from(beat_secs) * f64::from(sample_rate)).round();
    if samples >= MAX_WAV_SAMPLES as f64 {
        MAX_WAV_SAMPLES
    } else {
        samples as usize
    }
}

/// The sample a time `at` from a buffer's start lands on, the way rodio's
/// `skip_duration` counts it: nanoseconds times the rate over a billion,
/// floored.
fn sample_at(at: Duration, rate: NonZeroU32) -> usize {
    let sample = at.as_nanos() * u128::from(rate.get()) / NANOS_PER_SEC;
    usize::try_from(sample).unwrap_or(usize::MAX)
}

/// The earliest time that lands on `sample` under [`sample_at`]: the
/// sample's nanosecond, rounded up. Rounded down it would floor to the
/// sample before whenever the sample falls between two nanoseconds. Exact
/// for every rate up to a gigahertz, where a nanosecond is still no longer
/// than a sample.
fn time_of(sample: usize, rate: NonZeroU32) -> Duration {
    let nanos = (sample as u128 * NANOS_PER_SEC).div_ceil(u128::from(rate.get()));
    Duration::new(
        u64::try_from(nanos / NANOS_PER_SEC).unwrap_or(u64::MAX),
        (nanos % NANOS_PER_SEC) as u32,
    )
}

/// What a voice and its playhead share: the sample the voice plays next,
/// and a seek it has not taken yet.
#[derive(Debug)]
struct Shared {
    /// Written only by the voice's decoder, on the audio thread.
    at: AtomicUsize,
    /// Written only by [`LoopPlayhead::seek`]; taken, and cleared, by the
    /// decoder. [`NO_SEEK`] when nothing is asked.
    seek: AtomicUsize,
}

/// A mono buffer whose voice plays from its loop start to its end, and from
/// its end back to its loop start, for ever — and can be moved while it
/// plays.
///
/// See the [module docs](self) for when to use it rather than
/// `PlaybackSettings::start_position`. Play it under
/// `PlaybackSettings::ONCE`, never `LOOP`, once it is registered with
/// `App::add_audio_source::<LoopedSamples>()`.
///
/// The asset keeps its voice's [`LoopPlayhead`], and every decoder made of
/// it shares that playhead: play one voice of an asset at a time, or the
/// voices move each other.
#[derive(Asset, TypePath)]
pub struct LoopedSamples {
    samples: Arc<[f32]>,
    sample_rate: NonZeroU32,
    /// The sample the loop goes back to; always inside the buffer, or zero
    /// for an empty one.
    loop_start: usize,
    shared: Arc<Shared>,
}

impl LoopedSamples {
    /// `samples`, mono at `sample_rate`, looping from `loop_start`.
    ///
    /// `loop_start` is a time from the first sample, and lands on a sample
    /// the way `Source::skip_duration` does, so a loop start from
    /// [`sequence_loop_start`] is the sample a world player's
    /// `start_position` skips to. A loop start at or past the end loops the
    /// whole buffer, as a bake with no loop point does. The voice starts at
    /// the loop start.
    ///
    /// A zero sample rate cannot be played, and cannot be passed: refusing
    /// one is the caller's, where a `u32` rate becomes a [`NonZeroU32`].
    pub fn new(
        samples: impl Into<Arc<[f32]>>,
        sample_rate: NonZeroU32,
        loop_start: Duration,
    ) -> Self {
        let samples = samples.into();
        let loop_start = Some(sample_at(loop_start, sample_rate))
            .filter(|&sample| sample < samples.len())
            .unwrap_or(0);
        Self {
            samples,
            sample_rate,
            loop_start,
            shared: Arc::new(Shared {
                at: AtomicUsize::new(loop_start),
                seek: AtomicUsize::new(NO_SEEK),
            }),
        }
    }

    /// The buffer, from its first sample.
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// Samples per second.
    pub fn sample_rate(&self) -> NonZeroU32 {
        self.sample_rate
    }

    /// Where the loop starts, as a time from the first sample: the loop start
    /// [`Self::new`] was given, on the sample it landed on, or zero when it
    /// was past the end.
    pub fn loop_start(&self) -> Duration {
        time_of(self.loop_start, self.sample_rate)
    }

    /// The playhead of this asset's voice: where it is, and a way to move it.
    pub fn playhead(&self) -> LoopPlayhead {
        LoopPlayhead {
            shared: Arc::clone(&self.shared),
            sample_rate: self.sample_rate,
            len: self.samples.len(),
        }
    }
}

impl fmt::Debug for LoopedSamples {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The length rather than the samples: a seeded bed is 749 700 of them.
        f.debug_struct("LoopedSamples")
            .field("samples", &self.samples.len())
            .field("sample_rate", &self.sample_rate)
            .field("loop_start", &self.loop_start)
            .field("playhead", &self.playhead())
            .finish()
    }
}

impl Decodable for LoopedSamples {
    type Decoder = LoopedSamplesDecoder;

    fn decoder(&self) -> Self::Decoder {
        LoopedSamplesDecoder {
            samples: Arc::clone(&self.samples),
            sample_rate: self.sample_rate,
            loop_start: self.loop_start,
            // Wherever the playhead is: the loop start for a voice that has
            // not played. A seek asked for before the voice existed is taken
            // at its first sample.
            at: self.shared.at.load(Ordering::Relaxed),
            shared: Arc::clone(&self.shared),
        }
    }
}

/// Where a [`LoopedSamples`] voice is in its buffer, and a way to move it
/// while it plays. Cheap to clone, and every clone is the same playhead.
#[derive(Clone, Debug)]
pub struct LoopPlayhead {
    shared: Arc<Shared>,
    sample_rate: NonZeroU32,
    len: usize,
}

impl LoopPlayhead {
    /// Where the voice is, as a time from its buffer's first sample: the
    /// sample it plays next, or where a [`Self::seek`] it has not reached yet
    /// will put it.
    ///
    /// At the loop start before the voice has played anything, and there for
    /// good while nothing pulls samples from it — a host with no audio
    /// device.
    pub fn position(&self) -> Duration {
        // The seek is read first. The decoder publishes where a seek put it
        // before clearing the seek, so a read between the two finds the
        // target either way.
        let asked = self.shared.seek.load(Ordering::Relaxed);
        let at = if asked == NO_SEEK {
            self.shared.at.load(Ordering::Relaxed)
        } else {
            asked
        };
        time_of(at, self.sample_rate)
    }

    /// Move the voice to `to`, a time from its buffer's first sample,
    /// clamped into the buffer.
    ///
    /// Returns at once: the decoder takes the new place between two samples,
    /// and [`Self::position`] reports it from now on. A seek into a run-up,
    /// before the loop start, plays the run-up from there once.
    ///
    /// Unlike `AudioSinkPlayback::try_seek` this never waits for the audio
    /// thread, which is what makes it safe on the web. rodio's
    /// `Player::try_seek` waits on a channel for the audio callback to
    /// answer; on wasm32 that wait spins for ever, and the web's audio
    /// callback runs on the one thread there is (Overlands #1341).
    pub fn seek(&self, to: Duration) {
        let at = sample_at(to, self.sample_rate).min(self.len.saturating_sub(1));
        self.shared.seek.store(at, Ordering::Relaxed);
    }
}

/// The rodio source a [`LoopedSamples`] voice plays, one per voice, made by
/// bevy_audio through [`Decodable`].
pub struct LoopedSamplesDecoder {
    samples: Arc<[f32]>,
    sample_rate: NonZeroU32,
    loop_start: usize,
    /// The sample played next; inside the buffer unless the buffer is empty.
    at: usize,
    shared: Arc<Shared>,
}

impl Iterator for LoopedSamplesDecoder {
    type Item = Sample;

    fn next(&mut self) -> Option<Sample> {
        let len = self.samples.len();
        if len == 0 {
            // Nothing to loop: the voice ends rather than spinning.
            return None;
        }
        let asked = self.shared.seek.load(Ordering::Relaxed);
        if asked != NO_SEEK {
            // Published before the seek is cleared (see
            // `LoopPlayhead::position`), and cleared only if it is still the
            // seek taken: a newer one asked in between is taken next sample.
            self.at = asked;
            self.shared.at.store(asked, Ordering::Relaxed);
            let _ = self.shared.seek.compare_exchange(
                asked,
                NO_SEEK,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
        }
        let sample = self.samples[self.at];
        self.at = if self.at + 1 < len {
            self.at + 1
        } else {
            self.loop_start
        };
        self.shared.at.store(self.at, Ordering::Relaxed);
        Some(sample)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Endless, as rodio's own `Repeat` says it: no lower bound for a
        // queue to trust as a span length.
        if self.samples.is_empty() {
            (0, Some(0))
        } else {
            (0, None)
        }
    }
}

impl Source for LoopedSamplesDecoder {
    fn current_span_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> ChannelCount {
        ChannelCount::MIN
    }

    fn sample_rate(&self) -> SampleRate {
        self.sample_rate
    }

    fn total_duration(&self) -> Option<Duration> {
        None
    }

    /// Move to `pos`, a time from the buffer's first sample, clamped into the
    /// buffer. Never refused. rodio calls this on the audio thread for a
    /// native host's `AudioSinkPlayback::try_seek`, and the playhead sees it
    /// at once.
    fn try_seek(&mut self, pos: Duration) -> Result<(), SeekError> {
        self.at = sample_at(pos, self.sample_rate).min(self.samples.len().saturating_sub(1));
        self.shared.at.store(self.at, Ordering::Relaxed);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::{
        AudioPatch, Event, GraphNode, Instrument, NodeGraph, NodeId, NodeKind, SineOsc, Track,
        bake_sequence, samples_to_audio_source,
    };

    /// The seeded bed's shape: 34 s at 22 050 Hz looping from 2.0 s — 749 700
    /// samples, and a loop from sample 44 100 to the end.
    const BED_LEN: usize = 749_700;
    const BED_RATE: u32 = 22_050;
    const BED_LOOP: usize = 44_100;

    fn rate(hz: u32) -> NonZeroU32 {
        NonZeroU32::new(hz).expect("a rate")
    }

    /// A buffer whose every sample is its own index, so a sample says where
    /// it was played from. Exact in `f32` below 2^24.
    fn numbered(len: usize) -> Vec<f32> {
        (0..len).map(|i| i as f32).collect()
    }

    /// The first place `played` differs from `want`, as (index, played,
    /// wanted), with `None` for a sample it never played.
    fn first_difference(
        played: impl Iterator<Item = f32>,
        want: &[f32],
    ) -> Option<(usize, Option<f32>, f32)> {
        let mut played = played;
        want.iter().enumerate().find_map(|(n, &w)| {
            let p = played.next();
            (p != Some(w)).then_some((n, p, w))
        })
    }

    /// THE LOOP. The sample after the last is the loop start's, never sample
    /// 0, and the run-up before the loop start is never played at all.
    ///
    /// The control is HEAD's: 0.4.10's cursor wrapped a voice 34.006 s in to
    /// 0.006 s, inside the run-up, because the voice itself looped from
    /// sample 0 (Overlands #1341).
    #[test]
    fn after_the_last_sample_comes_the_loop_start_never_sample_zero() {
        let bed = LoopedSamples::new(numbered(BED_LEN), rate(BED_RATE), Duration::from_secs(2));
        assert_eq!(bed.loop_start(), Duration::from_secs(2));
        let mut voice = bed.decoder();
        let mut last = voice.next().expect("a voice");
        assert_eq!(last, BED_LOOP as f32, "the voice starts at its loop start");
        for n in 1..3 * (BED_LEN - BED_LOOP) {
            let sample = voice.next().expect("a loop never ends");
            let want = if last == (BED_LEN - 1) as f32 {
                BED_LOOP as f32
            } else {
                last + 1.0
            };
            assert_eq!(sample, want, "sample {n} of the voice came after {last}");
            last = sample;
        }
    }

    /// Bevy's own looping player, given [`sequence_loop_start`] as its
    /// `start_position`, loops the same window of the WAV a world bakes as
    /// the editor's voice does: bevy_audio 0.19's `Loop` arm appends exactly
    /// `skip_duration(start).repeat_infinite()`.
    #[test]
    fn a_world_player_from_the_same_loop_start_plays_the_same_loop() {
        const LEN: usize = 64_000;
        const START: usize = 16_000;
        let buffer: Vec<f32> = (0..LEN).map(|i| i as f32 / LEN as f32).collect();
        let recipe = SequenceRecipe {
            bpm: 60.0,
            sample_rate: 8_000,
            duration_beats: 8.0,
            loop_start_beats: Some(2.0),
            ..SequenceRecipe::default()
        };
        let start = sequence_loop_start(&recipe).expect("a loop point");
        let passes = 3 * (LEN - START) + 1;
        let want: Vec<f32> = (0..passes)
            .map(|n| buffer[START + n % (LEN - START)])
            .collect();

        let world = samples_to_audio_source(&buffer, 8_000)
            .decoder()
            .skip_duration(start)
            .repeat_infinite();
        assert_eq!(
            first_difference(world, &want),
            None,
            "the world's player (index, played, wanted)"
        );
        let voice = LoopedSamples::new(buffer.clone(), rate(8_000), start).decoder();
        assert_eq!(
            first_difference(voice, &want),
            None,
            "the editor's voice (index, played, wanted)"
        );
    }

    /// A seek moves the playhead at once and the voice at its next sample,
    /// and a seek into the run-up plays the run-up from there, once.
    #[test]
    fn a_seek_into_the_run_up_plays_it_once_and_then_loops_from_the_loop_start() {
        let bed = LoopedSamples::new(numbered(BED_LEN), rate(BED_RATE), Duration::from_secs(2));
        let playhead = bed.playhead();
        let mut voice = bed.decoder();
        voice.by_ref().take(10).for_each(drop);
        assert_eq!(playhead.position(), time_of(BED_LOOP + 10, rate(BED_RATE)));

        playhead.seek(Duration::from_secs(1));
        assert_eq!(
            playhead.position(),
            Duration::from_secs(1),
            "the playhead reports a seek before the voice has taken it"
        );
        assert_eq!(
            voice.next(),
            Some(22_050.0),
            "the voice plays from the seek"
        );
        assert_eq!(playhead.position(), time_of(22_051, rate(BED_RATE)));

        // The run-up plays through, once …
        let rest = BED_LEN - 22_051;
        let want: Vec<f32> = (22_051..BED_LEN).map(|i| i as f32).collect();
        assert_eq!(first_difference(voice.by_ref().take(rest), &want), None);
        // … and the loop never goes back into it.
        let want = numbered(BED_LEN)[BED_LOOP..].to_vec();
        assert_eq!(
            first_difference(voice.by_ref().take(want.len()), &want),
            None
        );
        assert_eq!(voice.next(), Some(BED_LOOP as f32));
    }

    /// A seek past the end asks for the end: the last sample plays, and the
    /// loop start after it.
    #[test]
    fn a_seek_past_the_end_plays_the_last_sample_and_then_the_loop_start() {
        let bed = LoopedSamples::new(numbered(1_000), rate(100), Duration::from_secs(2));
        let mut voice = bed.decoder();
        bed.playhead().seek(Duration::from_secs(60));
        assert_eq!(voice.next(), Some(999.0));
        assert_eq!(voice.next(), Some(200.0));
    }

    /// rodio's own seek, which a native host's `AudioSinkPlayback::try_seek`
    /// reaches on the audio thread, moves it too: clamped, never refused, and
    /// seen by the playhead. The trait's default refuses every seek; this
    /// source overrides it.
    #[test]
    fn rodios_own_seek_moves_the_voice_and_the_playhead_sees_it() {
        let bed = LoopedSamples::new(numbered(1_000), rate(100), Duration::from_secs(2));
        let mut voice = bed.decoder();
        voice
            .try_seek(Duration::from_millis(1_234))
            .expect("never refused");
        assert_eq!(bed.playhead().position(), Duration::from_millis(1_230));
        assert_eq!(voice.next(), Some(123.0));
        voice
            .try_seek(Duration::from_secs(60))
            .expect("past the end is the end");
        assert_eq!(voice.next(), Some(999.0));
        assert_eq!(
            voice.next(),
            Some(200.0),
            "and after the end, the loop start"
        );
    }

    /// An empty buffer has nothing to loop: its voice ends at once, rather
    /// than spinning a queue that asks it for a sample.
    #[test]
    fn an_empty_buffer_ends_instead_of_spinning() {
        let bed = LoopedSamples::new(Vec::new(), rate(8_000), Duration::from_secs(1));
        assert_eq!(bed.loop_start(), Duration::ZERO);
        let mut voice = bed.decoder();
        assert_eq!(voice.next(), None);
        voice
            .try_seek(Duration::from_secs(1))
            .expect("never refused");
        bed.playhead().seek(Duration::from_secs(1));
        assert_eq!(voice.next(), None);
        assert_eq!(voice.size_hint(), (0, Some(0)));
    }

    /// A loop start at or past the end loops the whole buffer, as a bake with
    /// no loop point does.
    #[test]
    fn a_loop_start_at_or_past_the_end_loops_the_whole_buffer() {
        for past in [Duration::from_secs(10), Duration::from_secs(11)] {
            let bed = LoopedSamples::new(numbered(1_000), rate(100), past);
            assert_eq!(bed.loop_start(), Duration::ZERO, "from {past:?}");
            let mut voice = bed.decoder();
            let want: Vec<f32> = (0..1_000).chain(0..10).map(|i| i as f32).collect();
            assert_eq!(first_difference(voice.by_ref(), &want), None, "{past:?}");
        }
    }

    // -- The loop start, against the mixdown ------------------------------

    /// The seeded bed: 34 beats at 60 BPM and 22 050 Hz looping from beat 2,
    /// which is two seconds and sample 44 100.
    #[test]
    fn the_seeded_beds_loop_starts_at_two_seconds() {
        let recipe = SequenceRecipe {
            bpm: 60.0,
            sample_rate: 22_050,
            duration_beats: 34.0,
            loop_start_beats: Some(2.0),
            loop_crossfade_beats: 2.0,
            ..SequenceRecipe::default()
        };
        assert_eq!(sequence_loop_start(&recipe), Some(Duration::from_secs(2)));
        assert_eq!(loop_start_sample(&recipe), Some(BED_LOOP));
    }

    /// One sine held through the recipe and its tail, so the tail the mixdown
    /// folds is not silence and the fold can be seen. 219.3 Hz so that no
    /// stretch these recipes compare is a whole number of cycles, which would
    /// make the tail and the loop region the same samples.
    fn sounding(recipe: SequenceRecipe) -> SequenceRecipe {
        let held = recipe.duration_beats + recipe.loop_crossfade_beats + 1.0;
        SequenceRecipe {
            instruments: vec![Instrument {
                id: "tone".into(),
                patch: AudioPatch {
                    seed: 0,
                    graph: NodeGraph {
                        nodes: vec![GraphNode {
                            id: NodeId(0),
                            kind: NodeKind::Sine(SineOsc {
                                freq_hz: 219.3,
                                ..SineOsc::default()
                            }),
                            inputs: BTreeMap::new(),
                        }],
                        output: NodeId(0),
                    },
                },
            }],
            tracks: vec![Track {
                events: vec![Event {
                    instrument_id: "tone".into(),
                    gate_beats: if held.is_finite() { held } else { 1.0 },
                    volume: 0.5,
                    ..Event::default()
                }],
            }],
            ..recipe
        }
    }

    /// Two seconds at 8 kHz, four beats of 120 BPM with a beat of tail,
    /// looping from beat 1.
    fn base() -> SequenceRecipe {
        SequenceRecipe {
            bpm: 120.0,
            sample_rate: 8_000,
            duration_beats: 4.0,
            loop_start_beats: Some(1.0),
            loop_crossfade_beats: 1.0,
            ..SequenceRecipe::default()
        }
    }

    /// Where `bake_sequence` folded its tail into `recipe`'s buffer: the
    /// first sample at which it differs from the same recipe baked with no
    /// loop point, which keeps its tail and folds nothing. `None` when the
    /// mixdown folded nothing.
    fn fold(recipe: &SequenceRecipe) -> Option<usize> {
        let looped = bake_sequence(recipe);
        let plain = bake_sequence(&SequenceRecipe {
            loop_start_beats: None,
            ..recipe.clone()
        });
        looped.iter().zip(&plain).position(|(a, b)| a != b)
    }

    /// The helper agrees with the mixdown it mirrors, found by BAKING rather
    /// than by reading: every `None` is a recipe whose bake folds nothing,
    /// and every `Some` is the sample the bake folded its tail into, reached
    /// by rodio's own arithmetic from the time the helper returns.
    #[test]
    fn the_loop_start_is_where_the_mixdown_folds_its_tail() {
        let none = [
            (
                "no loop point",
                SequenceRecipe {
                    loop_start_beats: None,
                    ..base()
                },
            ),
            (
                "a loop start at the end",
                SequenceRecipe {
                    loop_start_beats: Some(4.0),
                    ..base()
                },
            ),
            (
                "a loop start past the end",
                SequenceRecipe {
                    loop_start_beats: Some(9.0),
                    ..base()
                },
            ),
            (
                "a loop start that rounds onto the end",
                SequenceRecipe {
                    loop_start_beats: Some(3.9999),
                    ..base()
                },
            ),
            (
                "no length",
                SequenceRecipe {
                    duration_beats: 0.0,
                    loop_start_beats: Some(0.0),
                    ..base()
                },
            ),
            (
                "a NaN length",
                SequenceRecipe {
                    duration_beats: f32::NAN,
                    ..base()
                },
            ),
            ("no tempo", SequenceRecipe { bpm: 0.0, ..base() }),
            (
                "a NaN tempo",
                SequenceRecipe {
                    bpm: f32::NAN,
                    ..base()
                },
            ),
            (
                "no sample rate",
                SequenceRecipe {
                    sample_rate: 0,
                    ..base()
                },
            ),
        ];
        for (what, recipe) in none {
            let recipe = sounding(recipe);
            assert_eq!(sequence_loop_start(&recipe), None, "{what}");
            assert_eq!(fold(&recipe), None, "{what}: the mixdown folded its tail");
        }

        let some = [
            ("beat 1", base(), Some(4_000)),
            (
                "a start between two samples",
                SequenceRecipe {
                    loop_start_beats: Some(2.37),
                    ..base()
                },
                Some(9_480),
            ),
            (
                "the last sample",
                SequenceRecipe {
                    loop_start_beats: Some(3.99),
                    ..base()
                },
                Some(15_960),
            ),
            (
                "a negative start",
                SequenceRecipe {
                    loop_start_beats: Some(-1.0),
                    ..base()
                },
                Some(0),
            ),
            (
                "a NaN start",
                SequenceRecipe {
                    loop_start_beats: Some(f32::NAN),
                    ..base()
                },
                Some(0),
            ),
            (
                "another tempo and rate",
                SequenceRecipe {
                    bpm: 97.0,
                    sample_rate: 11_025,
                    duration_beats: 3.0,
                    loop_start_beats: Some(1.3),
                    loop_crossfade_beats: 0.5,
                    ..SequenceRecipe::default()
                },
                None,
            ),
        ];
        for (what, recipe, sample) in some {
            let recipe = sounding(recipe);
            let folded = fold(&recipe).unwrap_or_else(|| panic!("{what}: nothing folded"));
            if let Some(sample) = sample {
                assert_eq!(folded, sample, "{what}: the bake folded somewhere else");
            }
            let start = sequence_loop_start(&recipe).unwrap_or_else(|| {
                panic!("{what}: no loop start, and the bake folded at {folded}")
            });
            assert_eq!(
                sample_at(start, rate(recipe.sample_rate)),
                folded,
                "{what}: {start:?} lands somewhere other than the fold"
            );
        }

        // A hard cut folds nothing, having no tail — and still loops from
        // where it was set, which is why the crossfade's own early returns
        // are not mirrored.
        let hard = sounding(SequenceRecipe {
            loop_crossfade_beats: 0.0,
            ..base()
        });
        assert_eq!(fold(&hard), None, "a hard cut has no tail to fold");
        assert_eq!(sequence_loop_start(&hard), Some(Duration::from_millis(500)));
    }

    /// The time of a sample floors back to exactly that sample under rodio's
    /// arithmetic, and is the earliest time that does, across the rates a
    /// recipe can hold and the samples a WAV can.
    #[test]
    fn a_samples_time_lands_on_that_sample_and_is_the_earliest_that_does() {
        let rates = [
            1,
            8_000,
            11_025,
            22_050,
            44_100,
            48_000,
            96_000,
            192_000,
            1_000_000_000,
        ];
        let samples = [0, 1, 2, 3, 7, 441, 44_100, 749_699, MAX_WAV_SAMPLES - 1];
        for hz in rates {
            for sample in samples {
                let at = time_of(sample, rate(hz));
                assert_eq!(sample_at(at, rate(hz)), sample, "{hz} Hz, sample {sample}");
                if sample > 0 {
                    assert!(
                        sample_at(at - Duration::from_nanos(1), rate(hz)) < sample,
                        "{hz} Hz, sample {sample}: {at:?} is not the earliest"
                    );
                }
            }
        }
    }

    /// And rodio's real `skip_duration`, over a real decoder of the WAV a bake
    /// becomes, lands on that sample.
    #[test]
    fn rodios_skip_over_a_baked_wav_lands_on_the_same_sample() {
        for (hz, sample) in [
            (8_000, 1),
            (11_025, 7),
            (22_050, 44_100),
            (44_100, 12_345),
            (48_000, 47_999),
            (96_000, 95_999),
        ] {
            let len = sample + 16;
            let ramp: Vec<f32> = (0..len).map(|i| i as f32 / len as f32).collect();
            let skipped = samples_to_audio_source(&ramp, hz)
                .decoder()
                .skip_duration(time_of(sample, rate(hz)))
                .next();
            assert_eq!(skipped, Some(ramp[sample]), "{hz} Hz, sample {sample}");
        }
    }
}
