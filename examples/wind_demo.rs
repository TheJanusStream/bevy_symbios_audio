//! Offline demonstration of the full Phase 4 mixdown pipeline.
//!
//! Builds a `SequenceRecipe` with two instruments — a brown-noise wind
//! drone whose LP cutoff is modulated by a slow LFO, and a slightly-
//! detuned sine "voice" that fades in and out via ADSR — bakes them
//! through [`bake_sequence`], writes the result to `wind_demo.wav` in
//! the current working directory, and prints the buffer length.
//!
//! Run with:
//!
//! ```sh
//! cargo run --release --example wind_demo
//! ```
//!
//! The output WAV is ~5 s long at 44.1 kHz, plays as a seamless loop
//! when imported into any DAW (the loop_start_beats + crossfade are
//! configured so the buffer's tail is pre-mixed back into the loop
//! start).
//!
//! This is the same pipeline `SymbiosAudioPlugin` exposes inside a
//! running Bevy app, just hooked up to disk instead of an ECS world —
//! useful for video pipelines and offline sound design where running
//! `cargo run --example` is cheaper than spinning up the renderer.

use std::collections::BTreeMap;
use std::fs;

use bevy_symbios_audio::{
    AdsrCurve, AdsrEnvelope, AudioPatch, BiquadLowpass, BrownNoise, Connection, Event, Gate,
    GraphNode, Instrument, Lfo, LfoShape, NodeGraph, NodeId, NodeKind, SequenceRecipe, SineOsc,
    Track, bake_sequence, samples_to_wav_bytes,
};

/// Build a wind-drone patch: BrownNoise → LP with cutoff swept by a
/// 0.3 Hz sine LFO between roughly 200 and 2000 Hz.
fn wind_patch() -> AudioPatch {
    let mut lp_inputs = BTreeMap::new();
    lp_inputs.insert("in".to_string(), vec![Connection::from_node(NodeId(1))]);
    lp_inputs.insert(
        "cutoff_hz".to_string(),
        vec![Connection::modulation(NodeId(0), 1.0)],
    );
    AudioPatch {
        seed: 0xCAFE_BABE,
        graph: NodeGraph {
            nodes: vec![
                GraphNode {
                    id: NodeId(0),
                    kind: NodeKind::Lfo(Lfo {
                        rate_hz: 0.3,
                        shape: LfoShape::Sine,
                        depth: 900.0,
                        offset: 0.0,
                    }),
                    ..Default::default()
                },
                GraphNode {
                    id: NodeId(1),
                    kind: NodeKind::BrownNoise(BrownNoise { amplitude: 0.5 }),
                    ..Default::default()
                },
                GraphNode {
                    id: NodeId(2),
                    kind: NodeKind::BiquadLowpass(BiquadLowpass {
                        cutoff_hz: 1_100.0,
                        q: std::f32::consts::FRAC_1_SQRT_2,
                    }),
                    inputs: lp_inputs,
                },
            ],
            output: NodeId(2),
        },
    }
}

/// Build a tonal voice: a detuned sine through an ADSR that is gated by
/// a [`Gate`] node — so each sequenced swell attacks while its event's
/// gate is open, then releases and rings out across the event's
/// `release_beats` tail instead of cutting off abruptly.
fn voice_patch() -> AudioPatch {
    let mut adsr_inputs = BTreeMap::new();
    // Gate (node 2) drives the envelope: open for the event's gate_beats,
    // then closed so the release stage fires.
    adsr_inputs.insert("gate".to_string(), vec![Connection::from_node(NodeId(2))]);
    let mut sine_inputs = BTreeMap::new();
    sine_inputs.insert(
        "amplitude".to_string(),
        vec![Connection::modulation(NodeId(0), 1.0)],
    );
    AudioPatch {
        seed: 0,
        graph: NodeGraph {
            nodes: vec![
                GraphNode {
                    id: NodeId(0),
                    kind: NodeKind::Adsr(AdsrEnvelope {
                        attack_s: 0.05,
                        decay_s: 0.1,
                        sustain_level: 0.4,
                        release_s: 0.5,
                        curve: AdsrCurve::Exponential,
                    }),
                    inputs: adsr_inputs,
                },
                GraphNode {
                    id: NodeId(1),
                    kind: NodeKind::Sine(SineOsc {
                        freq_hz: 219.0,
                        phase_offset: 0.0,
                        amplitude: 0.0,
                    }),
                    inputs: sine_inputs,
                },
                GraphNode {
                    id: NodeId(2),
                    kind: NodeKind::Gate(Gate::default()),
                    ..Default::default()
                },
            ],
            output: NodeId(1),
        },
    }
}

fn main() {
    let sample_rate: u32 = 44_100;

    // 96 BPM × 8 beats = 5 s at the recipe's tempo, with a 1-beat tail
    // crossfade so the loop seam is buttery-smooth.
    let recipe = SequenceRecipe {
        bpm: 96.0,
        sample_rate,
        duration_beats: 8.0,
        loop_start_beats: Some(0.0),
        loop_crossfade_beats: 1.0,
        instruments: vec![
            Instrument {
                id: "wind".into(),
                patch: wind_patch(),
            },
            Instrument {
                id: "voice".into(),
                patch: voice_patch(),
            },
        ],
        tracks: vec![
            // Sustained wind layer through the whole loop — a drone, so
            // no gate release (release_beats: 0).
            Track {
                events: vec![Event {
                    time_beats: 0.0,
                    instrument_id: "wind".into(),
                    pitch_multiplier: 1.0,
                    volume: 0.6,
                    gate_beats: 8.0,
                    release_beats: 0.0,
                    ..Default::default()
                }],
            },
            // Voice swells at the start and mid-loop, each gated for two
            // beats then ringing out over a one-beat release tail.  The
            // last one's release is mixed into the loop start by the
            // crossfade.
            Track {
                events: vec![
                    Event {
                        time_beats: 0.0,
                        instrument_id: "voice".into(),
                        pitch_multiplier: 1.0,
                        volume: 0.3,
                        gate_beats: 2.0,
                        release_beats: 1.0,
                        ..Default::default()
                    },
                    Event {
                        time_beats: 4.0,
                        instrument_id: "voice".into(),
                        pitch_multiplier: 1.5,
                        volume: 0.3,
                        gate_beats: 2.0,
                        release_beats: 1.0,
                        ..Default::default()
                    },
                ],
            },
        ],
    };

    let samples = bake_sequence(&recipe);
    let wav = samples_to_wav_bytes(&samples, sample_rate);
    let out = "wind_demo.wav";
    fs::write(out, &wav).expect("write WAV");
    println!(
        "wrote {} samples ({:.2} s @ {} Hz) to {out}",
        samples.len(),
        samples.len() as f32 / sample_rate as f32,
        sample_rate,
    );
}
