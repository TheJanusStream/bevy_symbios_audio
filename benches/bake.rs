//! Throughput benchmarks for the bake hot path.
//!
//! Gives the allocation-free per-sample evaluator a measurable baseline:
//! a multi-node modulated patch (`bake`) and a small multi-voice recipe
//! with a gated, releasing instrument (`bake_sequence`).
//!
//! ```sh
//! cargo bench
//! ```

use std::collections::BTreeMap;
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

use bevy_symbios_audio::{
    AdsrCurve, AdsrEnvelope, AudioPatch, BiquadLowpass, BrownNoise, Connection, Event, Gate,
    GraphNode, Instrument, Lfo, LfoShape, NodeGraph, NodeId, NodeKind, SequenceRecipe, SineOsc,
    Track, bake, bake_sequence,
};

/// BrownNoise → LP, LP cutoff swept by a sine LFO — exercises stateful
/// nodes (noise integrator, biquad), RNG draws, and per-sample modulation
/// input resolution all at once.
fn wind_patch() -> AudioPatch {
    let mut lp_inputs: BTreeMap<String, Vec<Connection>> = BTreeMap::new();
    lp_inputs.insert("in".into(), vec![Connection::from_node(NodeId(1))]);
    lp_inputs.insert(
        "cutoff_hz".into(),
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
                        offset: 1_100.0,
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

/// Gate → ADSR → Sine amplitude — a gated, releasing voice for the
/// sequence benchmark.
fn voice_instrument() -> Instrument {
    let mut adsr_inputs: BTreeMap<String, Vec<Connection>> = BTreeMap::new();
    adsr_inputs.insert("gate".into(), vec![Connection::from_node(NodeId(2))]);
    let mut sine_inputs: BTreeMap<String, Vec<Connection>> = BTreeMap::new();
    sine_inputs.insert(
        "amplitude".into(),
        vec![Connection::modulation(NodeId(0), 1.0)],
    );
    Instrument {
        id: "voice".into(),
        patch: AudioPatch {
            seed: 0,
            graph: NodeGraph {
                nodes: vec![
                    GraphNode {
                        id: NodeId(0),
                        kind: NodeKind::Adsr(AdsrEnvelope {
                            attack_s: 0.05,
                            decay_s: 0.1,
                            sustain_level: 0.5,
                            release_s: 0.3,
                            curve: AdsrCurve::Exponential,
                        }),
                        inputs: adsr_inputs,
                    },
                    GraphNode {
                        id: NodeId(1),
                        kind: NodeKind::Sine(SineOsc {
                            freq_hz: 220.0,
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
        },
    }
}

fn voice_recipe() -> SequenceRecipe {
    SequenceRecipe {
        bpm: 96.0,
        sample_rate: 44_100,
        duration_beats: 8.0,
        loop_start_beats: Some(0.0),
        loop_crossfade_beats: 1.0,
        instruments: vec![voice_instrument()],
        tracks: vec![Track {
            events: vec![
                Event {
                    time_beats: 0.0,
                    instrument_id: "voice".into(),
                    pitch_multiplier: 1.0,
                    volume: 0.4,
                    gate_beats: 2.0,
                    release_beats: 1.0,
                },
                Event {
                    time_beats: 4.0,
                    instrument_id: "voice".into(),
                    pitch_multiplier: 1.5,
                    volume: 0.4,
                    gate_beats: 2.0,
                    release_beats: 1.0,
                },
            ],
        }],
    }
}

fn bench_bake(c: &mut Criterion) {
    let patch = wind_patch();
    c.bench_function("bake_wind_1s_44k", |b| {
        b.iter(|| bake(black_box(&patch), 44_100, black_box(1.0)));
    });
}

fn bench_bake_sequence(c: &mut Criterion) {
    let recipe = voice_recipe();
    c.bench_function("bake_sequence_voice_8beats", |b| {
        b.iter(|| bake_sequence(black_box(&recipe)));
    });
}

criterion_group!(benches, bench_bake, bench_bake_sequence);
criterion_main!(benches);
