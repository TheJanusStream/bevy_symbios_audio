//! Serde round-trip and shape tests for [`SequenceRecipe`].
//!
//! Phase 4 ticket #13 is pure schema work — the mixdown baker that
//! actually renders a recipe lands in #14.  These tests guarantee the
//! JSON form survives a round-trip with every field intact, including
//! the nested `AudioPatch` graphs inside `Instrument`.

use std::collections::BTreeMap;

use bevy_symbios_audio::{
    AdsrEnvelope, AudioPatch, BiquadLowpass, BrownNoise, Connection, Event, GraphNode, Instrument,
    NodeGraph, NodeId, NodeKind, SequenceRecipe, SineOsc, Track,
};

/// A realistic two-instrument recipe — a wind drone built from a
/// brown-noise → modulated LP graph, plus a sine-with-ADSR voice — so
/// the round-trip exercises every variant added across Phases 1-3.
fn realistic_recipe() -> SequenceRecipe {
    // Wind instrument: BrownNoise → BiquadLowpass(cutoff_hz modulated).
    let mut lp_inputs = BTreeMap::new();
    lp_inputs.insert("in".to_string(), Connection::from_node(NodeId(0)));
    let wind_patch = AudioPatch {
        seed: 0xABCD,
        graph: NodeGraph {
            nodes: vec![
                GraphNode {
                    id: NodeId(0),
                    kind: NodeKind::BrownNoise(BrownNoise { amplitude: 0.5 }),
                    inputs: BTreeMap::new(),
                },
                GraphNode {
                    id: NodeId(1),
                    kind: NodeKind::BiquadLowpass(BiquadLowpass {
                        cutoff_hz: 1_200.0,
                        q: 0.7,
                    }),
                    inputs: lp_inputs,
                },
            ],
            output: NodeId(1),
        },
    };

    // Bell instrument: SineOsc gated by ADSR.
    let mut adsr_inputs = BTreeMap::new();
    adsr_inputs.insert("gate".to_string(), Connection::constant(1.0));
    let mut sine_inputs = BTreeMap::new();
    sine_inputs.insert(
        "amplitude".to_string(),
        Connection::modulation(NodeId(0), 1.0),
    );
    let bell_patch = AudioPatch {
        seed: 7,
        graph: NodeGraph {
            nodes: vec![
                GraphNode {
                    id: NodeId(0),
                    kind: NodeKind::Adsr(AdsrEnvelope::default()),
                    inputs: adsr_inputs,
                },
                GraphNode {
                    id: NodeId(1),
                    kind: NodeKind::Sine(SineOsc {
                        freq_hz: 880.0,
                        phase_offset: 0.0,
                        amplitude: 0.0,
                    }),
                    inputs: sine_inputs,
                },
            ],
            output: NodeId(1),
        },
    };

    SequenceRecipe {
        bpm: 96.0,
        sample_rate: 48_000,
        duration_beats: 16.0,
        loop_start_beats: Some(8.0),
        loop_crossfade_beats: 2.0,
        instruments: vec![
            Instrument {
                id: "wind".into(),
                patch: wind_patch,
            },
            Instrument {
                id: "bell".into(),
                patch: bell_patch,
            },
        ],
        tracks: vec![
            // Wind: held for the full timeline with a slight retrigger
            // at the midpoint.
            Track {
                events: vec![
                    Event {
                        time_beats: 0.0,
                        instrument_id: "wind".into(),
                        pitch_multiplier: 1.0,
                        volume: 0.7,
                        gate_beats: 16.0,
                    },
                    Event {
                        time_beats: 8.0,
                        instrument_id: "wind".into(),
                        pitch_multiplier: 0.98,
                        volume: 0.5,
                        gate_beats: 8.0,
                    },
                ],
            },
            // Bell pattern.
            Track {
                events: vec![
                    Event {
                        time_beats: 0.0,
                        instrument_id: "bell".into(),
                        pitch_multiplier: 1.0,
                        volume: 0.9,
                        gate_beats: 1.0,
                    },
                    Event {
                        time_beats: 4.0,
                        instrument_id: "bell".into(),
                        pitch_multiplier: 1.5,
                        volume: 0.7,
                        gate_beats: 1.0,
                    },
                ],
            },
        ],
    }
}

#[test]
fn recipe_round_trips_through_json() {
    let original = realistic_recipe();
    let json = serde_json::to_string(&original).unwrap();
    let back: SequenceRecipe = serde_json::from_str(&json).unwrap();
    assert_eq!(back, original);
}

#[test]
fn recipe_round_trips_through_pretty_json() {
    let original = realistic_recipe();
    let pretty = serde_json::to_string_pretty(&original).unwrap();
    let back: SequenceRecipe = serde_json::from_str(&pretty).unwrap();
    assert_eq!(back, original);
}

#[test]
fn empty_recipe_round_trips() {
    let original = SequenceRecipe::default();
    let json = serde_json::to_string(&original).unwrap();
    let back: SequenceRecipe = serde_json::from_str(&json).unwrap();
    assert_eq!(back, original);
}

#[test]
fn loop_start_beats_optional_serialises_both_states() {
    // None form serialises with "loop_start_beats": null.
    let r = SequenceRecipe::default();
    let j = serde_json::to_string(&r).unwrap();
    assert!(j.contains("\"loop_start_beats\":null"));
    // Some form survives round-trip.
    let r2 = SequenceRecipe {
        loop_start_beats: Some(4.0),
        ..r
    };
    let back: SequenceRecipe = serde_json::from_str(&serde_json::to_string(&r2).unwrap()).unwrap();
    assert_eq!(back.loop_start_beats, Some(4.0));
}

#[test]
fn nested_audio_patches_survive_round_trip() {
    // Cross-check that the AudioPatch nested inside each Instrument
    // round-trips byte-for-byte — catches any accidental drift in the
    // schema when sequencer types are evolved.
    let original = realistic_recipe();
    let json = serde_json::to_string(&original).unwrap();
    let back: SequenceRecipe = serde_json::from_str(&json).unwrap();
    for (a, b) in original.instruments.iter().zip(back.instruments.iter()) {
        assert_eq!(a.id, b.id);
        assert_eq!(a.patch, b.patch);
    }
}
