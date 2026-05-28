//! End-to-end bake test for the ADSR envelope.
//!
//! Verifies that `bake()` correctly installs and threads through the
//! envelope's per-node state via the dispatch path defined on
//! `NodeKind::Adsr`.  A `Constant(1.0)` gate keeps the envelope held
//! through attack → decay → sustain so we can pin the ticket's three
//! gated-portion sample-index checks (`sample[4410] ≈ 1.0`,
//! `sample[8820] ≈ 0.5`, `sample[44100] ≈ 0.5`).
//!
//! The release phase requires a gate that goes low mid-bake, which Phase 1
//! has no node to express (gate-source generators arrive with the
//! sequencer in Phase 4 #13).  The release path is covered exhaustively
//! by the unit tests in `src/adsr.rs` that drive `AdsrEnvelope::sample`
//! directly with controlled gate transitions.

use std::collections::BTreeMap;

use bevy_symbios_audio::{
    AdsrCurve, AdsrEnvelope, AudioPatch, Connection, GraphNode, NodeGraph, NodeId, NodeKind, bake,
};

fn linear_test_envelope() -> AdsrEnvelope {
    AdsrEnvelope {
        attack_s: 0.1,
        decay_s: 0.1,
        sustain_level: 0.5,
        release_s: 0.1,
        curve: AdsrCurve::Linear,
    }
}

fn gated_envelope_patch(env: AdsrEnvelope) -> AudioPatch {
    let mut inputs = BTreeMap::new();
    inputs.insert("gate".to_string(), Connection::constant(1.0));
    AudioPatch {
        seed: 0,
        graph: NodeGraph {
            nodes: vec![GraphNode {
                id: NodeId(0),
                kind: NodeKind::Adsr(env),
                inputs,
            }],
            output: NodeId(0),
        },
    }
}

#[test]
fn baked_linear_envelope_hits_attack_decay_sustain_pins() {
    // Mirrors the ticket's acceptance criteria for the gated portion of
    // the envelope.  Tolerance is loose enough that f32 ramp quantization
    // can't flap the result.
    let p = gated_envelope_patch(linear_test_envelope());
    // 1 s + 0.1 s buffer so the test isn't running right up against the
    // tail.
    let buf = bake(&p, 44_100, 1.1);

    assert!(
        (buf[4_410] - 1.0).abs() < 1e-2,
        "end of attack: {}",
        buf[4_410]
    );
    assert!(
        (buf[8_820] - 0.5).abs() < 1e-2,
        "end of decay (start of sustain): {}",
        buf[8_820]
    );
    assert!(
        (buf[44_100] - 0.5).abs() < 1e-2,
        "deep in sustain: {}",
        buf[44_100]
    );
}

#[test]
fn baked_envelope_first_sample_is_zero() {
    // With a constant 1.0 gate, the rising edge fires on sample 0 — that
    // sample is the very first of the attack ramp, so output is 0.
    let p = gated_envelope_patch(linear_test_envelope());
    let buf = bake(&p, 44_100, 0.01);
    assert_eq!(buf[0], 0.0);
}

#[test]
fn baked_envelope_is_deterministic() {
    let p = gated_envelope_patch(linear_test_envelope());
    let a = bake(&p, 44_100, 0.5);
    let b = bake(&p, 44_100, 0.5);
    assert_eq!(a, b);
}

#[test]
fn baked_exponential_envelope_has_curved_attack_midpoint() {
    // Cross-check that AdsrCurve::Exponential survives the bake() path
    // and produces the ease-out shape — value at the attack midpoint
    // (sample 2205, α=0.5) should be 0.75, not 0.5.
    let env = AdsrEnvelope {
        curve: AdsrCurve::Exponential,
        ..linear_test_envelope()
    };
    let p = gated_envelope_patch(env);
    let buf = bake(&p, 44_100, 0.5);
    assert!(
        buf[2_205] > 0.7 && buf[2_205] < 0.8,
        "exp midpoint: {}",
        buf[2_205]
    );
}

#[test]
fn adsr_round_trips_through_json() {
    let kind = NodeKind::Adsr(AdsrEnvelope {
        attack_s: 0.123,
        decay_s: 0.456,
        sustain_level: 0.789,
        release_s: 0.321,
        curve: AdsrCurve::Exponential,
    });
    let p = gated_envelope_patch(match kind.clone() {
        NodeKind::Adsr(e) => e,
        _ => unreachable!(),
    });
    let json = serde_json::to_string(&p).unwrap();
    let back: AudioPatch = serde_json::from_str(&json).unwrap();
    assert_eq!(back, p);
}
