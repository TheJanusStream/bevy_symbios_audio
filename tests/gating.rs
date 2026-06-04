//! End-to-end test for sequencer-driven gating (#22).
//!
//! A [`Gate`] node feeds an [`AdsrEnvelope`], which modulates a sine's
//! amplitude.  The mixdown holds the gate open for an event's `gate_beats`
//! and then bakes `release_beats` of tail, so the envelope must:
//!   1. be loud while the gate is open (attack → sustain), and
//!   2. fall to silence *during* the tail, because the gate closed and the
//!      release stage ran — not merely because the buffer ended.
//!
//! The second point is what separates the fixed behaviour from the old
//! bug, where `gate_beats` only trimmed the bake and the note was cut off
//! at sustain with no release.

use std::collections::BTreeMap;

use bevy_symbios_audio::{
    AdsrCurve, AdsrEnvelope, AudioPatch, Connection, Event, Gate, GraphNode, Instrument, NodeGraph,
    NodeId, NodeKind, SequenceRecipe, SineOsc, Track, bake_sequence,
};

const SR: u32 = 44_100;

/// Gate(2) → ADSR(0).gate ; ADSR(0) → Sine(1).amplitude.  Short attack /
/// decay, 0.8 sustain, 0.1 s release.
fn gated_voice() -> Instrument {
    let mut adsr_inputs = BTreeMap::new();
    adsr_inputs.insert("gate".to_string(), vec![Connection::from_node(NodeId(2))]);
    let mut sine_inputs = BTreeMap::new();
    sine_inputs.insert(
        "amplitude".to_string(),
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
                            attack_s: 0.005,
                            decay_s: 0.005,
                            sustain_level: 0.8,
                            release_s: 0.1,
                            curve: AdsrCurve::Linear,
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
                        inputs: BTreeMap::new(),
                    },
                ],
                output: NodeId(1),
            },
        },
    }
}

/// Peak `|sample|` over `[start, end)`.
fn peak(buf: &[f32], start: usize, end: usize) -> f32 {
    buf[start..end.min(buf.len())]
        .iter()
        .fold(0.0_f32, |m, s| m.max(s.abs()))
}

fn one_note_recipe(gate_beats: f32, release_beats: f32) -> SequenceRecipe {
    SequenceRecipe {
        bpm: 60.0, // 1 beat = 1 s, so beats map straight to seconds
        sample_rate: SR,
        duration_beats: 1.0,
        loop_start_beats: None,
        loop_crossfade_beats: 0.0,
        instruments: vec![gated_voice()],
        tracks: vec![Track {
            events: vec![Event {
                time_beats: 0.0,
                instrument_id: "voice".into(),
                pitch_multiplier: 1.0,
                volume: 1.0,
                gate_beats,
                release_beats,
                ..Default::default()
            }],
        }],
    }
}

#[test]
fn gated_note_releases_to_silence_within_the_tail() {
    // Gate open 0.2 s (→ sample 8_820), then a 0.3 s tail.  The 0.1 s
    // release finishes ~sample 13_230, well before the event buffer ends
    // at 0.5 s (sample 22_050).
    let master = bake_sequence(&one_note_recipe(0.2, 0.3));

    let gate_close = (0.2 * SR as f32) as usize; // 8_820
    let release_done = gate_close + (0.1 * SR as f32) as usize; // 13_230

    // 1. Loud while the gate is open (sustain ≈ 0.8 → tanh ≈ 0.66).
    let sustain_peak = peak(&master, 3_000, 7_000);
    assert!(
        sustain_peak > 0.3,
        "voice should sound during the gate, peak={sustain_peak}"
    );

    // 2. Silent after the release completes — but still inside the baked
    //    tail.  This only holds if the gate CLOSED and the release ran; a
    //    stuck-open gate would keep sustaining here.
    let post_release_peak = peak(&master, release_done + 1_000, 21_000);
    assert!(
        post_release_peak < 0.02,
        "release should have brought the voice to silence, peak={post_release_peak}"
    );

    // 3. And the release is a genuine decay, not an instant cut: midway
    //    through the release window the voice is quieter than at sustain
    //    but not yet silent.
    let mid_release = peak(&master, gate_close + 1_000, gate_close + 3_000);
    assert!(
        mid_release < sustain_peak && mid_release > post_release_peak,
        "expected a decaying release ramp: sustain={sustain_peak}, \
         mid_release={mid_release}, post={post_release_peak}"
    );
}

#[test]
fn zero_release_reproduces_hard_cutoff() {
    // release_beats: 0 → the bake is exactly the gate length and the note
    // is cut at sustain (the pre-#22 behaviour), so the buffer's content
    // ends at the gate boundary still loud.
    let master = bake_sequence(&one_note_recipe(0.2, 0.0));
    let gate_close = (0.2 * SR as f32) as usize;
    // Just inside the gate the voice is loud...
    assert!(peak(&master, gate_close - 2_000, gate_close - 1) > 0.3);
    // ...and there is no baked content past the gate (the rest of the
    // 1-beat master is the empty region after the event buffer).
    assert!(peak(&master, gate_close + 100, master.len()) < 0.02);
}
