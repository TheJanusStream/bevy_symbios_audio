//! End-to-end bake tests for the Phase 4 mixdown baker.
//!
//! The acceptance criterion from the ticket: one instrument (Square
//! 440 Hz here — a Square peaks at +1 from sample 0 which makes timing
//! assertions clean), 4 events on the beat at 120 BPM, total 2 beats →
//! sample at each event timestamp is near peak.  Plus determinism,
//! pitch-multiplier-shortens-duration, volume scaling, dangling
//! instrument refs are skipped not panicked, and crossfade tail
//! reservation.

use std::collections::BTreeMap;

use bevy_symbios_audio::{
    AudioPatch, Event, GraphNode, Instrument, NodeGraph, NodeId, NodeKind, SequenceRecipe, SineOsc,
    SquareOsc, Track, bake_sequence,
};

const SR: u32 = 44_100;

fn square_instrument() -> Instrument {
    Instrument {
        id: "square".into(),
        patch: AudioPatch {
            seed: 0,
            graph: NodeGraph {
                nodes: vec![GraphNode {
                    id: NodeId(0),
                    kind: NodeKind::Square(SquareOsc {
                        freq_hz: 440.0,
                        duty: 0.5,
                        amplitude: 1.0,
                    }),
                    inputs: BTreeMap::new(),
                }],
                output: NodeId(0),
            },
        },
    }
}

fn sine_instrument() -> Instrument {
    Instrument {
        id: "sine".into(),
        patch: AudioPatch {
            seed: 0,
            graph: NodeGraph {
                nodes: vec![GraphNode {
                    id: NodeId(0),
                    kind: NodeKind::Sine(SineOsc {
                        freq_hz: 440.0,
                        phase_offset: 0.0,
                        amplitude: 1.0,
                    }),
                    inputs: BTreeMap::new(),
                }],
                output: NodeId(0),
            },
        },
    }
}

fn event(time_beats: f32, instrument_id: &str, volume: f32, gate_beats: f32) -> Event {
    Event {
        time_beats,
        instrument_id: instrument_id.into(),
        pitch_multiplier: 1.0,
        volume,
        gate_beats,
    }
}

// --- Master length ----------------------------------------------------------

#[test]
fn master_length_matches_duration_at_120bpm() {
    let recipe = SequenceRecipe {
        bpm: 120.0,
        sample_rate: SR,
        duration_beats: 2.0,
        loop_start_beats: None,
        loop_crossfade_beats: 0.0,
        instruments: vec![square_instrument()],
        tracks: vec![],
    };
    // 2 beats at 120 BPM = 1 second = 44_100 samples.
    let master = bake_sequence(&recipe);
    assert_eq!(master.len(), SR as usize);
}

#[test]
fn master_reserves_crossfade_tail() {
    let recipe = SequenceRecipe {
        bpm: 120.0,
        sample_rate: SR,
        duration_beats: 2.0,
        loop_start_beats: None,
        loop_crossfade_beats: 1.0,
        instruments: vec![square_instrument()],
        tracks: vec![],
    };
    // 2 main beats + 1 crossfade-tail beat = 3 beats at 120 BPM = 1.5 s.
    let master = bake_sequence(&recipe);
    assert_eq!(master.len(), (SR as f32 * 1.5) as usize);
    // Tail is silent in #14 — #15 will fill it.
    assert!(master[SR as usize..].iter().all(|s| *s == 0.0));
}

// --- Event timing ----------------------------------------------------------

#[test]
fn four_square_events_on_the_beat_land_at_expected_offsets() {
    // 4 events spaced 0.5 beats apart at 120 BPM = real-time spacing
    // of 0.25 s → sample-offset spacing of 11_025 samples.  A 440 Hz
    // Square peaks at +1 at sample 0 of each event (phase 0 < duty),
    // so master[start] = +volume modulo tanh's mild compression.
    let recipe = SequenceRecipe {
        bpm: 120.0,
        sample_rate: SR,
        duration_beats: 2.0,
        loop_start_beats: None,
        loop_crossfade_beats: 0.0,
        instruments: vec![square_instrument()],
        tracks: vec![Track {
            events: vec![
                event(0.0, "square", 0.3, 0.25),
                event(0.5, "square", 0.3, 0.25),
                event(1.0, "square", 0.3, 0.25),
                event(1.5, "square", 0.3, 0.25),
            ],
        }],
    };
    let master = bake_sequence(&recipe);
    // tanh(0.3) ≈ 0.291 — assert within a comfortable tolerance.
    for (i, t_beats) in [0.0, 0.5, 1.0, 1.5].iter().enumerate() {
        let sample_idx = (*t_beats * 0.5 * SR as f32).round() as usize;
        let v = master[sample_idx];
        assert!(
            (v - 0.291).abs() < 0.02,
            "event {i} at sample {sample_idx}: got {v}, expected ~0.291"
        );
    }
}

#[test]
fn gaps_between_short_events_are_silent() {
    // 4 events spaced 0.5 beats with 0.25-beat gate → 0.125 s on, 0.125
    // s off pattern.  Samples mid-gap should be zero.
    let recipe = SequenceRecipe {
        bpm: 120.0,
        sample_rate: SR,
        duration_beats: 2.0,
        loop_start_beats: None,
        loop_crossfade_beats: 0.0,
        instruments: vec![square_instrument()],
        tracks: vec![Track {
            events: vec![
                event(0.0, "square", 0.3, 0.25),
                event(0.5, "square", 0.3, 0.25),
            ],
        }],
    };
    let master = bake_sequence(&recipe);
    // Sample in the gap between event 1 (ends ~sample 5512) and event
    // 2 (starts at 11025) — middle of gap is roughly sample 8000.
    let mid_gap = 8000;
    assert!(
        master[mid_gap].abs() < 1e-6,
        "gap not silent: {}",
        master[mid_gap]
    );
}

// --- Pitch and volume ------------------------------------------------------

#[test]
fn pitch_multiplier_of_two_halves_event_length() {
    // Event with gate=1.0 beat at 120 BPM = 0.5 s = 22_050 samples
    // when played at pitch=1.0.  At pitch=2.0 the resampled buffer
    // covers half that — 11_025 samples.  Past the halfway point the
    // event has finished, so the master should be silent there
    // (assuming no other events overlap).
    let recipe = SequenceRecipe {
        bpm: 120.0,
        sample_rate: SR,
        duration_beats: 1.0,
        loop_start_beats: None,
        loop_crossfade_beats: 0.0,
        instruments: vec![square_instrument()],
        tracks: vec![Track {
            events: vec![Event {
                time_beats: 0.0,
                instrument_id: "square".into(),
                pitch_multiplier: 2.0,
                volume: 0.3,
                gate_beats: 1.0,
            }],
        }],
    };
    let master = bake_sequence(&recipe);
    // First half should be non-silent...
    assert!(master[100].abs() > 1e-3);
    // ...and the back half should be silent.
    let three_quarters = (SR as f32 * 0.4) as usize; // 17_640, well past 11_025.
    assert!(
        master[three_quarters].abs() < 1e-6,
        "expected silence past pitch-shortened event, got {}",
        master[three_quarters]
    );
}

#[test]
fn volume_scaler_attenuates_output() {
    let mut recipe = SequenceRecipe {
        bpm: 120.0,
        sample_rate: SR,
        duration_beats: 1.0,
        loop_start_beats: None,
        loop_crossfade_beats: 0.0,
        instruments: vec![square_instrument()],
        tracks: vec![Track {
            events: vec![event(0.0, "square", 0.5, 1.0)],
        }],
    };
    let loud = bake_sequence(&recipe);
    recipe.tracks[0].events[0].volume = 0.1;
    let quiet = bake_sequence(&recipe);
    // Quiet RMS should be substantially smaller than loud RMS.
    let loud_rms: f64 =
        (loud.iter().map(|s| (*s as f64).powi(2)).sum::<f64>() / loud.len() as f64).sqrt();
    let quiet_rms: f64 =
        (quiet.iter().map(|s| (*s as f64).powi(2)).sum::<f64>() / quiet.len() as f64).sqrt();
    assert!(
        loud_rms > 3.0 * quiet_rms,
        "loud_rms={loud_rms}, quiet_rms={quiet_rms}"
    );
}

// --- Multi-instrument / bake cache -----------------------------------------

#[test]
fn multi_instrument_and_repeated_gate_lengths_bake_correctly() {
    // Two instruments + two events per instrument with matching gate
    // lengths — exercises the bake cache and ensures both contribute
    // to the master.
    let recipe = SequenceRecipe {
        bpm: 120.0,
        sample_rate: SR,
        duration_beats: 1.0,
        loop_start_beats: None,
        loop_crossfade_beats: 0.0,
        instruments: vec![square_instrument(), sine_instrument()],
        tracks: vec![
            Track {
                events: vec![
                    event(0.0, "square", 0.2, 0.25),
                    event(0.5, "square", 0.2, 0.25),
                ],
            },
            Track {
                events: vec![event(0.0, "sine", 0.2, 0.25), event(0.5, "sine", 0.2, 0.25)],
            },
        ],
    };
    let master = bake_sequence(&recipe);
    let rms =
        (master.iter().map(|s| (*s as f64).powi(2)).sum::<f64>() / master.len() as f64).sqrt();
    assert!(rms > 0.05, "multi-instrument mix too quiet: rms={rms}");
}

// --- Dangling references ---------------------------------------------------

#[test]
fn dangling_instrument_id_is_skipped_not_panicked() {
    // A typo in instrument_id must not crash bake_sequence — the
    // event silently drops out of the mix.
    let recipe = SequenceRecipe {
        bpm: 120.0,
        sample_rate: SR,
        duration_beats: 1.0,
        loop_start_beats: None,
        loop_crossfade_beats: 0.0,
        instruments: vec![square_instrument()],
        tracks: vec![Track {
            events: vec![event(0.0, "does-not-exist", 0.5, 0.5)],
        }],
    };
    let master = bake_sequence(&recipe);
    assert!(master.iter().all(|s| *s == 0.0));
}

// --- Determinism -----------------------------------------------------------

#[test]
fn bake_sequence_is_deterministic() {
    let recipe = SequenceRecipe {
        bpm: 120.0,
        sample_rate: SR,
        duration_beats: 1.0,
        loop_start_beats: None,
        loop_crossfade_beats: 0.0,
        instruments: vec![square_instrument()],
        tracks: vec![Track {
            events: vec![event(0.0, "square", 0.3, 0.5)],
        }],
    };
    let a = bake_sequence(&recipe);
    let b = bake_sequence(&recipe);
    assert_eq!(a, b);
}

// --- Soft clip ------------------------------------------------------------

#[test]
fn master_is_bounded_by_one_via_tanh() {
    // Stack many overlapping max-volume events — the unclamped sum
    // would blow past ±1, tanh must hold it inside the rail.
    let recipe = SequenceRecipe {
        bpm: 120.0,
        sample_rate: SR,
        duration_beats: 1.0,
        loop_start_beats: None,
        loop_crossfade_beats: 0.0,
        instruments: vec![square_instrument()],
        tracks: vec![Track {
            events: (0..8).map(|_| event(0.0, "square", 1.0, 0.5)).collect(),
        }],
    };
    let master = bake_sequence(&recipe);
    for s in &master {
        assert!(s.abs() <= 1.0, "tanh failed to bound: {s}");
    }
}
