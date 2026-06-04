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
    AudioPatch, Event, GraphNode, Instrument, NodeGraph, NodeId, NodeKind, PitchMode,
    SequenceRecipe, SineOsc, SquareOsc, Track, bake_sequence,
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
                        ..Default::default()
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
        release_beats: 0.0,
        ..Default::default()
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
                release_beats: 0.0,
                ..Default::default()
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

/// Index of the last sample louder than `eps` — i.e. how far into the
/// master the event's audio actually reaches.
fn last_nonsilent(buf: &[f32], eps: f32) -> usize {
    buf.iter().rposition(|s| s.abs() > eps).unwrap_or(0)
}

/// Count zero crossings (sign changes) — a cheap, FFT-free pitch proxy.
fn zero_crossings(buf: &[f32]) -> usize {
    buf.windows(2)
        .filter(|w| (w[0] < 0.0) != (w[1] < 0.0))
        .count()
}

fn single_sine_event(pitch_multiplier: f32, pitch_mode: PitchMode) -> SequenceRecipe {
    // gate 1.0 beat @ 120 BPM = 0.5 s = 22_050 samples; no release.
    SequenceRecipe {
        bpm: 120.0,
        sample_rate: SR,
        duration_beats: 1.0,
        loop_start_beats: None,
        loop_crossfade_beats: 0.0,
        instruments: vec![sine_instrument()],
        tracks: vec![Track {
            events: vec![Event {
                time_beats: 0.0,
                instrument_id: "sine".into(),
                pitch_multiplier,
                volume: 0.8,
                gate_beats: 1.0,
                release_beats: 0.0,
                pitch_mode,
            }],
        }],
    }
}

#[test]
fn time_preserving_pitch_keeps_event_length_independent_of_pitch() {
    // The headline fix: a pitch-up event under TimePreserving fills its
    // whole gate slot, whereas the same multiplier under Varispeed (the
    // resample path) finishes in half the time.
    let master_len = bake_sequence(&single_sine_event(1.0, PitchMode::Varispeed)).len();

    let varispeed = bake_sequence(&single_sine_event(2.0, PitchMode::Varispeed));
    let preserving = bake_sequence(&single_sine_event(2.0, PitchMode::TimePreserving));

    let vari_reach = last_nonsilent(&varispeed, 1e-4);
    let pres_reach = last_nonsilent(&preserving, 1e-4);

    // Varispeed pitch-2.0 finishes around the halfway mark...
    assert!(
        vari_reach < master_len * 6 / 10,
        "varispeed reach {vari_reach} should be ~half of {master_len}"
    );
    // ...while TimePreserving runs essentially to the end of the gate.
    assert!(
        pres_reach > master_len * 9 / 10,
        "time-preserving reach {pres_reach} should fill {master_len}"
    );
}

#[test]
fn time_preserving_pitch_actually_raises_the_pitch() {
    // Retuning the oscillator must double the frequency at pitch 2.0 —
    // verified by a doubling of zero crossings over an identical-length
    // buffer (both TimePreserving, same gate).
    let unison = bake_sequence(&single_sine_event(1.0, PitchMode::TimePreserving));
    let octave = bake_sequence(&single_sine_event(2.0, PitchMode::TimePreserving));
    assert_eq!(
        unison.len(),
        octave.len(),
        "lengths must match for a fair count"
    );

    let base = zero_crossings(&unison) as f32;
    let up = zero_crossings(&octave) as f32;
    let ratio = up / base;
    assert!(
        (1.8..=2.2).contains(&ratio),
        "expected ~2x crossings at octave up, got {ratio} ({base} → {up})"
    );
}

#[test]
fn time_preserving_non_positive_pitch_is_skipped_not_panicked() {
    // A nonsense pitch must be dropped (like resample_linear's guard), not
    // crash the mixdown or emit garbage.
    let recipe = single_sine_event(0.0, PitchMode::TimePreserving);
    let master = bake_sequence(&recipe);
    assert_eq!(
        master.len(),
        bake_sequence(&single_sine_event(1.0, PitchMode::Varispeed)).len()
    );
    assert!(
        master.iter().all(|s| s.abs() < 1e-6),
        "skipped event should leave the master silent"
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

#[test]
fn invalid_instrument_graph_is_skipped_not_panicked() {
    // An instrument whose graph is structurally invalid (output points at
    // a missing node) must be skipped via try_bake's Result rather than
    // panicking the whole mixdown.
    let broken = Instrument {
        id: "broken".into(),
        patch: AudioPatch {
            seed: 0,
            graph: NodeGraph {
                nodes: vec![GraphNode {
                    id: NodeId(0),
                    kind: NodeKind::Square(SquareOsc {
                        freq_hz: 440.0,
                        duty: 0.5,
                        amplitude: 1.0,
                        ..Default::default()
                    }),
                    inputs: BTreeMap::new(),
                }],
                output: NodeId(99), // missing node → GraphError::MissingOutput
            },
        },
    };
    let recipe = SequenceRecipe {
        bpm: 120.0,
        sample_rate: SR,
        duration_beats: 1.0,
        loop_start_beats: None,
        loop_crossfade_beats: 0.0,
        instruments: vec![broken],
        tracks: vec![Track {
            events: vec![event(0.0, "broken", 0.5, 0.5)],
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
