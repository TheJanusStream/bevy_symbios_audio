//! End-to-end test for the Phase 4 seamless-loop tail-crossfade.
//!
//! The contract: when a recipe's `loop_start_beats` is set, the
//! mixdown baker truncates the output to exactly `duration_beats *
//! beat_secs * sample_rate` samples and pre-mixes a crossfade so the
//! seam between the last sample and `master[loop_start_sample]` has
//! no discontinuity.

use std::collections::BTreeMap;

use bevy_symbios_audio::{
    AdsrCurve, AdsrEnvelope, AudioPatch, Connection, Event, Gate, GraphNode, Instrument, NodeGraph,
    NodeId, NodeKind, SequenceRecipe, SineOsc, Track, bake_sequence,
};

const SR: u32 = 44_100;
const BPM: f32 = 120.0;

/// Gate-driven, ADSR-gated sine instrument.  A [`Gate`] node feeds the
/// envelope, so when an event's gate closes the ADSR's long release
/// (0.5 s) rings out across the event's `release_beats` tail — and an
/// event placed near `duration_beats` puts that release into the
/// crossfade region, exactly what the seamless loop folds back.
fn ringing_sine_instrument() -> Instrument {
    let mut adsr_inputs = BTreeMap::new();
    adsr_inputs.insert("gate".to_string(), vec![Connection::from_node(NodeId(2))]);
    let mut sine_inputs = BTreeMap::new();
    sine_inputs.insert(
        "amplitude".to_string(),
        vec![Connection::modulation(NodeId(0), 1.0)],
    );
    Instrument {
        id: "bell".into(),
        patch: AudioPatch {
            seed: 0,
            graph: NodeGraph {
                nodes: vec![
                    GraphNode {
                        id: NodeId(0),
                        kind: NodeKind::Adsr(AdsrEnvelope {
                            attack_s: 0.01,
                            decay_s: 0.05,
                            sustain_level: 0.8,
                            release_s: 0.5,
                            curve: AdsrCurve::Linear,
                        }),
                        inputs: adsr_inputs,
                    },
                    GraphNode {
                        id: NodeId(1),
                        kind: NodeKind::Sine(SineOsc {
                            // 219 Hz, not 220.  At 44.1 kHz a 220 Hz
                            // sine produces an integer-cycle-per-100-
                            // samples relationship that makes the
                            // event coincidentally zero-cross right at
                            // the seam in both crossfaded and
                            // non-crossfaded bakes — defeating the
                            // test.  Detuning by 1 Hz breaks the
                            // resonance and gives the seam click a
                            // proper magnitude to compare against.
                            freq_hz: 219.0,
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

fn looping_ringing_recipe(loop_crossfade_beats: f32) -> SequenceRecipe {
    SequenceRecipe {
        bpm: BPM,
        sample_rate: SR,
        duration_beats: 4.0,
        loop_start_beats: Some(0.0),
        loop_crossfade_beats,
        instruments: vec![ringing_sine_instrument()],
        tracks: vec![Track {
            events: vec![
                // First event at the start so the loop region has
                // actual content to crossfade against.
                Event {
                    time_beats: 0.0,
                    instrument_id: "bell".into(),
                    pitch_multiplier: 1.0,
                    volume: 0.5,
                    gate_beats: 0.5,
                    release_beats: 0.5,
                    ..Default::default()
                },
                // Trailing event whose ring-out extends past
                // duration_beats — the gate closes at beat 3.75 and the
                // 0.75-beat release tail rings through beat 4.5, so the
                // release lives in the crossfade window the loop folds
                // back onto loop start.
                Event {
                    time_beats: 3.5,
                    instrument_id: "bell".into(),
                    pitch_multiplier: 1.0,
                    volume: 0.5,
                    gate_beats: 0.25,
                    release_beats: 0.75,
                    ..Default::default()
                },
            ],
        }],
    }
}

// --- Buffer shape ----------------------------------------------------------

#[test]
fn looping_recipe_truncates_buffer_to_main_samples() {
    let recipe = looping_ringing_recipe(0.5);
    let master = bake_sequence(&recipe);
    // 4 beats × 0.5 s/beat × 44_100 Hz = 88_200 samples.  No tail.
    assert_eq!(master.len(), 88_200);
}

#[test]
fn looping_recipe_with_zero_crossfade_still_truncates() {
    let recipe = looping_ringing_recipe(0.0);
    let master = bake_sequence(&recipe);
    assert_eq!(master.len(), 88_200);
}

#[test]
fn non_looping_recipe_keeps_crossfade_tail() {
    // No loop_start_beats → fall back to #14 behaviour: master_len =
    // duration + tail.  The tail can carry overhang from late events.
    let mut recipe = looping_ringing_recipe(0.5);
    recipe.loop_start_beats = None;
    let master = bake_sequence(&recipe);
    // 4 beats main + 0.5 beats tail = 4.5 beats × 0.5 s × 44_100 = 99_225.
    assert_eq!(master.len(), 99_225);
}

// --- Seam continuity -------------------------------------------------------

#[test]
fn loop_seam_has_no_audible_click() {
    // Without crossfade: |buf[last] - buf[loop_start]| is large because
    // the timeline ends mid-release while loop_start sees a fresh
    // attack ramp.  With crossfade, buf[loop_start] gets blended with
    // the "next sample after duration_beats" from the tail, so the
    // step between buf[last] and buf[loop_start] is the continuous
    // event's per-sample delta — tiny.
    let with_xf = bake_sequence(&looping_ringing_recipe(0.5));
    let without_xf = bake_sequence(&looping_ringing_recipe(0.0));

    let main_samples = with_xf.len();
    assert_eq!(main_samples, without_xf.len());

    let loop_start = 0usize;
    let last = main_samples - 1;

    let click_with = (with_xf[last] - with_xf[loop_start]).abs();
    let click_without = (without_xf[last] - without_xf[loop_start]).abs();

    // The crossfaded seam should be at least 5× smaller than the
    // raw seam — generous so this doesn't flap on f32 rounding.
    assert!(
        click_with * 5.0 < click_without,
        "crossfade didn't reduce seam click: with={click_with}, without={click_without}"
    );
}

#[test]
fn crossfade_modifies_loop_start_region() {
    // The non-crossfaded master and the crossfaded one differ in
    // exactly the loop_start..(loop_start+crossfade_samples) window.
    // Outside that window the buffers are byte-identical.
    let with_xf = bake_sequence(&looping_ringing_recipe(0.5));
    let without_xf = bake_sequence(&looping_ringing_recipe(0.0));

    // Crossfade window in samples = 0.5 beats × 0.5 s × 44_100 = 11_025.
    let crossfade_samples = 11_025;
    // Inside the crossfade window: buffers must differ somewhere (the
    // trailing event's release contributes via the tail).
    let mut differed = false;
    for i in 0..crossfade_samples {
        if (with_xf[i] - without_xf[i]).abs() > 1e-6 {
            differed = true;
            break;
        }
    }
    assert!(
        differed,
        "loop region unchanged after crossfade — tail wasn't overlaid"
    );

    // Outside the window (sample crossfade_samples onward), buffers
    // must match.
    for i in crossfade_samples..with_xf.len() {
        assert!(
            (with_xf[i] - without_xf[i]).abs() < 1e-6,
            "buffer differs outside crossfade window at sample {i}"
        );
    }
}

// --- Determinism + tanh bound --------------------------------------------

#[test]
fn looping_bake_is_deterministic() {
    let recipe = looping_ringing_recipe(0.5);
    let a = bake_sequence(&recipe);
    let b = bake_sequence(&recipe);
    assert_eq!(a, b);
}

#[test]
fn crossfaded_loop_region_stays_inside_unit_interval() {
    let recipe = looping_ringing_recipe(0.5);
    let master = bake_sequence(&recipe);
    for s in &master {
        assert!(
            s.abs() <= 1.0,
            "sample escaped [-1, 1] after crossfade: {s}"
        );
    }
}

// --- loop_start past duration handled gracefully -------------------------

#[test]
fn loop_start_past_duration_does_not_panic() {
    let mut recipe = looping_ringing_recipe(0.5);
    recipe.loop_start_beats = Some(99.0); // past 4-beat duration
    let master = bake_sequence(&recipe);
    // Still truncated to main_samples; no crossfade was applied.
    assert_eq!(master.len(), 88_200);
}
