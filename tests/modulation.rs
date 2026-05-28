//! End-to-end modulation routing tests.
//!
//! Validates the cross-node parameter modulation system from Phase 2
//! ticket #8: an upstream node's output wired to a downstream node's
//! named modulation port, scaled by `Connection.amount`, added to the
//! downstream node's configured base value at sample time.
//!
//! Acceptance for this ticket: the "wind" example.  Brown noise into a
//! lowpass whose cutoff is swept by a slow sine LFO produces a sound that
//! breathes — measurable here as time-varying high-frequency energy.

use std::collections::BTreeMap;

use bevy_symbios_audio::{
    AudioPatch, BiquadLowpass, BrownNoise, Connection, GraphNode, Lfo, LfoShape, NodeGraph, NodeId,
    NodeKind, SineOsc, bake,
};

const SR: u32 = 44_100;

fn rms(buf: &[f32]) -> f32 {
    let sum_sq: f64 = buf.iter().map(|s| (*s as f64) * (*s as f64)).sum();
    (sum_sq / buf.len() as f64).sqrt() as f32
}

// --- Wind acceptance --------------------------------------------------------

/// Build the wind patch: BrownNoise → LP, with LP cutoff modulated by a
/// 0.3 Hz sine LFO sweeping ±900 around base 1100 (so 200..2000 Hz).
fn wind_patch() -> AudioPatch {
    let mut lp_inputs = BTreeMap::new();
    lp_inputs.insert("in".to_string(), Connection::from_node(NodeId(1)));
    // Connection amount is 1.0 here; the depth/offset live on the LFO so
    // the modulation range is fully described by the modulator.
    lp_inputs.insert(
        "cutoff_hz".to_string(),
        Connection::modulation(NodeId(0), 1.0),
    );
    AudioPatch {
        seed: 0xABCD,
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
                    inputs: BTreeMap::new(),
                },
                GraphNode {
                    id: NodeId(1),
                    kind: NodeKind::BrownNoise(BrownNoise { amplitude: 0.5 }),
                    inputs: BTreeMap::new(),
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

#[test]
fn wind_patch_produces_non_silent_output() {
    let p = wind_patch();
    let buf = bake(&p, SR, 4.0);
    let r = rms(&buf);
    assert!(r > 0.001, "wind is silent: rms={r}");
}

#[test]
fn wind_patch_is_deterministic_per_seed() {
    let p = wind_patch();
    let a = bake(&p, SR, 1.0);
    let b = bake(&p, SR, 1.0);
    assert_eq!(a, b);
}

#[test]
fn wind_patch_has_time_varying_spectral_content() {
    // A swept LP cutoff produces noticeable high-frequency energy
    // variation when the filter is "open" (high cutoff) vs "closed"
    // (low cutoff).  Plain RMS barely moves on brown noise (most of the
    // energy is in the lows that always pass), but the first-difference
    // RMS — a coarse proxy for high-frequency content — varies
    // strongly with the cutoff.  Split a ~6.7-second bake (two full
    // 0.3 Hz cycles) into 200 ms windows and check the diff-RMS ratio.
    fn diff_rms(buf: &[f32]) -> f32 {
        let mut acc = 0.0_f64;
        for w in buf.windows(2) {
            let d = (w[1] - w[0]) as f64;
            acc += d * d;
        }
        ((acc / buf.len() as f64).sqrt()) as f32
    }
    let p = wind_patch();
    let buf = bake(&p, SR, 6.7);
    let window = (SR as f32 * 0.2) as usize;
    let mut diffs = Vec::new();
    let mut i = 0;
    while i + window < buf.len() {
        diffs.push(diff_rms(&buf[i..i + window]));
        i += window;
    }
    let max = diffs.iter().cloned().fold(f32::MIN, f32::max);
    let min = diffs.iter().cloned().fold(f32::MAX, f32::min);
    assert!(
        max / min > 2.0,
        "expected ≥2:1 diff_rms variation across windows, got {}",
        max / min
    );
}

// --- FM (oscillator → oscillator.freq) -------------------------------------

#[test]
fn fm_sine_to_sine_changes_spectral_content() {
    // Sine modulator at 100 Hz drives a sine carrier at 440 Hz, with the
    // amount large enough (200 Hz) to produce audible sidebands.  The
    // baked output should have noticeably more energy at frequencies
    // away from 440 Hz than an unmodulated reference.
    let modulator = NodeKind::Sine(SineOsc {
        freq_hz: 100.0,
        phase_offset: 0.0,
        amplitude: 1.0,
    });
    let carrier = NodeKind::Sine(SineOsc {
        freq_hz: 440.0,
        phase_offset: 0.0,
        amplitude: 1.0,
    });
    let mut carrier_inputs = BTreeMap::new();
    carrier_inputs.insert("freq".to_string(), Connection::modulation(NodeId(0), 200.0));
    let fm_patch = AudioPatch {
        seed: 0,
        graph: NodeGraph {
            nodes: vec![
                GraphNode {
                    id: NodeId(0),
                    kind: modulator,
                    inputs: BTreeMap::new(),
                },
                GraphNode {
                    id: NodeId(1),
                    kind: carrier,
                    inputs: carrier_inputs,
                },
            ],
            output: NodeId(1),
        },
    };
    let buf = bake(&fm_patch, SR, 0.5);
    // The FM signal must be non-silent and bounded.
    let r = rms(&buf);
    let max = buf.iter().fold(0.0_f32, |a, b| a.max(b.abs()));
    assert!(r > 0.1, "FM signal too quiet: rms={r}");
    assert!(max <= 1.0 + 1e-3, "FM signal clipped: max={max}");

    // Cross-check: an unmodulated 440 Hz reference has a smaller
    // first-difference-RMS (a coarse high-freq energy proxy) than the
    // FM version, since the sidebands add high-frequency content.
    let plain = AudioPatch {
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
    };
    let plain_buf = bake(&plain, SR, 0.5);
    fn diff_rms(buf: &[f32]) -> f32 {
        let mut acc = 0.0_f64;
        for w in buf.windows(2) {
            let d = (w[1] - w[0]) as f64;
            acc += d * d;
        }
        ((acc / buf.len() as f64).sqrt()) as f32
    }
    let plain_d = diff_rms(&plain_buf);
    let fm_d = diff_rms(&buf);
    assert!(
        fm_d > plain_d,
        "FM diff_rms {fm_d} should exceed plain {plain_d}"
    );
}

// --- AM (LFO → oscillator.amplitude) ---------------------------------------

#[test]
fn am_lfo_into_sine_amplitude_creates_tremolo() {
    // 4 Hz tremolo on a 440 Hz sine with base amplitude 0.0 and depth
    // 0.5 means the sine's amplitude swings between -0.5 and +0.5
    // (the LFO is bipolar).  The envelope of the output should rise and
    // fall four times per second.
    let lfo = NodeKind::Lfo(Lfo {
        rate_hz: 4.0,
        shape: LfoShape::Sine,
        depth: 0.5,
        offset: 0.0,
    });
    let carrier = NodeKind::Sine(SineOsc {
        freq_hz: 440.0,
        phase_offset: 0.0,
        amplitude: 0.0,
    });
    let mut carrier_inputs = BTreeMap::new();
    carrier_inputs.insert(
        "amplitude".to_string(),
        Connection::modulation(NodeId(0), 1.0),
    );
    let am_patch = AudioPatch {
        seed: 0,
        graph: NodeGraph {
            nodes: vec![
                GraphNode {
                    id: NodeId(0),
                    kind: lfo,
                    inputs: BTreeMap::new(),
                },
                GraphNode {
                    id: NodeId(1),
                    kind: carrier,
                    inputs: carrier_inputs,
                },
            ],
            output: NodeId(1),
        },
    };
    let buf = bake(&am_patch, SR, 0.5);
    let r = rms(&buf);
    let max = buf.iter().fold(0.0_f32, |a, b| a.max(b.abs()));
    // Some sound is produced (the carrier isn't muted) and amplitude
    // stays bounded by the LFO depth.
    assert!(r > 0.05, "AM signal too quiet: rms={r}");
    assert!(max < 0.6, "AM signal exceeded depth ±0.5: max={max}");
}

// --- Schema round-trip -----------------------------------------------------

#[test]
fn modulation_connection_round_trips_through_json() {
    let p = wind_patch();
    let json = serde_json::to_string_pretty(&p).unwrap();
    let back: AudioPatch = serde_json::from_str(&json).unwrap();
    assert_eq!(back, p);
    // The serialised form must include the modulation amount.
    assert!(json.contains("\"amount\""));
}
