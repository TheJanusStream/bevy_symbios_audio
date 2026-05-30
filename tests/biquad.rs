//! End-to-end bake tests for the three biquad filters.
//!
//! The acceptance criterion: a lowpass with cutoff 1 kHz must attenuate a
//! sine at 8 kHz by more than 24 dB relative to a sine at 200 Hz.  We
//! verify this by baking the same patch shape twice — once with a sine
//! source at each test frequency — and comparing the filtered RMS.
//!
//! The patch graph is `Sine → BiquadLowpass(in)` etc.  This exercises the
//! full bake dispatch path: per-node state initialised via
//! `Node::init_state`, per-sample input resolution via the BTreeMap-based
//! connection map, the filter pulling its input from the `"in"` port.

use std::collections::BTreeMap;

use bevy_symbios_audio::{
    AudioPatch, BiquadBandpass, BiquadHighpass, BiquadLowpass, Connection, GraphNode, NodeGraph,
    NodeId, NodeKind, SineOsc, bake,
};

const SR: u32 = 44_100;
const SECS: f32 = 0.5;
/// Skip the start of the bake to let the filter's state settle.
const SETTLE: usize = 4_410;

fn rms(buf: &[f32]) -> f32 {
    let sum_sq: f64 = buf.iter().map(|s| (*s as f64) * (*s as f64)).sum();
    (sum_sq / buf.len() as f64).sqrt() as f32
}

/// Two-node patch: `Sine(freq) → filter(in)` with `filter` as the output.
fn sine_through(freq: f32, filter_kind: NodeKind) -> AudioPatch {
    let mut filter_inputs = BTreeMap::new();
    filter_inputs.insert("in".to_string(), vec![Connection::from_node(NodeId(0))]);
    AudioPatch {
        seed: 0,
        graph: NodeGraph {
            nodes: vec![
                GraphNode {
                    id: NodeId(0),
                    kind: NodeKind::Sine(SineOsc {
                        freq_hz: freq,
                        phase_offset: 0.0,
                        amplitude: 1.0,
                    }),
                    inputs: BTreeMap::new(),
                },
                GraphNode {
                    id: NodeId(1),
                    kind: filter_kind,
                    inputs: filter_inputs,
                },
            ],
            output: NodeId(1),
        },
    }
}

// --- Lowpass ----------------------------------------------------------------

#[test]
fn lowpass_at_1khz_drops_8khz_relative_to_200hz_by_more_than_24db() {
    let lp = || {
        NodeKind::BiquadLowpass(BiquadLowpass {
            cutoff_hz: 1_000.0,
            q: std::f32::consts::FRAC_1_SQRT_2,
        })
    };
    let buf_low = bake(&sine_through(200.0, lp()), SR, SECS);
    let buf_high = bake(&sine_through(8_000.0, lp()), SR, SECS);
    let r_low = rms(&buf_low[SETTLE..]);
    let r_high = rms(&buf_high[SETTLE..]);
    let atten_db = 20.0 * (r_low / r_high).log10();
    assert!(
        atten_db > 24.0,
        "LP attenuation 200Hz/8kHz: {atten_db} dB (need >24)"
    );
}

#[test]
fn lowpass_passes_below_cutoff_with_low_loss() {
    let lp = NodeKind::BiquadLowpass(BiquadLowpass {
        cutoff_hz: 5_000.0,
        q: std::f32::consts::FRAC_1_SQRT_2,
    });
    let buf = bake(&sine_through(200.0, lp), SR, SECS);
    let r = rms(&buf[SETTLE..]);
    // A 200 Hz sine well below a 5 kHz cutoff should retain most of its
    // 0.707 RMS amplitude.
    assert!(r > 0.6, "LP passband loss too high: rms={r}");
}

// --- Highpass ---------------------------------------------------------------

#[test]
fn highpass_at_1khz_drops_200hz_relative_to_8khz_by_more_than_24db() {
    let hp = || {
        NodeKind::BiquadHighpass(BiquadHighpass {
            cutoff_hz: 1_000.0,
            q: std::f32::consts::FRAC_1_SQRT_2,
        })
    };
    let buf_low = bake(&sine_through(200.0, hp()), SR, SECS);
    let buf_high = bake(&sine_through(8_000.0, hp()), SR, SECS);
    let r_low = rms(&buf_low[SETTLE..]);
    let r_high = rms(&buf_high[SETTLE..]);
    let atten_db = 20.0 * (r_high / r_low).log10();
    assert!(
        atten_db > 24.0,
        "HP attenuation 200Hz/8kHz: {atten_db} dB (need >24)"
    );
}

// --- Bandpass ---------------------------------------------------------------

#[test]
fn bandpass_peaks_at_centre_versus_far_band() {
    let bp = || {
        NodeKind::BiquadBandpass(BiquadBandpass {
            center_hz: 1_000.0,
            q: 4.0,
        })
    };
    let buf_centre = bake(&sine_through(1_000.0, bp()), SR, SECS);
    let buf_far = bake(&sine_through(100.0, bp()), SR, SECS);
    let r_c = rms(&buf_centre[SETTLE..]);
    let r_f = rms(&buf_far[SETTLE..]);
    assert!(r_c > r_f, "BP centre {r_c} should exceed far-band {r_f}");
    // High-Q bandpass should put the centre at least 12 dB above an
    // octave-out signal.
    let suppression_db = 20.0 * (r_c / r_f).log10();
    assert!(
        suppression_db > 12.0,
        "BP suppression too low: {suppression_db} dB"
    );
}

// --- Determinism + round-trip ----------------------------------------------

#[test]
fn baked_filter_is_deterministic() {
    let p = sine_through(
        440.0,
        NodeKind::BiquadLowpass(BiquadLowpass {
            cutoff_hz: 800.0,
            q: 2.0,
        }),
    );
    let a = bake(&p, SR, 0.1);
    let b = bake(&p, SR, 0.1);
    assert_eq!(a, b);
}

#[test]
fn biquad_kinds_round_trip_through_json() {
    let cases = [
        NodeKind::BiquadLowpass(BiquadLowpass::default()),
        NodeKind::BiquadHighpass(BiquadHighpass::default()),
        NodeKind::BiquadBandpass(BiquadBandpass::default()),
    ];
    for kind in cases {
        let p = sine_through(440.0, kind.clone());
        let json = serde_json::to_string(&p).unwrap();
        let back: AudioPatch = serde_json::from_str(&json).unwrap();
        assert_eq!(back, p);
    }
}
