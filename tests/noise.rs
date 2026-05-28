//! End-to-end bake tests for the Phase 1 noise generators.
//!
//! Validates that each colour:
//! - bakes with the same seed produce a bit-identical buffer (determinism
//!   is the load-bearing contract for DID-seeded ambient in Phase 5);
//! - stays bounded in `[−1, 1]` for the default amplitude;
//! - is not silent (rules out a broken patch wiring or a zeroed state);
//! - survives a serde JSON round-trip.
//!
//! Spectral colour (white ≈ flat, pink ≈ −3 dB/oct, brown ≈ −6 dB/oct) is
//! a property we trust from the well-known filter coefficients rather than
//! verifying numerically here — a meaningful FFT-slope test needs enough
//! samples and averaging to be more code than the rest of this file.

use std::collections::BTreeMap;

use bevy_symbios_audio::{
    AudioPatch, BrownNoise, GraphNode, NodeGraph, NodeId, NodeKind, PinkNoise, WhiteNoise, bake,
};

fn patch_with(seed: u32, kind: NodeKind) -> AudioPatch {
    AudioPatch {
        seed,
        graph: NodeGraph {
            nodes: vec![GraphNode {
                id: NodeId(0),
                kind,
                inputs: BTreeMap::new(),
            }],
            output: NodeId(0),
        },
    }
}

fn rms(buf: &[f32]) -> f32 {
    let sum_sq: f64 = buf.iter().map(|s| (*s as f64) * (*s as f64)).sum();
    (sum_sq / buf.len() as f64).sqrt() as f32
}

fn max_abs(buf: &[f32]) -> f32 {
    buf.iter().fold(0.0_f32, |acc, s| acc.max(s.abs()))
}

const SR: u32 = 44_100;
const SECS: f32 = 0.5;

// --- determinism ------------------------------------------------------------

#[test]
fn white_is_deterministic_across_bakes() {
    let p = patch_with(0xABCD_1234, NodeKind::WhiteNoise(WhiteNoise::default()));
    let a = bake(&p, SR, SECS);
    let b = bake(&p, SR, SECS);
    assert_eq!(a, b);
}

#[test]
fn pink_is_deterministic_across_bakes() {
    let p = patch_with(0xABCD_1234, NodeKind::PinkNoise(PinkNoise::default()));
    let a = bake(&p, SR, SECS);
    let b = bake(&p, SR, SECS);
    assert_eq!(a, b);
}

#[test]
fn brown_is_deterministic_across_bakes() {
    let p = patch_with(0xABCD_1234, NodeKind::BrownNoise(BrownNoise::default()));
    let a = bake(&p, SR, SECS);
    let b = bake(&p, SR, SECS);
    assert_eq!(a, b);
}

#[test]
fn different_seeds_yield_different_buffers() {
    let p1 = patch_with(1, NodeKind::WhiteNoise(WhiteNoise::default()));
    let p2 = patch_with(2, NodeKind::WhiteNoise(WhiteNoise::default()));
    let a = bake(&p1, SR, SECS);
    let b = bake(&p2, SR, SECS);
    assert_ne!(a, b);
}

// --- bounded / non-silent ---------------------------------------------------

#[test]
fn white_is_bounded_and_not_silent() {
    let p = patch_with(0, NodeKind::WhiteNoise(WhiteNoise { amplitude: 0.5 }));
    let buf = bake(&p, SR, SECS);
    assert!(max_abs(&buf) <= 0.5);
    let r = rms(&buf);
    // Uniform on [−0.5, 0.5] has RMS = 0.5/√3 ≈ 0.289 — give plenty of
    // tolerance.
    assert!(r > 0.1 && r < 0.5, "white RMS out of band: {r}");
}

#[test]
fn pink_is_bounded_and_not_silent() {
    let p = patch_with(0, NodeKind::PinkNoise(PinkNoise { amplitude: 0.5 }));
    let buf = bake(&p, SR, SECS);
    // Pink filter can produce occasional excursions slightly above
    // ±amplitude, so allow a tiny headroom.
    assert!(max_abs(&buf) < 0.6, "pink leaked past ±0.6");
    assert!(rms(&buf) > 0.02, "pink looks silent");
}

#[test]
fn brown_is_bounded_and_not_silent() {
    let p = patch_with(0, NodeKind::BrownNoise(BrownNoise { amplitude: 0.5 }));
    let buf = bake(&p, SR, SECS);
    // The integrator + clamp guarantees ±amplitude.
    assert!(max_abs(&buf) <= 0.5);
    assert!(rms(&buf) > 0.01, "brown looks silent");
}

// --- spectral ordering ------------------------------------------------------

#[test]
fn spectral_colour_orders_by_lpf_strength() {
    // Brown is the most low-passed — its first-difference RMS (a coarse
    // high-frequency proxy) is much smaller than pink's, which in turn is
    // smaller than white's.  We don't need an FFT to verify the slope
    // monotonically gets darker.
    fn diff_rms(buf: &[f32]) -> f32 {
        let mut acc = 0.0_f64;
        for w in buf.windows(2) {
            let d = (w[1] - w[0]) as f64;
            acc += d * d;
        }
        ((acc / buf.len() as f64).sqrt()) as f32
    }
    let pw = patch_with(7, NodeKind::WhiteNoise(WhiteNoise::default()));
    let pp = patch_with(7, NodeKind::PinkNoise(PinkNoise::default()));
    let pb = patch_with(7, NodeKind::BrownNoise(BrownNoise::default()));
    let w = diff_rms(&bake(&pw, SR, 1.0));
    let p = diff_rms(&bake(&pp, SR, 1.0));
    let b = diff_rms(&bake(&pb, SR, 1.0));
    assert!(w > p, "expected white diff_rms ({w}) > pink ({p})");
    assert!(p > b, "expected pink diff_rms ({p}) > brown ({b})");
}

// --- JSON round-trip --------------------------------------------------------

#[test]
fn noise_kinds_round_trip_through_json() {
    let cases = [
        NodeKind::WhiteNoise(WhiteNoise::default()),
        NodeKind::PinkNoise(PinkNoise::default()),
        NodeKind::BrownNoise(BrownNoise::default()),
    ];
    for kind in cases {
        let p = patch_with(0, kind.clone());
        let json = serde_json::to_string(&p).unwrap();
        let back: AudioPatch = serde_json::from_str(&json).unwrap();
        assert_eq!(back, p);
    }
}

// --- determinism in mixed graphs --------------------------------------------

#[test]
fn pink_state_resets_between_bakes() {
    // Two consecutive bakes share no carried state, so the second always
    // starts from PinkState::default() — this means consecutive bakes of
    // the same patch are not just deterministic against each other, but
    // also produce a *known* opening transient.  Without state reset,
    // bake() would be stateful at the function boundary.
    let p = patch_with(0xFEED_FACE, NodeKind::PinkNoise(PinkNoise::default()));
    let first = bake(&p, SR, 0.05);
    let second = bake(&p, SR, 0.05);
    assert_eq!(first, second);
    // And the first sample should be small (filter starting from 0).
    assert!(first[0].abs() < 0.5);
}
