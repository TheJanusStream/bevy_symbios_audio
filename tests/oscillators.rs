//! Spectral acceptance test for the Phase 1 oscillators.
//!
//! Phase 1 ticket #4 acceptance: a 1-second 440 Hz bake of each waveform
//! has its dominant FFT bin at the expected frequency.  This catches any
//! drift in phase, frequency, or waveform shape — including the kind of
//! off-by-2π or off-by-half-rate bugs that unit tests on individual
//! samples can miss.
//!
//! No runtime FFT dep: we ship a tiny iterative radix-2 Cooley-Tukey FFT
//! inline (~30 lines of arithmetic).  Numerical accuracy is fine for
//! 32_768-point transforms of synthetic signals — single-precision peak
//! detection only needs about 4 decimal digits.

use std::collections::BTreeMap;

use bevy_symbios_audio::{
    AudioPatch, GraphNode, NodeGraph, NodeId, NodeKind, SawPolarity, SawtoothOsc, SineOsc,
    SquareOsc, TriangleOsc, bake,
};

// --- FFT helper -------------------------------------------------------------

#[derive(Clone, Copy)]
struct Complex {
    re: f32,
    im: f32,
}

impl Complex {
    const ZERO: Self = Self { re: 0.0, im: 0.0 };
    fn from_real(x: f32) -> Self {
        Self { re: x, im: 0.0 }
    }
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
    fn magnitude(self) -> f32 {
        (self.re * self.re + self.im * self.im).sqrt()
    }
}

/// In-place iterative radix-2 Cooley-Tukey FFT.  `buf.len()` must be a
/// power of two.  Mutates the buffer to contain its forward DFT.
fn fft(buf: &mut [Complex]) {
    let n = buf.len();
    assert!(n.is_power_of_two(), "FFT length must be a power of two");
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation so the butterflies operate on the right
    // index pairs.
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }

    // Iterative butterflies — width doubles every level.
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let ang = -2.0 * std::f32::consts::PI / len as f32;
        let w_step = Complex {
            re: ang.cos(),
            im: ang.sin(),
        };
        let mut i = 0;
        while i < n {
            let mut w = Complex { re: 1.0, im: 0.0 };
            for k in 0..half {
                let t = w.mul(buf[i + k + half]);
                let u = buf[i + k];
                buf[i + k] = u.add(t);
                buf[i + k + half] = u.sub(t);
                w = w.mul(w_step);
            }
            i += len;
        }
        len *= 2;
    }
}

/// Return the bin index of the strongest spectral component in the lower
/// (non-DC, sub-Nyquist) half of the magnitude spectrum.
fn dominant_bin(samples: &[f32]) -> usize {
    let mut buf: Vec<Complex> = samples.iter().copied().map(Complex::from_real).collect();
    // Pad up to a power of two if needed.
    let n = samples.len().next_power_of_two();
    buf.resize(n, Complex::ZERO);
    fft(&mut buf);
    let half = n / 2;
    // Skip bin 0 (DC).  Pick the maximum magnitude bin in [1, n/2).
    let mut best = 1;
    let mut best_mag = buf[1].magnitude();
    for (i, c) in buf.iter().enumerate().take(half).skip(2) {
        let m = c.magnitude();
        if m > best_mag {
            best_mag = m;
            best = i;
        }
    }
    best
}

// --- Patch helpers ----------------------------------------------------------

fn patch_with(kind: NodeKind) -> AudioPatch {
    AudioPatch {
        seed: 0,
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

/// Bake a 1-second buffer at a sample rate that's a power of two so bin
/// spacing = 1 Hz and 440 Hz falls exactly on bin 440.  No leakage, no
/// neighbour-bin contamination.
const SR: u32 = 32_768;

// --- Tests ------------------------------------------------------------------

#[test]
fn fft_self_check_finds_planted_tone() {
    // Sanity-check the FFT helper before using it on oscillator output.
    // A 1 s pure cosine at 100 Hz sampled at 32_768 Hz must peak at bin 100.
    let mut buf = Vec::with_capacity(SR as usize);
    for n in 0..SR {
        let t = n as f32 / SR as f32;
        buf.push((2.0 * std::f32::consts::PI * 100.0 * t).cos());
    }
    assert_eq!(dominant_bin(&buf), 100);
}

#[test]
fn sine_at_440hz_peaks_at_bin_440() {
    let osc = SineOsc {
        freq_hz: 440.0,
        phase_offset: 0.0,
        amplitude: 1.0,
    };
    let p = patch_with(NodeKind::Sine(osc));
    let buf = bake(&p, SR, 1.0);
    assert_eq!(buf.len(), SR as usize);
    assert_eq!(dominant_bin(&buf), 440);
}

#[test]
fn square_at_440hz_peaks_at_bin_440() {
    let osc = SquareOsc {
        freq_hz: 440.0,
        duty: 0.5,
        amplitude: 1.0,
    };
    let p = patch_with(NodeKind::Square(osc));
    let buf = bake(&p, SR, 1.0);
    assert_eq!(dominant_bin(&buf), 440);
}

#[test]
fn sawtooth_up_at_440hz_peaks_at_bin_440() {
    let osc = SawtoothOsc {
        freq_hz: 440.0,
        polarity: SawPolarity::Up,
        amplitude: 1.0,
    };
    let p = patch_with(NodeKind::Sawtooth(osc));
    let buf = bake(&p, SR, 1.0);
    assert_eq!(dominant_bin(&buf), 440);
}

#[test]
fn sawtooth_down_at_440hz_peaks_at_bin_440() {
    let osc = SawtoothOsc {
        freq_hz: 440.0,
        polarity: SawPolarity::Down,
        amplitude: 1.0,
    };
    let p = patch_with(NodeKind::Sawtooth(osc));
    let buf = bake(&p, SR, 1.0);
    assert_eq!(dominant_bin(&buf), 440);
}

#[test]
fn triangle_at_440hz_peaks_at_bin_440() {
    let osc = TriangleOsc {
        freq_hz: 440.0,
        amplitude: 1.0,
    };
    let p = patch_with(NodeKind::Triangle(osc));
    let buf = bake(&p, SR, 1.0);
    assert_eq!(dominant_bin(&buf), 440);
}

#[test]
fn sine_at_220hz_peaks_at_bin_220() {
    // Cross-check that the frequency really controls the bin, not just
    // some 440-coincidence.
    let osc = SineOsc {
        freq_hz: 220.0,
        phase_offset: 0.0,
        amplitude: 1.0,
    };
    let p = patch_with(NodeKind::Sine(osc));
    let buf = bake(&p, SR, 1.0);
    assert_eq!(dominant_bin(&buf), 220);
}

#[test]
fn each_kind_round_trips_through_json() {
    // Schema sanity: every new NodeKind variant added in this ticket has
    // to survive a serde round-trip with its config.
    let cases = [
        NodeKind::Sine(SineOsc::default()),
        NodeKind::Square(SquareOsc::default()),
        NodeKind::Sawtooth(SawtoothOsc::default()),
        NodeKind::Triangle(TriangleOsc::default()),
    ];
    for kind in cases {
        let p = patch_with(kind.clone());
        let json = serde_json::to_string(&p).unwrap();
        let back: AudioPatch = serde_json::from_str(&json).unwrap();
        assert_eq!(back, p);
    }
}
