//! Four classic waveform generators — sine, square, sawtooth, triangle.
//!
//! All four share a single phase-accumulator state shape ([`OscPhase`])
//! and integrate the instantaneous frequency sample by sample, so wiring
//! another node into the `"freq"` input port gives proper per-sample
//! frequency modulation (an LFO sweep produces a vibrato, an audio-rate
//! oscillator produces FM sidebands).  Amplitude is similarly modulatable
//! through the `"amplitude"` input port.
//!
//! Aliasing is intentional: square/saw/triangle remain naïve generators
//! (no PolyBLEP, BLIT, or oversampling).  The Janus Stream targets
//! *texture* over purity — the audible grit of a 440 Hz square at 32 kHz
//! is part of the aesthetic.  Band-limited variants can be added later
//! as new `NodeKind` variants since the enum is `#[non_exhaustive]`.
//!
//! # Input ports
//!
//! - `"freq"` — added to `freq_hz` per sample.  Wire an LFO here for
//!   vibrato; wire an audio-rate oscillator for FM.
//! - `"amplitude"` — added to `amplitude` per sample.  Wire an envelope
//!   here for AM/tremolo or volume shaping.
//!
//! Unwired ports read zero, leaving the configured value untouched.

use std::any::Any;
use std::f32::consts::PI;

use serde::{Deserialize, Serialize};

use crate::node::{BakeContext, Node};

const TWO_PI: f32 = 2.0 * PI;

/// Shared phase accumulator for every oscillator in this module (and the
/// LFO, conceptually — though that one carries extra fields for the
/// sample-and-hold shape).
///
/// One running phase value in `[0, 1)` that advances by `freq / sr` per
/// sample.  Lifted to a struct rather than a bare `f32` so the
/// `#[non_exhaustive]` extension story stays open.
#[derive(Debug, Clone, Copy, Default)]
pub struct OscPhase {
    pub(crate) phase: f32,
}

/// Direction of a sawtooth ramp.  `Up` rises from −1 to +1 over the period
/// (the classic bright sawtooth ramp); `Down` falls from +1 to −1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SawPolarity {
    #[default]
    Up,
    Down,
}

fn default_amplitude() -> f32 {
    1.0
}

/// Advance the phase accumulator and return the *previous* phase so the
/// sample produced at index N reflects N's contribution rather than
/// N+1's.  Returns the phase to use for this sample.
#[inline]
fn step_phase(state: &mut OscPhase, effective_freq: f32, sample_rate: f32) -> f32 {
    let p = state.phase;
    state.phase = (p + effective_freq / sample_rate).rem_euclid(1.0);
    p
}

/// Stateless fallback: when no [`OscPhase`] state is installed (typical
/// for direct unit tests that bypass the baker), reconstruct phase from
/// `t * freq`.  Equivalent to the accumulator path when the frequency is
/// constant — different only under modulation.
#[inline]
fn stateless_phase(time_secs: f64, freq_hz: f32) -> f32 {
    (time_secs as f32 * freq_hz).rem_euclid(1.0)
}

// --- Sine -------------------------------------------------------------------

/// Pure-tone sine oscillator with a constant phase offset.
///
/// `phase_offset` is in units of one cycle (0.0 to 1.0 covers a full
/// rotation), so two sines at the same frequency with `phase_offset` 0.0
/// and 0.25 are 90° apart.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SineOsc {
    pub freq_hz: f32,
    pub phase_offset: f32,
    /// Output gain, multiplied with the waveform.  Wired modulation on
    /// the `"amplitude"` input is added to this value per sample.
    #[serde(default = "default_amplitude")]
    pub amplitude: f32,
}

impl Default for SineOsc {
    fn default() -> Self {
        Self {
            freq_hz: 440.0,
            phase_offset: 0.0,
            amplitude: 1.0,
        }
    }
}

impl Node for SineOsc {
    fn sample(&self, ctx: &mut BakeContext) -> f32 {
        let sr = ctx.sample_rate as f32;
        let freq = self.freq_hz + ctx.input("freq");
        let amp = self.amplitude + ctx.input("amplitude");
        let phase = match ctx.state_mut::<OscPhase>() {
            Some(s) => step_phase(s, freq, sr),
            None => stateless_phase(ctx.time_secs(), self.freq_hz),
        };
        let total = (phase + self.phase_offset).rem_euclid(1.0);
        (TWO_PI * total).sin() * amp
    }

    fn init_state(&self) -> Option<Box<dyn Any + Send>> {
        Some(Box::new(OscPhase::default()))
    }
}

// --- Square -----------------------------------------------------------------

/// Naïve pulse-width square at `freq_hz`.
///
/// `duty` is the fraction of the period spent at +1, clamped to `(0, 1)`
/// at sample time so the canonical 0.5 case yields a symmetric square
/// wave.  Duty values near 0 or 1 produce a thin pulse — useful as a
/// click train and as a target for PWM modulation later.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SquareOsc {
    pub freq_hz: f32,
    pub duty: f32,
    #[serde(default = "default_amplitude")]
    pub amplitude: f32,
}

impl Default for SquareOsc {
    fn default() -> Self {
        Self {
            freq_hz: 440.0,
            duty: 0.5,
            amplitude: 1.0,
        }
    }
}

impl Node for SquareOsc {
    fn sample(&self, ctx: &mut BakeContext) -> f32 {
        let sr = ctx.sample_rate as f32;
        let freq = self.freq_hz + ctx.input("freq");
        let amp = self.amplitude + ctx.input("amplitude");
        let phase = match ctx.state_mut::<OscPhase>() {
            Some(s) => step_phase(s, freq, sr),
            None => stateless_phase(ctx.time_secs(), self.freq_hz),
        };
        let duty = self.duty.clamp(f32::EPSILON, 1.0 - f32::EPSILON);
        let raw = if phase < duty { 1.0 } else { -1.0 };
        raw * amp
    }

    fn init_state(&self) -> Option<Box<dyn Any + Send>> {
        Some(Box::new(OscPhase::default()))
    }
}

// --- Sawtooth ---------------------------------------------------------------

/// Naïve sawtooth.  Polarity flips the ramp direction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SawtoothOsc {
    pub freq_hz: f32,
    pub polarity: SawPolarity,
    #[serde(default = "default_amplitude")]
    pub amplitude: f32,
}

impl Default for SawtoothOsc {
    fn default() -> Self {
        Self {
            freq_hz: 440.0,
            polarity: SawPolarity::Up,
            amplitude: 1.0,
        }
    }
}

impl Node for SawtoothOsc {
    fn sample(&self, ctx: &mut BakeContext) -> f32 {
        let sr = ctx.sample_rate as f32;
        let freq = self.freq_hz + ctx.input("freq");
        let amp = self.amplitude + ctx.input("amplitude");
        let phase = match ctx.state_mut::<OscPhase>() {
            Some(s) => step_phase(s, freq, sr),
            None => stateless_phase(ctx.time_secs(), self.freq_hz),
        };
        let raw = match self.polarity {
            SawPolarity::Up => 2.0 * phase - 1.0,
            SawPolarity::Down => 1.0 - 2.0 * phase,
        };
        raw * amp
    }

    fn init_state(&self) -> Option<Box<dyn Any + Send>> {
        Some(Box::new(OscPhase::default()))
    }
}

// --- Triangle ---------------------------------------------------------------

/// Naïve triangle.  Symmetric — peaks at +1 at the half-period mark and
/// bottoms out at −1 at the phase boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TriangleOsc {
    pub freq_hz: f32,
    #[serde(default = "default_amplitude")]
    pub amplitude: f32,
}

impl Default for TriangleOsc {
    fn default() -> Self {
        Self {
            freq_hz: 440.0,
            amplitude: 1.0,
        }
    }
}

impl Node for TriangleOsc {
    fn sample(&self, ctx: &mut BakeContext) -> f32 {
        let sr = ctx.sample_rate as f32;
        let freq = self.freq_hz + ctx.input("freq");
        let amp = self.amplitude + ctx.input("amplitude");
        let phase = match ctx.state_mut::<OscPhase>() {
            Some(s) => step_phase(s, freq, sr),
            None => stateless_phase(ctx.time_secs(), self.freq_hz),
        };
        let raw = 1.0 - 4.0 * (phase - 0.5).abs();
        raw * amp
    }

    fn init_state(&self) -> Option<Box<dyn Any + Send>> {
        Some(Box::new(OscPhase::default()))
    }
}

// --- Genotype ---------------------------------------------------------------

crate::impl_genotype!(SineOsc {
    freq_hz: f32_log(0.5, 20.0, 20_000.0),
    phase_offset: f32(0.05, 0.0, 1.0),
    amplitude: f32(0.1, 0.0, 1.0),
});

crate::impl_genotype!(SquareOsc {
    freq_hz: f32_log(0.5, 20.0, 20_000.0),
    duty: f32(0.05, 0.05, 0.95),
    amplitude: f32(0.1, 0.0, 1.0),
});

crate::impl_genotype!(SawtoothOsc {
    freq_hz: f32_log(0.5, 20.0, 20_000.0),
    polarity: enum_cycle([SawPolarity::Up, SawPolarity::Down]),
    amplitude: f32(0.1, 0.0, 1.0),
});

crate::impl_genotype!(TriangleOsc {
    freq_hz: f32_log(0.5, 20.0, 20_000.0),
    amplitude: f32(0.1, 0.0, 1.0),
});

// --- Tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;

    /// Drive an oscillator through N samples with installed state and no
    /// modulation, returning the per-sample buffer.  Matches what the
    /// production bake() path does for a one-node patch.
    fn drive<N: Node>(node: &N, sample_rate: u32, n: usize) -> Vec<f32> {
        let mut state = node.init_state();
        let inputs: &[(&str, f32)] = &[];
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let state_ref: Option<&mut (dyn Any + Send)> = state.as_deref_mut();
            let mut ctx =
                BakeContext::new(sample_rate, i as u64, n as u64, &mut rng, inputs, state_ref);
            out.push(node.sample(&mut ctx));
        }
        out
    }

    #[test]
    fn sine_zero_phase_at_t_zero() {
        let osc = SineOsc::default();
        let buf = drive(&osc, 44_100, 1);
        assert!(buf[0].abs() < 1e-6, "sample[0] = {}", buf[0]);
    }

    #[test]
    fn sine_quarter_period_is_one() {
        // sr / (4 * 440) ≈ 25 samples per quarter period at 440 Hz.
        let osc = SineOsc::default();
        let sr: u32 = 44_100;
        let quarter = (sr as f32 / (4.0 * 440.0)).round() as usize;
        let buf = drive(&osc, sr, quarter + 1);
        assert!(
            buf[quarter] > 0.99,
            "quarter-period sample: {}",
            buf[quarter]
        );
    }

    #[test]
    fn square_50pct_duty_is_bipolar() {
        let osc = SquareOsc::default();
        let sr: u32 = 44_100;
        // Sample 10 is in the first half-cycle (positive); 10 + half period is
        // in the second (negative).
        let half_period = (sr as f32 / (2.0 * 440.0)).round() as usize;
        let buf = drive(&osc, sr, 10 + half_period + 1);
        assert_eq!(buf[10], 1.0);
        assert_eq!(buf[10 + half_period], -1.0);
    }

    #[test]
    fn sawtooth_up_rises_linearly_within_period() {
        let osc = SawtoothOsc {
            freq_hz: 1.0,
            polarity: SawPolarity::Up,
            amplitude: 1.0,
        };
        let buf = drive(&osc, 1_000, 501);
        // Phase 0 → -1.  Phase 0.5 (sample 500 at 1 Hz sr=1000) → 0.
        assert!((buf[0] - -1.0).abs() < 1e-3);
        assert!(buf[500].abs() < 1e-2);
    }

    #[test]
    fn sawtooth_down_is_negated_up() {
        let up = SawtoothOsc {
            freq_hz: 1.0,
            polarity: SawPolarity::Up,
            amplitude: 1.0,
        };
        let down = SawtoothOsc {
            freq_hz: 1.0,
            polarity: SawPolarity::Down,
            amplitude: 1.0,
        };
        let up_buf = drive(&up, 1_000, 1_000);
        let down_buf = drive(&down, 1_000, 1_000);
        for i in (0..1_000).step_by(73) {
            let sum = up_buf[i] + down_buf[i];
            assert!(sum.abs() < 1e-3, "up+down ≠ 0 at i={i}: {sum}");
        }
    }

    #[test]
    fn triangle_is_symmetric_around_half_period() {
        let osc = TriangleOsc {
            freq_hz: 1.0,
            amplitude: 1.0,
        };
        let buf = drive(&osc, 1_000, 1_000);
        // f(0) = -1; f(0.25) = 0; f(0.5) = +1; f(0.75) = 0.
        assert!(buf[250].abs() < 1e-2);
        assert!((buf[500] - 1.0).abs() < 1e-2);
        assert!(buf[750].abs() < 1e-2);
    }

    #[test]
    fn freq_input_modulates_pitch() {
        // Phase-accumulator FM: drive a sine with a positive constant
        // freq input.  Total frequency should be config + mod_value, so
        // the output cycles faster than the config alone.
        let osc = SineOsc {
            freq_hz: 440.0,
            phase_offset: 0.0,
            amplitude: 1.0,
        };
        let sr: u32 = 44_100;
        let mut state = osc.init_state();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        // Mod input is +440 Hz, so total freq = 880 Hz.
        let inputs = [("freq", 440.0_f32)];
        let half_period_at_880 = (sr as f32 / (4.0 * 880.0)).round() as u64;
        let mut samples = Vec::new();
        for i in 0..=half_period_at_880 {
            let state_ref: Option<&mut (dyn Any + Send)> = state.as_deref_mut();
            let mut ctx = BakeContext::new(sr, i, 100, &mut rng, &inputs, state_ref);
            samples.push(osc.sample(&mut ctx));
        }
        // At sample sr/(4*880), a sine at 880 Hz should be at peak.
        let last = *samples.last().unwrap();
        assert!(last > 0.99, "modulated quarter-period: {last}");
    }

    #[test]
    fn amplitude_input_scales_output() {
        let osc = SineOsc {
            freq_hz: 440.0,
            phase_offset: 0.25, // start at peak
            amplitude: 0.0,
        };
        let sr: u32 = 44_100;
        let mut state = osc.init_state();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let inputs = [("amplitude", 0.5_f32)];
        let state_ref: Option<&mut (dyn Any + Send)> = state.as_deref_mut();
        let mut ctx = BakeContext::new(sr, 0, 1, &mut rng, &inputs, state_ref);
        // amplitude is 0.0 + 0.5 = 0.5; phase 0.25 means sin(π/2) = 1.0.
        // Output should be 0.5.
        let s = osc.sample(&mut ctx);
        assert!((s - 0.5).abs() < 1e-3, "amplitude-modulated peak: {s}");
    }

    #[test]
    fn genotype_clamps_frequencies_to_audible_range() {
        use symbios_genetics::Genotype;
        let mut osc = SineOsc {
            freq_hz: 19_500.0,
            phase_offset: 0.0,
            amplitude: 0.9,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        for _ in 0..200 {
            osc.mutate(&mut rng, 1.0);
            assert!((20.0..=20_000.0).contains(&osc.freq_hz));
            assert!((0.0..=1.0).contains(&osc.phase_offset));
            assert!((0.0..=1.0).contains(&osc.amplitude));
        }
    }
}
