//! Node trait and the closed enum of built-in node kinds.
//!
//! The crate's runtime contract is the [`Node`] trait, which produces a
//! single audio sample given a [`BakeContext`].  For the serializable
//! schema, [`NodeKind`] tags the concrete node variant.  The enum is
//! `#[non_exhaustive]`, so new built-in kinds can be added without
//! breaking downstream `match` expressions (downstream code must include
//! a wildcard arm).
//!
//! The trait takes `&self`: per-node runtime state (filter z-1, envelope
//! stage, etc.) does not live on the node config but is carried by the
//! evaluator — keeps configs pure data, serde-clean, and `Genotype`-friendly.

use std::any::Any;
use std::collections::BTreeMap;

use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::adsr::AdsrEnvelope;
use crate::filter::{BiquadBandpass, BiquadHighpass, BiquadLowpass};
use crate::lfo::Lfo;
use crate::noise::{BrownNoise, PinkNoise, WhiteNoise};
use crate::oscillator::{SawtoothOsc, SineOsc, SquareOsc, TriangleOsc};

/// Closed enum of every built-in node kind that can appear in a patch.
///
/// Variants are tagged on the JSON wire by `kind` so adding a new variant is
/// a forward-compatible operation as long as readers tolerate unknown tags
/// (callers can wrap deserialization in their own validation step).
///
/// Marked `#[non_exhaustive]`: external matches must include a wildcard arm,
/// so new variants in future phases don't break downstream callers.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
#[non_exhaustive]
pub enum NodeKind {
    /// Outputs 0.0 every sample.  Useful as a pad, a bypassed channel, or a
    /// schema placeholder before later phases populate richer node types.
    #[default]
    Silence,
    /// Pure-tone sine oscillator — see [`SineOsc`].
    Sine(SineOsc),
    /// Naïve pulse-width square — see [`SquareOsc`].
    Square(SquareOsc),
    /// Naïve sawtooth with selectable polarity — see [`SawtoothOsc`].
    Sawtooth(SawtoothOsc),
    /// Naïve triangle wave — see [`TriangleOsc`].
    Triangle(TriangleOsc),
    /// Uniform white noise — see [`WhiteNoise`].
    WhiteNoise(WhiteNoise),
    /// Paul Kellet 3-band pink noise — see [`PinkNoise`].
    PinkNoise(PinkNoise),
    /// Leaky-integrator brown noise — see [`BrownNoise`].
    BrownNoise(BrownNoise),
    /// Attack/Decay/Sustain/Release envelope — see [`AdsrEnvelope`].
    Adsr(AdsrEnvelope),
    /// Second-order biquad lowpass — see [`BiquadLowpass`].
    BiquadLowpass(BiquadLowpass),
    /// Second-order biquad highpass — see [`BiquadHighpass`].
    BiquadHighpass(BiquadHighpass),
    /// Second-order biquad bandpass — see [`BiquadBandpass`].
    BiquadBandpass(BiquadBandpass),
    /// Low-frequency oscillator (modulation source) — see [`Lfo`].
    Lfo(Lfo),
}

/// Per-sample context handed to every [`Node::sample`] invocation.
///
/// Carries the patch sample rate, the current sample index, the total
/// duration of the bake, a resolved view of the inputs wired to the node
/// being evaluated, and a borrowed reference to the patch's seeded RNG.
///
/// Read-only for node implementations: `sample_rate`, `sample_index`, and
/// `duration_samples` are advanced by the baker between samples and must
/// not be touched.  The RNG can be drawn from (which advances internal
/// state) but must not be replaced — this is what keeps two bakes from the
/// same seed bit-identical.
pub struct BakeContext<'a> {
    /// Target sample rate in Hz.
    pub sample_rate: u32,
    /// Index of the sample currently being produced, starting at 0.
    pub sample_index: u64,
    /// Total number of samples this bake will produce.
    pub duration_samples: u64,
    /// Inputs wired to the node currently being evaluated, resolved to
    /// their f32 sample values (constants or upstream node outputs).
    pub(crate) inputs: &'a BTreeMap<String, f32>,
    /// Seeded deterministic RNG, shared across the entire bake so the same
    /// patch + same seed always yields the same buffer.
    pub(crate) rng: &'a mut ChaCha8Rng,
    /// Per-node persistent state.  `None` for stateless nodes; for stateful
    /// ones it points at the `Box<dyn Any + Send>` the baker built from
    /// [`Node::init_state`] at bake start.  Type-erased so each node kind
    /// owns its own state struct; reach in with [`Self::state_mut`].
    pub(crate) state: Option<&'a mut (dyn Any + Send)>,
}

impl<'a> BakeContext<'a> {
    /// Construct a context for a single node evaluation.  Intended for
    /// evaluator code; user node implementations only read from `&self`.
    pub fn new(
        sample_rate: u32,
        sample_index: u64,
        duration_samples: u64,
        rng: &'a mut ChaCha8Rng,
        inputs: &'a BTreeMap<String, f32>,
        state: Option<&'a mut (dyn Any + Send)>,
    ) -> Self {
        Self {
            sample_rate,
            sample_index,
            duration_samples,
            inputs,
            rng,
            state,
        }
    }

    /// Mutably borrow the per-node state as a concrete type `S`.  Returns
    /// `None` for stateless nodes, or when the state's concrete type
    /// doesn't match `S` — node implementations always know their own
    /// state shape, so the latter only indicates a baker bug.
    #[inline]
    pub fn state_mut<S: Any>(&mut self) -> Option<&mut S> {
        self.state.as_deref_mut()?.downcast_mut::<S>()
    }

    /// Resolved value at the named input port.  Returns 0.0 if the port is
    /// unwired — matches the "missing connection reads zero" convention
    /// every modular synth uses.
    #[inline]
    pub fn input(&self, port: &str) -> f32 {
        self.inputs.get(port).copied().unwrap_or(0.0)
    }

    /// Wall-clock time at the current sample, in seconds.
    #[inline]
    pub fn time_secs(&self) -> f64 {
        self.sample_index as f64 / self.sample_rate as f64
    }

    /// Mutable access to the patch's seeded RNG.  Drawing from it advances
    /// the internal state — that's the whole point — but `BakeContext`
    /// does not let the node replace the RNG, which preserves the
    /// "same seed → bit-identical buffer" determinism guarantee.
    #[inline]
    pub fn rng(&mut self) -> &mut ChaCha8Rng {
        self.rng
    }
}

/// Per-sample audio producer.  Every variant of [`NodeKind`] implements this
/// (via the trait impl below); user-extensible alternate node types may also
/// implement it directly, though they won't be representable in [`NodeKind`]
/// without a crate-level extension.
pub trait Node: Send + Sync {
    /// Produce one audio sample (mono, normalised to roughly `[-1.0, 1.0]`)
    /// for the current evaluation step.
    fn sample(&self, ctx: &mut BakeContext) -> f32;

    /// Build the initial state container for this node, if any.  Stateless
    /// nodes (oscillators, white noise, silence) use the default `None`
    /// impl; stateful ones (filters, envelopes, pink/brown noise) return
    /// `Some(Box::new(MyState::default()))`.  The baker calls this once
    /// at the start of a bake and reuses the container across every
    /// sample for that node.
    fn init_state(&self) -> Option<Box<dyn Any + Send>> {
        None
    }
}

impl Node for NodeKind {
    fn sample(&self, ctx: &mut BakeContext) -> f32 {
        match self {
            NodeKind::Silence => 0.0,
            NodeKind::Sine(osc) => osc.sample(ctx),
            NodeKind::Square(osc) => osc.sample(ctx),
            NodeKind::Sawtooth(osc) => osc.sample(ctx),
            NodeKind::Triangle(osc) => osc.sample(ctx),
            NodeKind::WhiteNoise(n) => n.sample(ctx),
            NodeKind::PinkNoise(n) => n.sample(ctx),
            NodeKind::BrownNoise(n) => n.sample(ctx),
            NodeKind::Adsr(env) => env.sample(ctx),
            NodeKind::BiquadLowpass(f) => f.sample(ctx),
            NodeKind::BiquadHighpass(f) => f.sample(ctx),
            NodeKind::BiquadBandpass(f) => f.sample(ctx),
            NodeKind::Lfo(l) => l.sample(ctx),
        }
    }

    fn init_state(&self) -> Option<Box<dyn Any + Send>> {
        match self {
            NodeKind::Sine(o) => o.init_state(),
            NodeKind::Square(o) => o.init_state(),
            NodeKind::Sawtooth(o) => o.init_state(),
            NodeKind::Triangle(o) => o.init_state(),
            NodeKind::PinkNoise(n) => n.init_state(),
            NodeKind::BrownNoise(n) => n.init_state(),
            NodeKind::Adsr(env) => env.init_state(),
            NodeKind::BiquadLowpass(f) => f.init_state(),
            NodeKind::BiquadHighpass(f) => f.init_state(),
            NodeKind::BiquadBandpass(f) => f.init_state(),
            NodeKind::Lfo(l) => l.init_state(),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use super::*;

    fn rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(0)
    }

    #[test]
    fn silence_samples_zero() {
        let inputs = BTreeMap::new();
        let mut r = rng();
        let mut ctx = BakeContext::new(44_100, 0, 44_100, &mut r, &inputs, None);
        assert_eq!(NodeKind::Silence.sample(&mut ctx), 0.0);
    }

    #[test]
    fn input_defaults_to_zero_when_unwired() {
        let inputs = BTreeMap::new();
        let mut r = rng();
        let ctx = BakeContext::new(48_000, 100, 48_000, &mut r, &inputs, None);
        assert_eq!(ctx.input("anything"), 0.0);
    }

    #[test]
    fn input_returns_wired_value() {
        let mut inputs = BTreeMap::new();
        inputs.insert("freq".to_string(), 440.0_f32);
        let mut r = rng();
        let ctx = BakeContext::new(44_100, 0, 44_100, &mut r, &inputs, None);
        assert_eq!(ctx.input("freq"), 440.0);
    }

    #[test]
    fn time_secs_advances_with_sample_index() {
        let inputs = BTreeMap::new();
        let mut r = rng();
        let ctx = BakeContext::new(44_100, 22_050, 44_100, &mut r, &inputs, None);
        let t = ctx.time_secs();
        assert!((t - 0.5).abs() < 1e-9, "expected ~0.5s, got {t}");
    }

    #[test]
    fn rng_is_deterministic_for_same_seed() {
        let inputs = BTreeMap::new();
        let mut r1 = ChaCha8Rng::seed_from_u64(42);
        let mut r2 = ChaCha8Rng::seed_from_u64(42);
        let mut ctx1 = BakeContext::new(44_100, 0, 100, &mut r1, &inputs, None);
        let mut ctx2 = BakeContext::new(44_100, 0, 100, &mut r2, &inputs, None);
        let a: u32 = ctx1.rng().random();
        let b: u32 = ctx2.rng().random();
        assert_eq!(a, b);
    }
}
