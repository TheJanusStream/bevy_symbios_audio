//! Genetics controls for the editor — mutate a node, a patch, or reseed.
//!
//! Every node config implements [`symbios_genetics::Genotype`] (via the crate's
//! `impl_genotype!` macro), so the editor can offer "🎲 Mutate" buttons that
//! nudge parameters along the same axes the evolutionary search uses.  These
//! helpers dispatch the closed [`NodeKind`] enum to the right inner config's
//! mutation and provide a clock-seeded RNG for one-shot, interactive clicks.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use symbios_genetics::Genotype;

use crate::node::NodeKind;
use crate::patch::AudioPatch;

/// A fresh RNG seeded from the wall clock — good enough for interactive,
/// non-reproducible "mutate" / "randomize" button clicks (we deliberately
/// don't thread a persistent editor RNG through every widget).
pub(crate) fn fresh_rng() -> ChaCha8Rng {
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    ChaCha8Rng::seed_from_u64(seed)
}

/// Mutate one node's configuration in place at the given `rate` (probability a
/// field moves).  [`NodeKind::Silence`] has no parameters and is left as-is.
pub fn mutate_node_kind(kind: &mut NodeKind, rng: &mut impl Rng, rate: f32) {
    match kind {
        NodeKind::Silence => {}
        NodeKind::Sine(o) => o.mutate(rng, rate),
        NodeKind::Square(o) => o.mutate(rng, rate),
        NodeKind::Sawtooth(o) => o.mutate(rng, rate),
        NodeKind::Triangle(o) => o.mutate(rng, rate),
        NodeKind::WhiteNoise(n) => n.mutate(rng, rate),
        NodeKind::PinkNoise(n) => n.mutate(rng, rate),
        NodeKind::BrownNoise(n) => n.mutate(rng, rate),
        NodeKind::Adsr(e) => e.mutate(rng, rate),
        NodeKind::BiquadLowpass(f) => f.mutate(rng, rate),
        NodeKind::BiquadHighpass(f) => f.mutate(rng, rate),
        NodeKind::BiquadBandpass(f) => f.mutate(rng, rate),
        NodeKind::Lfo(l) => l.mutate(rng, rate),
        NodeKind::Mix(m) => m.mutate(rng, rate),
        NodeKind::Gain(g) => g.mutate(rng, rate),
        NodeKind::Gate(g) => g.mutate(rng, rate),
        NodeKind::Chorus(c) => c.mutate(rng, rate),
        NodeKind::Reverb(r) => r.mutate(rng, rate),
    }
}

/// Mutate every node's configuration in `patch`.  The graph topology and the
/// patch `seed` are left untouched — use [`randomize_seed`] for the seed.
pub fn mutate_patch(patch: &mut AudioPatch, rng: &mut impl Rng, rate: f32) {
    for node in &mut patch.graph.nodes {
        mutate_node_kind(&mut node.kind, rng, rate);
    }
}

/// Replace the patch's deterministic seed with a fresh random one — the cheap
/// way to reroll any stochastic nodes (noise, random LFO) without touching
/// their tuned parameters.
pub fn randomize_seed(patch: &mut AudioPatch, rng: &mut impl Rng) {
    patch.seed = rng.random();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oscillator::SineOsc;

    fn seeded() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(12345)
    }

    #[test]
    fn mutate_node_kind_changes_a_sine_at_full_rate() {
        let mut kind = NodeKind::Sine(SineOsc::default());
        let before = kind.clone();
        mutate_node_kind(&mut kind, &mut seeded(), 1.0);
        assert_ne!(kind, before, "rate 1.0 mutation should move some field");
    }

    #[test]
    fn mutate_node_kind_leaves_silence_alone() {
        let mut kind = NodeKind::Silence;
        mutate_node_kind(&mut kind, &mut seeded(), 1.0);
        assert_eq!(kind, NodeKind::Silence);
    }

    #[test]
    fn mutate_rate_zero_is_identity() {
        let mut kind = NodeKind::Lfo(Default::default());
        let before = kind.clone();
        mutate_node_kind(&mut kind, &mut seeded(), 0.0);
        assert_eq!(kind, before);
    }

    #[test]
    fn randomize_seed_changes_the_seed() {
        let mut patch = AudioPatch {
            seed: 7,
            ..AudioPatch::default()
        };
        // Drive several values forward so a draw differs from the start value.
        let mut rng = seeded();
        randomize_seed(&mut patch, &mut rng);
        // Vanishingly unlikely to land back on 7.
        assert_ne!(patch.seed, 7);
    }
}
