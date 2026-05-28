//! Sample-loop evaluator — turns an [`AudioPatch`] into a mono `Vec<f32>`.
//!
//! Phase 1 ticket #3.  Single-threaded, single-buffer, deterministic.  The
//! inner loop iterates from sample 0 to `duration_samples`, evaluates every
//! node once per sample in topological order, then pushes the output
//! node's value into the buffer.
//!
//! # Determinism
//!
//! Two bakes of the same patch with the same sample rate, duration, and
//! seed produce a bit-identical buffer.  This is guaranteed by:
//! - [`crate::patch::topo_sort`] yielding a deterministic order (Kahn's
//!   with sorted tie-breaking, no `HashMap` iteration anywhere).
//! - `BTreeMap` rather than `HashMap` for all per-sample lookups.
//! - A single [`ChaCha8Rng`] seeded from `AudioPatch::seed`, advanced only
//!   by node draws — never reset or reseeded mid-bake.
//!
//! See the `tests` module at the bottom for the regression hash test.
//!
//! # Scope and out-of-scope
//!
//! Correctness over speed.  The rayon-backed parallel bake pool lives in
//! [`crate::async_gen`] — the inner loop here stays single-threaded per
//! patch, which lets the pool just dispatch one bake per pending request.
//! Soft-clipping / master gain belongs to the mixdown baker
//! ([`crate::mixdown::bake_sequence`]).

use std::any::Any;
use std::collections::BTreeMap;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::node::{BakeContext, Node};
use crate::patch::{AudioPatch, Connection, GraphNode, NodeId, topo_sort};

/// Bake `patch` into a mono `Vec<f32>` at `sample_rate` Hz for
/// `duration_secs` seconds.
///
/// Returns an empty buffer if `duration_secs <= 0.0`.  Panics if the
/// underlying [`crate::patch::NodeGraph`] is structurally invalid
/// (cycle, dangling reference, duplicate id, or missing output) — callers
/// who can't trust their patch should validate it with
/// [`crate::patch::topo_sort`] beforehand.
///
/// # Determinism
///
/// `bake(p, sr, d) == bake(p, sr, d)` bit-for-bit for any well-formed
/// patch, sample rate, and duration.  Seed lives on [`AudioPatch::seed`].
pub fn bake(patch: &AudioPatch, sample_rate: u32, duration_secs: f32) -> Vec<f32> {
    let duration_samples = duration_samples(sample_rate, duration_secs);
    if duration_samples == 0 {
        return Vec::new();
    }

    let order = topo_sort(&patch.graph).expect("bake: AudioPatch.graph is structurally invalid");

    // id → node lookup; built once, used every sample.
    let nodes: BTreeMap<NodeId, &GraphNode> = patch.graph.nodes.iter().map(|n| (n.id, n)).collect();

    let mut rng = ChaCha8Rng::seed_from_u64(u64::from(patch.seed));

    // Per-node persistent state, built once before the sample loop.
    // Type-erased so each node kind can own its own state shape; node
    // implementations downcast via BakeContext::state_mut::<S>().
    let mut states: BTreeMap<NodeId, Box<dyn Any + Send>> = BTreeMap::new();
    for node in &patch.graph.nodes {
        if let Some(state) = node.kind.init_state() {
            states.insert(node.id, state);
        }
    }

    // Reusable per-sample scratch space.
    let mut outputs: BTreeMap<NodeId, f32> = order.iter().map(|id| (*id, 0.0_f32)).collect();
    let mut inputs_scratch: BTreeMap<String, f32> = BTreeMap::new();

    let mut buffer = Vec::with_capacity(duration_samples as usize);

    for sample_index in 0..duration_samples {
        for &node_id in &order {
            let node = nodes[&node_id];
            inputs_scratch.clear();
            for (port, conn) in &node.inputs {
                let value = match conn {
                    Connection::Constant { value } => *value,
                    Connection::Node { id, amount, .. } => {
                        outputs.get(id).copied().unwrap_or(0.0) * amount
                    }
                };
                inputs_scratch.insert(port.clone(), value);
            }
            let state_ref: Option<&mut (dyn Any + Send)> =
                states.get_mut(&node_id).map(|b| &mut **b);
            let mut ctx = BakeContext::new(
                sample_rate,
                sample_index,
                duration_samples,
                &mut rng,
                &inputs_scratch,
                state_ref,
            );
            let s = node.kind.sample(&mut ctx);
            outputs.insert(node_id, s);
        }
        buffer.push(outputs.get(&patch.graph.output).copied().unwrap_or(0.0));
    }

    buffer
}

/// Convert a `f32` seconds duration into a sample count for the given rate.
///
/// Computes in `f64` to keep precision sane for long bakes (a 32-bit float
/// loses ~1 sample of accuracy by ~30 seconds at 48 kHz).  Rounds to the
/// nearest sample so user-friendly values like `0.01` (which isn't exactly
/// representable in `f32`) yield the expected 441 samples at 44.1 kHz
/// rather than 440.  Clamps negative values to zero.
#[inline]
fn duration_samples(sample_rate: u32, duration_secs: f32) -> u64 {
    if duration_secs <= 0.0 {
        return 0;
    }
    (f64::from(duration_secs) * f64::from(sample_rate)).round() as u64
}

#[cfg(test)]
mod tests {
    use crate::node::NodeKind;
    use crate::patch::{AudioPatch, Connection, GraphNode, NodeGraph, NodeId};

    use super::*;

    /// Stable, version-portable hash of an `f32` buffer.  FNV-1a over each
    /// sample's IEEE-754 little-endian bit pattern.  Not cryptographic;
    /// good enough to detect any change to the bake output.
    fn fnv1a_64(samples: &[f32]) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf29ce4_84222325;
        const FNV_PRIME: u64 = 0x100000001b3;
        let mut h = FNV_OFFSET;
        for s in samples {
            for byte in s.to_bits().to_le_bytes() {
                h ^= u64::from(byte);
                h = h.wrapping_mul(FNV_PRIME);
            }
        }
        h
    }

    fn silence_patch(seed: u32) -> AudioPatch {
        AudioPatch {
            seed,
            graph: NodeGraph {
                nodes: vec![GraphNode {
                    id: NodeId(0),
                    kind: NodeKind::Silence,
                    inputs: BTreeMap::new(),
                }],
                output: NodeId(0),
            },
        }
    }

    #[test]
    fn zero_duration_yields_empty_buffer() {
        let p = silence_patch(0);
        assert!(bake(&p, 44_100, 0.0).is_empty());
    }

    #[test]
    fn negative_duration_yields_empty_buffer() {
        let p = silence_patch(0);
        assert!(bake(&p, 44_100, -1.0).is_empty());
    }

    #[test]
    fn buffer_length_matches_rate_times_duration() {
        let p = silence_patch(0);
        // 0.5 s @ 48 kHz = 24 000 samples.
        let buf = bake(&p, 48_000, 0.5);
        assert_eq!(buf.len(), 24_000);
        // 1 s @ 44.1 kHz = 44 100 samples.
        let buf2 = bake(&p, 44_100, 1.0);
        assert_eq!(buf2.len(), 44_100);
    }

    #[test]
    fn silence_only_patch_produces_all_zeros() {
        let p = silence_patch(0);
        let buf = bake(&p, 44_100, 0.01); // 441 samples
        assert_eq!(buf.len(), 441);
        for (i, s) in buf.iter().enumerate() {
            assert_eq!(*s, 0.0_f32, "sample {i} not zero: {s}");
        }
    }

    #[test]
    fn bake_is_deterministic_across_repeated_calls() {
        // A graph with constant-wired and node-wired inputs, plus a
        // multi-node DAG, exercises the per-sample input resolution and
        // BTreeMap iteration paths even though Silence ignores its inputs.
        let mut n1_inputs = BTreeMap::new();
        n1_inputs.insert("a".to_string(), Connection::from_node(NodeId(0)));
        n1_inputs.insert("b".to_string(), Connection::constant(0.25));
        let patch = AudioPatch {
            seed: 0xDEAD_BEEF,
            graph: NodeGraph {
                nodes: vec![
                    GraphNode {
                        id: NodeId(0),
                        kind: NodeKind::Silence,
                        inputs: BTreeMap::new(),
                    },
                    GraphNode {
                        id: NodeId(1),
                        kind: NodeKind::Silence,
                        inputs: n1_inputs,
                    },
                ],
                output: NodeId(1),
            },
        };
        let a = bake(&patch, 44_100, 0.1);
        let b = bake(&patch, 44_100, 0.1);
        assert_eq!(a, b);
    }

    #[test]
    fn bake_hash_pinned_for_silent_quarter_second() {
        // Regression pin: every f32 in the buffer is +0.0, and the FNV-1a
        // hash of 11_025 zero bytes-worth-of-zero-f32s is a constant.  Any
        // change to bake's output shape (length, samples, alignment) flips
        // this hash and the test fails loudly.
        let p = silence_patch(0);
        let buf = bake(&p, 44_100, 0.25);
        assert_eq!(buf.len(), 11_025);
        let h = fnv1a_64(&buf);
        // Computed once and pinned.  This is the hash of 11_025 f32 zeros.
        assert_eq!(h, 0xC7D0_6137_6364_38F5);
    }

    #[test]
    fn fnv1a_self_check_against_known_input() {
        // Lock down the hash function itself — if the FNV constants drift,
        // this fails before the bake-hash test does, giving a cleaner
        // diagnosis.
        let h = fnv1a_64(&[0.0_f32]);
        // FNV-1a over four zero bytes (the bit pattern of +0.0_f32).
        let expected: u64 = {
            const FNV_OFFSET: u64 = 0xcbf29ce4_84222325;
            const FNV_PRIME: u64 = 0x100000001b3;
            let mut h = FNV_OFFSET;
            for _ in 0..4 {
                h = h.wrapping_mul(FNV_PRIME);
            }
            h
        };
        assert_eq!(h, expected);
    }

    #[test]
    #[should_panic(expected = "structurally invalid")]
    fn invalid_graph_panics_loudly() {
        // Output points at a node id that doesn't exist.
        let p = AudioPatch {
            seed: 0,
            graph: NodeGraph {
                nodes: vec![GraphNode {
                    id: NodeId(0),
                    kind: NodeKind::Silence,
                    inputs: BTreeMap::new(),
                }],
                output: NodeId(99),
            },
        };
        let _ = bake(&p, 44_100, 0.01);
    }
}
