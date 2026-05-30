//! `bevy_symbios_audio` — algorithmic audio generation for Bevy.
//!
//! A DAG-of-nodes synth (sine / square / sawtooth / triangle oscillators,
//! white / pink / brown noise, ADSR envelopes, biquad LP/HP/BP filters,
//! LFOs and cross-node modulation routing) producing deterministic
//! `Vec<f32>` buffers off the main thread via a private rayon pool, with
//! an optional content-addressed [`PatchCache`] (memory + on-disk WAV).
//!
//! # Architecture
//!
//! The crate is monolithic but layered.  Lower layers are Bevy-free and
//! unit-testable in isolation:
//!
//! - [`patch`] — schema + topology.
//! - [`node`] — `Node` trait + `BakeContext` + the closed `NodeKind`
//!   enum that all built-in nodes plug into.
//! - [`oscillator`], [`noise`], [`adsr`], [`filter`], [`lfo`], [`mix`]
//!   (Mix/Gain), [`gate`] — the built-in node implementations.
//! - [`mod@bake`] — turns one [`AudioPatch`] into `Vec<f32>`.
//! - [`sequence`] + [`mixdown`] — the timeline-of-events layer and the
//!   seamless-loop-aware [`bake_sequence`].
//! - [`genetics`] — declarative [`impl_genotype!`] macro and shared
//!   mutation helpers that wire every config struct into
//!   `symbios-genetics`.
//!
//! Bevy enters at:
//!
//! - [`audio_source`] — the `Vec<f32>` → Bevy `AudioSource` bridge
//!   (writes an in-memory WAV blob).
//! - [`cache`] — Bevy `Resource` wrapper over the cache backends.
//! - [`async_gen`] — `PendingAudioPatch`/`AudioPatchReady` ECS handover.
//! - [`SymbiosAudioPlugin`] — the plugin entry point.
//!
//! Every config type implements [`symbios_genetics::Genotype`] via the
//! [`impl_genotype!`] declarative macro, so the entire DSP language plugs
//! into the evolutionary search algorithms in the `symbios-genetics`
//! crate.
//!
//! # Three usage modes
//!
//! ## 1. One-shot bake of a single [`AudioPatch`] → `AudioSource`
//!
//! Build a patch programmatically (or load one from JSON), spawn a
//! [`PendingAudioPatch`] onto an entity, and the polling system attaches
//! an [`AudioPatchReady`] with a `Handle<AudioSource>` once the bake
//! completes.  Use [`SymbiosAudioPlugin`] to register the poller.
//!
//! ```rust,ignore
//! use bevy::prelude::*;
//! use bevy_symbios_audio::{
//!     AudioPatch, NodeGraph, GraphNode, NodeId, NodeKind, SineOsc,
//!     PendingAudioPatch, SymbiosAudioPlugin,
//! };
//!
//! App::new()
//!     .add_plugins((DefaultPlugins, SymbiosAudioPlugin::default()))
//!     .add_systems(Startup, |mut commands: Commands| {
//!         let patch = AudioPatch {
//!             seed: 0,
//!             graph: NodeGraph {
//!                 nodes: vec![GraphNode {
//!                     id: NodeId(0),
//!                     kind: NodeKind::Sine(SineOsc::default()),
//!                     ..Default::default()
//!                 }],
//!                 output: NodeId(0),
//!             },
//!         };
//!         commands.spawn(PendingAudioPatch::new(patch, 44_100, 1.0));
//!     });
//! ```
//!
//! ## 2. Sequence bake of a [`SequenceRecipe`] → loopable `AudioSource`
//!
//! For ambient tracks, build a [`SequenceRecipe`] with instruments
//! (named [`AudioPatch`]es) and timed [`struct@Event`]s.  Call [`bake_sequence`]
//! directly (offline) or wire it through the same plugin/cache path used
//! for one-shot bakes.  Set `loop_start_beats` for seamless loops — the
//! mixdown baker pre-mixes a tail crossfade so the buffer hard-loops
//! through rodio's `Source::loop_..()` with no click at the seam.
//!
//! ## 3. CLI export — `symbios-audio-cli`
//!
//! A standalone binary ships with the crate for offline baking outside
//! the Bevy app:
//!
//! ```text
//! cargo run --release --bin symbios-audio-cli -- bake \
//!     --sample-rate 48000 --duration 5.0 patch.json out.wav
//! ```
//!
//! The CLI reads a serde-JSON [`AudioPatch`], bakes it, and writes a
//! mono IEEE-float WAV — handy for video-pipeline use and sound-design
//! iteration.
//!
//! # Serde
//!
//! [`AudioPatch`] and [`SequenceRecipe`] are serde-JSON only.  DAG-CBOR
//! conversion is the consumer's concern (Overlands' PDS layer mirrors
//! these as `Sovereign*` types with `Fp`/`Fp64` numeric wrappers).
//!
//! The public types are all `Default`-constructible (including
//! [`Connection`], [`GraphNode`], [`NodeId`]) so mirror shims can
//! `..Default::default()` their way to a known empty starting point
//! without enumerating every field.

pub mod adsr;
pub mod async_gen;
pub mod audio_source;
pub mod bake;
pub mod cache;
pub mod filter;
pub mod gate;
pub mod genetics;
pub mod lfo;
pub mod mix;
pub mod mixdown;
pub mod node;
pub mod noise;
pub mod oscillator;
pub mod patch;
pub mod sequence;

pub use adsr::{AdsrCurve, AdsrEnvelope};
pub use async_gen::{
    AsyncAudioConfig, AudioPatchReady, CacheOrPending, DEFAULT_POOL_THREADS, PendingAudioPatch,
    bake_with_cache,
};
pub use audio_source::{samples_to_audio_source, samples_to_wav_bytes};
pub use bake::{bake, try_bake};
pub use cache::{
    DEFAULT_MEMORY_CACHE_ENTRIES, FileStore, MemoryStore, PatchCache, PatchCacheKey,
    PatchCacheStore,
};
pub use filter::{BiquadBandpass, BiquadHighpass, BiquadLowpass, BiquadState};
pub use gate::Gate;
pub use lfo::{Lfo, LfoShape};
pub use mix::{Gain, Mix};
pub use mixdown::bake_sequence;
pub use node::{BakeContext, Node, NodeKind};
pub use noise::{BrownNoise, PinkNoise, WhiteNoise};
pub use oscillator::{OscPhase, SawPolarity, SawtoothOsc, SineOsc, SquareOsc, TriangleOsc};
pub use patch::{AudioPatch, Connection, GraphError, GraphNode, NodeGraph, NodeId, topo_sort};
pub use sequence::{Event, Instrument, SequenceRecipe, Track};

use bevy::prelude::*;

/// Bevy plugin — registers the async-bake polling system and applies the
/// [`AsyncAudioConfig`] to the private audio-bake thread pool.
///
/// Construct via [`Default`] for the standard [`DEFAULT_POOL_THREADS`]
/// cap, or set [`SymbiosAudioPlugin::config`] explicitly for custom pool
/// sizing.  The configuration is applied exactly once; later additions of
/// the plugin (or other paths that call [`async_gen::set_pool_config`]
/// first) are silently ignored.
#[derive(Default)]
pub struct SymbiosAudioPlugin {
    /// Configuration for the private audio-bake thread pool.
    pub config: AsyncAudioConfig,
}

impl Plugin for SymbiosAudioPlugin {
    fn build(&self, app: &mut App) {
        let _ = async_gen::set_pool_config(self.config.clone());
        app.insert_resource(self.config.clone());
        app.add_systems(Update, async_gen::poll_audio_tasks);
    }
}
