//! `bevy_symbios_audio` — algorithmic audio generation for Bevy.
//!
//! A DAG-of-nodes synth (sine / square / sawtooth / triangle oscillators,
//! white / pink / brown noise, ADSR envelopes, biquad LP/HP/BP filters,
//! LFOs, [`Mix`] / [`Gain`] (VCA) combiners, a sequencer-driven [`Gate`],
//! [`Chorus`] and [`Reverb`] delay-line effects, and cross-node modulation
//! routing) producing deterministic `Vec<f32>` buffers off the main thread
//! via a private rayon pool, with an optional content-addressed
//! [`PatchCache`] (memory + on-disk WAV).
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
//!   (Mix/Gain), [`gate`], [`chorus`], [`reverb`] — the built-in node
//!   implementations.
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
//! - `ui` — optional `bevy_egui` patch / sequence editor widgets, behind
//!   the `egui` Cargo feature.
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
//! For a Bevy-free `Vec<f32>` (unit tests, CLI, offline tooling) skip the
//! ECS entirely and call [`bake()`] / [`try_bake`] on the patch directly.
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

// Bevy-coupled modules kept in the wrapper.  The pure DSP language
// (adsr, bake, chorus, filter, gate, genetics, lfo, mix, mixdown, node,
// noise, oscillator, patch, reverb, sequence, wav) now lives in the
// Bevy-free `symbios-audio` core crate and is re-exported below.
pub mod async_gen;
pub mod audio_source;
pub mod cache;

// Egui editor widgets for the patch schema, behind the `egui` Cargo
// feature.  The module documents itself via its own `//!` header (see
// `ui/mod.rs`); keeping the description there — rather than in an outer
// `///` here — means its intra-doc links resolve in the `ui` scope.
#[cfg(feature = "egui")]
pub mod ui;

// Re-export the entire Bevy-free core so the public API
// (`bevy_symbios_audio::adsr`, `::bake`, `::AudioPatch`, `::NodeKind`,
// `::bake_sequence`, the `impl_genotype!` macro, …) is preserved
// byte-for-byte.  This pulls in every pure module + the pure type/fn
// re-exports the core's `lib.rs` declares.
pub use symbios_audio::*;

// `Event`, `Mix`, and `Node` also exist in `bevy::prelude` (glob-imported
// below for the plugin's `App`/`Plugin`/system items).  Re-export the
// audio ones explicitly so they win unambiguously over the prelude glob —
// `bevy_symbios_audio::{Event, Mix, Node}` resolve to the audio types,
// exactly as before the split.
pub use symbios_audio::{Event, Mix, Node};

// `impl_genotype!` is `#[macro_export]`ed from the core crate (so it lands
// at `symbios_audio::impl_genotype`); glob re-exports don't cover macros,
// so re-export it explicitly to keep `bevy_symbios_audio::impl_genotype!`
// available exactly as before the split.
pub use symbios_audio::impl_genotype;

// Wrapper-only re-exports (the Bevy-coupled surface) — the async-bake
// handover, the WAV→AudioSource bridge, and the `Resource` cache.  The
// pure `samples_to_wav_bytes` / `MAX_WAV_SAMPLES` already arrive via the
// `symbios_audio::*` glob above (and `audio_source` re-exports them too),
// so they're intentionally omitted here to avoid an ambiguous glob.
pub use async_gen::{
    AsyncAudioConfig, AudioPatchReady, CacheOrPending, DEFAULT_POOL_THREADS, PendingAudioPatch,
    bake_with_cache,
};
pub use audio_source::samples_to_audio_source;
pub use cache::{
    DEFAULT_MEMORY_CACHE_ENTRIES, FileStore, MemoryStore, PatchCache, PatchCacheKey,
    PatchCacheStore,
};

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
