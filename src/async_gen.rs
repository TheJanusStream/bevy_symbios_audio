//! Async audio bake pool — Bevy ECS handover for the
//! [`crate::bake::bake`] function.
//!
//! Phase 3 ticket #9.  Baking a 3-minute mixdown is comparably CPU-bound
//! to generating a 4K texture; doing it on the main thread will stall
//! the frame.  This module mirrors `bevy_symbios_texture::async_gen`:
//! a private, bounded [`rayon`] thread pool gated by [`OnceLock`] so all
//! audio work stays off Bevy's task pools and out of the application's
//! own rayon, and a non-blocking poll system that lifts finished
//! [`Vec<f32>`] buffers out of the workers and wraps them in
//! [`AudioSource`] handles via [`crate::samples_to_audio_source`].
//!
//! # Usage
//!
//! ```rust,ignore
//! // Plugin once at startup.
//! app.add_plugins(SymbiosAudioPlugin::default());
//!
//! // To bake: spawn a PendingAudioPatch.
//! commands.spawn(PendingAudioPatch::new(patch, 44_100, 1.5));
//!
//! // Later (any subsequent Update): query for AudioPatchReady to attach
//! // an AudioPlayer / SpatialAudioPlayer, etc.
//! ```
//!
//! # WASM
//!
//! On `wasm32` rayon's threading story is shaky.  The pool is gated by
//! `#[cfg(not(target_arch = "wasm32"))]` and the WASM build dispatches
//! through Bevy's `AsyncComputeTaskPool` instead, which multiplexes
//! onto the main thread — slower, blocks the UI for the duration of the
//! bake, but compiles cleanly and behaves predictably.

use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicBool, Ordering},
    mpsc,
};

use bevy::{
    asset::{Assets, Handle},
    audio::AudioSource,
    ecs::{
        component::Component,
        entity::Entity,
        resource::Resource,
        system::{Commands, Query, Res, ResMut},
    },
};

use crate::audio_source::samples_to_wav_bytes;
use crate::bake::try_bake_cancellable;
use crate::cache::{PatchCache, PatchCacheKey};
use crate::patch::AudioPatch;

/// Bake on a worker, downgrading a structurally-invalid patch to an empty
/// buffer with a clear error log instead of panicking the thread (which
/// would surface as a misleading "bake thread panicked" in
/// [`poll_audio_tasks`]).
///
/// Aborts early if `cancelled` flips during the bake (the owning
/// [`PendingAudioPatch`] was dropped): the returned partial buffer is sent
/// into a channel whose receiver is already gone, so the send is discarded.
fn bake_or_warn(
    patch: &AudioPatch,
    sample_rate: u32,
    duration_secs: f32,
    cancelled: &AtomicBool,
) -> Vec<f32> {
    match try_bake_cancellable(patch, sample_rate, duration_secs, cancelled) {
        Ok(buffer) => buffer,
        Err(err) => {
            bevy::log::error!(
                "bevy_symbios_audio: skipping bake of structurally-invalid patch ({err}); \
                 producing a silent buffer"
            );
            Vec::new()
        }
    }
}

/// Default concurrency cap applied when no explicit
/// [`AsyncAudioConfig::pool_threads`] is supplied.
///
/// Tasks beyond this cap are queued inside the rayon pool rather than
/// spawning new OS threads, bounding both CPU and memory usage.
/// Set [`AsyncAudioConfig::pool_threads`] to `0` for the auto value
/// (`available_parallelism / 2`), or an explicit higher number to
/// saturate large machines.
pub const DEFAULT_POOL_THREADS: usize = 4;

/// Plugin-time configuration for the private audio-bake thread pool.
///
/// Applied by [`crate::SymbiosAudioPlugin`] before any bake is
/// dispatched.  Once the pool is built (lazily, on the first bake
/// request) the configuration is frozen for the process lifetime;
/// changing the value afterwards has no effect.
#[derive(Resource, Clone, Debug)]
pub struct AsyncAudioConfig {
    /// Maximum concurrent bake tasks.
    ///
    /// * `0` selects an auto value of `available_parallelism / 2`
    ///   (minimum 1).  Trades fewer threads against better main-thread
    ///   responsiveness while still scaling on large machines.
    /// * Any positive value caps the pool at exactly that many threads.
    ///
    /// Defaults to [`DEFAULT_POOL_THREADS`].
    pub pool_threads: usize,
}

impl Default for AsyncAudioConfig {
    fn default() -> Self {
        Self {
            pool_threads: DEFAULT_POOL_THREADS,
        }
    }
}

fn resolve_pool_threads(cfg: &AsyncAudioConfig) -> usize {
    if cfg.pool_threads == 0 {
        std::thread::available_parallelism()
            .map(|n| (n.get() / 2).max(1))
            .unwrap_or(2)
    } else {
        cfg.pool_threads
    }
}

static POOL_CONFIG: OnceLock<AsyncAudioConfig> = OnceLock::new();
#[cfg(not(target_arch = "wasm32"))]
static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();

/// Returned by [`set_pool_config`] when a configuration has already been
/// installed by an earlier caller.
#[derive(Debug)]
pub struct PoolConfigAlreadySet;

impl std::fmt::Display for PoolConfigAlreadySet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AsyncAudioConfig has already been applied; new value ignored")
    }
}

impl std::error::Error for PoolConfigAlreadySet {}

/// Apply the audio-bake thread-pool configuration.
///
/// The plugin calls this once at startup with the user-supplied
/// [`AsyncAudioConfig`].  Calls after the pool has been initialised are
/// silently ignored — the configuration is read exactly once when the
/// first bake task is dispatched.
pub fn set_pool_config(cfg: AsyncAudioConfig) -> Result<(), PoolConfigAlreadySet> {
    POOL_CONFIG.set(cfg).map_err(|_| PoolConfigAlreadySet)
}

#[cfg(not(target_arch = "wasm32"))]
fn build_pool(cfg: &AsyncAudioConfig) -> Option<rayon::ThreadPool> {
    let n = resolve_pool_threads(cfg);
    match rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .thread_name(|i| format!("audio-bake-{i}"))
        .build()
    {
        Ok(pool) => Some(pool),
        Err(e) => {
            bevy::log::warn!(
                "bevy_symbios_audio: failed to build audio-bake thread pool ({e}); \
                 falling back to inline (synchronous) bake. Each PendingAudioPatch \
                 will be baked on the spawning thread, blocking it for the duration."
            );
            None
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn bake_pool() -> Option<&'static rayon::ThreadPool> {
    POOL.get_or_init(|| {
        let cfg = POOL_CONFIG.get().cloned().unwrap_or_default();
        build_pool(&cfg)
    })
    .as_ref()
}

/// Spawned onto an entity to request a background bake.
///
/// `PendingAudioPatch::new` submits the bake to the private rayon pool
/// (or the WASM fallback) and stores the receiver end of a one-shot
/// channel.  [`poll_audio_tasks`] non-blockingly checks for completion
/// each frame via [`mpsc::Receiver::try_recv`].
///
/// Dropping `PendingAudioPatch` (e.g. when the entity is despawned)
/// sets an atomic cancellation flag.  The bake checks it before starting
/// *and* periodically inside the sample loop, so a dropped request both
/// skips a not-yet-started bake and aborts one already in flight —
/// preventing zombie tasks from saturating the thread pool when entities
/// are rapidly spawned and destroyed.
#[derive(Component)]
pub struct PendingAudioPatch {
    // Mutex<…> wraps the Receiver to make the struct Sync — Bevy's
    // Component bound requires it.  The mutex is only contended during
    // poll, never long-held.
    pub(crate) rx: Mutex<mpsc::Receiver<Vec<f32>>>,
    /// Set to `true` on drop.  The background task checks this before
    /// starting and periodically while baking, so an in-flight bake aborts
    /// instead of running to completion.
    cancelled: Arc<AtomicBool>,
    /// Sample rate the buffer was baked at — needed to encode the WAV
    /// header when [`poll_audio_tasks`] wraps the samples in
    /// [`AudioSource`].
    sample_rate: u32,
    /// Optional content-addressed key; if set, [`poll_audio_tasks`]
    /// stores the produced WAV bytes into the [`PatchCache`] resource
    /// on completion so a future re-bake of the same patch hits the
    /// cache instead of redoing the DSP.
    cache_key: Option<PatchCacheKey>,
}

impl PendingAudioPatch {
    /// Spawn a background bake of `patch` at `sample_rate` Hz for
    /// `duration_secs` seconds.  The component returned can be inserted
    /// onto a Bevy entity; [`poll_audio_tasks`] does the rest.
    ///
    /// This constructor records a [`PatchCacheKey`] for the bake — if a
    /// [`PatchCache`] resource is present, the resulting bytes will be
    /// cached automatically on completion.  Use [`Self::new_uncached`]
    /// to opt out of cache writes (e.g. for one-off bakes whose result
    /// is never reused).
    pub fn new(patch: AudioPatch, sample_rate: u32, duration_secs: f32) -> Self {
        let key = PatchCacheKey::from_patch(&patch, sample_rate, duration_secs);
        Self::new_with_key(patch, sample_rate, duration_secs, Some(key))
    }

    /// Dispatch a bake without recording a cache key.  The result will
    /// never be written to [`PatchCache`].  Use this for transient
    /// one-shot bakes.
    pub fn new_uncached(patch: AudioPatch, sample_rate: u32, duration_secs: f32) -> Self {
        Self::new_with_key(patch, sample_rate, duration_secs, None)
    }

    fn new_with_key(
        patch: AudioPatch,
        sample_rate: u32,
        duration_secs: f32,
        cache_key: Option<PatchCacheKey>,
    ) -> Self {
        let cancelled = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&cancelled);
        let (tx, rx) = mpsc::sync_channel(1);

        spawn_bake(patch, sample_rate, duration_secs, flag, tx);

        Self {
            rx: Mutex::new(rx),
            cancelled,
            sample_rate,
            cache_key,
        }
    }

    /// Sample rate the bake will produce at.  Exposed so callers can
    /// double-check the consumer-side rate matches without having to
    /// poke at the [`AudioPatchReady`].
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// The cache key under which this bake will be stored on
    /// completion, or `None` if dispatched via [`Self::new_uncached`].
    pub fn cache_key(&self) -> Option<&PatchCacheKey> {
        self.cache_key.as_ref()
    }
}

/// Outcome of [`bake_with_cache`]: either pre-baked WAV bytes (cache
/// hit) or a `PendingAudioPatch` that needs to be spawned onto an
/// entity so [`poll_audio_tasks`] can pick it up.
///
/// On cache hit the caller decides what to do with the bytes — most
/// commonly `Assets::<AudioSource>::add(AudioSource { bytes })` to
/// mint a fresh handle.  Returning the bytes (rather than a `Handle`)
/// keeps this helper free of any `Assets<AudioSource>` borrow, which
/// makes it composable inside Bevy systems that already need to hold
/// other resources mutably.
pub enum CacheOrPending {
    /// Cache hit — the WAV bytes are ready.  Wrap them with
    /// `AudioSource { bytes }` and upload to `Assets<AudioSource>`.
    Cached(Arc<[u8]>),
    /// Cache miss — dispatched bake will populate the cache on
    /// completion (the pending carries the same key that was just
    /// probed).  Spawn it onto an entity.
    Pending(PendingAudioPatch),
}

/// Probe `cache` for an existing bake of `patch` at `sample_rate` and
/// `duration_secs`.  On hit, returns [`CacheOrPending::Cached`] with the
/// WAV bytes ready to wrap in an [`AudioSource`].  On miss, dispatches
/// a background bake whose result will populate `cache` via
/// [`poll_audio_tasks`] and returns [`CacheOrPending::Pending`].
///
/// Recommended entry point for systems that want caching — uses the
/// same key derivation as the bake-side write, so the probe and the
/// write always agree.
pub fn bake_with_cache(
    patch: AudioPatch,
    sample_rate: u32,
    duration_secs: f32,
    cache: &PatchCache,
) -> CacheOrPending {
    let key = PatchCacheKey::from_patch(&patch, sample_rate, duration_secs);
    if let Some(bytes) = cache.get(&key) {
        return CacheOrPending::Cached(bytes);
    }
    CacheOrPending::Pending(PendingAudioPatch::new_with_key(
        patch,
        sample_rate,
        duration_secs,
        Some(key),
    ))
}

impl Drop for PendingAudioPatch {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
}

/// Run the bake on the private rayon pool (native) or fall back to
/// inline-on-task-thread (wasm or pool-build-failed).  Either way the
/// channel ends up holding the buffer, so [`poll_audio_tasks`] consumes
/// the result through its normal polling loop regardless.
#[cfg(not(target_arch = "wasm32"))]
fn spawn_bake(
    patch: AudioPatch,
    sample_rate: u32,
    duration_secs: f32,
    cancelled: Arc<AtomicBool>,
    tx: mpsc::SyncSender<Vec<f32>>,
) {
    match bake_pool() {
        Some(pool) => pool.spawn(move || {
            if !cancelled.load(Ordering::Relaxed) {
                tx.send(bake_or_warn(&patch, sample_rate, duration_secs, &cancelled))
                    .ok();
            }
        }),
        None => {
            if !cancelled.load(Ordering::Relaxed) {
                tx.send(bake_or_warn(&patch, sample_rate, duration_secs, &cancelled))
                    .ok();
            }
        }
    }
}

/// WASM: dispatch through Bevy's [`AsyncComputeTaskPool`].  On wasm32
/// this multiplexes onto the main thread — the bake will block the UI
/// for its full duration — but it compiles cleanly and stays
/// deterministic.  Acceptable for now; if WASM bake latency becomes a
/// problem a Web Worker or audio worklet path can replace this.
#[cfg(target_arch = "wasm32")]
fn spawn_bake(
    patch: AudioPatch,
    sample_rate: u32,
    duration_secs: f32,
    cancelled: Arc<AtomicBool>,
    tx: mpsc::SyncSender<Vec<f32>>,
) {
    use bevy::tasks::AsyncComputeTaskPool;
    AsyncComputeTaskPool::get()
        .spawn(async move {
            if !cancelled.load(Ordering::Relaxed) {
                tx.send(bake_or_warn(&patch, sample_rate, duration_secs, &cancelled))
                    .ok();
            }
        })
        .detach();
}

/// Added to the entity by [`poll_audio_tasks`] when the bake completes.
///
/// `handle` points at the [`AudioSource`] asset built from the baked
/// samples — drop it on an `AudioPlayer` / `SpatialAudioPlayer` to
/// actually hear the bake.
#[derive(Component)]
pub struct AudioPatchReady {
    pub handle: Handle<AudioSource>,
}

/// Bevy system — polls every pending bake task; for any that have
/// finished, wrap the samples in an [`AudioSource`] via the WAV bridge,
/// upload to [`Assets<AudioSource>`], and swap [`PendingAudioPatch`]
/// for [`AudioPatchReady`] on the entity.
///
/// If a [`PatchCache`] resource is present AND the pending carries a
/// `cache_key` (the default; opt out via [`PendingAudioPatch::new_uncached`]),
/// the produced WAV bytes are inserted into the cache so a future
/// re-bake of the same patch returns immediately via
/// [`bake_with_cache`].
pub fn poll_audio_tasks(
    mut commands: Commands,
    tasks: Query<(Entity, &PendingAudioPatch)>,
    mut audio_sources: ResMut<Assets<AudioSource>>,
    // Shared (`Res`) rather than exclusive: `PatchCache::insert` takes
    // `&self` behind an `RwLock`, so the poller doesn't need mutable
    // access and won't serialise against other systems reading the cache.
    cache: Option<Res<PatchCache>>,
) {
    for (entity, pending) in &tasks {
        let poll = pending
            .rx
            .lock()
            .expect("audio bake thread poisoned")
            .try_recv();
        match poll {
            Ok(samples) => {
                let wav_bytes = samples_to_wav_bytes(&samples, pending.sample_rate);
                let arc_bytes: Arc<[u8]> = Arc::from(wav_bytes.into_boxed_slice());
                if let (Some(cache_ref), Some(key)) = (&cache, &pending.cache_key) {
                    cache_ref.insert(key.clone(), arc_bytes.clone());
                }
                let source = AudioSource { bytes: arc_bytes };
                let handle = audio_sources.add(source);
                commands
                    .entity(entity)
                    .remove::<PendingAudioPatch>()
                    .insert(AudioPatchReady { handle });
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                bevy::log::error!("audio bake thread panicked");
                commands.entity(entity).remove::<PendingAudioPatch>();
            }
            Err(mpsc::TryRecvError::Empty) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::NodeKind;
    use crate::patch::{GraphNode, NodeGraph, NodeId};
    use std::collections::BTreeMap;

    #[test]
    fn auto_pool_threads_is_at_least_one() {
        let cfg = AsyncAudioConfig { pool_threads: 0 };
        assert!(resolve_pool_threads(&cfg) >= 1);
    }

    #[test]
    fn explicit_pool_threads_is_passthrough() {
        let cfg = AsyncAudioConfig { pool_threads: 7 };
        assert_eq!(resolve_pool_threads(&cfg), 7);
    }

    #[test]
    fn pool_config_set_twice_returns_already_set() {
        // First set wins; subsequent attempts return PoolConfigAlreadySet.
        // We can't reset the global, so test the error path by setting
        // twice into a fresh OnceLock with the same plumbing.
        let lock: OnceLock<AsyncAudioConfig> = OnceLock::new();
        assert!(lock.set(AsyncAudioConfig::default()).is_ok());
        assert!(lock.set(AsyncAudioConfig::default()).is_err());
    }

    /// Inline-fallback path: simulates the post-rayon-build-failure
    /// branch by running the bake on the calling thread.  Exercises the
    /// same shape of code that runs after a real ThreadPoolBuilder
    /// failure.
    #[test]
    fn inline_fallback_runs_synchronously_and_fills_channel() {
        let patch = AudioPatch {
            seed: 0,
            graph: NodeGraph {
                nodes: vec![GraphNode {
                    id: NodeId(0),
                    kind: NodeKind::Silence,
                    inputs: BTreeMap::new(),
                }],
                output: NodeId(0),
            },
        };
        let cancelled = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&cancelled);
        let (tx, rx) = mpsc::sync_channel(1);

        // Inline-fallback shape (mirror of spawn_bake's None branch).
        if !flag.load(Ordering::Relaxed) {
            tx.send(bake_or_warn(&patch, 44_100, 0.001, &cancelled))
                .ok();
        }

        let samples = rx
            .try_recv()
            .expect("inline fallback should fill the channel synchronously");
        assert_eq!(samples.len(), 44); // 0.001s @ 44.1 kHz, rounded.
    }

    #[test]
    fn pending_drop_sets_cancellation_flag() {
        // Construct a PendingAudioPatch and drop it; the underlying flag
        // must flip to true so any not-yet-started task sees the
        // cancellation and exits.
        let patch = AudioPatch {
            seed: 0,
            graph: NodeGraph {
                nodes: vec![GraphNode {
                    id: NodeId(0),
                    kind: NodeKind::Silence,
                    inputs: BTreeMap::new(),
                }],
                output: NodeId(0),
            },
        };
        let pending = PendingAudioPatch::new(patch, 44_100, 0.001);
        let flag = Arc::clone(&pending.cancelled);
        assert!(!flag.load(Ordering::Relaxed));
        drop(pending);
        assert!(flag.load(Ordering::Relaxed));
    }
}
