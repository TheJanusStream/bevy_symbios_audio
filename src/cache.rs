//! Patch cache: content-addressed store for baked WAV bytes.
//!
//! Phase 3 ticket #11.  A baked 3-minute mixdown is a few hundred
//! kilobytes of `f32` samples (plus WAV header) that took non-trivial
//! CPU to produce — re-baking on every room re-entry would melt the
//! battery.  This module mirrors `bevy_symbios_texture::cache`: a small
//! [`PatchCacheStore`] trait with [`MemoryStore`] (bounded FIFO) and
//! [`FileStore`] (disk-backed) implementations, wrapped in a
//! [`PatchCache`] Bevy resource.
//!
//! The cache value type is `Arc<[u8]>` — the raw WAV bytes produced by
//! [`crate::samples_to_wav_bytes`].  This matches `AudioSource.bytes`
//! exactly, so a cache hit is a single
//! `AudioSource { bytes: arc.clone() }` away from a playable handle.
//!
//! # Key shape
//!
//! [`PatchCacheKey`] hashes the canonical JSON serialisation of the
//! [`AudioPatch`] together with `sample_rate` and `duration_secs` (as
//! `f32::to_bits` so floats stay `Hash + Eq`-able).  The schema uses
//! `BTreeMap` and structs only — no `HashMap` — so serialisation is
//! deterministic across runs.  Different Rust versions may emit
//! different `DefaultHasher` outputs, so the on-disk cache is not
//! portable across compiler upgrades; bump
//! [`PatchCache::manifest_version`] when that matters.
//!
//! # Overlands integration note
//!
//! For the DID-seeded ambient layer in the Overlands repo, the cache key
//! is effectively the room's audio seed → one ambient track per room.
//! The `FileStore` flavour is what lets the player skip the bake on
//! re-entry.

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use bevy::ecs::resource::Resource;
use serde::{Deserialize, Serialize};

use crate::patch::AudioPatch;

/// Default maximum number of entries kept in [`MemoryStore`].
///
/// Each entry is the raw WAV blob of one bake — typically tens to
/// hundreds of kilobytes for a 1-second mixdown, scaling linearly with
/// duration.  64 entries hovers around a few megabytes of RAM for
/// short bakes and is enough to keep a small ambient palette warm
/// without thrashing.
pub const DEFAULT_MEMORY_CACHE_ENTRIES: usize = 64;

/// Content-addressed key for a baked patch.
///
/// `patch_fingerprint` is the [`std::hash::DefaultHasher`] digest of the
/// patch's canonical JSON form (struct field order + sorted BTreeMap
/// keys make this deterministic).  `sample_rate` and `duration_bits`
/// (the f32 duration's IEEE bit pattern) round out the cache identity
/// — re-baking the same patch at a different rate or duration is a
/// different cache entry.
#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct PatchCacheKey {
    pub patch_fingerprint: u64,
    pub sample_rate: u32,
    /// `f32::to_bits` of the requested duration in seconds.  Stored as
    /// `u32` so the key is `Hash + Eq`.
    pub duration_bits: u32,
}

impl PatchCacheKey {
    /// Derive a cache key from a patch and the bake parameters.
    ///
    /// Panics if the patch fails to serialise to JSON — in practice the
    /// only way that happens is a non-finite `f32` field (NaN /
    /// infinity), which represents a malformed patch and warrants
    /// the panic.
    pub fn from_patch(patch: &AudioPatch, sample_rate: u32, duration_secs: f32) -> Self {
        use std::hash::{DefaultHasher, Hasher};
        let mut h = DefaultHasher::new();
        let json = serde_json::to_vec(patch)
            .expect("AudioPatch must serialise to JSON for cache fingerprinting");
        h.write(&json);
        Self {
            patch_fingerprint: h.finish(),
            sample_rate,
            duration_bits: duration_secs.to_bits(),
        }
    }
}

/// Trait implemented by patch cache backends.
///
/// Implementations must be `Send + Sync` — the resource lookup hands a
/// `&mut PatchCache` to systems on the main scheduling thread, but the
/// trait object may be queried from any thread that holds a reference.
pub trait PatchCacheStore: Send + Sync {
    /// Returns the WAV bytes previously stored under `key`, or `None`
    /// on miss.  Implementations that load lazily (e.g. [`FileStore`])
    /// should perform the I/O here.
    fn get(&mut self, key: &PatchCacheKey) -> Option<Arc<[u8]>>;

    /// Stores `bytes` under `key`, evicting older entries if needed.
    fn put(&mut self, key: PatchCacheKey, bytes: Arc<[u8]>);
}

/// Bevy resource wrapper for any [`PatchCacheStore`] implementation.
///
/// Insert this resource before adding [`crate::SymbiosAudioPlugin`] to
/// enable caching:
///
/// ```rust,ignore
/// app.insert_resource(PatchCache::memory(DEFAULT_MEMORY_CACHE_ENTRIES));
/// ```
#[derive(Resource)]
pub struct PatchCache {
    /// Application-supplied schema version.  Not consumed by the
    /// built-in stores — entries are keyed on [`PatchCacheKey`] alone —
    /// but exposed so callers can rotate caches out-of-band when DSP
    /// internals change without a config-field change.
    pub manifest_version: u32,
    inner: Mutex<Box<dyn PatchCacheStore>>,
}

impl PatchCache {
    /// Wrap any [`PatchCacheStore`] in a [`PatchCache`] resource.
    pub fn new(store: Box<dyn PatchCacheStore>, manifest_version: u32) -> Self {
        Self {
            manifest_version,
            inner: Mutex::new(store),
        }
    }

    /// Convenience: in-memory cache with bounded capacity.
    pub fn memory(max_entries: usize) -> Self {
        Self::new(Box::new(MemoryStore::new(max_entries)), 0)
    }

    /// Convenience: file-backed cache rooted at `dir`.  The directory
    /// is created if missing.  Each entry produces one `<hash>.wav`
    /// file containing the bytes from [`crate::samples_to_wav_bytes`].
    pub fn file(dir: impl Into<PathBuf>, manifest_version: u32) -> std::io::Result<Self> {
        Ok(Self::new(
            Box::new(FileStore::new(dir.into())?),
            manifest_version,
        ))
    }

    /// Look up cached WAV bytes for `key`.  Returns `None` on miss or
    /// if the internal lock is poisoned (in which case the cache is
    /// degraded but the bake path still works).
    pub fn get(&self, key: &PatchCacheKey) -> Option<Arc<[u8]>> {
        self.inner.lock().ok()?.get(key)
    }

    /// Store WAV `bytes` under `key`, evicting older entries if the
    /// backend's capacity is reached.
    pub fn insert(&self, key: PatchCacheKey, bytes: Arc<[u8]>) {
        if let Ok(mut store) = self.inner.lock() {
            store.put(key, bytes);
        }
    }
}

// --- MemoryStore ------------------------------------------------------------

/// In-memory cache with bounded capacity and FIFO eviction.
///
/// FIFO on insertion order rather than full LRU — simpler, adequate
/// for the typical access pattern (palettes loaded in bulk, hits
/// clustered around hot patches).  Re-inserting an existing key
/// updates in place without evicting the next-oldest.
pub struct MemoryStore {
    max_entries: usize,
    entries: HashMap<PatchCacheKey, Arc<[u8]>>,
    insertion_order: VecDeque<PatchCacheKey>,
}

impl MemoryStore {
    /// Build a memory store bounded by `max_entries`.  Values below
    /// `1` are rounded up — a zero-sized cache is never useful.
    pub fn new(max_entries: usize) -> Self {
        let cap = max_entries.max(1);
        Self {
            max_entries: cap,
            entries: HashMap::with_capacity(cap),
            insertion_order: VecDeque::with_capacity(cap),
        }
    }
}

impl PatchCacheStore for MemoryStore {
    fn get(&mut self, key: &PatchCacheKey) -> Option<Arc<[u8]>> {
        self.entries.get(key).cloned()
    }

    fn put(&mut self, key: PatchCacheKey, bytes: Arc<[u8]>) {
        if let std::collections::hash_map::Entry::Occupied(mut e) = self.entries.entry(key.clone())
        {
            e.insert(bytes);
            return;
        }
        if self.entries.len() >= self.max_entries
            && let Some(oldest) = self.insertion_order.pop_front()
        {
            self.entries.remove(&oldest);
        }
        self.insertion_order.push_back(key.clone());
        self.entries.insert(key, bytes);
    }
}

// --- FileStore --------------------------------------------------------------

/// Disk-backed cache.  Each entry is a single `<hash>.wav` blob in
/// `dir` containing the bytes from [`crate::samples_to_wav_bytes`].
///
/// The on-disk filename is `<DefaultHasher(key)>.wav`.  Entries from
/// older builds may be unreadable if the patch schema changes — the
/// loader returns `None` on any I/O error and treats stale files as
/// inert rather than fatal.
pub struct FileStore {
    root: PathBuf,
}

impl FileStore {
    /// Open or create a file-backed store rooted at `root`.  The
    /// directory is created if it does not exist; any I/O error is
    /// returned unchanged so callers can decide whether to fall back
    /// to an in-memory store or abort startup.
    pub fn new(root: PathBuf) -> std::io::Result<Self> {
        fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    fn path_for(&self, key: &PatchCacheKey) -> PathBuf {
        use std::hash::{DefaultHasher, Hash, Hasher};
        let mut h = DefaultHasher::new();
        key.hash(&mut h);
        self.root.join(format!("{:016x}.wav", h.finish()))
    }
}

impl PatchCacheStore for FileStore {
    fn get(&mut self, key: &PatchCacheKey) -> Option<Arc<[u8]>> {
        let path = self.path_for(key);
        let bytes = fs::read(&path).ok()?;
        Some(Arc::from(bytes.into_boxed_slice()))
    }

    fn put(&mut self, key: PatchCacheKey, bytes: Arc<[u8]>) {
        let path = self.path_for(&key);
        if let Err(e) = fs::write(&path, bytes.as_ref()) {
            bevy::log::warn!("FileStore::put failed for {}: {e}", path.display());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use tempfile::TempDir;

    use super::*;
    use crate::node::NodeKind;
    use crate::patch::{GraphNode, NodeGraph, NodeId};

    fn sample_patch(seed: u32) -> AudioPatch {
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

    fn fake_wav(byte: u8) -> Arc<[u8]> {
        Arc::from(vec![byte; 64].into_boxed_slice())
    }

    fn key(seed: u32) -> PatchCacheKey {
        PatchCacheKey::from_patch(&sample_patch(seed), 44_100, 1.0)
    }

    // --- PatchCacheKey -----------------------------------------------------

    #[test]
    fn key_is_deterministic_for_same_patch_and_params() {
        let a = key(7);
        let b = key(7);
        assert_eq!(a, b);
    }

    #[test]
    fn key_differs_for_different_seed() {
        assert_ne!(key(1), key(2));
    }

    #[test]
    fn key_differs_for_different_sample_rate() {
        let p = sample_patch(0);
        let a = PatchCacheKey::from_patch(&p, 44_100, 1.0);
        let b = PatchCacheKey::from_patch(&p, 48_000, 1.0);
        assert_ne!(a, b);
    }

    #[test]
    fn key_differs_for_different_duration() {
        let p = sample_patch(0);
        let a = PatchCacheKey::from_patch(&p, 44_100, 1.0);
        let b = PatchCacheKey::from_patch(&p, 44_100, 2.0);
        assert_ne!(a, b);
    }

    // --- MemoryStore -------------------------------------------------------

    #[test]
    fn memory_store_round_trips_bytes() {
        let mut s = MemoryStore::new(8);
        let k = key(0);
        assert!(s.get(&k).is_none());
        s.put(k.clone(), fake_wav(0xAB));
        assert_eq!(s.get(&k).unwrap()[0], 0xAB);
    }

    #[test]
    fn memory_store_evicts_oldest_at_capacity() {
        let mut s = MemoryStore::new(2);
        s.put(key(1), fake_wav(1));
        s.put(key(2), fake_wav(2));
        s.put(key(3), fake_wav(3));
        assert!(s.get(&key(1)).is_none(), "oldest should be evicted");
        assert!(s.get(&key(2)).is_some());
        assert!(s.get(&key(3)).is_some());
    }

    #[test]
    fn memory_store_treats_replace_as_no_evict() {
        let mut s = MemoryStore::new(2);
        s.put(key(1), fake_wav(1));
        s.put(key(2), fake_wav(2));
        // Re-insert existing key — must not push key(2) out.
        s.put(key(1), fake_wav(0xFF));
        assert!(s.get(&key(1)).is_some());
        assert!(s.get(&key(2)).is_some());
        assert_eq!(s.get(&key(1)).unwrap()[0], 0xFF, "value should refresh");
    }

    #[test]
    fn memory_store_clamps_zero_capacity_to_one() {
        let mut s = MemoryStore::new(0);
        s.put(key(1), fake_wav(1));
        assert!(s.get(&key(1)).is_some());
    }

    // --- FileStore ---------------------------------------------------------

    #[test]
    fn file_store_round_trips_bytes() {
        let dir = TempDir::new().unwrap();
        let mut s = FileStore::new(dir.path().to_path_buf()).unwrap();
        let k = key(0);
        assert!(s.get(&k).is_none());
        let original = fake_wav(0xCD);
        s.put(k.clone(), original.clone());
        let back = s.get(&k).unwrap();
        assert_eq!(back.as_ref(), original.as_ref());
    }

    #[test]
    fn file_store_persists_across_instances() {
        // Open a store, write an entry, drop the store, open a fresh
        // store on the same dir, read the entry back.  This is the
        // load-bearing behaviour for ambient-on-re-entry — survives
        // process restarts, not just session restarts.
        let dir = TempDir::new().unwrap();
        let k = key(0);
        {
            let mut s = FileStore::new(dir.path().to_path_buf()).unwrap();
            s.put(k.clone(), fake_wav(0xEE));
        }
        let mut s2 = FileStore::new(dir.path().to_path_buf()).unwrap();
        assert_eq!(s2.get(&k).unwrap()[0], 0xEE);
    }

    #[test]
    fn file_store_get_on_missing_returns_none() {
        let dir = TempDir::new().unwrap();
        let mut s = FileStore::new(dir.path().to_path_buf()).unwrap();
        assert!(s.get(&key(999)).is_none());
    }

    #[test]
    fn file_store_filename_uses_wav_extension() {
        let dir = TempDir::new().unwrap();
        let mut s = FileStore::new(dir.path().to_path_buf()).unwrap();
        s.put(key(0), fake_wav(1));
        let mut found = false;
        for entry in fs::read_dir(dir.path()).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) == Some("wav") {
                found = true;
            }
        }
        assert!(found, "FileStore must write .wav files");
    }

    // --- PatchCache resource ----------------------------------------------

    #[test]
    fn patch_cache_memory_wrapper_round_trips() {
        let cache = PatchCache::memory(8);
        assert!(cache.get(&key(0)).is_none());
        cache.insert(key(0), fake_wav(0x42));
        assert_eq!(cache.get(&key(0)).unwrap()[0], 0x42);
    }
}
