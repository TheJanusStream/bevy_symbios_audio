//! End-to-end test for the Phase 3 patch cache integrated with the
//! async bake pool.
//!
//! Builds a Bevy `App` with `MinimalPlugins` + `AssetPlugin` +
//! `init_asset::<AudioSource>()` + `SymbiosAudioPlugin`, inserts a
//! `PatchCache::memory(...)` resource, then dispatches a bake twice
//! through `bake_with_cache`:
//!
//! 1. First call: cache miss → `CacheOrPending::Pending(...)`.  Spawn
//!    it onto an entity, run frames until `poll_audio_tasks` records
//!    the bytes into the cache and attaches `AudioPatchReady`.
//! 2. Second call: cache hit → `CacheOrPending::Cached(handle)` returns
//!    immediately, no bake dispatched.
//!
//! The on-disk `FileStore` variant gets a separate test with `tempfile`
//! to confirm bytes survive a process boundary (well, a `PatchCache`
//! reconstruction, which is the load-bearing case for "ambient survives
//! room re-entry across app launches").

use std::collections::BTreeMap;
use std::time::Duration;

use bevy::asset::{AssetPlugin, Assets};
use bevy::audio::AudioSource;
use bevy::prelude::*;
use bevy_symbios_audio::{
    AudioPatch, AudioPatchReady, BrownNoise, CacheOrPending, GraphNode, NodeGraph, NodeId,
    NodeKind, PatchCache, PatchCacheKey, PendingAudioPatch, SymbiosAudioPlugin, bake,
    bake_with_cache, samples_to_wav_bytes,
};
use tempfile::TempDir;

fn small_patch() -> AudioPatch {
    AudioPatch {
        seed: 0xCAFE,
        graph: NodeGraph {
            nodes: vec![GraphNode {
                id: NodeId(0),
                kind: NodeKind::BrownNoise(BrownNoise { amplitude: 0.5 }),
                inputs: BTreeMap::new(),
            }],
            output: NodeId(0),
        },
    }
}

fn make_app() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(AssetPlugin::default())
        .init_asset::<AudioSource>()
        .add_plugins(SymbiosAudioPlugin::default());
    app
}

fn run_until_ready(app: &mut App, entity: Entity, max_frames: u32) {
    for _ in 0..max_frames {
        app.update();
        if app.world().get::<AudioPatchReady>(entity).is_some() {
            return;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    panic!("AudioPatchReady never attached after {max_frames} frames");
}

// --- in-memory PatchCache ---------------------------------------------------

#[test]
fn second_bake_through_bake_with_cache_hits_memory_cache() {
    let mut app = make_app();
    app.insert_resource(PatchCache::memory(8));

    // First dispatch: cache miss → pending.
    let outcome = {
        let cache = app.world().resource::<PatchCache>();
        bake_with_cache(small_patch(), 44_100, 0.1, cache)
    };
    let pending = match outcome {
        CacheOrPending::Pending(p) => p,
        CacheOrPending::Cached(_) => panic!("first dispatch should miss"),
    };
    let entity = app.world_mut().spawn(pending).id();
    run_until_ready(&mut app, entity, 200);

    // Cache must now contain the produced bytes.
    let key = PatchCacheKey::from_patch(&small_patch(), 44_100, 0.1);
    {
        let cache = app.world().resource::<PatchCache>();
        assert!(cache.get(&key).is_some(), "cache miss after completed bake");
    }

    // Second dispatch: cache hit → bytes returned directly, no
    // PendingAudioPatch dispatched.
    let outcome2 = {
        let cache = app.world().resource::<PatchCache>();
        bake_with_cache(small_patch(), 44_100, 0.1, cache)
    };
    let bytes = match outcome2 {
        CacheOrPending::Cached(b) => b,
        CacheOrPending::Pending(_) => panic!("second dispatch should hit"),
    };
    assert_eq!(&bytes[0..4], b"RIFF");
    // And the bytes can be wrapped into a fresh AudioSource handle.
    let mut audio_sources = app.world_mut().resource_mut::<Assets<AudioSource>>();
    let handle = audio_sources.add(AudioSource { bytes });
    assert!(audio_sources.get(&handle).is_some());
}

#[test]
fn pending_audio_patch_new_records_into_cache_on_completion() {
    // The default constructor (PendingAudioPatch::new, not _uncached)
    // already attaches a cache_key.  Just spawning one onto an app with
    // a PatchCache resource records the bytes on completion — no
    // bake_with_cache helper needed.
    let mut app = make_app();
    app.insert_resource(PatchCache::memory(8));

    let entity = app
        .world_mut()
        .spawn(PendingAudioPatch::new(small_patch(), 44_100, 0.05))
        .id();
    run_until_ready(&mut app, entity, 200);

    let key = PatchCacheKey::from_patch(&small_patch(), 44_100, 0.05);
    let cache = app.world().resource::<PatchCache>();
    assert!(cache.get(&key).is_some());
}

#[test]
fn pending_audio_patch_new_uncached_does_not_touch_cache() {
    let mut app = make_app();
    app.insert_resource(PatchCache::memory(8));

    let entity = app
        .world_mut()
        .spawn(PendingAudioPatch::new_uncached(small_patch(), 44_100, 0.05))
        .id();
    run_until_ready(&mut app, entity, 200);

    let key = PatchCacheKey::from_patch(&small_patch(), 44_100, 0.05);
    let cache = app.world().resource::<PatchCache>();
    assert!(cache.get(&key).is_none(), "uncached path must not write");
}

#[test]
fn cached_bytes_match_a_fresh_bake_of_the_same_patch() {
    // Determinism cross-check: the bytes the cache holds should match
    // exactly what samples_to_wav_bytes(bake(patch, sr, dur), sr)
    // produces.  Otherwise something silently re-baked at a different
    // seed or shape.
    let mut app = make_app();
    app.insert_resource(PatchCache::memory(8));
    let patch = small_patch();

    let entity = app
        .world_mut()
        .spawn(PendingAudioPatch::new(patch.clone(), 44_100, 0.05))
        .id();
    run_until_ready(&mut app, entity, 200);

    let key = PatchCacheKey::from_patch(&patch, 44_100, 0.05);
    let cached = {
        let cache = app.world().resource::<PatchCache>();
        cache.get(&key).expect("cache miss")
    };

    let fresh_samples = bake(&patch, 44_100, 0.05);
    let fresh_bytes = samples_to_wav_bytes(&fresh_samples, 44_100);
    assert_eq!(cached.as_ref(), fresh_bytes.as_slice());
}

// --- file PatchCache --------------------------------------------------------

#[test]
fn file_store_survives_cache_reconstruction() {
    // Bake once with a FileStore-backed cache, drop the cache resource,
    // build a fresh FileStore on the same directory, and confirm the
    // entry is still there.  This is the "ambient survives a process
    // restart" contract Overlands relies on.
    let dir = TempDir::new().unwrap();
    let patch = small_patch();
    let key = PatchCacheKey::from_patch(&patch, 44_100, 0.05);

    // First app: bake → file cache.
    {
        let mut app = make_app();
        app.insert_resource(PatchCache::file(dir.path(), 0).unwrap());
        let entity = app
            .world_mut()
            .spawn(PendingAudioPatch::new(patch.clone(), 44_100, 0.05))
            .id();
        run_until_ready(&mut app, entity, 200);
        let cache = app.world().resource::<PatchCache>();
        assert!(cache.get(&key).is_some());
    }

    // Fresh app — same directory, no in-memory state.
    {
        let app_cache = PatchCache::file(dir.path(), 0).unwrap();
        let bytes = app_cache.get(&key).expect("file cache lost the entry");
        assert_eq!(&bytes[0..4], b"RIFF");
    }
}
