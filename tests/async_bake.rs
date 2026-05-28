//! End-to-end test for the Phase 3 async bake pool.
//!
//! Builds a minimal Bevy `App` with the asset plumbing required for
//! `Assets<AudioSource>` and the `SymbiosAudioPlugin`, spawns a
//! `PendingAudioPatch`, runs `app.update()` repeatedly until the
//! pending entity is replaced by an `AudioPatchReady`, and then verifies
//! the produced `AudioSource` has the expected WAV byte signature.
//!
//! This is the real-world contract the Overlands integration leans on:
//! "spawn a pending audio request, get a handle to a playable
//! `AudioSource` back a few frames later, no main-thread stall in
//! between".

use std::collections::BTreeMap;
use std::time::Duration;

use bevy::asset::{AssetPlugin, Assets};
use bevy::audio::AudioSource;
use bevy::prelude::*;
use bevy_symbios_audio::{
    AudioPatch, AudioPatchReady, GraphNode, NodeGraph, NodeId, NodeKind, PendingAudioPatch,
    SineOsc, SymbiosAudioPlugin,
};

/// Build a single-node patch that bakes a 100 ms 440 Hz sine — enough
/// to verify the dispatch path without slowing the test runner.
fn small_sine_patch() -> AudioPatch {
    AudioPatch {
        seed: 0,
        graph: NodeGraph {
            nodes: vec![GraphNode {
                id: NodeId(0),
                kind: NodeKind::Sine(SineOsc {
                    freq_hz: 440.0,
                    phase_offset: 0.0,
                    amplitude: 1.0,
                }),
                inputs: BTreeMap::new(),
            }],
            output: NodeId(0),
        },
    }
}

fn make_app() -> App {
    let mut app = App::new();
    // MinimalPlugins gives us the schedule + task pool + time stepping
    // needed for Update.  AssetPlugin sets up the asset server; the
    // explicit init_asset registers Assets<AudioSource> as a resource
    // without pulling in AudioPlugin (which wants an audio backend).
    app.add_plugins(MinimalPlugins)
        .add_plugins(AssetPlugin::default())
        .init_asset::<AudioSource>()
        .add_plugins(SymbiosAudioPlugin::default());
    app
}

/// Run `app.update()` up to `max_frames` times, returning the entity
/// once `AudioPatchReady` is attached.  Panics if the budget elapses
/// without completion — keeps a runaway bake from hanging the test.
fn run_until_ready(app: &mut App, entity: Entity, max_frames: u32) -> Handle<AudioSource> {
    for _ in 0..max_frames {
        app.update();
        if let Some(ready) = app.world().get::<AudioPatchReady>(entity) {
            return ready.handle.clone();
        }
        // Small sleep so the bake thread has time to make progress on
        // multi-core machines without burning CPU spinning.
        std::thread::sleep(Duration::from_millis(2));
    }
    panic!("AudioPatchReady never attached after {max_frames} frames");
}

#[test]
fn pending_audio_patch_becomes_audio_patch_ready() {
    let mut app = make_app();
    let entity = app
        .world_mut()
        .spawn(PendingAudioPatch::new(small_sine_patch(), 44_100, 0.1))
        .id();
    let handle = run_until_ready(&mut app, entity, 200);

    // The pending component should be gone after completion.
    assert!(app.world().get::<PendingAudioPatch>(entity).is_none());
    // And the AudioSource bytes must be a valid WAV with the right rate.
    let sources = app.world().resource::<Assets<AudioSource>>();
    let source = sources.get(&handle).expect("AudioSource asset present");
    let bytes = source.bytes.as_ref();
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
}

#[test]
fn baked_buffer_length_matches_requested_duration() {
    let mut app = make_app();
    let entity = app
        .world_mut()
        .spawn(PendingAudioPatch::new(small_sine_patch(), 44_100, 0.1))
        .id();
    let handle = run_until_ready(&mut app, entity, 200);
    let sources = app.world().resource::<Assets<AudioSource>>();
    let bytes = sources.get(&handle).unwrap().bytes.as_ref();
    // Walk to the data chunk size at the fixed offset for our header
    // shape (RIFF=12 + fmt=24 + fact=12 = offset 48, then "data" id + size).
    let data_size = u32::from_le_bytes([bytes[52], bytes[53], bytes[54], bytes[55]]);
    // 0.1 s @ 44.1 kHz = 4410 samples * 4 bytes/sample = 17640 bytes.
    assert_eq!(data_size, 4410 * 4);
}

#[test]
fn multiple_concurrent_pending_patches_all_complete() {
    let mut app = make_app();
    let mut entities = Vec::new();
    for seed in 0..4 {
        let mut p = small_sine_patch();
        p.seed = seed;
        let e = app
            .world_mut()
            .spawn(PendingAudioPatch::new(p, 44_100, 0.05))
            .id();
        entities.push(e);
    }
    for &e in &entities {
        run_until_ready(&mut app, e, 200);
    }
    let sources = app.world().resource::<Assets<AudioSource>>();
    assert_eq!(sources.len(), entities.len());
}

#[test]
fn despawning_pending_entity_does_not_panic_subsequent_polls() {
    // Cancellation path: spawn and immediately despawn.  The drop
    // handler sets the flag; subsequent updates must not panic.
    let mut app = make_app();
    let entity = app
        .world_mut()
        .spawn(PendingAudioPatch::new(small_sine_patch(), 44_100, 0.5))
        .id();
    app.world_mut().despawn(entity);
    for _ in 0..20 {
        app.update();
    }
}
