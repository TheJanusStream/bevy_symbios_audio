//! Integration tests for the `symbios-audio-cli` binary.
//!
//! Each test invokes the freshly-built binary via the
//! `CARGO_BIN_EXE_symbios-audio-cli` env var Cargo injects into test
//! processes, feeding a hand-rolled patch JSON written to a tempdir
//! and asserting the produced WAV file is structurally correct.

use std::collections::BTreeMap;
use std::fs;
use std::process::Command;

use bevy_symbios_audio::{
    AudioPatch, Event, GraphNode, Instrument, NodeGraph, NodeId, NodeKind, SequenceRecipe, SineOsc,
    Track,
};
use tempfile::TempDir;

const CLI: &str = env!("CARGO_BIN_EXE_symbios-audio-cli");

fn write_patch(dir: &TempDir, name: &str, patch: &AudioPatch) -> std::path::PathBuf {
    let path = dir.path().join(name);
    let json = serde_json::to_string_pretty(patch).unwrap();
    fs::write(&path, json).unwrap();
    path
}

fn write_recipe(dir: &TempDir, name: &str, recipe: &SequenceRecipe) -> std::path::PathBuf {
    let path = dir.path().join(name);
    let json = serde_json::to_string_pretty(recipe).unwrap();
    fs::write(&path, json).unwrap();
    path
}

fn sine_patch() -> AudioPatch {
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

/// A two-event recipe at 120 BPM: a single sine instrument struck twice.
/// `duration_beats = 2` at 120 BPM (0.5 s/beat) and 48 kHz is exactly
/// 48 000 samples, which the data-size assertions below pin.
fn sine_recipe() -> SequenceRecipe {
    SequenceRecipe {
        bpm: 120.0,
        sample_rate: 48_000,
        duration_beats: 2.0,
        loop_start_beats: None,
        loop_crossfade_beats: 0.0,
        instruments: vec![Instrument {
            id: "lead".into(),
            patch: sine_patch(),
        }],
        tracks: vec![Track {
            events: vec![
                Event {
                    time_beats: 0.0,
                    instrument_id: "lead".into(),
                    pitch_multiplier: 1.0,
                    volume: 0.8,
                    gate_beats: 0.5,
                    ..Default::default()
                },
                Event {
                    time_beats: 1.0,
                    instrument_id: "lead".into(),
                    pitch_multiplier: 1.5,
                    volume: 0.8,
                    gate_beats: 0.5,
                    ..Default::default()
                },
            ],
        }],
    }
}

#[test]
fn bake_writes_a_valid_wav_at_the_requested_rate_and_duration() {
    let dir = TempDir::new().unwrap();
    let input = write_patch(&dir, "in.json", &sine_patch());
    let output = dir.path().join("out.wav");

    let status = Command::new(CLI)
        .args([
            "bake",
            "--sample-rate",
            "48000",
            "--duration",
            "0.25",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .status()
        .expect("spawning the CLI must succeed");
    assert!(status.success(), "CLI exited with {status}");

    let bytes = fs::read(&output).unwrap();
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    // 0.25 s @ 48 kHz = 12_000 samples × 4 bytes/sample = 48_000 bytes.
    let data_size = u32::from_le_bytes([bytes[52], bytes[53], bytes[54], bytes[55]]);
    assert_eq!(data_size, 12_000 * 4);
}

#[test]
fn defaults_match_44100hz_1s() {
    let dir = TempDir::new().unwrap();
    let input = write_patch(&dir, "in.json", &sine_patch());
    let output = dir.path().join("out.wav");

    let status = Command::new(CLI)
        .args(["bake", input.to_str().unwrap(), output.to_str().unwrap()])
        .status()
        .unwrap();
    assert!(status.success());
    let bytes = fs::read(&output).unwrap();
    // Sample rate field at offset 24 in our WAV layout.
    let sr = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    assert_eq!(sr, 44_100);
    // 1 s @ 44.1 kHz = 44_100 samples × 4 bytes.
    let data_size = u32::from_le_bytes([bytes[52], bytes[53], bytes[54], bytes[55]]);
    assert_eq!(data_size, 44_100 * 4);
}

#[test]
fn missing_input_file_exits_with_io_error_code() {
    let dir = TempDir::new().unwrap();
    let missing = dir.path().join("does-not-exist.json");
    let output = dir.path().join("out.wav");
    let status = Command::new(CLI)
        .args(["bake", missing.to_str().unwrap(), output.to_str().unwrap()])
        .status()
        .unwrap();
    assert!(!status.success());
    assert_eq!(status.code(), Some(1));
}

#[test]
fn malformed_patch_json_exits_with_io_error_code() {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("bad.json");
    fs::write(&input, "{ not real json").unwrap();
    let output = dir.path().join("out.wav");
    let status = Command::new(CLI)
        .args(["bake", input.to_str().unwrap(), output.to_str().unwrap()])
        .status()
        .unwrap();
    assert!(!status.success());
    assert_eq!(status.code(), Some(1));
}

#[test]
fn zero_sample_rate_exits_with_arg_error_code() {
    let dir = TempDir::new().unwrap();
    let input = write_patch(&dir, "in.json", &sine_patch());
    let output = dir.path().join("out.wav");
    let status = Command::new(CLI)
        .args([
            "bake",
            "--sample-rate",
            "0",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(!status.success());
    assert_eq!(status.code(), Some(2));
}

#[test]
fn negative_duration_exits_with_arg_error_code() {
    let dir = TempDir::new().unwrap();
    let input = write_patch(&dir, "in.json", &sine_patch());
    let output = dir.path().join("out.wav");
    let status = Command::new(CLI)
        .args([
            "bake",
            "--duration",
            "-1.0",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(!status.success());
    assert_eq!(status.code(), Some(2));
}

// --- bake-sequence ----------------------------------------------------------

#[test]
fn bake_sequence_writes_a_valid_wav_at_the_recipes_rate_and_length() {
    let dir = TempDir::new().unwrap();
    let input = write_recipe(&dir, "recipe.json", &sine_recipe());
    let output = dir.path().join("out.wav");

    let status = Command::new(CLI)
        .args([
            "bake-sequence",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .status()
        .expect("spawning the CLI must succeed");
    assert!(status.success(), "CLI exited with {status}");

    let bytes = fs::read(&output).unwrap();
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    // Sample rate comes from the recipe, not a flag.
    let sr = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    assert_eq!(sr, 48_000);
    // duration_beats 2 at 120 BPM = 1.0 s; 1.0 s @ 48 kHz = 48 000 samples.
    let data_size = u32::from_le_bytes([bytes[52], bytes[53], bytes[54], bytes[55]]);
    assert_eq!(data_size, 48_000 * 4);
}

#[test]
fn bake_sequence_zero_sample_rate_exits_with_arg_error_code() {
    let dir = TempDir::new().unwrap();
    let recipe = SequenceRecipe {
        sample_rate: 0,
        ..sine_recipe()
    };
    let input = write_recipe(&dir, "recipe.json", &recipe);
    let output = dir.path().join("out.wav");
    let status = Command::new(CLI)
        .args([
            "bake-sequence",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(!status.success());
    assert_eq!(status.code(), Some(2));
}

#[test]
fn bake_sequence_empty_timeline_exits_with_io_error_code() {
    // duration_beats 0 and no crossfade tail → the mixdown produces no
    // samples, which the CLI treats as a (loud) failure rather than writing
    // a zero-sample WAV.
    let dir = TempDir::new().unwrap();
    let recipe = SequenceRecipe {
        duration_beats: 0.0,
        loop_crossfade_beats: 0.0,
        ..sine_recipe()
    };
    let input = write_recipe(&dir, "recipe.json", &recipe);
    let output = dir.path().join("out.wav");
    let status = Command::new(CLI)
        .args([
            "bake-sequence",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(!status.success());
    assert_eq!(status.code(), Some(1));
}

#[test]
fn bake_sequence_malformed_json_exits_with_io_error_code() {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("bad.json");
    fs::write(&input, "{ not real json").unwrap();
    let output = dir.path().join("out.wav");
    let status = Command::new(CLI)
        .args([
            "bake-sequence",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(!status.success());
    assert_eq!(status.code(), Some(1));
}
