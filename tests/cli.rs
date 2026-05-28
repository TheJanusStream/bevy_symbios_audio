//! Integration tests for the `symbios-audio-cli` binary.
//!
//! Each test invokes the freshly-built binary via the
//! `CARGO_BIN_EXE_symbios-audio-cli` env var Cargo injects into test
//! processes, feeding a hand-rolled patch JSON written to a tempdir
//! and asserting the produced WAV file is structurally correct.

use std::collections::BTreeMap;
use std::fs;
use std::process::Command;

use bevy_symbios_audio::{AudioPatch, GraphNode, NodeGraph, NodeId, NodeKind, SineOsc};
use tempfile::TempDir;

const CLI: &str = env!("CARGO_BIN_EXE_symbios-audio-cli");

fn write_patch(dir: &TempDir, name: &str, patch: &AudioPatch) -> std::path::PathBuf {
    let path = dir.path().join(name);
    let json = serde_json::to_string_pretty(patch).unwrap();
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
