//! Offline patch baker — read an `AudioPatch` JSON file, bake it, write
//! the result as a WAV file.
//!
//! Phase 3 ticket #12.  Intended for the Janus Stream video pipeline
//! and for sound-design iteration outside the Bevy app.  Mono WAV only
//! (IEEE float, the same encoding the in-Bevy `AudioSource` bridge
//! produces) — Ogg Vorbis / Opus are deliberately out of scope
//! because the pure-Rust encoder ecosystem is still rough.
//!
//! # Example
//!
//! ```text
//! symbios-audio-cli bake patch.json out.wav
//! symbios-audio-cli bake --sample-rate 48000 --duration 5.0 patch.json out.wav
//! ```

use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use bevy_symbios_audio::{AudioPatch, bake, samples_to_wav_bytes};

#[derive(Parser, Debug)]
#[command(
    name = "symbios-audio-cli",
    version,
    about = "Offline AudioPatch JSON → WAV baker"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Bake a JSON patch file into a WAV file.
    Bake {
        /// Path to an AudioPatch JSON file.
        input: PathBuf,
        /// Path to write the WAV output to.  Existing files are
        /// overwritten.
        output: PathBuf,
        /// Sample rate in Hz.
        #[arg(long, default_value_t = 44_100)]
        sample_rate: u32,
        /// Bake duration in seconds.
        #[arg(long, default_value_t = 1.0)]
        duration: f32,
    },
}

fn main() -> ExitCode {
    match Cli::parse().cmd {
        Cmd::Bake {
            input,
            output,
            sample_rate,
            duration,
        } => bake_command(&input, &output, sample_rate, duration),
    }
}

fn bake_command(
    input: &PathBuf,
    output: &PathBuf,
    sample_rate: u32,
    duration_secs: f32,
) -> ExitCode {
    if sample_rate == 0 {
        eprintln!("error: --sample-rate must be > 0");
        return ExitCode::from(2);
    }
    if !(duration_secs.is_finite() && duration_secs > 0.0) {
        eprintln!("error: --duration must be a positive, finite number of seconds");
        return ExitCode::from(2);
    }

    let json = match fs::read_to_string(input) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("error: read {}: {e}", input.display());
            return ExitCode::from(1);
        }
    };

    let patch: AudioPatch = match serde_json::from_str(&json) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: parse {}: {e}", input.display());
            return ExitCode::from(1);
        }
    };

    let samples = bake(&patch, sample_rate, duration_secs);
    let bytes = samples_to_wav_bytes(&samples, sample_rate);

    if let Err(e) = fs::write(output, &bytes) {
        eprintln!("error: write {}: {e}", output.display());
        return ExitCode::from(1);
    }

    eprintln!(
        "baked {} samples ({:.3}s @ {} Hz) → {}",
        samples.len(),
        duration_secs,
        sample_rate,
        output.display()
    );
    ExitCode::SUCCESS
}
