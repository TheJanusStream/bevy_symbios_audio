//! Offline patch baker — read an `AudioPatch` (or `SequenceRecipe`) JSON
//! file, bake it, write the result as a WAV file.
//!
//! Phase 3 ticket #12.  Intended for the Janus Stream video pipeline
//! and for sound-design iteration outside the Bevy app.  Mono WAV only
//! (IEEE float, the same encoding the in-Bevy `AudioSource` bridge
//! produces) — Ogg Vorbis / Opus are deliberately out of scope
//! because the pure-Rust encoder ecosystem is still rough.
//!
//! Two subcommands:
//! - `bake` renders a single [`AudioPatch`] over a `--duration` window at
//!   a `--sample-rate` chosen on the command line (the patch is
//!   rate-agnostic).
//! - `bake-sequence` renders a [`SequenceRecipe`] via
//!   [`bevy_symbios_audio::bake_sequence`].  Sample rate and length live
//!   inside the recipe (`sample_rate`, `bpm`, `duration_beats`), so the
//!   subcommand takes only the input and output paths.
//!
//! # Example
//!
//! ```text
//! symbios-audio-cli bake patch.json out.wav
//! symbios-audio-cli bake --sample-rate 48000 --duration 5.0 patch.json out.wav
//! symbios-audio-cli bake-sequence recipe.json out.wav
//! ```

use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use bevy_symbios_audio::{AudioPatch, SequenceRecipe, bake, bake_sequence, samples_to_wav_bytes};

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
    /// Bake a single-patch JSON file into a WAV file.
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
    /// Bake a sequence-recipe JSON file into a WAV file.
    ///
    /// Unlike `bake`, the sample rate and length are read from the recipe
    /// itself (`sample_rate`, `bpm`, `duration_beats`), so no `--sample-rate`
    /// / `--duration` flags are accepted here.
    BakeSequence {
        /// Path to a SequenceRecipe JSON file.
        input: PathBuf,
        /// Path to write the WAV output to.  Existing files are
        /// overwritten.
        output: PathBuf,
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
        Cmd::BakeSequence { input, output } => bake_sequence_command(&input, &output),
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

fn bake_sequence_command(input: &PathBuf, output: &PathBuf) -> ExitCode {
    let json = match fs::read_to_string(input) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("error: read {}: {e}", input.display());
            return ExitCode::from(1);
        }
    };

    let recipe: SequenceRecipe = match serde_json::from_str(&json) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: parse {}: {e}", input.display());
            return ExitCode::from(1);
        }
    };

    // The recipe carries its own sample rate; it has to be valid for the WAV
    // header even though `bake_sequence` would otherwise tolerate a zero rate
    // by producing an empty buffer.
    if recipe.sample_rate == 0 {
        eprintln!("error: recipe sample_rate must be > 0");
        return ExitCode::from(2);
    }

    let sample_rate = recipe.sample_rate;
    let samples = bake_sequence(&recipe);

    // An empty mixdown means the recipe renders nothing — a non-positive
    // duration/bpm, no tracks, or every event referencing an unknown or
    // invalid instrument (those are skipped with a warning by the baker).
    // Writing a zero-sample WAV would mask the mistake, so fail loudly.
    if samples.is_empty() {
        eprintln!(
            "error: recipe produced no samples — check bpm, duration_beats, and that \
             tracks reference defined, well-formed instruments"
        );
        return ExitCode::from(1);
    }

    let bytes = samples_to_wav_bytes(&samples, sample_rate);

    if let Err(e) = fs::write(output, &bytes) {
        eprintln!("error: write {}: {e}", output.display());
        return ExitCode::from(1);
    }

    eprintln!(
        "baked {} samples ({:.3}s @ {} Hz, {} BPM) → {}",
        samples.len(),
        samples.len() as f32 / sample_rate as f32,
        sample_rate,
        recipe.bpm,
        output.display()
    );
    ExitCode::SUCCESS
}
