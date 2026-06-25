//! Bridge from a baked `Vec<f32>` to Bevy's `AudioSource` asset.
//!
//! Phase 3 ticket #10.  The audio engine in Bevy 0.18 (rodio under the
//! hood) consumes [`AudioSource`] — a `bytes::Bytes`-equivalent blob that
//! it re-decodes on play.  Rather than implementing a custom rodio
//! `Decoder` for `Vec<f32>` (doable but ties us to rodio internals and a
//! moving target across Bevy releases), this module wraps the raw samples
//! in a minimal in-memory RIFF/WAVE blob with the IEEE-float (format
//! code `0x0003`) flavour, then hands those bytes to `AudioSource`.
//!
//! The pure RIFF/WAVE encoder ([`samples_to_wav_bytes`] / [`MAX_WAV_SAMPLES`])
//! lives in the Bevy-free [`symbios_audio::wav`] core module and is
//! re-exported here so the public API is unchanged; this module only adds
//! the Bevy-coupled [`AudioSource`] wrapper on top.
//!
//! # Format
//!
//! - PCM container: RIFF / WAVE.
//! - Codec: IEEE float, 32-bit little-endian.
//! - Channels: mono only.  Both `bake` and `bake_sequence` produce mono
//!   buffers; stereo and multichannel routing remain out of scope.
//! - Includes a `fact` chunk so strict decoders (which require it for
//!   non-PCM formats per the WAV spec) accept the blob without warning.
//!
//! # Bevy feature
//!
//! `AudioSource`'s WAV decoding requires Bevy's `wav` feature.  This crate
//! enables it in its own `Cargo.toml`, so downstream users pulling this
//! crate get WAV support automatically — they don't need to enable it
//! again on their own `bevy` line.

use std::sync::Arc;

use bevy::audio::AudioSource;

// Re-export the pure encoder so the public API
// (`bevy_symbios_audio::{samples_to_wav_bytes, MAX_WAV_SAMPLES}`) is
// preserved byte-for-byte after the split.
pub use symbios_audio::wav::{MAX_WAV_SAMPLES, samples_to_wav_bytes};

/// Convert a mono `f32` buffer to a Bevy [`AudioSource`] backed by an
/// in-memory WAV blob.
///
/// `sample_rate` is written into the WAV header — the resulting
/// `AudioSource`, once decoded by rodio, plays back at exactly that rate.
///
/// See [`samples_to_wav_bytes`] for the underlying byte layout.
pub fn samples_to_audio_source(samples: &[f32], sample_rate: u32) -> AudioSource {
    let bytes = samples_to_wav_bytes(samples, sample_rate);
    AudioSource {
        bytes: Arc::from(bytes.into_boxed_slice()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_source_construction_holds_wav_bytes() {
        let samples = vec![0.0_f32, 0.25, -0.25];
        let source = samples_to_audio_source(&samples, 48_000);
        // The bridge keeps the bytes intact and identical to the helper.
        let direct = samples_to_wav_bytes(&samples, 48_000);
        assert_eq!(source.bytes.as_ref(), direct.as_slice());
    }
}
