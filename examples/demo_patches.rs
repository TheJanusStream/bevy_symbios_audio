//! Generates the demo `patches/` JSON files used with `symbios-audio-cli`.
//!
//! Run from the crate root:
//!
//! ```sh
//! cargo run --example demo_patches
//! # then, for any of the files it wrote:
//! cargo run --release --bin symbios-audio-cli -- bake patches/crow_triple_caw.json out.wav
//! cargo run --release --bin symbios-audio-cli -- bake-sequence patches/seq_rock_groove.json out.wav
//! ```
//!
//! It writes two parallel *sets* of the same three sounds:
//!
//! - **Single-graph patches** (`bake`): one [`AudioPatch`] DAG rendered over a
//!   fixed duration.  There is no sequencer here, so rhythm is built from
//!   LFO-square waves driving [`AdsrEnvelope`] gate ports (edge-triggered, so
//!   each rising edge is a new note) — including an *inverted* square (negative
//!   `depth`) to land a backbeat on the off-beat, which the LFO's lack of a
//!   phase offset otherwise can't reach.
//! - **Sequence recipes** (`bake-sequence`): a [`SequenceRecipe`] whose
//!   instruments wire a [`Gate`] into their envelopes, so timed `Event`s open
//!   the gate window and play real notes — the natural way to express rhythm
//!   and melody.
//!
//! Because [`bake`] does no master limiting (only [`bake_sequence`] tanh
//! soft-clips), the single-graph patches are auto-gain-staged here: each is
//! baked once, its peak measured, and its master mixer's gain rescaled so the
//! written patch peaks at [`TARGET_PEAK`].

use std::fs;
use std::path::Path;

use bevy_symbios_audio::{
    AdsrCurve, AdsrEnvelope, AntiAlias, AudioPatch, BiquadBandpass, BiquadHighpass, BiquadLowpass,
    BrownNoise, Chorus, Connection, Event, Gain, Gate, GraphNode, Instrument, Lfo, LfoShape, Mix,
    NodeGraph, NodeId, NodeKind, PinkNoise, PitchMode, Reverb, SawPolarity, SawtoothOsc,
    SequenceRecipe, SineOsc, Track, TriangleOsc, WhiteNoise, bake, bake_sequence,
};

/// Peak the single-graph patches are normalised to (leaves a little headroom
/// under full scale so playback chains don't clip).
const TARGET_PEAK: f32 = 0.9;
const OUT_DIR: &str = "patches";

fn main() {
    let dir = Path::new(OUT_DIR);
    fs::create_dir_all(dir).expect("create patches/ directory");

    // --- Single-graph patches (the `bake` subcommand). -------------------
    // The fourth field is the id of the master mixer node to rescale for
    // gain-staging (not the seed — that lives inside each patch).
    let single: [(&str, AudioPatch, f32, u32); 3] = [
        ("crow_triple_caw", crow_patch(), 1.6, 7),
        ("rock_groove", rock_patch(), 8.0, 30),
        ("cyberpunk_noir_ambient", ambient_patch(), 14.0, 25),
    ];
    for (name, mut patch, duration, master_id) in single {
        let (raw, achieved) = autoscale(&mut patch, master_id, duration);
        write_json(dir, &format!("{name}.json"), &patch);
        println!(
            "  {name}.json  (single-graph, {duration:.1}s)  raw peak {raw:.3} → {achieved:.3}"
        );
    }

    // --- Sequence recipes (the `bake-sequence` subcommand). --------------
    let sequences: [(&str, SequenceRecipe); 3] = [
        ("seq_crow_triple_caw", crow_sequence()),
        ("seq_rock_groove", rock_sequence()),
        ("seq_cyberpunk_noir_ambient", ambient_sequence()),
    ];
    for (name, recipe) in &sequences {
        let buf = bake_sequence(recipe);
        let peak = peak_abs(&buf);
        write_json(dir, &format!("{name}.json"), recipe);
        let secs = buf.len() as f32 / recipe.sample_rate.max(1) as f32;
        println!(
            "  {name}.json  (sequence, {secs:.1}s @ {} BPM)  peak {peak:.3}",
            recipe.bpm
        );
    }

    println!("\nwrote 6 files to {OUT_DIR}/");
}

// ============================================================================
// Tiny builders — keep the graph definitions below readable.
// ============================================================================

fn gnode(id: u32, kind: NodeKind) -> GraphNode {
    GraphNode {
        id: NodeId(id),
        kind,
        ..Default::default()
    }
}

fn from(id: u32) -> Connection {
    Connection::from_node(NodeId(id))
}

fn modc(id: u32, amount: f32) -> Connection {
    Connection::modulation(NodeId(id), amount)
}

fn lfo(rate_hz: f32, shape: LfoShape, depth: f32, offset: f32) -> NodeKind {
    NodeKind::Lfo(Lfo {
        rate_hz,
        shape,
        depth,
        offset,
    })
}

fn adsr(
    attack_s: f32,
    decay_s: f32,
    sustain_level: f32,
    release_s: f32,
    curve: AdsrCurve,
) -> NodeKind {
    NodeKind::Adsr(AdsrEnvelope {
        attack_s,
        decay_s,
        sustain_level,
        release_s,
        curve,
    })
}

fn sine(freq_hz: f32, amplitude: f32) -> NodeKind {
    NodeKind::Sine(SineOsc {
        freq_hz,
        phase_offset: 0.0,
        amplitude,
    })
}

fn saw(freq_hz: f32, amplitude: f32, anti_alias: AntiAlias) -> NodeKind {
    NodeKind::Sawtooth(SawtoothOsc {
        freq_hz,
        polarity: SawPolarity::Up,
        amplitude,
        anti_alias,
    })
}

fn tri(freq_hz: f32, amplitude: f32) -> NodeKind {
    NodeKind::Triangle(TriangleOsc {
        freq_hz,
        amplitude,
        anti_alias: AntiAlias::Naive,
    })
}

fn white(amplitude: f32) -> NodeKind {
    NodeKind::WhiteNoise(WhiteNoise { amplitude })
}

fn pink(amplitude: f32) -> NodeKind {
    NodeKind::PinkNoise(PinkNoise { amplitude })
}

fn brown(amplitude: f32) -> NodeKind {
    NodeKind::BrownNoise(BrownNoise { amplitude })
}

fn lowpass(cutoff_hz: f32, q: f32) -> NodeKind {
    NodeKind::BiquadLowpass(BiquadLowpass { cutoff_hz, q })
}

fn highpass(cutoff_hz: f32, q: f32) -> NodeKind {
    NodeKind::BiquadHighpass(BiquadHighpass { cutoff_hz, q })
}

fn bandpass(center_hz: f32, q: f32) -> NodeKind {
    NodeKind::BiquadBandpass(BiquadBandpass { center_hz, q })
}

fn mix(gain: f32) -> NodeKind {
    NodeKind::Mix(Mix { gain })
}

fn vca() -> NodeKind {
    // Base gain 0.0 — a textbook VCA whose level is the "gain" CV alone.
    NodeKind::Gain(Gain { gain: 0.0 })
}

fn gate() -> NodeKind {
    NodeKind::Gate(Gate { invert: false })
}

fn chorus(rate_hz: f32, depth_ms: f32, base_delay_ms: f32, feedback: f32, mix: f32) -> NodeKind {
    NodeKind::Chorus(Chorus {
        rate_hz,
        depth_ms,
        base_delay_ms,
        feedback,
        mix,
    })
}

fn reverb(room_size: f32, damping: f32, mix: f32) -> NodeKind {
    NodeKind::Reverb(Reverb {
        room_size,
        damping,
        mix,
    })
}

fn patch(seed: u32, nodes: Vec<GraphNode>, output: u32) -> AudioPatch {
    AudioPatch {
        seed,
        graph: NodeGraph {
            nodes,
            output: NodeId(output),
        },
    }
}

fn instrument(id: &str, seed: u32, nodes: Vec<GraphNode>, output: u32) -> Instrument {
    Instrument {
        id: id.to_string(),
        patch: patch(seed, nodes, output),
    }
}

#[allow(clippy::too_many_arguments)]
fn ev(
    time_beats: f32,
    instrument_id: &str,
    pitch_multiplier: f32,
    volume: f32,
    gate_beats: f32,
    release_beats: f32,
    pitch_mode: PitchMode,
) -> Event {
    Event {
        time_beats,
        instrument_id: instrument_id.to_string(),
        pitch_multiplier,
        volume,
        gate_beats,
        release_beats,
        pitch_mode,
    }
}

fn peak_abs(buf: &[f32]) -> f32 {
    buf.iter().fold(0.0_f32, |m, &s| m.max(s.abs()))
}

/// Bake `patch`, measure its peak, and rescale the master mixer (`master_id`)
/// so the written patch peaks at [`TARGET_PEAK`].  Returns `(raw_peak,
/// achieved_peak)` — the second is measured from a re-bake so the printed
/// number reflects the file that was actually written, not an assumption.
///
/// Panics if `master_id` is not a [`Mix`]/[`Gain`] node in the graph: that
/// means the trim point is mis-wired and the patch would ship un-normalised.
fn autoscale(patch: &mut AudioPatch, master_id: u32, duration: f32) -> (f32, f32) {
    let raw = peak_abs(&bake(patch, 44_100, duration));
    let mut scaled = false;
    if raw > 0.0 {
        let factor = TARGET_PEAK / raw;
        for node in &mut patch.graph.nodes {
            if node.id == NodeId(master_id) {
                match &mut node.kind {
                    NodeKind::Mix(m) => {
                        m.gain *= factor;
                        scaled = true;
                    }
                    NodeKind::Gain(g) => {
                        g.gain *= factor;
                        scaled = true;
                    }
                    _ => {}
                }
            }
        }
        assert!(
            scaled,
            "autoscale: node {master_id} is not a Mix/Gain master trim",
        );
    }
    let achieved = peak_abs(&bake(patch, 44_100, duration));
    (raw, achieved)
}

fn write_json<T: serde::Serialize>(dir: &Path, name: &str, value: &T) {
    let json = serde_json::to_string_pretty(value).expect("serialise patch");
    fs::write(dir.join(name), json).expect("write patch file");
}

// ============================================================================
// 1. Triple-caw of a crow.
// ============================================================================

/// Three caws from one fixed-length bake.  A 1.8 Hz square LFO edge-triggers
/// the envelope; over a 1.6 s bake its rising edges land at 0.00 s, 0.56 s and
/// 1.11 s — exactly three caws (the fourth would fall at 1.67 s, past the
/// window).  Each caw is a bright sawtooth honk shaped by a bandpass formant,
/// a noise rasp, a fast tremolo "buzz" VCA for roughness, and a per-caw
/// downward pitch glide from a sawtooth LFO synced to the trigger.
fn crow_patch() -> AudioPatch {
    let nodes = vec![
        gnode(0, lfo(1.8, LfoShape::Square, 0.5, 0.5)), // caw trigger (0/1 gate)
        gnode(1, adsr(0.01, 0.12, 0.25, 0.12, AdsrCurve::Exponential)).with_input("gate", from(0)),
        gnode(2, lfo(1.8, LfoShape::Saw, -220.0, 40.0)), // per-caw downward glide
        gnode(3, saw(540.0, 1.0, AntiAlias::Naive)).with_input("freq", from(2)),
        gnode(4, bandpass(1200.0, 2.2)).with_input("in", from(3)), // nasal formant
        gnode(5, white(0.6)),
        gnode(6, bandpass(2600.0, 1.4)).with_input("in", from(5)), // rasp band
        gnode(7, mix(1.0)) // master trim (autoscaled)
            .with_input("tone", modc(4, 0.95))
            .with_input("rasp", modc(6, 0.25)),
        gnode(8, lfo(80.0, LfoShape::Sine, 0.3, 0.7)), // 80 Hz roughness, 0.4..1.0
        gnode(9, vca())
            .with_input("in", from(7))
            .with_input("gain", from(8)),
        gnode(10, vca())
            .with_input("in", from(9))
            .with_input("gain", from(1)),
        gnode(11, reverb(0.3, 0.5, 0.18)).with_input("in", from(10)),
    ];
    patch(7, nodes, 11)
}

/// The crow voice as a sequenced instrument: a [`Gate`] feeds the amplitude
/// envelope (so an `Event`'s gate window is the caw), and a second gate-driven
/// envelope sweeps the pitch down at each onset.
fn crow_instrument() -> Instrument {
    let nodes = vec![
        gnode(0, gate()),
        gnode(1, adsr(0.008, 0.1, 0.5, 0.18, AdsrCurve::Exponential)).with_input("gate", from(0)),
        gnode(2, adsr(0.001, 0.16, 0.0, 0.05, AdsrCurve::Linear)).with_input("gate", from(0)),
        gnode(3, saw(540.0, 1.0, AntiAlias::Naive)).with_input("freq", modc(2, 200.0)),
        gnode(4, bandpass(1200.0, 2.2)).with_input("in", from(3)),
        gnode(5, white(0.6)),
        gnode(6, bandpass(2600.0, 1.4)).with_input("in", from(5)),
        // Trim is higher than the single-graph patch's because the mixdown's
        // tanh soft-clip (not the raw `bake` path) governs the final level
        // here, so the caw can sit louder without clipping.
        gnode(7, mix(2.2))
            .with_input("tone", modc(4, 0.95))
            .with_input("rasp", modc(6, 0.25)),
        gnode(8, lfo(80.0, LfoShape::Sine, 0.3, 0.7)),
        gnode(9, vca())
            .with_input("in", from(7))
            .with_input("gain", from(8)),
        gnode(10, vca())
            .with_input("in", from(9))
            .with_input("gain", from(1)),
        gnode(11, reverb(0.3, 0.5, 0.2)).with_input("in", from(10)),
    ];
    instrument("caw", 7, nodes, 11)
}

fn crow_sequence() -> SequenceRecipe {
    let caws = vec![
        ev(0.0, "caw", 1.0, 0.9, 0.35, 0.9, PitchMode::Varispeed),
        ev(1.0, "caw", 1.06, 0.85, 0.35, 0.9, PitchMode::Varispeed),
        ev(2.0, "caw", 0.96, 0.9, 0.4, 1.2, PitchMode::Varispeed),
    ];
    SequenceRecipe {
        bpm: 120.0,
        sample_rate: 44_100,
        duration_beats: 4.0,
        loop_start_beats: None,
        loop_crossfade_beats: 0.0,
        instruments: vec![crow_instrument()],
        tracks: vec![Track { events: caws }],
    }
}

// ============================================================================
// 2. Short rock groove — drums, bass, guitar.
// ============================================================================

/// 120 BPM, 4 bars of 4/4 with a two-chord-per-progression vamp.  Kick lands
/// on beats 1 & 3 (1 Hz square), the snare backbeat on 2 & 4 (an *inverted*
/// 1 Hz square — negative `depth` flips the half-cycle the gate opens on),
/// hats and bass pump straight 8ths (4 Hz), and the guitar strums power chords
/// on the quarter note (2 Hz).  A slow 0.25 Hz square LFO adds Hz to the
/// bass/guitar oscillators to walk the riff E → G across each 4 s span.
fn rock_patch() -> AudioPatch {
    let nodes = vec![
        // Timing LFOs (square 0/1 gates).
        gnode(0, lfo(1.0, LfoShape::Square, 0.5, 0.5)), // kick: beats 1 & 3
        gnode(1, lfo(1.0, LfoShape::Square, -0.5, 0.5)), // snare: inverted → 2 & 4
        gnode(2, lfo(4.0, LfoShape::Square, 0.5, 0.5)), // 8ths (hat + bass)
        gnode(3, lfo(2.0, LfoShape::Square, 0.5, 0.5)), // quarters (guitar)
        gnode(4, lfo(0.25, LfoShape::Square, -7.8, 7.8)), // chord walk: +0 then +15.6 Hz
        // Kick: pitch-blip sine through an amp VCA.
        gnode(5, adsr(0.001, 0.045, 0.0, 0.01, AdsrCurve::Linear)).with_input("gate", from(0)),
        gnode(6, adsr(0.002, 0.16, 0.0, 0.02, AdsrCurve::Linear)).with_input("gate", from(0)),
        gnode(7, sine(50.0, 1.0)).with_input("freq", modc(5, 95.0)),
        gnode(8, vca())
            .with_input("in", from(7))
            .with_input("gain", from(6)),
        // Snare: highpassed noise + a short tonal body.
        gnode(9, adsr(0.001, 0.13, 0.0, 0.02, AdsrCurve::Linear)).with_input("gate", from(1)),
        gnode(10, white(0.8)),
        gnode(11, highpass(1400.0, 0.8)).with_input("in", from(10)),
        gnode(12, tri(190.0, 0.6)),
        gnode(13, mix(1.0))
            .with_input("noise", modc(11, 0.7))
            .with_input("body", modc(12, 0.5)),
        gnode(14, vca())
            .with_input("in", from(13))
            .with_input("gain", from(9)),
        // Hi-hat: bright highpassed noise, very short.
        gnode(15, adsr(0.001, 0.03, 0.0, 0.005, AdsrCurve::Linear)).with_input("gate", from(2)),
        gnode(16, white(0.7)),
        gnode(17, highpass(8000.0, 0.8)).with_input("in", from(16)),
        gnode(18, vca())
            .with_input("in", from(17))
            .with_input("gain", from(15)),
        // Bass: lowpassed saw, root walks with the chord LFO.
        gnode(19, adsr(0.004, 0.12, 0.35, 0.04, AdsrCurve::Linear)).with_input("gate", from(2)),
        gnode(20, saw(82.41, 1.0, AntiAlias::PolyBlep)).with_input("freq", modc(4, 1.0)),
        gnode(21, lowpass(420.0, 1.0)).with_input("in", from(20)),
        gnode(22, vca())
            .with_input("in", from(21))
            .with_input("gain", from(19)),
        // Guitar: root + fifth power chord (the fifth tracks 1.5× the Hz walk).
        gnode(23, adsr(0.005, 0.12, 0.5, 0.16, AdsrCurve::Linear)).with_input("gate", from(3)),
        gnode(24, saw(164.81, 0.5, AntiAlias::Naive)).with_input("freq", modc(4, 2.0)),
        gnode(25, saw(246.94, 0.5, AntiAlias::Naive)).with_input("freq", modc(4, 3.0)),
        gnode(26, mix(1.0))
            .with_input("root", from(24))
            .with_input("fifth", from(25)),
        gnode(27, lowpass(2200.0, 1.2)).with_input("in", from(26)),
        gnode(28, vca())
            .with_input("in", from(27))
            .with_input("gain", from(23)),
        gnode(29, chorus(0.7, 3.0, 9.0, 0.1, 0.3)).with_input("in", from(28)),
        // Master mix (autoscaled) + glue reverb.
        gnode(30, mix(1.0))
            .with_input("kick", modc(8, 0.95))
            .with_input("snare", modc(14, 0.6))
            .with_input("hat", modc(18, 0.35))
            .with_input("bass", modc(22, 0.7))
            .with_input("guitar", modc(29, 0.6)),
        gnode(31, reverb(0.25, 0.6, 0.1)).with_input("in", from(30)),
    ];
    patch(1, nodes, 31)
}

fn kick_instrument() -> Instrument {
    let nodes = vec![
        gnode(0, gate()),
        gnode(1, adsr(0.002, 0.18, 0.0, 0.02, AdsrCurve::Linear)).with_input("gate", from(0)),
        gnode(2, adsr(0.001, 0.05, 0.0, 0.01, AdsrCurve::Linear)).with_input("gate", from(0)),
        gnode(3, sine(50.0, 1.0)).with_input("freq", modc(2, 95.0)),
        gnode(4, vca())
            .with_input("in", from(3))
            .with_input("gain", from(1)),
    ];
    instrument("kick", 1, nodes, 4)
}

fn snare_instrument() -> Instrument {
    let nodes = vec![
        gnode(0, gate()),
        gnode(1, adsr(0.001, 0.13, 0.0, 0.02, AdsrCurve::Linear)).with_input("gate", from(0)),
        gnode(2, white(0.8)),
        gnode(3, highpass(1400.0, 0.8)).with_input("in", from(2)),
        gnode(4, tri(190.0, 0.6)),
        gnode(5, mix(1.0))
            .with_input("noise", modc(3, 0.7))
            .with_input("body", modc(4, 0.5)),
        gnode(6, vca())
            .with_input("in", from(5))
            .with_input("gain", from(1)),
    ];
    instrument("snare", 2, nodes, 6)
}

fn hat_instrument() -> Instrument {
    let nodes = vec![
        gnode(0, gate()),
        gnode(1, adsr(0.001, 0.03, 0.0, 0.005, AdsrCurve::Linear)).with_input("gate", from(0)),
        gnode(2, white(0.7)),
        gnode(3, highpass(8000.0, 0.8)).with_input("in", from(2)),
        gnode(4, vca())
            .with_input("in", from(3))
            .with_input("gain", from(1)),
    ];
    instrument("hat", 3, nodes, 4)
}

fn bass_instrument() -> Instrument {
    let nodes = vec![
        gnode(0, gate()),
        gnode(1, adsr(0.004, 0.1, 0.7, 0.06, AdsrCurve::Linear)).with_input("gate", from(0)),
        gnode(2, saw(82.41, 1.0, AntiAlias::PolyBlep)),
        gnode(3, lowpass(500.0, 1.0)).with_input("in", from(2)),
        gnode(4, vca())
            .with_input("in", from(3))
            .with_input("gain", from(1)),
    ];
    instrument("bass", 1, nodes, 4)
}

fn guitar_instrument() -> Instrument {
    let nodes = vec![
        gnode(0, gate()),
        gnode(1, adsr(0.005, 0.12, 0.6, 0.18, AdsrCurve::Linear)).with_input("gate", from(0)),
        gnode(2, saw(164.81, 0.5, AntiAlias::Naive)), // E3 root
        gnode(3, saw(246.94, 0.5, AntiAlias::Naive)), // B3 fifth
        gnode(4, mix(1.0))
            .with_input("root", from(2))
            .with_input("fifth", from(3)),
        gnode(5, lowpass(2400.0, 1.1)).with_input("in", from(4)),
        gnode(6, vca())
            .with_input("in", from(5))
            .with_input("gain", from(1)),
        gnode(7, chorus(0.6, 3.0, 9.0, 0.1, 0.3)).with_input("in", from(6)),
    ];
    instrument("guitar", 1, nodes, 7)
}

fn rock_sequence() -> SequenceRecipe {
    // One chord per bar — an E–G–A–G power-chord vamp (semitone ratios from E).
    let chords = [1.0_f32, 1.1892, 1.3348, 1.1892];
    let mut kick = Vec::new();
    let mut snare = Vec::new();
    let mut hat = Vec::new();
    let mut bass = Vec::new();
    let mut guitar = Vec::new();

    for (bar, &chord) in chords.iter().enumerate() {
        let base = bar as f32 * 4.0;
        for &b in &[0.0_f32, 2.0] {
            kick.push(ev(
                base + b,
                "kick",
                1.0,
                0.95,
                0.25,
                0.05,
                PitchMode::Varispeed,
            ));
        }
        for &b in &[1.0_f32, 3.0] {
            snare.push(ev(
                base + b,
                "snare",
                1.0,
                0.7,
                0.25,
                0.05,
                PitchMode::Varispeed,
            ));
        }
        let mut eighth = 0.0;
        while eighth < 4.0 {
            hat.push(ev(
                base + eighth,
                "hat",
                1.0,
                0.4,
                0.12,
                0.02,
                PitchMode::Varispeed,
            ));
            bass.push(ev(
                base + eighth,
                "bass",
                chord,
                0.7,
                0.45,
                0.05,
                PitchMode::TimePreserving,
            ));
            eighth += 0.5;
        }
        for &b in &[0.0_f32, 1.0, 2.0, 3.0] {
            guitar.push(ev(
                base + b,
                "guitar",
                chord,
                0.55,
                0.9,
                0.2,
                PitchMode::TimePreserving,
            ));
        }
    }

    SequenceRecipe {
        bpm: 120.0,
        sample_rate: 44_100,
        duration_beats: 16.5, // 16 beats + a little ring-out for the last chord
        loop_start_beats: None,
        loop_crossfade_beats: 0.0,
        instruments: vec![
            kick_instrument(),
            snare_instrument(),
            hat_instrument(),
            bass_instrument(),
            guitar_instrument(),
        ],
        tracks: vec![
            Track { events: kick },
            Track { events: snare },
            Track { events: hat },
            Track { events: bass },
            Track { events: guitar },
        ],
    }
}

// ============================================================================
// 3. Moody cyberpunk-noir ambient soundscape.
// ============================================================================

/// A dark drone bed: two slightly-detuned saws over a sub sine, a tritone pad
/// with chorus shimmer, swept pink/brown noise for rain-and-rumble air, and a
/// sparse sine "ping" (a distant neon signal) — all under a big reverb.  Slow
/// sine LFOs breathe the filter cutoffs and the pad volume.
fn ambient_patch() -> AudioPatch {
    let nodes = vec![
        // Drone: detuned saws + sub, swept lowpass.
        gnode(0, saw(55.0, 0.5, AntiAlias::PolyBlep)),
        gnode(1, saw(55.35, 0.5, AntiAlias::PolyBlep)), // detune → slow beating
        gnode(2, sine(27.5, 0.6)),
        gnode(3, mix(1.0))
            .with_input("a", modc(0, 0.5))
            .with_input("b", modc(1, 0.5))
            .with_input("sub", modc(2, 0.7)),
        gnode(4, lfo(0.06, LfoShape::Sine, 525.0, 375.0)), // cutoff 150..1200 Hz
        gnode(5, lowpass(300.0, 3.0))
            .with_input("in", from(3))
            .with_input("cutoff_hz", from(4)),
        // Pad: A + tritone Eb + a fifth shimmer, tremolo'd and chorused.
        gnode(6, tri(220.0, 0.3)),
        gnode(7, tri(311.13, 0.22)), // tritone above A — the noir interval
        gnode(8, sine(660.0, 0.12)),
        gnode(9, mix(1.0))
            .with_input("a", modc(6, 0.5))
            .with_input("b", modc(7, 0.5))
            .with_input("c", modc(8, 0.5)),
        gnode(10, lfo(0.15, LfoShape::Sine, 0.35, 0.6)), // breathing tremolo 0.25..0.95
        gnode(11, vca())
            .with_input("in", from(9))
            .with_input("gain", from(10)),
        gnode(12, lowpass(1800.0, 1.0)).with_input("in", from(11)),
        gnode(13, chorus(0.4, 4.0, 14.0, 0.2, 0.5)).with_input("in", from(12)),
        // Noise bed: swept pink-noise "rain".
        gnode(14, pink(0.5)),
        gnode(15, lfo(0.08, LfoShape::Sine, 500.0, 700.0)), // cutoff 200..1200 Hz
        gnode(16, lowpass(800.0, 0.7))
            .with_input("in", from(14))
            .with_input("cutoff_hz", from(15)),
        // Rumble: brown-noise sub floor.
        gnode(17, brown(0.5)),
        gnode(18, lowpass(120.0, 0.7)).with_input("in", from(17)),
        // Ping: sparse two-partial sine bell, struck every ~5.6 s.
        gnode(19, lfo(0.18, LfoShape::Square, 0.5, 0.5)),
        gnode(20, adsr(0.005, 1.2, 0.0, 0.5, AdsrCurve::Exponential)).with_input("gate", from(19)),
        gnode(21, sine(880.0, 0.5)),
        gnode(22, sine(1318.0, 0.2)),
        gnode(23, mix(1.0))
            .with_input("a", modc(21, 0.6))
            .with_input("b", modc(22, 0.4)),
        gnode(24, vca())
            .with_input("in", from(23))
            .with_input("gain", from(20)),
        // Master mix (autoscaled) → big reverb → DC/subsonic highpass.
        gnode(25, mix(1.0))
            .with_input("drone", modc(5, 0.6))
            .with_input("pad", modc(13, 0.5))
            .with_input("rain", modc(16, 0.5))
            .with_input("rumble", modc(18, 0.5))
            .with_input("ping", modc(24, 0.5)),
        gnode(26, reverb(0.85, 0.45, 0.45)).with_input("in", from(25)),
        gnode(27, highpass(30.0, 0.7)).with_input("in", from(26)),
    ];
    // Master trim is node 25 (the mix feeding the reverb); rescaling it stays
    // linear through the reverb + highpass tail.
    patch(42, nodes, 27)
}

fn drone_instrument() -> Instrument {
    let nodes = vec![
        gnode(0, gate()),
        gnode(1, adsr(2.0, 1.0, 0.8, 3.0, AdsrCurve::Exponential)).with_input("gate", from(0)),
        gnode(2, saw(55.0, 0.5, AntiAlias::PolyBlep)),
        gnode(3, saw(55.35, 0.5, AntiAlias::PolyBlep)),
        gnode(4, sine(27.5, 0.6)),
        gnode(5, mix(1.0))
            .with_input("a", modc(2, 0.5))
            .with_input("b", modc(3, 0.5))
            .with_input("sub", modc(4, 0.7)),
        gnode(6, lfo(0.06, LfoShape::Sine, 525.0, 375.0)),
        gnode(7, lowpass(300.0, 3.0))
            .with_input("in", from(5))
            .with_input("cutoff_hz", from(6)),
        gnode(8, vca())
            .with_input("in", from(7))
            .with_input("gain", from(1)),
    ];
    instrument("drone", 42, nodes, 8)
}

fn pad_instrument() -> Instrument {
    let nodes = vec![
        gnode(0, gate()),
        gnode(1, adsr(1.5, 1.0, 0.7, 2.5, AdsrCurve::Exponential)).with_input("gate", from(0)),
        gnode(2, tri(220.0, 0.3)),
        gnode(3, tri(311.13, 0.22)),
        gnode(4, sine(660.0, 0.12)),
        gnode(5, mix(1.0))
            .with_input("a", modc(2, 0.5))
            .with_input("b", modc(3, 0.5))
            .with_input("c", modc(4, 0.5)),
        gnode(6, lfo(0.15, LfoShape::Sine, 0.35, 0.6)),
        gnode(7, vca())
            .with_input("in", from(5))
            .with_input("gain", from(6)),
        gnode(8, lowpass(1800.0, 1.0)).with_input("in", from(7)),
        gnode(9, chorus(0.4, 4.0, 14.0, 0.2, 0.5)).with_input("in", from(8)),
        gnode(10, vca())
            .with_input("in", from(9))
            .with_input("gain", from(1)),
    ];
    instrument("pad", 42, nodes, 10)
}

fn noise_instrument() -> Instrument {
    let nodes = vec![
        gnode(0, gate()),
        gnode(1, adsr(2.0, 2.0, 0.6, 3.0, AdsrCurve::Exponential)).with_input("gate", from(0)),
        gnode(2, pink(0.5)),
        gnode(3, lfo(0.08, LfoShape::Sine, 500.0, 700.0)),
        gnode(4, lowpass(800.0, 0.7))
            .with_input("in", from(2))
            .with_input("cutoff_hz", from(3)),
        gnode(5, vca())
            .with_input("in", from(4))
            .with_input("gain", from(1)),
    ];
    instrument("noise", 99, nodes, 5)
}

fn ping_instrument() -> Instrument {
    let nodes = vec![
        gnode(0, gate()),
        gnode(1, adsr(0.005, 1.2, 0.0, 0.6, AdsrCurve::Exponential)).with_input("gate", from(0)),
        gnode(2, sine(880.0, 0.5)),
        gnode(3, sine(1318.0, 0.2)),
        gnode(4, mix(1.0))
            .with_input("a", modc(2, 0.6))
            .with_input("b", modc(3, 0.4)),
        gnode(5, vca())
            .with_input("in", from(4))
            .with_input("gain", from(1)),
        gnode(6, reverb(0.7, 0.4, 0.4)).with_input("in", from(5)),
    ];
    instrument("ping", 42, nodes, 6)
}

fn ambient_sequence() -> SequenceRecipe {
    // 60 BPM → 1 beat = 1 s.  16 s window, seamless loop from beat 4.
    let drone = vec![
        ev(0.0, "drone", 1.0, 0.6, 16.0, 4.0, PitchMode::TimePreserving),
        ev(8.0, "drone", 1.5, 0.3, 8.0, 4.0, PitchMode::TimePreserving), // fifth enters
    ];
    let pad = vec![ev(1.0, "pad", 1.0, 0.45, 14.0, 3.0, PitchMode::Varispeed)];
    let noise = vec![ev(0.0, "noise", 1.0, 0.3, 16.0, 3.0, PitchMode::Varispeed)];
    let ping = vec![
        ev(3.0, "ping", 1.0, 0.4, 0.1, 2.0, PitchMode::TimePreserving),
        ev(7.0, "ping", 1.5, 0.4, 0.1, 2.0, PitchMode::TimePreserving),
        ev(
            11.0,
            "ping",
            1.3348,
            0.35,
            0.1,
            2.0,
            PitchMode::TimePreserving,
        ),
        ev(15.0, "ping", 2.0, 0.4, 0.1, 2.5, PitchMode::TimePreserving),
    ];
    SequenceRecipe {
        bpm: 60.0,
        sample_rate: 44_100,
        duration_beats: 16.0,
        loop_start_beats: Some(4.0),
        loop_crossfade_beats: 2.0,
        instruments: vec![
            drone_instrument(),
            pad_instrument(),
            noise_instrument(),
            ping_instrument(),
        ],
        tracks: vec![
            Track { events: drone },
            Track { events: pad },
            Track { events: noise },
            Track { events: ping },
        ],
    }
}
