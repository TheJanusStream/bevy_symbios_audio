# bevy_symbios_audio

Algorithmic audio generator for Bevy — a DAG-of-nodes synth (sine /
square / sawtooth / triangle oscillators, white / pink / brown noise,
ADSR envelopes, biquad LP/HP/BP filters, LFOs, `Mix` / `Gain` (VCA)
combiners, a sequencer-driven `Gate`, `Chorus` / `Reverb` delay-line
effects, and cross-node modulation routing where several sources can feed
one input port and are summed) that bakes
deterministic `Vec<f32>` buffers off the main thread via a private
[`rayon`](https://crates.io/crates/rayon) pool, with an optional
content-addressed `PatchCache` (memory + on-disk WAV) so DID-seeded
ambient survives a process restart.

The DSP layer is Bevy-free and unit-testable on its own; the Bevy
integration adds `SymbiosAudioPlugin` with a `PendingAudioPatch` →
`AudioPatchReady` ECS handover, plus a `SequenceRecipe` /
`bake_sequence` pipeline for multi-instrument ambient tracks with
seamless tail-crossfade loops.

Every config struct (oscillators, filters, envelopes, LFOs, sequence
recipes) implements
[`symbios_genetics::Genotype`](https://crates.io/crates/symbios-genetics)
through the `impl_genotype!` macro, so the whole DSP language plugs
straight into the evolutionary-search algorithms in that crate.

## Usage

The crate exposes three layered entry points.  Pick whichever matches
your problem.

### 1. One-shot patch → `Vec<f32>` (no Bevy)

```rust
use bevy_symbios_audio::{
    AudioPatch, NodeGraph, GraphNode, NodeId, NodeKind, SineOsc, bake,
};

let patch = AudioPatch {
    seed: 0,
    graph: NodeGraph {
        nodes: vec![GraphNode {
            id: NodeId(0),
            kind: NodeKind::Sine(SineOsc::default()), // 440 Hz
            ..Default::default()
        }],
        output: NodeId(0),
    },
};

let samples: Vec<f32> = bake(&patch, 44_100, 1.0); // 1 s @ 44.1 kHz
```

`bake` is single-threaded, fully deterministic for a given
`(patch, sample_rate, duration)` triple, and Bevy-free — use it in
unit tests, CLI tools, or anywhere you just want raw samples.  It
panics on a structurally invalid graph; call `try_bake` instead for a
`Result<Vec<f32>, GraphError>` when the patch can't be trusted.

To combine voices inside a single patch, wire several connections into
one port (they're summed) or use a `Mix` node; `Gain` is a VCA whose
`"gain"` port multiplies the signal (clean tremolo / ring-mod, as
opposed to an oscillator's *additive* `"amplitude"` port).

### 2. Async bake inside a Bevy app

Add the plugin once at startup, then spawn a `PendingAudioPatch` for
every bake; a polling system attaches an `AudioPatchReady` with a
`Handle<AudioSource>` once the worker thread finishes.

```rust,ignore
use bevy::prelude::*;
use bevy_symbios_audio::{PendingAudioPatch, SymbiosAudioPlugin};

App::new()
    .add_plugins((DefaultPlugins, SymbiosAudioPlugin::default()))
    .add_systems(Startup, |mut commands: Commands| {
        commands.spawn(PendingAudioPatch::new(patch, 44_100, 1.0));
    });
```

Configure the worker-pool size by setting
`SymbiosAudioPlugin { config: AsyncAudioConfig { pool_threads: N } }`
(0 selects `available_parallelism / 2`; defaults to
`DEFAULT_POOL_THREADS = 4`).  Dropping the `PendingAudioPatch`
component (e.g. when its entity despawns) cancels the bake — both a
not-yet-started one and one already in flight (the worker checks the
flag periodically as it bakes) — so rapid spawn/despawn doesn't
saturate the pool with work that can no longer be used.

### 3. Sequenced multi-voice tracks → loopable buffer

For ambient drones, layered textures, and anything longer than a
single voice, build a `SequenceRecipe` of named `Instrument`s and
timed `Event`s, then call `bake_sequence`:

```rust,ignore
use bevy_symbios_audio::{bake_sequence, SequenceRecipe};

let buffer: Vec<f32> = bake_sequence(&recipe);
```

Each `Event` has a real note shape: the gate is held open for
`gate_beats` and then the bake continues through `release_beats` of
tail.  Wire a `Gate` node into an `AdsrEnvelope`'s `"gate"` port and the
envelope attacks/sustains while the gate is open, then *releases* and
rings out across the tail — `release_beats: 0.0` reproduces a hard
one-shot.  Instruments with an unresolvable graph (or a typo'd
`instrument_id`) are skipped with a warning rather than aborting the
mixdown.  The sequence editor marks a note whose `instrument_id` names no
instrument, and offers to reassign it.

Set `recipe.loop_start_beats = Some(b)` and a non-zero
`loop_crossfade_beats` to get a seamless loop — the mixdown baker
pre-mixes a tail crossfade onto the loop region so a hard
`Source::loop_..()` is click-free at the seam.  See
`examples/wind_demo.rs` for a complete recipe (brown-noise wind drone
with LFO-swept cutoff plus an ADSR-gated sine voice).

```sh
cargo run --release --example wind_demo
```

### Caching baked WAVs

For workloads where the same `(patch, sample_rate, duration)` triple
will be baked repeatedly (room re-entry, scrubbing, A/B comparisons),
insert a `PatchCache` resource *before* adding `SymbiosAudioPlugin`:

```rust,ignore
use bevy_symbios_audio::{
    DEFAULT_MEMORY_CACHE_ENTRIES, PatchCache, SymbiosAudioPlugin,
};

app.insert_resource(PatchCache::memory(DEFAULT_MEMORY_CACHE_ENTRIES));
//  or, to survive process restart:
app.insert_resource(PatchCache::file("/path/to/audio_cache", 0)?);
```

`PendingAudioPatch::new` records a content-addressed cache key for
each dispatch; the poll system writes the produced WAV bytes into the
resource on completion.  Probe the cache up front with
`bake_with_cache`, which returns either pre-baked bytes (cache hit) or
a `PendingAudioPatch` to spawn.  Use `PendingAudioPatch::new_uncached`
to opt out of cache writes for one-off bakes.

## CLI

A `symbios-audio-cli` binary ships with the crate for offline baking
outside the Bevy app — handy for video pipelines and sound-design
iteration.  It has two subcommands.

`bake` renders a single [`AudioPatch`](src/patch.rs); the patch is
rate- and length-agnostic, so the rate and duration are chosen on the
command line:

```sh
# Bake `patch.json` to `out.wav` at default 44.1 kHz / 1.0 s.
cargo run --release --bin symbios-audio-cli -- bake patch.json out.wav

# Custom rate and duration.
cargo run --release --bin symbios-audio-cli -- bake \
    --sample-rate 48000 --duration 5.0 patch.json out.wav
```

`bake-sequence` renders a [`SequenceRecipe`](src/sequence.rs) through
[`bake_sequence`](src/mixdown.rs) — multi-instrument, event-timed
tracks with `tanh` soft-clipping and seamless-loop crossfades.  The
sample rate and length live *inside* the recipe (`sample_rate`, `bpm`,
`duration_beats`), so the subcommand takes only the input and output
paths:

```sh
cargo run --release --bin symbios-audio-cli -- bake-sequence recipe.json out.wav
```

Both `patch.json` and `recipe.json` are the serde-JSON form of their
respective types.  The output is a mono IEEE-float WAV file — playable
in every video editor and DAW.  Ogg Vorbis / Opus exports are
deliberately out of scope because the pure-Rust encoder ecosystem is
still rough.

Ready-made example patches for both subcommands live in
[`patches/`](patches/) (a crow caw, a rock groove, and a cyberpunk-noir
ambient bed, each in single-graph and sequenced form); regenerate them
with `cargo run --example demo_patches`.

## Editor (`egui` feature)

Enable the `egui` feature to pull in
[`bevy_egui`](https://crates.io/crates/bevy_egui) and the `ui` module — a
set of composable editor widgets any `bevy_egui` host (Overlands) can
embed:

- per-node config editors (one widget group per `NodeKind`) — a
  two-column grid, name on the left and control on the right, with the
  unit written into the value — plus a kind picker, captioned
  (`node_kind_editor`) or bare (`node_kind_picker`, for a host that puts
  the name in a title of its own),
- a pannable / zoomable node-graph canvas (`audio_patch_canvas`) that
  edits a whole `AudioPatch` — drag nodes, wire ports, choose the output.
  Each input port's dot sits beside its own named row in the node's
  "Inputs" list; a wire can be dropped on the dot or anywhere on the row,
  and while it is dragged the port it would connect to is highlighted and
  named. A graph that cannot bake says why by name ("#2 Lowpass and #3
  Gain feed each other in a loop") and outlines the nodes at fault. A wire
  can be hovered, picked, re-routed by its end and deleted, and picking one
  opens a panel on its curve with what it drives, its amount and a cross;
  a port's row says what driving it does, and a wire that sweeps one states
  the range it really produces ("about 150 ± 250 Hz"). Add node is a menu
  grouped by role, also on a right-click on clear canvas. A node the output
  cannot be reached from is dimmed and wears a "not heard" badge, and a
  node's own menu offers "Hear this node", which auditions a copy
  (`patch_hearing`) and never edits the patch. Ctrl+Z undoes a committed
  edit, Ctrl+Shift+Z and Ctrl+Y redo, and `forget_history` drops the ring
  when a host re-seeds the value under an open editor,
- a DAW-style sequence-recipe timeline (`sequence_recipe_editor`) with
  transport, instruments, and draggable track / event lanes. Renaming an
  instrument takes its notes and its canvas layout with it, applied on
  Enter and only for a name that is unique, not empty, and within
  symbios-audio's `Envelope` byte limit. A note is painted in its
  instrument's own colour and says its pitch when the block has room;
  Fit goes back to the zoom that shows the whole sequence; a double- or
  right-click on a lane adds a note of that lane's instrument and Ctrl+D
  duplicates what is picked. Each lane's gutter carries an M and an S that
  decide what the AUDITION hears (solo wins over mute, and the recipe is
  not touched). A note whose instrument is gone is tinted, hatched and
  outlined in the theme's error colour, labelled `missing: <id>` in a tone
  that reads on that tint, and the inspector offers to reassign every note
  of that id,
- a pure-egui `waveform` widget plus a Bevy audition monitor
  (`AudioEditorPlugin`) for auditioning edits. A replaced or stopped patch
  bake is cancelled, not left to run. The monitor publishes where its voice
  has got to (`position_secs`, wrapped into the loop, with `loop_secs`), and
  `waveform_with_cursor` draws that as a playhead with a seconds axis;
  `SequenceEditorState::set_playhead` puts the same position on the
  timeline in beats. What does not need a bake — the monitor's own level,
  mute — is a `MonitorControl` rather than a `MonitorRequest`,
- the audition strip (`audition_strip`), the row a host puts above an
  editor to hear it: Audition, Stop, an Auto toggle that re-bakes a playing
  audition shortly after each committed edit, a status chip (Idle, Baking
  with the seconds so far, Playing, Muted, Error), a caption saying what is
  played ("1.0 s at 22.05 kHz, looped"), a Level for the monitor's own
  output, and the waveform with the playhead on it. It returns the
  `MonitorRequest` to write rather than writing it, and its
  `MonitorControl`s through `AuditionState::take_controls`,
- `EditorLimits`, the caps every Add is held to. They default to
  symbios-audio's own `Envelope`, the bound the record sanitiser enforces
  anyway, so an editor nobody configured refuses exactly what would
  otherwise have been deleted after the edit: the control is disabled *with
  the reason* and a `N / cap` readout counts against it, and a removal that
  loses work asks first. `set_sample_rates` says which rates the transport
  offers,
- `symbios_genetics`-backed Mutate / Reroll seed helpers (`mutate_patch`,
  `randomize_seed`) and a reusable JSON copy/paste section (`json_io`),
- `EditorStyle`, the colour roles the editors paint with: canvas ground,
  node boxes, wires and ports, the timeline's lanes, loop markers and
  notes, the waveform, and ok / warn / error. With nothing set, every
  widget derives one from the `Visuals` it is drawn with
  (`EditorStyle::from_visuals`), so the editors follow egui's dark and light
  themes. A host with its own palette calls
  `set_editor_style(ctx, style)` when its theme changes.

Buttons say what they do in words ("Add node", "Fit view", "Delete note").
The few glyphs left match Overlands' own: the remove cross `✖`, the valid
check `✔`, the audition's `▶` and `⏹`, and the pencil of an instrument's
`✏ Edit`.

Every editor returns an `EditorResponse { changed, rebake }` so a host
can persist mid-drag edits (`changed`) but only re-bake on commit
(`rebake`).  Two runnable examples drive the widgets end to end:

```sh
cargo run --example patch_editor --features egui
cargo run --example sequence_editor --features egui
```

Both canvases take all the space left in the `Ui` they are given, so draw
them **last**. Put anything that must stay visible above them and a
sequence editor beside them in a panel. In an `egui::Window`, content drawn
after a canvas is never seen, and the window grows every frame until it
reaches the screen edge. `host_window` shows both editors inside windows,
laid out the way that works. It doubles as a screenshot harness:
`--shot <path>` saves a picture and quits, and `--light` switches theme.
`--status <idle|baking|playing|playing-sequence|muted|error>` puts the
audition strips in that state through real requests. The rest drive the
editors the way a user would, each for a picture of one thing:

| flag | what it sets up |
| --- | --- |
| `--broken` | the patch opens with a loop in its graph |
| `--orphan` | the sequence opens with notes that name no instrument |
| `--rename <name>` | types a name over the open instrument's and presses Enter |
| `--drag` | holds a wire over a port's row, mid-drag |
| `--wire` / `--pick` | parks the pointer on a wire, or clicks it |
| `--menu <add\|node>` | holds the Add menu, or a node's own menu, open |
| `--unheard` | points the output at one node, so the rest are not heard |
| `--notes <picked\|box\|snap\|track>` | drives the timeline's notes |
| `--limits` / `--confirm` / `--beyond` | a cap reached, a removal asked about, a value outside its own slider |
| `--hover <label>` | parks the pointer on one widget, for its tooltip |
| `--notice <text>` / `--notice-live <text>` | the host's own line above the editors |

There is no flag that pins the cursor to a chosen second: a seek on a
looping sink is refused by the backend (see Limitations), so
`--status playing-sequence` waits for the voice to be past 0.2 s instead.

```sh
cargo run --example host_window --features egui -- --shot host_window.png
cargo run --example host_window --features egui -- --status error --shot error.png
cargo run --example host_window --features egui -- --orphan --shot orphan.png
```

## Determinism

A bake of the same `(patch, sample_rate, duration)` returns a
bit-identical `Vec<f32>` every time, on every machine:

- [`patch::topo_sort`](src/patch.rs) uses Kahn's algorithm with
  sorted tie-breaking — no `HashMap` iteration.  The graph is compiled
  once into a flat, index-addressed plan in that order; per-sample
  evaluation is `Vec`-indexed (no map iteration), so output identity
  hinges only on the topo order and the RNG draw order.
- A single `ChaCha8Rng` seeded from `AudioPatch::seed` drives every
  stochastic node; it is never reset or reseeded mid-bake, and stateful
  nodes (e.g. the sample-and-hold LFO) draw from it at well-defined
  points only.

This is the contract the DID-seeded ambient layer in the Overlands
integration relies on — one stable ambient track per room seed.

## Features

- **`egui`** (off by default) — pulls in
  [`bevy_egui`](https://crates.io/crates/bevy_egui) and the `ui` module:
  composable patch / sequence editor widgets you can embed in a
  `bevy_egui` host.  See [Editor](#editor-egui-feature).

The crate's own `bevy` dependency enables Bevy's `wav` feature so
downstream users don't need to enable it on their own `bevy = ...`
line for `AudioSource` to decode the in-memory WAV the bridge
produces.

## Limitations

- **Mono only.**  Both `bake` and `bake_sequence` produce a single
  channel.  Stereo / multichannel routing is out of scope.
- **Oscillator aliasing is opt-out.**  Square / saw / triangle default
  to naïve generators (`anti_alias: Naive`) — the audible grit is part of
  the aesthetic.  Set `anti_alias: PolyBlep` on any of them to band-limit
  the discontinuities (PolyBLEP for square/saw value steps, polyBLAMP for
  triangle slope corners); it stays pure-arithmetic deterministic and
  tracks per-sample `"freq"` modulation.  Residual aliasing remains at
  extreme high pitches; a wavetable/mipmap path would go further.
- **Pitch shift is per-event.**  Each sequencer `Event` picks a
  `pitch_mode`: `Varispeed` (the default) resamples — pitch and time are
  coupled, so pitch-up plays shorter than its gate and pitch-down hangs
  past it (tape-style, handy for SFX); `TimePreserving` retunes the
  instrument's oscillators at synthesis time and bakes the event at its
  true length, so pitch and duration are independent with no resampling
  artifacts.  Because retuning happens at synthesis, no PSOLA / phase
  vocoder is needed — but it only re-pitches *oscillators* (not sampled
  or noise-based material), and LFO/filter settings stay fixed.
- **A looping audition cannot be seeked.** `PlaybackMode::Loop` appends
  rodio's `repeat_infinite()`, which wraps the source in a `Buffered` whose
  `try_seek` is an unconditional `NotSupported`, so `MonitorControl::Seek`
  is refused and moves nothing — including the cursor. Nothing in the
  editors offers a click-to-seek for that reason. A custom `Source` that
  plays a run-up once and then loops a window, forwarding `try_seek`, is
  what would recover it.
- **Buffer ≤ ~6 hours.**  `data_size` in the WAV header is a 32-bit
  field, capping ~1.07 G samples (≈ 6.7 h @ 44.1 kHz).  `samples_to_wav_bytes`
  now panics rather than emitting a silently-wrapped (corrupt) header, so
  split longer bakes into segments.
