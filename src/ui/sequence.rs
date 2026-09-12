//! Timeline editor for a [`SequenceRecipe`] — the "arrange view".
//!
//! Sits above the patch canvas the way the schema does: a recipe owns named
//! [`Instrument`]s (each an [`AudioPatch`]) and parallel [`Track`]s of
//! [`Event`]s scheduled in beats.  This module edits the *sequence* layer:
//!
//! - **Transport** — BPM, sample rate, total duration, and the optional loop
//!   window (start + crossfade), which is drawn as markers on the timeline.
//! - **Instruments** — add / remove / rename; pick one as *active* to edit its
//!   embedded patch in the Phase-2 canvas (see [`active_instrument_canvas`]).
//!   Events name their instrument by id, so a rename carries the instrument's
//!   events on every track, and its canvas layout, to the new id. It applies
//!   on Enter or when the field loses focus, never per keystroke, and only
//!   for a name that is non-empty after trimming, unique, and within
//!   symbios-audio's [`Envelope`] byte limit; until then the field shows the
//!   reason in the error colour, and Esc puts the name back.
//! - **Timeline** — one lane per track; events are blocks whose x is
//!   `time_beats`, width is `gate_beats`, with a translucent tail for
//!   `release_beats`.  Drag a block to move it, drag its right edge to resize
//!   the gate, double-click an empty lane to add an event.  An event whose
//!   instrument id names no instrument bakes to silence; its block is drawn
//!   hatched in the error colour and labelled `missing: <id>`.
//! - **Inspector** — full numeric editing of the selected event, plus delete.
//!   For an event with no instrument it says so and offers to reassign every
//!   event of that id to an existing instrument.
//!
//! Every colour the timeline paints — its ground, grid and ruler, the lanes,
//! the loop markers, note blocks and their names, the error colour of a
//! missing note or a refused name — comes from the [`EditorStyle`] in effect
//! ([`crate::ui::style`]), so it follows the host's theme (#58).
//!
//! Like the rest of [`crate::ui`] this is pure egui returning an
//! [`EditorResponse`]; the host drives the bake-and-play monitor (Phase 3) off
//! `rebake`, baking the whole recipe with [`crate::mixdown::bake_sequence`].
//!
//! Editor-only view state (active instrument, per-instrument canvas layout,
//! selection, zoom, names being typed) lives in [`SequenceEditorState`],
//! never in the serialized recipe.

use std::collections::{HashMap, HashSet};

use bevy_egui::egui::{self, Align2, Color32, Id, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};

use symbios_genetics::Genotype;

use crate::Envelope;
use crate::patch::AudioPatch;
use crate::sequence::{Event, Instrument, PitchMode, SequenceRecipe, Track};

use super::evolve::fresh_rng;
use super::history::EditHistory;
use super::io::json_io;
use super::style::{EditorStyle, editor_style};
use super::{
    EditorResponse, JsonIoState, PatchEditorState, audio_patch_canvas, drag_debounced,
    slider_debounced,
};

const RULER_H: f32 = 22.0;
const LANE_H: f32 = 34.0;
/// Left margin inside the timeline reserved for the per-lane remove button.
const GUTTER: f32 = 22.0;
const DEFAULT_PPB: f32 = 48.0;
/// Beat grid that drags snap to on release.
const SNAP: f32 = 0.25;
/// Smallest gate a block can be resized to (beats).
const MIN_GATE: f32 = 0.1;

/// Whether an in-progress block drag is moving the event or resizing its gate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum DragMode {
    #[default]
    Move,
    Resize,
}

/// Editor-side view state for the sequence editor — kept out of the
/// serialized [`SequenceRecipe`].  Construct with [`Default`].
#[derive(Clone, Debug)]
pub struct SequenceEditorState {
    /// Index of the instrument whose patch is open in the canvas, if any.
    active_instrument: Option<usize>,
    /// Per-instrument patch-canvas layout, keyed by instrument id so each
    /// instrument keeps its node positions when you switch away and back.
    /// A rename moves the entry to the new id.
    canvas_states: HashMap<String, PatchEditorState>,
    /// Selected `(track, event)` for the inspector.
    selected_event: Option<(usize, usize)>,
    /// Timeline zoom, pixels per beat.
    px_per_beat: f32,
    /// Move-vs-resize for the active block drag.
    drag_mode: DragMode,
    /// Mutation rate for the "Mutate recipe" button.
    mutate_rate: f32,
    /// Buffer + last error for the JSON import/export section.
    json: JsonIoState,
    /// Names being typed into instrument rows, keyed by row. The recipe is
    /// renamed only when one is committed; see [`NameEdit`].
    name_edits: HashMap<usize, NameEdit>,
    /// Undo/redo over the whole recipe (#60). An instrument's patch is part
    /// of the recipe, so this is the *only* history in the sequence editor:
    /// the embedded canvas's own is turned off in
    /// [`active_instrument_canvas`].
    history: EditHistory<SequenceRecipe>,
    /// The sequence editor took this frame's keyboard.
    owns_keys: bool,
    /// It acted on an Escape this frame.
    took_escape: bool,
}

impl Default for SequenceEditorState {
    fn default() -> Self {
        Self {
            active_instrument: None,
            canvas_states: HashMap::new(),
            selected_event: None,
            px_per_beat: DEFAULT_PPB,
            drag_mode: DragMode::Move,
            mutate_rate: 0.3,
            json: JsonIoState::default(),
            name_edits: HashMap::new(),
            history: EditHistory::default(),
            owns_keys: false,
            took_escape: false,
        }
    }
}

/// A name typed into an instrument's row, held here until it is committed.
///
/// Notes find their instrument by id, so a rename has to move them too, and
/// it must not do that per keystroke: typing "inst1" -> "inst" -> "inst2"
/// while an "inst2" exists would pour inst1's notes into inst2's, and no
/// later keystroke could tell them apart again (#55). So the field edits
/// this buffer, and Enter or leaving the field commits it once, if the name
/// is valid ([`check_name`]). Esc drops it.
#[derive(Clone, Debug)]
struct NameEdit {
    /// The row's id when typing began. A row that no longer has it (a
    /// removal moved the rows up, or the host swapped the recipe) drops the
    /// edit rather than apply it to another instrument.
    from: String,
    /// What the field shows.
    text: String,
    /// The field had the focus the last time it was drawn, so its next
    /// frame without focus is a commit. A refused commit clears this, so a
    /// refused name waits for the user and never applies itself later
    /// because some other instrument moved out of its way.
    typing: bool,
}

/// Why a typed name cannot be an instrument's id.
#[derive(Clone, Debug, PartialEq, Eq)]
enum NameRefusal {
    /// Nothing but whitespace.
    Empty,
    /// Longer than the byte limit.
    TooLong { bytes: usize, limit: usize },
    /// Another instrument already answers to it.
    Taken(String),
}

impl std::fmt::Display for NameRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "A name can't be empty"),
            Self::TooLong { bytes, limit } => {
                write!(f, "Too long: {bytes} bytes, the limit is {limit}")
            }
            Self::Taken(name) => write!(f, "Another instrument is called '{name}'"),
        }
    }
}

/// The longest instrument id, in bytes: the bound the record sanitiser
/// holds a recipe to (symbios-audio's [`Envelope`]), so a name the editor
/// accepts is never cut short on its way into a record.
fn instrument_id_byte_limit() -> usize {
    Envelope::default().max_instrument_id_bytes
}

/// The id `text` would give instrument `row`, trimmed, or why it can't.
fn check_name<'a>(
    recipe: &SequenceRecipe,
    row: usize,
    text: &'a str,
    limit: usize,
) -> Result<&'a str, NameRefusal> {
    let name = text.trim();
    if name.is_empty() {
        Err(NameRefusal::Empty)
    } else if name.len() > limit {
        Err(NameRefusal::TooLong {
            bytes: name.len(),
            limit,
        })
    } else if recipe
        .instruments
        .iter()
        .enumerate()
        .any(|(i, inst)| i != row && inst.id == name)
    {
        Err(NameRefusal::Taken(name.to_owned()))
    } else {
        Ok(name)
    }
}

/// Point every note of `from`, on every track, at `to`.
fn reassign_notes(recipe: &mut SequenceRecipe, from: &str, to: &str) {
    for event in recipe.tracks.iter_mut().flat_map(|t| t.events.iter_mut()) {
        if event.instrument_id == from {
            event.instrument_id = to.to_owned();
        }
    }
}

/// Rename instrument `row` to `new`, which [`check_name`] accepted, in one
/// step: its notes on every track and its canvas layout go with it, so the
/// rename is a pure relabel and the recipe bakes the same.
fn rename_instrument(
    recipe: &mut SequenceRecipe,
    canvas_states: &mut HashMap<String, PatchEditorState>,
    row: usize,
    new: &str,
) {
    let old = std::mem::replace(&mut recipe.instruments[row].id, new.to_owned());
    // A recipe from outside the editor can hold two instruments with one id.
    // The other still answers to `old`, so the notes and the layout stay
    // with it: which of the two a note meant cannot be known.
    if recipe.instruments.iter().any(|inst| inst.id == old) {
        return;
    }
    reassign_notes(recipe, &old, new);
    match canvas_states.remove(&old) {
        Some(layout) => {
            canvas_states.insert(new.to_owned(), layout);
        }
        // Never opened, so it has no layout; one already under `new` was
        // left by a removed instrument and would scatter this one's nodes.
        None => {
            canvas_states.remove(new);
        }
    }
}

/// How the timeline and the inspector name a note whose instrument is gone.
fn missing_label(instrument_id: &str) -> String {
    if instrument_id.is_empty() {
        "missing: (none)".to_owned()
    } else {
        format!("missing: {instrument_id}")
    }
}

/// What is wrong with a note whose instrument is gone, as a sentence.
fn no_instrument_text(instrument_id: &str) -> String {
    if instrument_id.is_empty() {
        "This note names no instrument".to_owned()
    } else {
        format!("No instrument called '{instrument_id}'")
    }
}

/// A block for a note whose instrument is gone: the error colour, hatched
/// and outlined, so it reads as broken without relying on the hue alone.
/// Every colour is `error` (the host's `error_fg_color`) at some strength.
fn paint_missing_block(painter: &egui::Painter, body: Rect, selected: bool, error: Color32) {
    painter.rect_filled(
        body,
        3.0,
        error.gamma_multiply(if selected { 0.35 } else { 0.2 }),
    );
    let hatch = painter.with_clip_rect(body.intersect(painter.clip_rect()));
    let stroke = Stroke::new(1.0, error.gamma_multiply(0.45));
    let mut x = body.left() - body.height();
    while x < body.right() {
        hatch.line_segment(
            [
                Pos2::new(x, body.bottom()),
                Pos2::new(x + body.height(), body.top()),
            ],
            stroke,
        );
        x += 6.0;
    }
    painter.rect_stroke(
        body,
        3.0,
        Stroke::new(if selected { 2.0 } else { 1.0 }, error),
        StrokeKind::Inside,
    );
}

/// A note's block: the style's note fill, outlined in `note_selected` when
/// selected. The fill does not change with the selection, so the note's
/// name reads the same either way, and the selection is a shape as well as
/// a colour.
fn paint_note_block(painter: &egui::Painter, body: Rect, selected: bool, style: &EditorStyle) {
    painter.rect_filled(body, 3.0, style.note_fill);
    let edge = if selected {
        Stroke::new(2.0, style.note_selected)
    } else {
        // The ground's colour, so two notes that touch stay two.
        Stroke::new(1.0, style.timeline_ground)
    };
    painter.rect_stroke(body, 3.0, edge, StrokeKind::Inside);
}

impl SequenceEditorState {
    /// Index of the instrument currently open for patch editing, if any.
    pub fn active_instrument(&self) -> Option<usize> {
        self.active_instrument
    }

    /// Open instrument `index` in [`active_instrument_canvas`], or close the
    /// canvas with `None`. It does the same as clicking the instrument's
    /// Edit toggle, so a host can open an editor on a chosen instrument. An index
    /// past the end of the recipe's instruments is cleared on the next draw.
    pub fn set_active_instrument(&mut self, index: Option<usize>) {
        self.active_instrument = index;
    }

    /// The `(track, event)` open in the event inspector, if any.
    pub fn selected_event(&self) -> Option<(usize, usize)> {
        self.selected_event
    }

    /// Open event `event` of track `track` in the event inspector, or clear
    /// the selection with `None`. It does the same as clicking the event's
    /// block. A pair that names no event is cleared on the next draw.
    pub fn set_selected_event(&mut self, selected: Option<(usize, usize)>) {
        self.selected_event = selected;
    }
}

/// Snap a beat value to the [`SNAP`] grid, clamped to `>= 0`.
fn snap_beat(beats: f32) -> f32 {
    ((beats / SNAP).round() * SNAP).max(0.0)
}

/// A fresh `instN` id not already used by an instrument in `recipe`.
fn unique_instrument_id(recipe: &SequenceRecipe) -> String {
    let mut n = 1;
    loop {
        let candidate = format!("inst{n}");
        if !recipe.instruments.iter().any(|i| i.id == candidate) {
            return candidate;
        }
        n += 1;
    }
}

impl SequenceEditorState {
    /// Step the recipe back to before the last committed edit. `true` if
    /// anything moved.
    ///
    /// There is one history for the whole recipe, an open instrument's
    /// patch included: the patch is part of the recipe, so an edit made in
    /// the embedded canvas is a step on this stack and not on that canvas's
    /// (#60, Overlands #1333).
    pub fn undo(&mut self, recipe: &mut SequenceRecipe) -> bool {
        self.history.undo(recipe)
    }

    /// Step the recipe forward again after an [`Self::undo`].
    pub fn redo(&mut self, recipe: &mut SequenceRecipe) -> bool {
        self.history.redo(recipe)
    }

    /// Whether there is a committed edit to undo.
    pub fn can_undo(&self) -> bool {
        self.history.can_undo()
    }

    /// Whether there is an undone edit to redo.
    pub fn can_redo(&self) -> bool {
        self.history.can_redo()
    }

    /// Tell the editor its recipe was replaced from outside, so the swap
    /// becomes a step in its history. See
    /// [`crate::ui::PatchEditorState::note_external_change`].
    pub fn note_external_change(&mut self, recipe: &SequenceRecipe) {
        self.history.commit(recipe);
    }

    /// Whether the sequence editor took this frame's keyboard. See
    /// [`crate::ui::PatchEditorState::wants_keyboard`].
    pub fn wants_keyboard(&self) -> bool {
        self.owns_keys
    }

    /// Whether it acted on an Escape this frame by clearing its selection.
    /// See [`crate::ui::PatchEditorState::took_escape`].
    pub fn took_escape(&self) -> bool {
        self.took_escape
    }
}

/// Edit a whole [`SequenceRecipe`]: transport, instruments, the track/event
/// timeline, and the selected-event inspector.
///
/// `state` persists view/selection across frames — pass the same instance each
/// frame.  Does **not** draw the active instrument's patch; call
/// [`active_instrument_canvas`] for that (typically in a separate panel).
pub fn sequence_recipe_editor(
    ui: &mut egui::Ui,
    recipe: &mut SequenceRecipe,
    state: &mut SequenceEditorState,
    id: Id,
) -> EditorResponse {
    let mut res = EditorResponse::NONE;
    let style = editor_style(ui);
    state.history.begin(recipe);

    egui::CollapsingHeader::new("Transport")
        .default_open(true)
        .show(ui, |ui| res.merge(transport(ui, recipe)));

    ui.horizontal(|ui| {
        if ui
            .button("Mutate recipe")
            .on_hover_text("Nudge BPM and event volumes via symbios-genetics")
            .clicked()
        {
            recipe.mutate(&mut fresh_rng(), state.mutate_rate);
            res.changed = true;
            res.rebake = true;
        }
        ui.add(egui::Slider::new(&mut state.mutate_rate, 0.0..=1.0).text("rate"));
    });
    res.merge(json_io(ui, recipe, &mut state.json, id.with("recipe_json")));

    ui.separator();
    res.merge(instruments_panel(ui, recipe, state, id, &style));

    ui.separator();
    res.merge(timeline(ui, recipe, state, id, &style));

    ui.separator();
    egui::CollapsingHeader::new("Event inspector")
        .default_open(true)
        .show(ui, |ui| {
            res.merge(event_inspector(ui, recipe, state, &style))
        });

    res.merge(recipe_keys(ui, recipe, state, ui.min_rect()));
    if res.rebake {
        state.history.commit(recipe);
    }
    res
}

/// The sequence editor's keyboard: undo and redo over the whole recipe, and
/// Escape to clear the selected event.
///
/// The same ownership rule as the canvas's — the pointer over the editor,
/// nothing holding text focus — so typing an instrument's name keeps its
/// own keys. The *canvas* half of the sequence editor reads its keys
/// through this too, via [`active_instrument_canvas`]: one recipe, one
/// history, one place the keys land.
fn recipe_keys(
    ui: &mut egui::Ui,
    recipe: &mut SequenceRecipe,
    state: &mut SequenceEditorState,
    region: egui::Rect,
) -> EditorResponse {
    use egui::{Key, Modifiers};

    let mut res = EditorResponse::NONE;
    state.took_escape = false;
    let typing = ui.memory(|m| m.focused()).is_some();
    state.owns_keys = !typing && ui.rect_contains_pointer(region);
    if !state.owns_keys {
        return res;
    }
    // Most specific first: `consume_key` ignores extra Shift, so Ctrl+Z
    // would swallow Ctrl+Shift+Z the other way round.
    let (redo, undo, escape) = ui.input_mut(|i| {
        (
            i.consume_key(Modifiers::COMMAND | Modifiers::SHIFT, Key::Z)
                | i.consume_key(Modifiers::COMMAND, Key::Y),
            i.consume_key(Modifiers::COMMAND, Key::Z),
            i.consume_key(Modifiers::NONE, Key::Escape),
        )
    });
    if redo {
        res.rebake = state.history.redo(recipe);
    } else if undo {
        res.rebake = state.history.undo(recipe);
    }
    res.changed = res.rebake;
    if escape && state.selected_event.take().is_some() {
        state.took_escape = true;
    }
    res
}

/// Draw the active instrument's patch in the Phase-2 node canvas, or a hint if
/// none is selected.  Each instrument keeps its own canvas layout (keyed by
/// id) in `state`.
///
/// # The canvas claims the rest of the `Ui` — draw it last
///
/// With an instrument open this is [`audio_patch_canvas`], which takes all
/// the space left in `ui`. Anything drawn after it is laid out below that
/// space, and inside an [`egui::Window`] it also makes the window grow to its
/// constraint (the same heading on [`audio_patch_canvas`] has the mechanism).
/// This one is easy to miss because it only shows up once an instrument is
/// open: with none open, the hint label is short and the content after it
/// fits. Give it the last region, usually an `egui::CentralPanel` next to an
/// `egui::Panel::left` that holds [`sequence_recipe_editor`] in a
/// `ScrollArea`, as the `sequence_editor` and `host_window` examples do.
pub fn active_instrument_canvas(
    ui: &mut egui::Ui,
    recipe: &mut SequenceRecipe,
    state: &mut SequenceEditorState,
    id: Id,
) -> EditorResponse {
    let Some(i) = state
        .active_instrument
        .filter(|i| *i < recipe.instruments.len())
    else {
        state.active_instrument = None;
        ui.label("Press an instrument's \u{270F} Edit to open its patch here.");
        return EditorResponse::NONE;
    };
    let inst_id = recipe.instruments[i].id.clone();
    ui.label(format!("Patch for instrument \u{201C}{inst_id}\u{201D}"));
    state.history.begin(recipe);
    let canvas_state = state.canvas_states.entry(inst_id).or_default();
    // This patch is part of the recipe, so the recipe's history owns it.
    // With both on, one Ctrl+Z would walk two stacks at once and the
    // recipe's would go stale (#60, Overlands #1333).
    canvas_state.disable_history();
    let mut res = audio_patch_canvas(
        ui,
        &mut recipe.instruments[i].patch,
        canvas_state,
        id.with("inst_canvas"),
    );
    // The canvas's own keys are its own (delete, duplicate, nudge, fit);
    // undo and redo are the recipe's, and reach it from here.
    let region = ui.min_rect();
    res.merge(recipe_keys(ui, recipe, state, region));
    if res.rebake {
        state.history.commit(recipe);
    }
    res
}

// ---------------------------------------------------------------------------
// Transport
// ---------------------------------------------------------------------------

fn transport(ui: &mut egui::Ui, recipe: &mut SequenceRecipe) -> EditorResponse {
    let mut res = EditorResponse::NONE;

    res.merge(slider_debounced(
        ui,
        egui::Slider::new(&mut recipe.bpm, 20.0..=300.0)
            .logarithmic(true)
            .text("BPM"),
    ));

    ui.horizontal(|ui| {
        ui.label("Sample rate");
        egui::ComboBox::from_id_salt("seq_sample_rate")
            .selected_text(recipe.sample_rate.to_string())
            .show_ui(ui, |ui| {
                for sr in [22_050_u32, 32_000, 44_100, 48_000, 96_000] {
                    if ui
                        .selectable_label(recipe.sample_rate == sr, sr.to_string())
                        .clicked()
                    {
                        recipe.sample_rate = sr;
                        res.changed = true;
                        res.rebake = true;
                    }
                }
            });
    });

    res.merge(drag_debounced(
        ui,
        "duration (beats)",
        &mut recipe.duration_beats,
        0.25,
        0.25..=512.0,
    ));

    let dur = recipe.duration_beats.max(0.0);
    let mut looping = recipe.loop_start_beats.is_some();
    if ui.checkbox(&mut looping, "seamless loop").changed() {
        recipe.loop_start_beats = looping.then_some(0.0);
        res.changed = true;
        res.rebake = true;
    }
    if let Some(loop_start) = recipe.loop_start_beats.as_mut() {
        res.merge(drag_debounced(
            ui,
            "loop start (beats)",
            loop_start,
            0.25,
            0.0..=dur,
        ));
    }
    if recipe.loop_start_beats.is_some() {
        res.merge(drag_debounced(
            ui,
            "crossfade (beats)",
            &mut recipe.loop_crossfade_beats,
            0.25,
            0.0..=dur.max(0.25),
        ));
    }

    res
}

// ---------------------------------------------------------------------------
// Instruments
// ---------------------------------------------------------------------------

fn instruments_panel(
    ui: &mut egui::Ui,
    recipe: &mut SequenceRecipe,
    state: &mut SequenceEditorState,
    id: Id,
    style: &EditorStyle,
) -> EditorResponse {
    let mut res = EditorResponse::NONE;
    let mut add = false;
    let mut remove: Option<usize> = None;
    let mut rename: Option<(usize, String)> = None;

    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Instruments").strong());
        if ui.button("Add instrument").clicked() {
            add = true;
        }
    });

    state.name_edits.retain(|row, edit| {
        recipe
            .instruments
            .get(*row)
            .is_some_and(|inst| inst.id == edit.from)
    });
    for i in 0..recipe.instruments.len() {
        let field = ui.horizontal(|ui| {
            let active = state.active_instrument == Some(i);
            if ui
                // The pencil and the word, as Overlands says it ("✏ Edit
                // audio…"): the one glyph the vocabulary keeps for editing
                // (#58). U+270F, the emoji-presentation pencil, is in egui's
                // embedded Noto Emoji, where the text-only U+270E is not, so
                // it draws in any host that ships egui's default fonts (#52).
                .selectable_label(active, "\u{270F} Edit")
                .on_hover_text("Open this instrument's patch in the canvas")
                .clicked()
            {
                state.active_instrument = if active { None } else { Some(i) };
            }
            let field = name_field(
                ui,
                recipe,
                i,
                &mut state.name_edits,
                id.with(("instrument_name", i)),
                style.error,
            );
            let nodes = recipe.instruments[i].patch.graph.nodes.len();
            ui.label(egui::RichText::new(format!("{nodes} node(s)")).weak());
            if ui
                .button("\u{2716}")
                .on_hover_text("Remove instrument")
                .clicked()
            {
                remove = Some(i);
            }
            field
        });
        let NameField {
            rect,
            focused,
            commit,
        } = field.inner;
        if let Some(name) = commit {
            rename = Some((i, name));
        }
        // Under the field: why the typed name is refused, or that the name
        // it shows is not the instrument's yet. The second happens only when
        // a refused name became valid because another instrument moved out
        // of its way; it waits for the user rather than apply itself.
        let note = state.name_edits.get(&i).and_then(|edit| {
            match check_name(recipe, i, &edit.text, instrument_id_byte_limit()) {
                Err(refusal) => Some((refusal.to_string(), style.error)),
                Ok(_) if !focused => Some((
                    "Not applied yet: Enter in the field applies it, Esc puts the name back"
                        .to_owned(),
                    ui.visuals().weak_text_color(),
                )),
                Ok(_) => None,
            }
        });
        if let Some((note, colour)) = note {
            ui.horizontal(|ui| {
                ui.add_space(rect.left() - ui.cursor().left());
                ui.label(egui::RichText::new(note).small().color(colour));
            });
        }
    }

    if let Some((i, name)) = rename {
        rename_instrument(recipe, &mut state.canvas_states, i, &name);
        res.changed = true;
        res.rebake = true;
    }
    if add {
        let id = unique_instrument_id(recipe);
        recipe.instruments.push(Instrument {
            id,
            patch: AudioPatch::default(),
        });
        res.changed = true;
        res.rebake = true;
    }
    if let Some(i) = remove {
        recipe.instruments.remove(i);
        match state.active_instrument {
            Some(a) if a == i => state.active_instrument = None,
            Some(a) if a > i => state.active_instrument = Some(a - 1),
            _ => {}
        }
        // The rows below move up, and a name being typed moves with its row.
        state.name_edits = std::mem::take(&mut state.name_edits)
            .into_iter()
            .filter_map(|(row, edit)| match row.cmp(&i) {
                std::cmp::Ordering::Less => Some((row, edit)),
                std::cmp::Ordering::Equal => None,
                std::cmp::Ordering::Greater => Some((row - 1, edit)),
            })
            .collect();
        res.changed = true;
        res.rebake = true;
    }

    res
}

/// What one instrument row's name field did this frame.
struct NameField {
    /// Where the field is, so the line under it can line up with it.
    rect: Rect,
    /// The field has the keyboard focus.
    focused: bool,
    /// A valid name committed this frame, to rename the instrument to.
    commit: Option<String>,
}

/// The name field of instrument `row`: it edits the row's [`NameEdit`], not
/// the id, and reports a commit only for a name [`check_name`] accepts.
/// While the typed name is refused, the text and the frame take `error`.
fn name_field(
    ui: &mut egui::Ui,
    recipe: &SequenceRecipe,
    row: usize,
    edits: &mut HashMap<usize, NameEdit>,
    field_id: Id,
    error: Color32,
) -> NameField {
    let current = &recipe.instruments[row].id;
    let limit = instrument_id_byte_limit();
    let refused = |edits: &HashMap<usize, NameEdit>| {
        edits
            .get(&row)
            .is_some_and(|edit| check_name(recipe, row, &edit.text, limit).is_err())
    };

    let mut text = edits
        .get(&row)
        .map_or_else(|| current.clone(), |edit| edit.text.clone());
    let mut field = egui::TextEdit::singleline(&mut text)
        .id(field_id)
        .desired_width(140.0);
    if refused(edits) {
        field = field.text_color(error);
    }
    let r = ui.add(field).on_hover_text(
        "Rename: Enter or clicking away applies it, and the notes follow; Esc puts the name back",
    );

    if r.changed() {
        let from = edits
            .get(&row)
            .map_or_else(|| current.clone(), |edit| edit.from.clone());
        edits.insert(
            row,
            NameEdit {
                from,
                text,
                typing: true,
            },
        );
        // The colour above was chosen before this keystroke.
        ui.ctx().request_repaint();
    }
    if refused(edits) {
        ui.painter().rect_stroke(
            r.rect,
            ui.visuals().widgets.inactive.corner_radius,
            Stroke::new(1.0, error),
            StrokeKind::Outside,
        );
    }

    let rect = r.rect;
    let focused = r.has_focus();
    let nothing = NameField {
        rect,
        focused,
        commit: None,
    };
    if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Escape)) {
        edits.remove(&row);
        return nothing;
    }
    if focused {
        if let Some(edit) = edits.get_mut(&row) {
            edit.typing = true;
        }
        return nothing;
    }
    // Not focused: Enter, Tab or a click elsewhere took the focus, or the
    // field was not drawn while it had it (a collapsed header, a closed
    // window). An edit typed into until now is committed, once.
    let Some(edit) = edits.get_mut(&row) else {
        return nothing;
    };
    if !std::mem::take(&mut edit.typing) {
        return nothing;
    }
    let commit = match check_name(recipe, row, &edit.text, limit) {
        // Refused: the edit and its reason stay until fixed or Esc'd.
        Err(_) => return nothing,
        Ok(name) if name == current.as_str() => None,
        Ok(name) => Some(name.to_owned()),
    };
    edits.remove(&row);
    NameField { commit, ..nothing }
}

// ---------------------------------------------------------------------------
// Timeline
// ---------------------------------------------------------------------------

fn timeline(
    ui: &mut egui::Ui,
    recipe: &mut SequenceRecipe,
    state: &mut SequenceEditorState,
    id: Id,
    style: &EditorStyle,
) -> EditorResponse {
    let mut res = EditorResponse::NONE;

    ui.horizontal(|ui| {
        if ui.button("Add track").clicked() {
            recipe.tracks.push(Track::default());
            res.changed = true;
            res.rebake = true;
        }
        res.merge(slider_debounced(
            ui,
            egui::Slider::new(&mut state.px_per_beat, 12.0..=160.0).text("px/beat"),
        ));
        ui.label(format!("{} track(s)", recipe.tracks.len()));
    });

    let ppb = state.px_per_beat.max(4.0);
    let dur = recipe.duration_beats.max(1.0);
    let loop_start = recipe.loop_start_beats;
    let crossfade = recipe.loop_crossfade_beats;
    let default_inst = recipe
        .instruments
        .first()
        .map(|i| i.id.clone())
        .unwrap_or_default();
    // Notes whose id is not in here name no instrument and bake to nothing.
    let known: HashSet<String> = recipe.instruments.iter().map(|i| i.id.clone()).collect();
    let error = style.error;

    let mut total = dur;
    for t in &recipe.tracks {
        for e in &t.events {
            total = total.max(e.time_beats + e.gate_beats + e.release_beats);
        }
    }
    total = (total + 4.0).ceil();
    let lanes = recipe.tracks.len();

    let mut add_event: Option<(usize, f32)> = None;
    let mut remove_track: Option<usize> = None;

    egui::ScrollArea::horizontal()
        .id_salt(id.with("timeline_scroll"))
        .show(ui, |ui| {
            let width = GUTTER + total * ppb + 16.0;
            let height = RULER_H + (lanes.max(1) as f32) * LANE_H;
            let (rect, _) = ui.allocate_exact_size(Vec2::new(width, height), Sense::hover());
            let painter = ui.painter_at(rect);
            let bx = |beat: f32| rect.left() + GUTTER + beat * ppb;

            painter.rect_filled(rect, 0.0, style.timeline_ground);

            // Beat ruler + grid lines.
            let step = if ppb < 20.0 {
                4
            } else if ppb < 40.0 {
                2
            } else {
                1
            };
            let mut beat = 0i32;
            while (beat as f32) <= total {
                let x = bx(beat as f32);
                painter.line_segment(
                    [Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())],
                    Stroke::new(1.0, style.timeline_grid),
                );
                if beat % step == 0 {
                    painter.text(
                        Pos2::new(x + 2.0, rect.top() + 1.0),
                        Align2::LEFT_TOP,
                        beat.to_string(),
                        egui::FontId::proportional(10.0),
                        style.ground_text,
                    );
                }
                beat += 1;
            }

            // Loop markers + crossfade shade.
            let lanes_top = rect.top() + RULER_H;
            if let Some(ls) = loop_start {
                let x = bx(ls);
                painter.line_segment(
                    [Pos2::new(x, lanes_top), Pos2::new(x, rect.bottom())],
                    Stroke::new(2.0, style.loop_start),
                );
            }
            let x_end = bx(dur);
            painter.line_segment(
                [Pos2::new(x_end, lanes_top), Pos2::new(x_end, rect.bottom())],
                Stroke::new(2.0, style.loop_end),
            );
            if crossfade > 0.0 {
                let x0 = bx((dur - crossfade).max(0.0));
                painter.rect_filled(
                    Rect::from_min_max(Pos2::new(x0, lanes_top), Pos2::new(x_end, rect.bottom())),
                    0.0,
                    style.crossfade_band,
                );
            }

            // Lanes + events.
            for (ti, track) in recipe.tracks.iter_mut().enumerate() {
                let lane_top = lanes_top + ti as f32 * LANE_H;
                painter.rect_filled(
                    Rect::from_min_max(
                        Pos2::new(rect.left(), lane_top),
                        Pos2::new(rect.right(), lane_top + LANE_H),
                    ),
                    0.0,
                    if ti % 2 == 0 {
                        style.lane
                    } else {
                        style.lane_alt
                    },
                );

                // Per-lane remove button.
                let x_rect = Rect::from_min_size(
                    Pos2::new(rect.left() + 3.0, lane_top + (LANE_H - 14.0) * 0.5),
                    Vec2::splat(14.0),
                );
                let x_resp = ui.interact(x_rect, id.with(("rm_track", ti)), Sense::click());
                painter.text(
                    x_rect.center(),
                    Align2::CENTER_CENTER,
                    "\u{2716}",
                    egui::FontId::proportional(13.0),
                    if x_resp.hovered() {
                        error
                    } else {
                        style.ground_text
                    },
                );
                if x_resp.clicked() {
                    remove_track = Some(ti);
                }

                // Empty-lane double-click adds an event at that beat.
                let bg = Rect::from_min_max(
                    Pos2::new(rect.left() + GUTTER, lane_top),
                    Pos2::new(rect.right(), lane_top + LANE_H),
                );
                let bg_resp = ui.interact(bg, id.with(("lane_bg", ti)), Sense::click());
                if bg_resp.double_clicked()
                    && let Some(p) = bg_resp.interact_pointer_pos()
                {
                    add_event = Some((ti, snap_beat((p.x - bx(0.0)) / ppb)));
                }

                for (ei, ev) in track.events.iter_mut().enumerate() {
                    let ex = bx(ev.time_beats);
                    let gate_w = (ev.gate_beats * ppb).max(6.0);
                    let etop = lane_top + 4.0;
                    let eh = LANE_H - 8.0;
                    let body = Rect::from_min_size(Pos2::new(ex, etop), Vec2::new(gate_w, eh));
                    let selected = state.selected_event == Some((ti, ei));
                    let missing = !known.contains(&ev.instrument_id);

                    let tail_w = ev.release_beats * ppb;
                    if tail_w > 0.5 {
                        painter.rect_filled(
                            Rect::from_min_size(
                                Pos2::new(body.right(), etop),
                                Vec2::new(tail_w, eh),
                            ),
                            2.0,
                            if missing {
                                error.gamma_multiply(0.15)
                            } else {
                                style.release_tail
                            },
                        );
                    }
                    if missing {
                        paint_missing_block(&painter, body, selected, error);
                    } else {
                        paint_note_block(&painter, body, selected, style);
                    }
                    painter.text(
                        Pos2::new(body.left() + 4.0, body.center().y),
                        Align2::LEFT_CENTER,
                        if missing {
                            missing_label(&ev.instrument_id)
                        } else {
                            ev.instrument_id.clone()
                        },
                        egui::FontId::proportional(11.0),
                        if missing { error } else { style.note_text },
                    );

                    let resp = ui.interact(body, id.with(("ev", ti, ei)), Sense::click_and_drag());
                    let resp = if missing {
                        resp.on_hover_text(format!(
                            "{}, so this note is silent. Select it to reassign it.",
                            no_instrument_text(&ev.instrument_id)
                        ))
                    } else {
                        resp
                    };
                    if resp.clicked() {
                        state.selected_event = Some((ti, ei));
                    }
                    if resp.drag_started() {
                        let near_right = resp
                            .interact_pointer_pos()
                            .is_some_and(|p| p.x >= body.right() - 8.0);
                        state.drag_mode = if near_right {
                            DragMode::Resize
                        } else {
                            DragMode::Move
                        };
                        state.selected_event = Some((ti, ei));
                    }
                    if resp.dragged() {
                        let dx = resp.drag_delta().x / ppb;
                        match state.drag_mode {
                            DragMode::Move => ev.time_beats = (ev.time_beats + dx).max(0.0),
                            DragMode::Resize => ev.gate_beats = (ev.gate_beats + dx).max(MIN_GATE),
                        }
                        res.changed = true;
                    }
                    if resp.drag_stopped() {
                        match state.drag_mode {
                            DragMode::Move => ev.time_beats = snap_beat(ev.time_beats),
                            DragMode::Resize => {
                                ev.gate_beats = snap_beat(ev.gate_beats).max(MIN_GATE)
                            }
                        }
                        res.rebake = true;
                    }
                }
            }
        });

    if let Some(ti) = remove_track {
        if ti < recipe.tracks.len() {
            recipe.tracks.remove(ti);
        }
        state.selected_event = None;
        res.changed = true;
        res.rebake = true;
    }
    if let Some((ti, beat)) = add_event
        && ti < recipe.tracks.len()
    {
        recipe.tracks[ti].events.push(Event {
            time_beats: beat,
            instrument_id: default_inst,
            volume: 0.8,
            ..Event::default()
        });
        state.selected_event = Some((ti, recipe.tracks[ti].events.len() - 1));
        res.changed = true;
        res.rebake = true;
    }

    res
}

// ---------------------------------------------------------------------------
// Event inspector
// ---------------------------------------------------------------------------

fn event_inspector(
    ui: &mut egui::Ui,
    recipe: &mut SequenceRecipe,
    state: &mut SequenceEditorState,
    style: &EditorStyle,
) -> EditorResponse {
    let mut res = EditorResponse::NONE;

    let Some((ti, ei)) = state.selected_event else {
        ui.label("No event selected — click an event, or double-click a lane to add one.");
        return res;
    };
    if ti >= recipe.tracks.len() || ei >= recipe.tracks[ti].events.len() {
        state.selected_event = None;
        ui.label("No event selected.");
        return res;
    }

    let inst_ids: Vec<String> = recipe.instruments.iter().map(|i| i.id.clone()).collect();
    let dur = recipe.duration_beats.max(1.0);
    let error = style.error;
    let mut delete = false;
    // Set by the reassign offer: every note of the first id takes the second.
    let mut reassign_all: Option<(String, String)> = None;
    let sharing_id = {
        let id = &recipe.tracks[ti].events[ei].instrument_id;
        recipe
            .tracks
            .iter()
            .flat_map(|t| &t.events)
            .filter(|e| e.instrument_id == *id)
            .count()
    };

    {
        let ev = &mut recipe.tracks[ti].events[ei];
        ui.label(format!("Track {ti}, event #{ei}"));
        let missing = !inst_ids.contains(&ev.instrument_id);
        if missing {
            ui.colored_label(error, no_instrument_text(&ev.instrument_id));
        }

        ui.horizontal_wrapped(|ui| {
            egui::ComboBox::from_id_salt("evt_instrument")
                .selected_text(if missing {
                    egui::RichText::new(missing_label(&ev.instrument_id)).color(error)
                } else {
                    egui::RichText::new(ev.instrument_id.as_str())
                })
                .show_ui(ui, |ui| {
                    for name in &inst_ids {
                        if ui
                            .selectable_label(ev.instrument_id == *name, name)
                            .clicked()
                            && ev.instrument_id != *name
                        {
                            ev.instrument_id = name.clone();
                            res.changed = true;
                            res.rebake = true;
                        }
                    }
                })
                .response
                .on_hover_text("The instrument this note plays");
            if missing {
                let whose = if ev.instrument_id.is_empty() {
                    "Reassign all notes with no instrument to\u{2026}".to_owned()
                } else {
                    format!("Reassign all notes of '{}' to\u{2026}", ev.instrument_id)
                };
                ui.add_enabled_ui(!inst_ids.is_empty(), |ui| {
                    ui.menu_button(whose, |ui| {
                        for name in &inst_ids {
                            if ui.button(name).clicked() {
                                reassign_all = Some((ev.instrument_id.clone(), name.clone()));
                                ui.close();
                            }
                        }
                    })
                    .response
                    .on_hover_text(format!(
                        "All {sharing_id} note(s) that name it, on every track, not only this one"
                    ));
                })
                .response
                .on_disabled_hover_text("There are no instruments yet: add one above");
            }
        });

        res.merge(drag_debounced(
            ui,
            "time (beats)",
            &mut ev.time_beats,
            0.05,
            0.0..=dur * 4.0,
        ));
        res.merge(drag_debounced(
            ui,
            "gate (beats)",
            &mut ev.gate_beats,
            0.05,
            MIN_GATE..=dur * 4.0,
        ));
        res.merge(drag_debounced(
            ui,
            "release (beats)",
            &mut ev.release_beats,
            0.05,
            0.0..=32.0,
        ));
        res.merge(slider_debounced(
            ui,
            egui::Slider::new(&mut ev.pitch_multiplier, 0.25..=4.0)
                .logarithmic(true)
                .text("pitch \u{00D7}"),
        ));
        // Pitch mode: tape varispeed (pitch ↔ time coupled) vs. synthesis-
        // time retune (note keeps its slot regardless of pitch).
        ui.horizontal(|ui| {
            ui.label("pitch mode");
            for (variant, label) in [
                (PitchMode::Varispeed, "Varispeed"),
                (PitchMode::TimePreserving, "Time-preserving"),
            ] {
                let selected = ev.pitch_mode == variant;
                if ui.selectable_label(selected, label).clicked() && !selected {
                    ev.pitch_mode = variant;
                    res.changed = true;
                    res.rebake = true;
                }
            }
        });
        res.merge(slider_debounced(
            ui,
            egui::Slider::new(&mut ev.volume, 0.0..=1.0).text("volume"),
        ));

        if ui.button("Delete note").clicked() {
            delete = true;
        }
    }

    if let Some((from, to)) = reassign_all {
        reassign_notes(recipe, &from, &to);
        res.changed = true;
        res.rebake = true;
    }
    if delete {
        recipe.tracks[ti].events.remove(ei);
        state.selected_event = None;
        res.changed = true;
        res.rebake = true;
    }

    res
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recipe_with(instruments: &[&str], tracks: usize) -> SequenceRecipe {
        SequenceRecipe {
            instruments: instruments
                .iter()
                .map(|id| Instrument {
                    id: (*id).to_string(),
                    patch: AudioPatch::default(),
                })
                .collect(),
            tracks: (0..tracks).map(|_| Track::default()).collect(),
            ..SequenceRecipe::default()
        }
    }

    #[test]
    fn snap_beat_rounds_to_quarter_grid_and_clamps() {
        assert_eq!(snap_beat(0.12), 0.0);
        assert_eq!(snap_beat(0.13), 0.25);
        assert_eq!(snap_beat(1.6), 1.5);
        assert_eq!(snap_beat(-3.0), 0.0);
    }

    #[test]
    fn unique_instrument_id_avoids_collisions() {
        let recipe = recipe_with(&["inst1", "inst2", "drum"], 0);
        assert_eq!(unique_instrument_id(&recipe), "inst3");
    }

    #[test]
    fn active_instrument_accessor_reflects_state() {
        let mut state = SequenceEditorState::default();
        assert_eq!(state.active_instrument(), None);
        state.active_instrument = Some(2);
        assert_eq!(state.active_instrument(), Some(2));
    }

    #[test]
    fn editor_renders_headless_without_panicking() {
        let mut recipe = recipe_with(&["wind", "kick"], 2);
        recipe.tracks[0].events.push(Event {
            time_beats: 0.0,
            instrument_id: "wind".into(),
            gate_beats: 4.0,
            ..Event::default()
        });
        recipe.loop_start_beats = Some(2.0);
        recipe.loop_crossfade_beats = 1.0;
        let mut state = SequenceEditorState {
            active_instrument: Some(0),
            ..Default::default()
        };

        let ctx = egui::Context::default();
        for _ in 0..3 {
            let _ = ctx.run_ui(egui::RawInput::default(), |root| {
                egui::Panel::left("seq").show(root, |ui| {
                    sequence_recipe_editor(ui, &mut recipe, &mut state, Id::new("seq"));
                });
                egui::CentralPanel::default().show(root, |ui| {
                    active_instrument_canvas(ui, &mut recipe, &mut state, Id::new("seq_canvas"));
                });
            });
        }
    }

    #[test]
    fn empty_recipe_renders_headless_without_panicking() {
        let mut recipe = SequenceRecipe::default();
        let mut state = SequenceEditorState::default();
        let ctx = egui::Context::default();
        let _ = ctx.run_ui(egui::RawInput::default(), |root| {
            egui::CentralPanel::default().show(root, |ui| {
                sequence_recipe_editor(ui, &mut recipe, &mut state, Id::new("seq"));
            });
        });
    }

    // -----------------------------------------------------------------------
    // Renaming instruments and missing instruments (#55, Overlands #1328)
    // -----------------------------------------------------------------------
    //
    // These drive the editor through its real widgets, the way a user does:
    // a name field is found by the text it shows (AccessKit), focused, and
    // typed into, and what the editor paints is read back from its shapes.
    // Nothing here reaches past the widgets to call the rename directly, so
    // the same tests ran against the field that edited `inst.id` in place.

    use egui::accesskit;

    use crate::ui::style::set_editor_style;
    use crate::ui::style::tests::{AA, distinct_style};
    use crate::ui::test_paint::{button_labels, colours, contrast_on, glyph_labels, shapes};

    /// A note of `instrument` at `time_beats`.
    fn note(instrument: &str, time_beats: f32) -> Event {
        Event {
            time_beats,
            instrument_id: instrument.into(),
            gate_beats: 0.5,
            ..Event::default()
        }
    }

    /// `inst1` and `inst2`, with `inst1` playing on two of three tracks.
    fn two_instrument_recipe() -> SequenceRecipe {
        let mut recipe = recipe_with(&["inst1", "inst2"], 3);
        recipe.tracks[0].events = vec![note("inst1", 0.0), note("inst2", 1.0)];
        recipe.tracks[1].events = vec![note("inst1", 2.0), note("inst1", 3.0)];
        recipe.tracks[2].events = vec![note("inst2", 0.0)];
        recipe
    }

    /// Every event's instrument id, track by track.
    fn note_ids(recipe: &SequenceRecipe) -> Vec<Vec<&str>> {
        recipe
            .tracks
            .iter()
            .map(|t| t.events.iter().map(|e| e.instrument_id.as_str()).collect())
            .collect()
    }

    fn instrument_ids(recipe: &SequenceRecipe) -> Vec<&str> {
        recipe.instruments.iter().map(|i| i.id.as_str()).collect()
    }

    fn key(key: egui::Key) -> egui::Event {
        egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }
    }

    fn typed(text: &str) -> egui::Event {
        egui::Event::Text(text.into())
    }

    fn backspaces(n: usize) -> Vec<egui::Event> {
        (0..n).map(|_| key(egui::Key::Backspace)).collect()
    }

    /// The text of every shape in `shapes`, nested ones included.
    fn shape_texts(shapes: &[egui::epaint::ClippedShape]) -> Vec<String> {
        fn walk(shape: &egui::Shape, out: &mut Vec<String>) {
            match shape {
                egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| walk(s, out)),
                egui::Shape::Text(text) => out.push(text.galley.text().to_owned()),
                _ => {}
            }
        }
        let mut out = Vec::new();
        for clipped in shapes {
            walk(&clipped.shape, &mut out);
        }
        out
    }

    /// The sequence editor on a headless context, laid out the way the
    /// `host_window` example lays it out: the editor in a left panel, the
    /// active instrument's canvas in the central panel.
    struct Driver {
        ctx: egui::Context,
        recipe: SequenceRecipe,
        state: SequenceEditorState,
        out: egui::FullOutput,
    }

    impl Driver {
        fn new(recipe: SequenceRecipe) -> Self {
            Self::with_state(recipe, SequenceEditorState::default())
        }

        /// With [`distinct_style`] set, so each role the editor paints has
        /// a colour nothing else does.
        fn with_state(recipe: SequenceRecipe, state: SequenceEditorState) -> Self {
            let ctx = egui::Context::default();
            set_editor_style(&ctx, distinct_style());
            Self::on(ctx, recipe, state)
        }

        /// Under `visuals`, with no style set.
        fn themed(
            recipe: SequenceRecipe,
            state: SequenceEditorState,
            visuals: egui::Visuals,
        ) -> Self {
            let ctx = egui::Context::default();
            ctx.set_visuals(visuals);
            Self::on(ctx, recipe, state)
        }

        fn on(ctx: egui::Context, recipe: SequenceRecipe, state: SequenceEditorState) -> Self {
            ctx.enable_accesskit();
            let mut driver = Self {
                ctx,
                recipe,
                state,
                out: egui::FullOutput::default(),
            };
            // A new area's first pass is an invisible sizing pass; act from
            // the third frame on.
            driver.frame(Vec::new());
            driver.frame(Vec::new());
            driver
        }

        /// One frame with `events` as its input; the editor's response.
        fn frame(&mut self, events: Vec<egui::Event>) -> EditorResponse {
            let Self {
                ctx,
                recipe,
                state,
                out,
            } = self;
            let mut res = EditorResponse::NONE;
            let input = egui::RawInput {
                events,
                ..Default::default()
            };
            *out = ctx.run_ui(input, |root| {
                egui::Panel::left("seq")
                    .default_size(480.0)
                    .show(root, |ui| {
                        res.merge(sequence_recipe_editor(ui, recipe, state, Id::new("seq")));
                    });
                egui::CentralPanel::default().show(root, |ui| {
                    res.merge(active_instrument_canvas(
                        ui,
                        recipe,
                        state,
                        Id::new("seq_canvas"),
                    ));
                });
            });
            res
        }

        fn nodes(&self) -> impl Iterator<Item = (accesskit::NodeId, &accesskit::Node)> {
            self.out
                .platform_output
                .accesskit_update
                .iter()
                .flat_map(|update| update.nodes.iter().map(|(id, node)| (*id, node)))
        }

        /// The values of every single-line text field, top to bottom.
        fn fields(&self) -> Vec<String> {
            let mut fields: Vec<(f64, String)> = self
                .nodes()
                .filter(|(_, n)| n.role() == accesskit::Role::TextInput)
                .map(|(_, n)| {
                    let top = n.bounds().map_or(f64::MAX, |b| b.y0);
                    (top, n.value().unwrap_or_default().to_owned())
                })
                .collect();
            fields.sort_by(|a, b| a.0.total_cmp(&b.0));
            fields.into_iter().map(|(_, v)| v).collect()
        }

        /// The topmost single-line text field showing exactly `text`.
        fn field(&self, text: &str) -> Option<accesskit::NodeId> {
            self.nodes()
                .filter(|(_, n)| n.role() == accesskit::Role::TextInput && n.value() == Some(text))
                .min_by(|a, b| {
                    let top = |n: &accesskit::Node| n.bounds().map_or(f64::MAX, |b| b.y0);
                    top(a.1).total_cmp(&top(b.1))
                })
                .map(|(id, _)| id)
        }

        /// The topmost button whose label starts with `prefix`.
        fn button(&self, prefix: &str) -> Option<accesskit::NodeId> {
            self.nodes()
                .filter(|(_, n)| {
                    n.role() == accesskit::Role::Button
                        && n.label().is_some_and(|l| l.starts_with(prefix))
                })
                .min_by(|a, b| {
                    let top = |n: &accesskit::Node| n.bounds().map_or(f64::MAX, |b| b.y0);
                    top(a.1).total_cmp(&top(b.1))
                })
                .map(|(id, _)| id)
        }

        fn act(&mut self, target: accesskit::NodeId, action: accesskit::Action) -> EditorResponse {
            self.frame(vec![egui::Event::AccessKitActionRequest(
                accesskit::ActionRequest {
                    action,
                    target_tree: accesskit::TreeId::ROOT,
                    target_node: target,
                    data: None,
                },
            )])
        }

        /// Focus the topmost text field showing `text`, as a click into it
        /// does; the cursor lands at the end.
        fn focus(&mut self, text: &str) {
            let field = self
                .field(text)
                .unwrap_or_else(|| panic!("no field shows {text:?}; fields: {:?}", self.fields()));
            self.act(field, accesskit::Action::Focus);
            assert!(
                self.ctx.memory(|m| m.focused()).is_some(),
                "the field showing {text:?} did not take the focus"
            );
        }

        /// Click the first button whose label starts with `prefix`.
        fn click(&mut self, prefix: &str) -> EditorResponse {
            let button = self
                .button(prefix)
                .unwrap_or_else(|| panic!("no button labelled {prefix:?}\u{2026}"));
            self.act(button, accesskit::Action::Click)
        }

        /// The text of everything the last frame painted.
        fn texts(&self) -> Vec<String> {
            shape_texts(&self.out.shapes)
        }

        fn paints(&self, needle: &str) -> bool {
            self.texts().iter().any(|t| t.contains(needle))
        }
    }

    #[test]
    fn renaming_an_instrument_moves_its_notes_on_every_track_and_its_canvas_layout() {
        let mut state = SequenceEditorState::default();
        state.set_active_instrument(Some(0));
        let mut driver = Driver::with_state(two_instrument_recipe(), state);
        // The canvas laid inst1's nodes out while it was open; close it so
        // nothing redraws that layout under the rename.
        let layout = format!("{:?}", driver.state.canvas_states["inst1"]);
        driver.state.set_active_instrument(None);
        driver.frame(Vec::new());

        driver.focus("inst1");
        driver.frame(backspaces(5));
        driver.frame(vec![typed("lead")]);
        let res = driver.frame(vec![key(egui::Key::Enter)]);

        assert_eq!(instrument_ids(&driver.recipe), ["lead", "inst2"]);
        assert_eq!(
            note_ids(&driver.recipe),
            [vec!["lead", "inst2"], vec!["lead", "lead"], vec!["inst2"]],
            "every note of the renamed instrument follows it, on every track"
        );
        assert!(res.changed && res.rebake, "the rename commits: {res:?}");
        let keys: Vec<&String> = driver.state.canvas_states.keys().collect();
        assert_eq!(keys, ["lead"], "the canvas layout is re-keyed, not dropped");
        assert_eq!(format!("{:?}", driver.state.canvas_states["lead"]), layout);
    }

    #[test]
    fn a_rename_commits_when_the_field_loses_focus() {
        let mut driver = Driver::new(two_instrument_recipe());
        driver.focus("inst1");
        driver.frame(vec![typed("x")]);
        // Tab moves the focus on at the end of its frame, so the field sees
        // the loss on the next one.
        driver.frame(vec![key(egui::Key::Tab)]);
        driver.frame(Vec::new());

        assert_eq!(instrument_ids(&driver.recipe), ["inst1x", "inst2"]);
        assert_eq!(
            note_ids(&driver.recipe),
            [
                vec!["inst1x", "inst2"],
                vec!["inst1x", "inst1x"],
                vec!["inst2"]
            ]
        );
    }

    #[test]
    fn empty_duplicate_and_over_long_names_are_refused_with_a_reason() {
        // "é" is two bytes: 62 of them after "inst1" is 129 bytes in only 67
        // characters, so the limit is counted in bytes, as the sanitiser
        // counts it.
        let over_long = format!("inst1{}", "\u{E9}".repeat(62));
        let cases: [(&str, Vec<Vec<egui::Event>>, &str, &str); 4] = [
            ("empty", vec![backspaces(5)], "", "empty"),
            (
                "blank",
                vec![backspaces(5), vec![typed("   ")]],
                "   ",
                "empty",
            ),
            (
                "duplicate",
                vec![backspaces(1), vec![typed("2")]],
                "inst2",
                "Another instrument is called 'inst2'",
            ),
            (
                "over-long",
                vec![vec![typed(&"\u{E9}".repeat(62))]],
                &over_long,
                "the limit is 128",
            ),
        ];
        for (case, keystrokes, shown, reason) in cases {
            let mut driver = Driver::new(two_instrument_recipe());
            let before = driver.recipe.clone();
            driver.focus("inst1");
            for events in keystrokes {
                driver.frame(events);
            }
            assert!(
                driver.paints(reason),
                "{case}: the reason shows while typing"
            );
            let res = driver.frame(vec![key(egui::Key::Enter)]);
            driver.frame(Vec::new());

            assert_eq!(
                driver.recipe, before,
                "{case}: a refused name changes nothing"
            );
            assert_eq!(res, EditorResponse::NONE, "{case}: nothing to write back");
            assert!(
                driver.paints(reason),
                "{case}: the reason stays after the refusal; painted: {:?}",
                driver.texts()
            );
            assert_eq!(
                driver.fields()[0],
                shown,
                "{case}: the field keeps what was typed, to fix or to Esc"
            );

            // Esc puts the name back and takes the reason away.
            driver.focus(shown);
            driver.frame(vec![key(egui::Key::Escape)]);
            driver.frame(Vec::new());
            assert_eq!(driver.fields()[0], "inst1", "{case}: Esc restores the name");
            assert!(!driver.paints(reason), "{case}: Esc clears the reason");
            assert_eq!(driver.recipe, before, "{case}: Esc changes nothing");
        }
    }

    #[test]
    fn typing_through_a_colliding_name_moves_no_notes() {
        // "inst1" -> "inst" -> "inst2" -> "inst23", with inst2 in the recipe.
        // Were each keystroke applied, "inst2" would merge inst1's notes into
        // inst2's and nothing could tell them apart again.
        let mut driver = Driver::new(two_instrument_recipe());
        let before = driver.recipe.clone();
        driver.focus("inst1");
        for step in [backspaces(1), vec![typed("2")], vec![typed("3")]] {
            let res = driver.frame(step);
            assert_eq!(driver.recipe, before, "nothing moves while typing");
            assert!(!res.changed, "typing is not an edit of the recipe");
        }
        driver.frame(vec![key(egui::Key::Enter)]);

        assert_eq!(instrument_ids(&driver.recipe), ["inst23", "inst2"]);
        assert_eq!(
            note_ids(&driver.recipe),
            [
                vec!["inst23", "inst2"],
                vec!["inst23", "inst23"],
                vec!["inst2"]
            ],
            "only inst1's notes follow it; inst2 keeps its own"
        );
    }

    #[test]
    fn esc_restores_the_name_and_applies_nothing() {
        let mut driver = Driver::new(two_instrument_recipe());
        let before = driver.recipe.clone();
        driver.focus("inst1");
        driver.frame(vec![typed("zzz")]);
        assert_eq!(driver.recipe, before);
        driver.frame(vec![key(egui::Key::Escape)]);
        for _ in 0..3 {
            driver.frame(Vec::new());
        }
        assert_eq!(driver.recipe, before, "nothing is applied later either");
        assert_eq!(driver.fields()[..2], ["inst1", "inst2"]);
    }

    /// One sine at `freq_hz`: an instrument that is audible in a bake.
    fn sine(freq_hz: f32) -> AudioPatch {
        use crate::{GraphNode, NodeGraph, NodeId, NodeKind, SineOsc};
        AudioPatch {
            seed: 0,
            graph: NodeGraph {
                nodes: vec![GraphNode {
                    id: NodeId(0),
                    kind: NodeKind::Sine(SineOsc {
                        freq_hz,
                        phase_offset: 0.0,
                        amplitude: 0.5,
                    }),
                    inputs: Default::default(),
                }],
                output: NodeId(0),
            },
        }
    }

    #[test]
    fn a_rename_is_a_pure_relabel_of_the_mixdown() {
        use crate::mixdown::bake_sequence;
        let mut recipe = two_instrument_recipe();
        recipe.instruments[0].patch = sine(330.0);
        recipe.instruments[1].patch = sine(495.0);
        recipe.sample_rate = 8_000;
        recipe.bpm = 120.0;
        recipe.duration_beats = 4.0;
        let bits = |samples: Vec<f32>| samples.into_iter().map(f32::to_bits).collect::<Vec<_>>();
        let before = bits(bake_sequence(&recipe));

        // The control: a rename that leaves the notes behind silences them,
        // so this bake can see a rename that is not a pure relabel.
        let mut in_place = recipe.clone();
        in_place.instruments[0].id = "lead".into();
        assert_ne!(bits(bake_sequence(&in_place)), before);

        let mut driver = Driver::new(recipe);
        driver.focus("inst1");
        driver.frame(backspaces(5));
        driver.frame(vec![typed("lead")]);
        driver.frame(vec![key(egui::Key::Enter)]);
        assert_eq!(instrument_ids(&driver.recipe), ["lead", "inst2"]);
        assert_eq!(
            bits(bake_sequence(&driver.recipe)),
            before,
            "the renamed recipe bakes sample for sample the same"
        );
    }

    /// `wind` and `kick`, with notes of `ghost` on two tracks: an instrument
    /// that was renamed or removed while notes still used it.
    fn ghost_recipe() -> SequenceRecipe {
        let mut recipe = recipe_with(&["wind", "kick"], 2);
        recipe.tracks[0].events = vec![note("ghost", 0.0), note("wind", 1.0)];
        recipe.tracks[1].events = vec![note("ghost", 2.0)];
        recipe
    }

    #[test]
    fn a_note_naming_no_instrument_is_marked_and_offers_a_reassign() {
        let mut state = SequenceEditorState::default();
        state.set_selected_event(Some((0, 0)));
        let mut driver = Driver::with_state(ghost_recipe(), state);

        assert!(
            driver.paints("missing: ghost"),
            "the block says whose note it is; painted: {:?}",
            driver.texts()
        );
        assert!(
            !driver.paints("missing: wind"),
            "a note whose instrument exists is not marked"
        );
        assert!(driver.paints("No instrument called 'ghost'"));
        assert!(driver.paints("Reassign all notes of 'ghost' to"));

        driver.click("Reassign all notes of 'ghost' to");
        // The menu is a new area: its first frame is a sizing pass.
        driver.frame(Vec::new());
        driver.frame(Vec::new());
        let res = driver.click("kick");
        assert_eq!(
            note_ids(&driver.recipe),
            [vec!["kick", "wind"], vec!["kick"]],
            "every ghost note, on every track, and nothing else"
        );
        assert!(res.changed && res.rebake, "{res:?}");
        driver.frame(Vec::new());
        assert!(!driver.paints("missing:"), "no note is missing any more");
    }

    #[test]
    fn missing_instrument_notes_render_headless_without_panicking() {
        // No instruments at all, a note with an empty id and one with an id
        // nothing answers to, each open in the inspector in turn.
        let mut recipe = recipe_with(&[], 1);
        recipe.tracks[0].events = vec![note("", 0.0), note("ghost", 1.0)];
        for selected in [(0, 0), (0, 1)] {
            let mut state = SequenceEditorState::default();
            state.set_selected_event(Some(selected));
            let mut driver = Driver::with_state(recipe.clone(), state);
            driver.frame(Vec::new());
            assert!(driver.paints("missing:"), "{selected:?}");
        }
    }

    #[test]
    fn a_refused_name_moves_with_its_row_and_never_applies_itself() {
        let mut driver = Driver::new(two_instrument_recipe());
        // inst2 typed to "inst1": refused while inst1 exists.
        driver.focus("inst2");
        driver.frame(backspaces(1));
        driver.frame(vec![typed("1")]);
        driver.frame(vec![key(egui::Key::Enter)]);
        assert!(driver.paints("Another instrument is called 'inst1'"));

        // Removing inst1 frees the name and moves inst2's row up; the typed
        // name moves with it but waits for the user.
        driver.click("\u{2716}");
        for _ in 0..3 {
            driver.frame(Vec::new());
        }
        assert_eq!(instrument_ids(&driver.recipe), ["inst2"]);
        assert_eq!(driver.fields()[0], "inst1", "the row keeps what was typed");
        assert!(driver.paints("Not applied yet"));
        assert!(
            driver.paints("missing: inst1"),
            "the removed instrument's notes are marked"
        );

        // Enter in the field applies it: inst2's notes follow, and the notes
        // the removal left behind find an instrument by that name again.
        driver.focus("inst1");
        driver.frame(vec![key(egui::Key::Enter)]);
        driver.frame(Vec::new());
        assert_eq!(instrument_ids(&driver.recipe), ["inst1"]);
        assert_eq!(
            note_ids(&driver.recipe),
            [
                vec!["inst1", "inst1"],
                vec!["inst1", "inst1"],
                vec!["inst1"]
            ]
        );
        assert!(!driver.paints("missing:"));
        assert!(!driver.paints("Not applied yet"));
    }

    #[test]
    fn names_are_trimmed_and_limited_to_symbios_audios_byte_bound() {
        assert_eq!(
            instrument_id_byte_limit(),
            128,
            "the bound Overlands' sanitiser pins"
        );
        let recipe = recipe_with(&["inst1", "inst2"], 0);
        let at_limit = "a".repeat(128);
        let padded = format!("  {at_limit}\t");
        assert_eq!(
            check_name(&recipe, 0, &at_limit, 128),
            Ok(at_limit.as_str())
        );
        assert_eq!(check_name(&recipe, 0, &padded, 128), Ok(at_limit.as_str()));
        assert_eq!(
            check_name(&recipe, 0, &"a".repeat(129), 128),
            Err(NameRefusal::TooLong {
                bytes: 129,
                limit: 128
            })
        );
        // A row's own name is not a collision with itself.
        assert_eq!(check_name(&recipe, 0, "inst1", 128), Ok("inst1"));
        assert_eq!(
            check_name(&recipe, 0, " inst2 ", 128),
            Err(NameRefusal::Taken("inst2".into()))
        );
    }

    #[test]
    fn a_rename_in_a_recipe_with_a_duplicate_id_leaves_the_notes_with_the_other() {
        // Only a recipe from outside the editor has two instruments with one
        // id. Which of them a note meant cannot be known, so the notes stay
        // with the name, which the other still has.
        let mut recipe = recipe_with(&["a", "a"], 1);
        recipe.tracks[0].events = vec![note("a", 0.0)];
        let mut layouts = HashMap::from([("a".to_owned(), PatchEditorState::default())]);
        rename_instrument(&mut recipe, &mut layouts, 0, "b");
        assert_eq!(instrument_ids(&recipe), ["b", "a"]);
        assert_eq!(note_ids(&recipe), [vec!["a"]]);
        assert!(layouts.contains_key("a") && !layouts.contains_key("b"));
    }

    #[test]
    fn a_rename_drops_a_layout_a_removed_instrument_left_under_the_new_name() {
        let mut recipe = recipe_with(&["inst1"], 0);
        let mut layouts = HashMap::from([("gone".to_owned(), PatchEditorState::default())]);
        rename_instrument(&mut recipe, &mut layouts, 0, "gone");
        assert!(
            layouts.is_empty(),
            "a never-opened instrument starts from a fresh layout, not a stranger's"
        );
    }

    // -----------------------------------------------------------------------
    // Colours from the editor style, words on the buttons (#58, Overlands
    // #1331 E1 E2)
    // -----------------------------------------------------------------------

    /// `wind` plays one long note with a release tail on lane 0, selected;
    /// `kick` one short note on lane 1. The recipe loops from beat 2 with a
    /// one-beat crossfade.
    fn two_lane_recipe() -> (SequenceRecipe, SequenceEditorState) {
        let mut recipe = recipe_with(&["wind", "kick"], 2);
        recipe.tracks[0].events = vec![Event {
            gate_beats: 4.0,
            release_beats: 1.0,
            ..note("wind", 0.0)
        }];
        recipe.tracks[1].events = vec![note("kick", 1.0)];
        recipe.loop_start_beats = Some(2.0);
        recipe.loop_crossfade_beats = 1.0;
        let mut state = SequenceEditorState::default();
        state.set_selected_event(Some((0, 0)));
        (recipe, state)
    }

    /// The block a note label is painted on: the last opaque fill of a
    /// note's height under the label's centre. An outline is painted as a
    /// rect too, with a transparent fill, after the block it outlines.
    fn block_under(out: &egui::FullOutput, label: &egui::Rect) -> Option<egui::epaint::RectShape> {
        let block_h = LANE_H - 8.0;
        shapes(out)
            .into_iter()
            .filter_map(|s| match s {
                egui::Shape::Rect(r)
                    if (r.rect.height() - block_h).abs() < 1.0
                        && r.fill.is_opaque()
                        && r.rect.contains(label.center()) =>
                {
                    Some(r)
                }
                _ => None,
            })
            .last()
    }

    /// Every note label reading `name` on the timeline, with its colour and
    /// the fill of the block under it.
    fn note_labels(out: &egui::FullOutput, name: &str) -> Vec<(Color32, Color32)> {
        shapes(out)
            .into_iter()
            .filter_map(|s| match s {
                egui::Shape::Text(t) if t.galley.text() == name => {
                    let rect = t.visual_bounding_rect();
                    let block = block_under(out, &rect)?;
                    Some((crate::ui::test_paint::text_colour(&t), block.fill))
                }
                _ => None,
            })
            .collect()
    }

    /// The acceptance of #1331 for notes: in egui's dark and light themes, a
    /// note's name reads on its block, selected or not.
    #[test]
    fn a_notes_name_reads_on_its_block_in_dark_and_light() {
        for (theme, visuals) in [
            ("dark", egui::Visuals::dark()),
            ("light", egui::Visuals::light()),
        ] {
            let (recipe, state) = two_lane_recipe();
            let driver = Driver::themed(recipe, state, visuals);
            for name in ["wind", "kick"] {
                let found = note_labels(&driver.out, name);
                assert_eq!(found.len(), 1, "{theme}: one {name:?} note label");
                let (text, fill) = found[0];
                let ratio = contrast_on(text, fill);
                assert!(
                    ratio >= AA,
                    "{theme}: {name:?} is {ratio:.2}:1 on its block ({text:?} on {fill:?})"
                );
            }
        }
    }

    /// A style the host set is what the timeline paints, role by role.
    #[test]
    fn a_set_style_is_what_the_timeline_paints() {
        let s = distinct_style();
        let (recipe, state) = two_lane_recipe();
        let driver = Driver::with_state(recipe, state);
        let painted = colours(&driver.out);
        for (role, colour) in [
            ("timeline_ground", s.timeline_ground),
            ("timeline_grid", s.timeline_grid),
            ("ground_text", s.ground_text),
            ("lane", s.lane),
            ("lane_alt", s.lane_alt),
            ("loop_start", s.loop_start),
            ("loop_end", s.loop_end),
            ("crossfade_band", s.crossfade_band),
            ("note_fill", s.note_fill),
            ("note_selected", s.note_selected),
            ("release_tail", s.release_tail),
        ] {
            assert!(painted.contains(&colour), "{role} is not painted");
        }
        for name in ["wind", "kick"] {
            assert_eq!(
                note_labels(&driver.out, name),
                [(s.note_text, s.note_fill)],
                "{name}: the label in note_text on a note_fill block"
            );
        }
    }

    /// A note with no instrument and a refused name are in the style's
    /// error colour.
    #[test]
    fn missing_notes_and_refused_names_are_in_the_styles_error_colour() {
        let s = distinct_style();
        let mut state = SequenceEditorState::default();
        state.set_selected_event(Some((0, 0)));
        let mut driver = Driver::with_state(ghost_recipe(), state);
        assert!(colours(&driver.out).contains(&s.error), "the ghost note");
        let (_, missing) = crate::ui::test_paint::text_painted(&driver.out, "missing: ghost")
            .expect("the ghost note's label");
        assert_eq!(missing, s.error);

        driver.focus("wind");
        driver.frame(backspaces(4));
        driver.frame(vec![typed("kick")]);
        let (_, refusal) =
            crate::ui::test_paint::text_painted(&driver.out, "Another instrument is called 'kick'")
                .expect("the reason");
        assert_eq!(refusal, s.error);
    }

    /// E2: the sequence editor's buttons say what they do in words. The
    /// remove cross stays (Overlands' `affordances::CROSS`), and so does the
    /// pencil of the instrument's Edit toggle, which Overlands draws for the
    /// same act ("✏ Edit audio…").
    #[test]
    fn the_sequence_buttons_say_what_they_do_in_words() {
        let (recipe, state) = two_lane_recipe();
        let driver = Driver::with_state(recipe, state);
        let labels = button_labels(&driver.out);
        for word in [
            "Add instrument",
            "Add track",
            "Mutate recipe",
            "Delete note",
            "\u{270F} Edit",
        ] {
            assert!(
                labels.iter().any(|l| l == word),
                "no {word:?} button; buttons: {labels:?}"
            );
        }
        assert_eq!(
            glyph_labels(&labels, &['\u{2716}', '\u{270F}']),
            Vec::<String>::new(),
            "buttons labelled with a glyph where the vocabulary says a word"
        );
    }

    // ---- step 7a (#60, Overlands #1333): one history per recipe ---------

    /// An instrument's patch lives inside the recipe, so an edit to it in
    /// the embedded canvas is an edit to the recipe — and the recipe's
    /// history is the one that takes the step. Were the canvas's own
    /// history on as well, one Ctrl+Z would walk two stacks at once.
    #[test]
    fn the_sequence_editor_has_exactly_one_history() {
        let state = SequenceEditorState {
            active_instrument: Some(0),
            ..Default::default()
        };
        let mut driver = Driver::with_state(two_instrument_recipe(), state);
        let before = driver.recipe.clone();
        assert!(!driver.state.can_undo(), "nothing edited yet");

        // Add a node to the open instrument's patch, through its canvas.
        driver.click("Add node");
        assert_ne!(driver.recipe, before, "the canvas edited the recipe");

        // The embedded canvas keeps no history of its own.
        let canvas = driver
            .state
            .canvas_states
            .values()
            .next()
            .expect("the open instrument has a canvas state");
        assert!(
            !canvas.history_is_enabled(),
            "the embedded patch canvas is keeping a second history of the recipe"
        );
        assert!(!canvas.can_undo(), "and it offers no undo of its own");

        // The recipe's history has it, and one undo takes it all back.
        assert!(driver.state.can_undo());
        assert!(driver.state.undo(&mut driver.recipe));
        assert_eq!(driver.recipe, before);
        assert!(driver.state.redo(&mut driver.recipe));
        assert_ne!(driver.recipe, before);
    }

    /// An edit made in the recipe's own widgets goes on the same one
    /// history as an edit made in the embedded canvas.
    #[test]
    fn the_recipes_own_edits_and_its_patches_share_the_one_history() {
        let state = SequenceEditorState {
            active_instrument: Some(0),
            ..Default::default()
        };
        let mut driver = Driver::with_state(two_instrument_recipe(), state);
        let start = driver.recipe.clone();

        driver.click("Add track");
        let after_track = driver.recipe.clone();
        assert_ne!(after_track, start);

        driver.click("Add node");
        let after_node = driver.recipe.clone();
        assert_ne!(after_node, after_track);

        assert!(driver.state.undo(&mut driver.recipe));
        assert_eq!(driver.recipe, after_track, "the node edit came off first");
        assert!(driver.state.undo(&mut driver.recipe));
        assert_eq!(driver.recipe, start, "then the track edit");
        assert!(!driver.state.can_undo());
    }
}
