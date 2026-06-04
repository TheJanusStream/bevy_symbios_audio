//! JSON import / export for editor values (patches and recipes).
//!
//! A small reusable section that copies the current value to the clipboard as
//! pretty JSON and parses pasted JSON back in.  Paste-and-Apply (rather than a
//! direct clipboard read) keeps it portable, including on wasm where clipboard
//! reads are restricted.

use bevy_egui::egui;
use serde::Serialize;
use serde::de::DeserializeOwned;

use super::EditorResponse;

/// Persistent state for a [`json_io`] section: the editable text buffer and the
/// last parse error.  Store one per value you expose (patch, recipe, …).
#[derive(Clone, Debug, Default)]
pub struct JsonIoState {
    buffer: String,
    error: Option<String>,
}

/// A collapsible "Import / Export JSON" section for any serde value.
///
/// - **Copy JSON** copies the current `value`, pretty-printed, to the clipboard.
/// - **Load current** fills the text box with the current value (a starting
///   point to tweak by hand).
/// - **Apply** parses the text box; on success it replaces `value` and sets
///   `rebake`, on failure it shows the parse error inline.
pub fn json_io<T: Serialize + DeserializeOwned>(
    ui: &mut egui::Ui,
    value: &mut T,
    state: &mut JsonIoState,
    id: egui::Id,
) -> EditorResponse {
    let mut res = EditorResponse::NONE;
    egui::CollapsingHeader::new("Import / Export JSON")
        .id_salt(id)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("\u{1F4CB} Copy JSON").clicked()
                    && let Ok(json) = serde_json::to_string_pretty(value)
                {
                    ui.ctx().copy_text(json);
                }
                if ui
                    .button("Load current")
                    .on_hover_text("Fill the box with the current value")
                    .clicked()
                    && let Ok(json) = serde_json::to_string_pretty(value)
                {
                    state.buffer = json;
                    state.error = None;
                }
            });

            ui.add(
                egui::TextEdit::multiline(&mut state.buffer)
                    .desired_rows(6)
                    .code_editor()
                    .desired_width(f32::INFINITY)
                    .hint_text("Paste JSON here, then Apply"),
            );

            if ui.button("Apply").clicked() {
                match serde_json::from_str::<T>(&state.buffer) {
                    Ok(parsed) => {
                        *value = parsed;
                        state.error = None;
                        res.changed = true;
                        res.rebake = true;
                    }
                    Err(e) => state.error = Some(e.to_string()),
                }
            }
            if let Some(err) = &state.error {
                ui.colored_label(egui::Color32::from_rgb(220, 120, 120), err);
            }
        });
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch::AudioPatch;

    #[test]
    fn apply_parses_a_round_tripped_patch() {
        // Mirrors what the Apply button does: serialize → buffer → parse back.
        let original = AudioPatch::default();
        let json = serde_json::to_string_pretty(&original).unwrap();
        let parsed: AudioPatch = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn json_io_renders_headless_without_panicking() {
        let ctx = egui::Context::default();
        let mut value = AudioPatch::default();
        let mut state = JsonIoState::default();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                json_io(ui, &mut value, &mut state, egui::Id::new("json"));
            });
        });
    }
}
