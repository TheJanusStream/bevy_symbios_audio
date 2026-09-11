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
///   `rebake`, on failure it shows the parse error inline, in the
///   [`EditorStyle`](super::EditorStyle)'s error colour.
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
                if ui.button("Copy JSON").clicked()
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
                ui.colored_label(super::style::editor_style(ui).error, err);
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

    /// The JSON section, open, with `text` pasted in: one frame per call,
    /// `events` its input; what it painted.
    struct Section {
        ctx: egui::Context,
        value: AudioPatch,
        state: JsonIoState,
        out: egui::FullOutput,
    }

    impl Section {
        fn open_with(text: &str) -> Self {
            use crate::ui::style::set_editor_style;
            use crate::ui::style::tests::distinct_style;
            let ctx = egui::Context::default();
            ctx.enable_accesskit();
            set_editor_style(&ctx, distinct_style());
            let mut section = Self {
                ctx,
                value: AudioPatch::default(),
                state: JsonIoState {
                    buffer: text.to_owned(),
                    error: None,
                },
                out: egui::FullOutput::default(),
            };
            section.frame(Vec::new());
            section.click("Import / Export JSON");
            section.settle();
            section
        }

        /// Quiet frames until the header's opening animation is over: while
        /// it runs, the section is clipped and its lower widgets unpainted.
        fn settle(&mut self) {
            for _ in 0..12 {
                self.frame(Vec::new());
            }
        }

        fn frame(&mut self, events: Vec<egui::Event>) {
            let Self {
                ctx,
                value,
                state,
                out,
            } = self;
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(900.0, 600.0),
                )),
                events,
                ..Default::default()
            };
            *out = ctx.run_ui(input, |root| {
                egui::CentralPanel::default().show(root, |ui| {
                    json_io(ui, value, state, egui::Id::new("json"));
                });
            });
        }

        fn click(&mut self, label: &str) {
            use egui::accesskit;
            let target = self
                .out
                .platform_output
                .accesskit_update
                .iter()
                .flat_map(|update| &update.nodes)
                .find(|(_, n)| n.label() == Some(label))
                .map(|(id, _)| *id)
                .unwrap_or_else(|| panic!("no widget labelled {label:?}"));
            self.frame(vec![egui::Event::AccessKitActionRequest(
                accesskit::ActionRequest {
                    action: accesskit::Action::Click,
                    target_tree: accesskit::TreeId::ROOT,
                    target_node: target,
                    data: None,
                },
            )]);
        }
    }

    /// A paste that does not parse says why, in the style's error colour.
    #[test]
    fn a_parse_error_is_shown_in_the_styles_error_colour() {
        use crate::ui::style::tests::distinct_style;
        use crate::ui::test_paint::shapes;
        let mut section = Section::open_with("{ not json");
        section.click("Apply");
        section.settle();
        let error = section
            .state
            .error
            .clone()
            .expect("the paste did not parse");
        let painted: Vec<(String, egui::Color32)> = shapes(&section.out)
            .into_iter()
            .filter_map(|s| match s {
                egui::Shape::Text(t) => Some((
                    t.galley.text().to_owned(),
                    crate::ui::test_paint::text_colour(&t),
                )),
                _ => None,
            })
            .collect();
        let colour = painted.iter().find(|(t, _)| *t == error).map(|(_, c)| *c);
        assert_eq!(
            colour,
            Some(distinct_style().error),
            "the error {error:?}; painted {painted:?}"
        );
    }

    /// E2: the section's buttons are words.
    #[test]
    fn the_json_buttons_say_what_they_do_in_words() {
        use crate::ui::test_paint::{button_labels, glyph_labels};
        let section = Section::open_with("");
        let labels = button_labels(&section.out);
        for word in ["Copy JSON", "Load current", "Apply"] {
            assert!(labels.iter().any(|l| l == word), "no {word:?}: {labels:?}");
        }
        assert_eq!(glyph_labels(&labels, &[]), Vec::<String>::new());
    }

    #[test]
    fn json_io_renders_headless_without_panicking() {
        let ctx = egui::Context::default();
        let mut value = AudioPatch::default();
        let mut state = JsonIoState::default();
        let _ = ctx.run_ui(egui::RawInput::default(), |root| {
            egui::CentralPanel::default().show(root, |ui| {
                json_io(ui, &mut value, &mut state, egui::Id::new("json"));
            });
        });
    }
}
