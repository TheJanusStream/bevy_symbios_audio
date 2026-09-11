//! What an editor painted, read back from egui's output: the helpers the
//! colour and label tests share (#58).
//!
//! The tests read the shapes a frame produced, not what the code meant to
//! paint, so a colour that bypasses the editor style is caught wherever it
//! is chosen.

use bevy_egui::egui::{self, Color32, accesskit};

/// Every shape `out` painted, nested ones flattened out.
pub(crate) fn shapes(out: &egui::FullOutput) -> Vec<egui::Shape> {
    fn walk(shape: &egui::Shape, into: &mut Vec<egui::Shape>) {
        match shape {
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| walk(s, into)),
            other => into.push(other.clone()),
        }
    }
    let mut flat = Vec::new();
    for clipped in &out.shapes {
        walk(&clipped.shape, &mut flat);
    }
    flat
}

/// The colour a text shape's glyphs are drawn in: an override, else the
/// first section's own colour, else the fallback the painter was given.
pub(crate) fn text_colour(text: &egui::epaint::TextShape) -> Color32 {
    if let Some(colour) = text.override_text_color {
        return colour;
    }
    match text.galley.job.sections.first() {
        Some(section) if section.format.color != Color32::PLACEHOLDER => section.format.color,
        _ => text.fallback_color,
    }
}

/// The text shape reading exactly `text`, and the colour it is drawn in.
pub(crate) fn text_painted(out: &egui::FullOutput, text: &str) -> Option<(egui::Rect, Color32)> {
    shapes(out).into_iter().find_map(|shape| match shape {
        egui::Shape::Text(t) if t.galley.text() == text => {
            Some((t.visual_bounding_rect(), text_colour(&t)))
        }
        _ => None,
    })
}

/// Every colour `out` painted: fills, strokes, lines and text.
pub(crate) fn colours(out: &egui::FullOutput) -> Vec<Color32> {
    let mut found = Vec::new();
    for shape in shapes(out) {
        match shape {
            egui::Shape::Rect(r) => found.extend([r.fill, r.stroke.color]),
            egui::Shape::Circle(c) => found.extend([c.fill, c.stroke.color]),
            egui::Shape::LineSegment { stroke, .. } => found.push(stroke.color),
            egui::Shape::Path(p) => {
                found.push(p.fill);
                if let egui::epaint::ColorMode::Solid(colour) = p.stroke.color {
                    found.push(colour);
                }
            }
            egui::Shape::Text(t) => found.push(text_colour(&t)),
            _ => {}
        }
    }
    found
}

/// The label of every button in `out`'s AccessKit tree.
pub(crate) fn button_labels(out: &egui::FullOutput) -> Vec<String> {
    out.platform_output
        .accesskit_update
        .iter()
        .flat_map(|update| &update.nodes)
        .filter(|(_, n)| n.role() == accesskit::Role::Button)
        .filter_map(|(_, n)| n.label().map(str::to_owned))
        .collect()
}

/// The labels among `labels` carrying a non-ASCII character that is not in
/// `allowed`: a glyph where the editors' vocabulary says a word.
pub(crate) fn glyph_labels(labels: &[String], allowed: &[char]) -> Vec<String> {
    labels
        .iter()
        .filter(|l| l.chars().any(|c| !c.is_ascii() && !allowed.contains(&c)))
        .cloned()
        .collect()
}

/// WCAG 2.1 contrast of two colours as the screen shows them: a
/// translucent `fg` is composited over the opaque `bg` first.
pub(crate) fn contrast_on(fg: Color32, bg: Color32) -> f32 {
    super::style::contrast_ratio(bg.blend(fg), bg)
}
