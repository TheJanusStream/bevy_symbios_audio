//! The editors' colours: [`EditorStyle`], a set of named roles that every
//! widget in [`crate::ui`] paints with (#58, Overlands #1331).
//!
//! # Why `Visuals` alone is not enough
//!
//! egui's `Visuals` names a window's colours: its text tiers, its fills, a
//! selection, a warning and an error. The editors paint more than a window
//! does: a canvas ground, node boxes, wires and ports, a timeline's lanes and
//! loop markers, note blocks, a waveform. Until 0.4.4 each of those was a
//! literal picked for egui's dark theme, so under a light theme the node
//! boxes stayed charcoal while the theme drew dark text on them, and the
//! node titles and port labels disappeared.
//!
//! # Where the style comes from
//!
//! [`EditorStyle::from_visuals`] derives every role from a `Visuals`, and the
//! editors use it whenever the host has set nothing. A host that only
//! switches egui's theme therefore gets editors that follow the switch.
//!
//! A host with its own palette builds a style, usually `from_visuals` with
//! the roles it wants to own overwritten, and passes it to
//! [`set_editor_style`] each time its theme changes. The style is kept in
//! the egui context's data, so every editor drawn in that context reads it
//! and no signature changes. A set style is not re-derived when the
//! `Visuals` change: set it again, or call [`clear_editor_style`] to go back
//! to following the `Visuals`.
//!
//! # What the defaults are held to
//!
//! Against egui's own dark and light `Visuals`, [`EditorStyle::from_visuals`]
//! holds every piece of text it paints to WCAG AA and every mark that is not
//! text to `NON_TEXT_FLOOR` — the two tests at the bottom of this module
//! carry the roster and the measurements. That is why two roles are derived
//! rather than taken from the `Visuals` as they stand: see `edge_for` and
//! `accent_for`, each of which records the numbers that made it necessary.
//! A host that sets its own roles owns that bar for them; Overlands holds
//! its three palettes to the same one.
//!
//! [`EditorStyle`] is `#[non_exhaustive]`: new roles may be added in a
//! compatible release, each with a `from_visuals` default, so a host builds
//! one from `from_visuals` and never with a struct literal.

use bevy_egui::egui::{self, Color32};

/// The success green on a dark ground. egui's `Visuals` has no success
/// colour, so [`EditorStyle::from_visuals`] picks between this and
/// [`OK_ON_LIGHT`], whichever reads better on the window. Both are the
/// Overlands palette's `status.ok` for its dark and light themes.
const OK_ON_DARK: Color32 = Color32::from_rgb(130, 200, 130);
/// The success green on a light ground; see [`OK_ON_DARK`].
const OK_ON_LIGHT: Color32 = Color32::from_rgb(30, 130, 50);
/// How far the canvas grid leans from the canvas ground toward the text
/// colour: visible as a ground, never a competitor to the wires on it.
const GRID_CONTRAST: f32 = 0.10;
/// How far the alternate lane stripe leans from the window fill toward the
/// text colour: enough to tell two lanes apart, in either direction.
const LANE_STRIPE: f32 = 0.06;
/// Opacity of the crossfade band over the lanes.
const CROSSFADE_ALPHA: f32 = 0.12;
/// Opacity of a note's release tail, a faint copy of its fill.
const TAIL_ALPHA: f32 = 0.35;
/// The contrast WCAG 1.4.11 asks of a mark that is not text: the boundary
/// of a component has to be found at a glance. Text is held to the higher
/// AA bar instead.
pub(crate) const NON_TEXT_FLOOR: f32 = 3.0;
/// Where `edge_for` starts looking, as a lean from the ground toward the
/// text colour, and how far it moves per try.
const EDGE_LEAN: f32 = 0.5;
/// See [`EDGE_LEAN`].
const EDGE_STEP: f32 = 0.05;
/// What `edge_for` aims for: a little over `NON_TEXT_FLOOR`, so the
/// edge that ships is over the bar rather than exactly on it.
const EDGE_TARGET: f32 = 3.3;

/// The editors' colour roles. See the [module docs](self).
///
/// Build one with [`EditorStyle::from_visuals`], change the roles you want,
/// and give it to [`set_editor_style`]. Text inside a node box, the
/// toolbars and the inspectors is drawn by ordinary egui widgets and follows
/// the host's `Visuals`; these roles are the colours the editors paint
/// themselves.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct EditorStyle {
    /// Behind the patch canvas's nodes.
    pub canvas_ground: Color32,
    /// The canvas's border, which says where the pannable ground ends.
    pub canvas_edge: Color32,
    /// The canvas's grid, which says that the ground moves under a drag and
    /// how far it has moved. Faint: the node boxes are the content.
    pub canvas_grid: Color32,
    /// A node box. Its widgets draw with the host's `Visuals`, so this has to
    /// be a surface the host's text reads on; the default is `window_fill`.
    pub node_fill: Color32,
    /// A node box's edge.
    pub node_stroke: Color32,
    /// A node's title ("#2  Lowpass").
    pub node_title: Color32,
    /// The selected node's edge.
    pub node_selected: Color32,
    /// The edge of the node the patch plays (the graph's output).
    pub node_output: Color32,
    /// A wire between two nodes.
    pub wire: Color32,
    /// The wire being dragged, and the port it would connect to.
    pub wire_active: Color32,
    /// The port dots on a node's edges.
    pub port: Color32,
    /// "Valid graph".
    pub ok: Color32,
    /// A warning: the audition strip's Muted chip.
    pub warn: Color32,
    /// An error: a broken graph's outline and reason, a refused name, a note
    /// with no instrument, a JSON parse failure, a failed bake.
    pub error: Color32,
    /// Behind the timeline's ruler and lanes.
    pub timeline_ground: Color32,
    /// The timeline's beat lines.
    pub timeline_grid: Color32,
    /// Text painted straight onto a ground: the ruler's beat numbers, the
    /// empty waveform's "no signal", a lane's remove cross at rest.
    pub ground_text: Color32,
    /// Even-numbered lanes.
    pub lane: Color32,
    /// Odd-numbered lanes.
    pub lane_alt: Color32,
    /// The marker where a loop starts.
    pub loop_start: Color32,
    /// The marker at the end of the sequence.
    pub loop_end: Color32,
    /// The shade over the crossfade before the end. Usually translucent.
    pub crossfade_band: Color32,
    /// A note block.
    pub note_fill: Color32,
    /// The selected note's outline. The fill stays [`Self::note_fill`], so
    /// its text reads the same selected or not.
    pub note_selected: Color32,
    /// The instrument name on a note block.
    pub note_text: Color32,
    /// A note's release tail after its block. Usually translucent.
    pub release_tail: Color32,
    /// Behind a waveform.
    pub waveform_ground: Color32,
    /// A waveform's zero line.
    pub waveform_zero: Color32,
    /// A waveform's trace.
    pub waveform_trace: Color32,
    /// The playhead: the line on a waveform and on the timeline that says
    /// where the monitor is in what it is playing (#65, Overlands #1338
    /// D1).
    ///
    /// Deliberately neither [`Self::loop_start`]'s accent nor
    /// [`Self::loop_end`]'s strong text: all three are vertical lines on
    /// the same timeline, and the two that do not move must not be
    /// mistaken for the one that does.
    pub playhead: Color32,
}

impl EditorStyle {
    /// Every role from `visuals`: the style the editors use when the host
    /// sets none.
    ///
    /// - Grounds (canvas, timeline, waveform) are `extreme_bg_color`, the
    ///   inset ground egui gives a text field; node boxes and lanes are
    ///   `window_fill`, where the theme's own text reads.
    /// - Text is the theme's: a node title in `strong_text_color`, text on
    ///   a ground in `text_color`, a note's name in the selection's text
    ///   colour on the selection fill.
    /// - A node's edge and the canvas border bound a component, so they
    ///   are `edge_for`: the ground leaned toward the text until they
    ///   clear the 3:1 WCAG 1.4.11 asks. The timeline's grid and the
    ///   waveform's zero line stay at the theme's separator colour — a
    ///   grid is a ground, not a boundary — and the canvas grid is its
    ///   ground leaned a tenth of the way toward the text.
    /// - The selected node, the dragged wire and the loop start are
    ///   `accent_for`: `hyperlink_color`, the theme's interactive
    ///   accent, or the selection's text colour where the link colour
    ///   cannot be found on the grounds it is drawn on. The output node
    ///   and the sequence end are `strong_text_color`.
    /// - `warn` and `error` are the theme's. `Visuals` has no success
    ///   colour, so `ok` (and the waveform trace) is a green chosen for the
    ///   window: a pale one on a dark window, a deep one on a light window.
    pub fn from_visuals(visuals: &egui::Visuals) -> Self {
        let text = visuals.text_color();
        let strong = visuals.strong_text_color();
        let ground = visuals.extreme_bg_color;
        let surface = visuals.window_fill;
        let separator = visuals.widgets.noninteractive.bg_stroke.color;
        let edge = edge_for(ground, text);
        let accent = accent_for(visuals, ground, surface);
        let ok = ok_for(surface);
        Self {
            canvas_ground: ground,
            canvas_edge: edge,
            canvas_grid: ground.lerp_to_gamma(text, GRID_CONTRAST),
            node_fill: surface,
            node_stroke: edge,
            node_title: strong,
            node_selected: accent,
            node_output: strong,
            wire: text,
            wire_active: accent,
            port: visuals.widgets.inactive.fg_stroke.color,
            ok,
            warn: visuals.warn_fg_color,
            error: visuals.error_fg_color,
            timeline_ground: ground,
            timeline_grid: separator,
            ground_text: text,
            lane: surface,
            lane_alt: surface.lerp_to_gamma(text, LANE_STRIPE),
            loop_start: accent,
            loop_end: strong,
            crossfade_band: accent.gamma_multiply(CROSSFADE_ALPHA),
            note_fill: visuals.selection.bg_fill,
            note_selected: strong,
            note_text: visuals.selection.stroke.color,
            release_tail: visuals.selection.bg_fill.gamma_multiply(TAIL_ALPHA),
            waveform_ground: ground,
            waveform_zero: separator,
            waveform_trace: ok,
            playhead: accent.lerp_to_gamma(strong, 0.5),
        }
    }
}

/// The colour a boundary is drawn in on `ground`: the ground leaned toward
/// `text` far enough that the boundary is found at a glance.
///
/// The separator colour a `Visuals` offers is meant to divide two halves of
/// a window rather than to bound a component, and it measures 1.79:1 on the
/// canvas under egui's stock dark theme and 1.86:1 under its light one —
/// both under `NON_TEXT_FLOOR`. Leaning from the ground toward the text
/// keeps the theme's own hue, and stopping at [`EDGE_TARGET`] lands short of
/// the text itself, so a box's edge stays quieter than the wires drawn over
/// it: 3.40:1 dark and 3.32:1 light, against a wire's 5.89 and 8.06
/// (Overlands #1339).
fn edge_for(ground: Color32, text: Color32) -> Color32 {
    for step in 0..=10_u8 {
        let lean = EDGE_LEAN + f32::from(step) * EDGE_STEP;
        let edge = ground.lerp_to_gamma(text, lean);
        if contrast_ratio(edge, ground) >= EDGE_TARGET {
            return edge;
        }
    }
    text
}

/// The accent the selected node, the dragged wire and the loop start are
/// drawn in: the theme's link colour where it can be found on both the
/// canvas and a lane, and the selection's text colour where it cannot.
///
/// egui's stock light blue is 2.94:1 on the canvas and 2.77:1 on a lane,
/// under `NON_TEXT_FLOOR`; its dark one is 8.13:1 and clears it, so a dark
/// theme keeps the accent it chose. A `Visuals` whose selection text is no
/// better than its link colour keeps the link colour: the roles are there
/// to be overwritten by a host that wants a third answer (Overlands #1339).
fn accent_for(visuals: &egui::Visuals, ground: Color32, surface: Color32) -> Color32 {
    let worst = |c: Color32| contrast_ratio(c, ground).min(contrast_ratio(c, surface));
    let accent = visuals.hyperlink_color;
    if worst(accent) >= NON_TEXT_FLOOR {
        return accent;
    }
    let stand_in = visuals.selection.stroke.color;
    if worst(stand_in) > worst(accent) {
        stand_in
    } else {
        accent
    }
}

/// The success green that reads better on `surface`.
fn ok_for(surface: Color32) -> Color32 {
    if contrast_ratio(OK_ON_DARK, surface) >= contrast_ratio(OK_ON_LIGHT, surface) {
        OK_ON_DARK
    } else {
        OK_ON_LIGHT
    }
}

/// The key the style is kept under in the context's data.
fn style_key() -> egui::Id {
    egui::Id::new("bevy_symbios_audio::ui::EditorStyle")
}

/// Make `style` the colours of every editor drawn in `ctx`, until the next
/// call or [`clear_editor_style`]. Call it when the host's theme changes.
pub fn set_editor_style(ctx: &egui::Context, style: EditorStyle) {
    ctx.data_mut(|data| data.insert_temp(style_key(), style));
}

/// Forget a style set with [`set_editor_style`]: the editors in `ctx` go
/// back to [`EditorStyle::from_visuals`] of the `Visuals` they are drawn
/// with.
pub fn clear_editor_style(ctx: &egui::Context) {
    ctx.data_mut(|data| data.remove::<EditorStyle>(style_key()));
}

/// The style the editors paint `ui` with: the one set on its context, or
/// else [`EditorStyle::from_visuals`] of `ui`'s own `Visuals`.
pub fn editor_style(ui: &egui::Ui) -> EditorStyle {
    set_style(ui.ctx()).unwrap_or_else(|| EditorStyle::from_visuals(ui.visuals()))
}

/// The style set on `ctx`, if any.
pub(crate) fn set_style(ctx: &egui::Context) -> Option<EditorStyle> {
    ctx.data(|data| data.get_temp::<EditorStyle>(style_key()))
}

/// WCAG 2.1 relative luminance of an opaque sRGB colour.
pub(crate) fn relative_luminance(c: Color32) -> f32 {
    let linear = |v: u8| {
        let s = f32::from(v) / 255.0;
        if s <= 0.040_45 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    0.2126 * linear(c.r()) + 0.7152 * linear(c.g()) + 0.0722 * linear(c.b())
}

/// WCAG 2.1 contrast ratio of two opaque colours: 1 for one colour against
/// itself, 21 for black on white. Normal text needs 4.5.
pub(crate) fn contrast_ratio(a: Color32, b: Color32) -> f32 {
    let (la, lb) = (relative_luminance(a), relative_luminance(b));
    (la.max(lb) + 0.05) / (la.min(lb) + 0.05)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::ui::sequence;
    use crate::ui::test_paint::contrast_on;

    /// WCAG AA for normal text.
    pub(crate) const AA: f32 = 4.5;

    fn both_themes() -> [(&'static str, egui::Visuals); 2] {
        [
            ("dark", egui::Visuals::dark()),
            ("light", egui::Visuals::light()),
        ]
    }

    /// A style whose every role is a colour of its own that egui's stock
    /// visuals never paint, so a test can tell which role painted what.
    pub(crate) fn distinct_style() -> EditorStyle {
        let mut style = EditorStyle::from_visuals(&egui::Visuals::dark());
        let roles = [
            &mut style.canvas_ground,
            &mut style.canvas_edge,
            &mut style.canvas_grid,
            &mut style.node_fill,
            &mut style.node_stroke,
            &mut style.node_title,
            &mut style.node_selected,
            &mut style.node_output,
            &mut style.wire,
            &mut style.wire_active,
            &mut style.port,
            &mut style.ok,
            &mut style.warn,
            &mut style.error,
            &mut style.timeline_ground,
            &mut style.timeline_grid,
            &mut style.ground_text,
            &mut style.lane,
            &mut style.lane_alt,
            &mut style.loop_start,
            &mut style.loop_end,
            &mut style.crossfade_band,
            &mut style.note_fill,
            &mut style.note_selected,
            &mut style.note_text,
            &mut style.release_tail,
            &mut style.waveform_ground,
            &mut style.waveform_zero,
            &mut style.waveform_trace,
            &mut style.playhead,
        ];
        for (i, role) in roles.into_iter().enumerate() {
            let step = u8::try_from(i).expect("fewer than 32 roles") * 8;
            *role = Color32::from_rgb(201, 3 + step, 57);
        }
        style
    }

    /// The canvas's border and grid are visible on its ground in both of
    /// egui's themes, and neither competes with the wires drawn over them
    /// (#59, Overlands #1332 B16).
    ///
    /// A grid nobody can see is a grid that does not say the ground moves;
    /// one as strong as the content is a grid that gets in the way. So the
    /// bar here is an ordering — ground, then grid, then border, then the
    /// wires over them — and not the 3:1 WCAG asks of a non-text mark.
    ///
    /// Against egui's stock visuals that ordering is: grid 1.10:1 dark and
    /// 1.16:1 light on the ground, border 3.40:1 and 3.32:1 over it, a
    /// wire 5.33:1 and 6.95:1 over the grid. The border is the same colour
    /// as a node's edge, and since Overlands #1339 that is `edge_for`
    /// rather than the theme's separator — a boundary has to clear
    /// `NON_TEXT_FLOOR`, which the separator missed at 1.79 and 1.86.
    /// The grid did not follow it up: a grid as strong as the content is a
    /// grid that gets in the way, so it stays where it was, and the top of
    /// the ordering is held here as well as the bottom.
    #[test]
    fn the_canvas_border_and_grid_are_visible_on_the_ground_in_dark_and_light() {
        for (theme, visuals) in both_themes() {
            let s = EditorStyle::from_visuals(&visuals);
            let edge = contrast_on(s.canvas_edge, s.canvas_ground);
            assert!(
                edge > 1.5,
                "{theme}: the canvas border is {edge:.2}:1 on its ground"
            );
            let grid = contrast_on(s.canvas_grid, s.canvas_ground);
            assert!(
                grid > 1.05,
                "{theme}: the canvas grid is {grid:.2}:1 on its ground — invisible"
            );
            assert!(
                grid < edge,
                "{theme}: the grid ({grid:.2}:1) is no fainter than the border \
                 ({edge:.2}:1); it is meant to sit behind the content"
            );
            assert!(
                contrast_on(s.wire, s.canvas_grid) > grid,
                "{theme}: a wire does not stand out from the grid it crosses"
            );
            let wire = contrast_on(s.wire, s.canvas_ground);
            assert!(
                edge < wire,
                "{theme}: the border ({edge:.2}:1) is no quieter than the wires \
                 over it ({wire:.2}:1); it is a boundary, not content"
            );
        }
    }

    /// Every mark the editors draw that is not text is found at a glance
    /// in egui's own dark and light themes: WCAG 1.4.11's 3:1 for the
    /// boundary of a component (Overlands #1339, E3).
    ///
    /// The roster is Overlands'
    /// `the_audio_editors_lines_and_edges_clear_the_non_text_floor_in_every_palette`,
    /// which holds the same marks against that host's three palettes, plus
    /// the canvas border. Two marks are measured through the function that
    /// paints them rather than a role, because no single role is what
    /// reaches the screen: a note block is filled with its instrument's
    /// tint, derived from `note_fill`, so [`sequence::note_edge`] is what
    /// bounds it against the lane.
    ///
    /// The grids are deliberately NOT here — `canvas_grid` and
    /// `timeline_grid` are grounds, not marks, and
    /// [`Self::the_canvas_border_and_grid_are_visible_on_the_ground_in_dark_and_light`]
    /// holds them to an ordering instead.
    #[test]
    fn from_visuals_holds_the_editors_marks_to_the_non_text_floor_in_dark_and_light() {
        let mut short = Vec::new();
        for (theme, visuals) in both_themes() {
            let s = EditorStyle::from_visuals(&visuals);
            let marks = [
                ("node edge on the canvas", s.node_stroke, s.canvas_ground),
                (
                    "canvas border on its ground",
                    s.canvas_edge,
                    s.canvas_ground,
                ),
                ("selected node's edge", s.node_selected, s.canvas_ground),
                ("output node's edge", s.node_output, s.canvas_ground),
                ("wire", s.wire, s.canvas_ground),
                ("dragged wire", s.wire_active, s.canvas_ground),
                ("port on its box", s.port, s.node_fill),
                ("loop start marker", s.loop_start, s.lane),
                ("sequence end marker", s.loop_end, s.lane),
                ("a note's edge on its lane", sequence::note_edge(&s), s.lane),
                ("selected note's outline", s.note_selected, s.note_fill),
            ];
            for (what, fg, bg) in marks {
                let ratio = contrast_on(fg, bg);
                if ratio < NON_TEXT_FLOOR {
                    short.push(format!(
                        "{theme}: {what} is {ratio:.2}:1 ({fg:?} on {bg:?})"
                    ));
                }
            }
        }
        assert!(
            short.is_empty(),
            "{} mark(s) under {NON_TEXT_FLOOR}:1:\n  {}",
            short.len(),
            short.join("\n  ")
        );
    }

    /// The acceptance of Overlands #1331: in egui's own dark and light
    /// themes, the text the editors paint reads on what it is painted on.
    #[test]
    fn from_visuals_holds_the_editors_text_to_aa_in_dark_and_light() {
        for (theme, visuals) in both_themes() {
            let s = EditorStyle::from_visuals(&visuals);
            let pairs = [
                ("node title on node fill", s.node_title, s.node_fill),
                ("note text on note fill", s.note_text, s.note_fill),
                (
                    "waveform trace on its ground",
                    s.waveform_trace,
                    s.waveform_ground,
                ),
                (
                    "ruler text on the timeline",
                    s.ground_text,
                    s.timeline_ground,
                ),
                (
                    "'no signal' on the waveform",
                    s.ground_text,
                    s.waveform_ground,
                ),
                ("valid graph on the window", s.ok, visuals.window_fill),
            ];
            for (what, fg, bg) in pairs {
                let ratio = contrast_on(fg, bg);
                assert!(
                    ratio >= AA,
                    "{theme}: {what} is {ratio:.2}:1 ({fg:?} on {bg:?})"
                );
            }
            // Overlands #1342: a note whose instrument is gone is labelled
            // on a block the error colour tints, so the pair that has to
            // read is the label over that composite — in both selection
            // states, and on both of the lane stripes it composites over.
            for (which, lane) in [("lane", s.lane), ("odd lane", s.lane_alt)] {
                for (state, strength) in [
                    ("unselected", sequence::MISSING_TINT),
                    ("selected", sequence::MISSING_TINT_SELECTED),
                ] {
                    let tint = lane.blend(s.error.gamma_multiply(strength));
                    let fg = sequence::missing_label_colour(&s);
                    let ratio = contrast_on(fg, tint);
                    assert!(
                        ratio >= AA,
                        "{theme}: a {state} missing note's label on the {which} \
                         is {ratio:.2}:1 ({fg:?} on {tint:?})"
                    );
                }
            }
        }
    }

    /// A selected note's outline is drawn inside its block, so it has to
    /// read against the note's fill: WCAG 1.4.11's 3:1. Overlands' first
    /// mapping outlined its bright teal notes in light grey, at 2.16:1.
    #[test]
    fn from_visuals_outlines_a_selected_note_where_it_can_be_seen() {
        for (theme, visuals) in both_themes() {
            let s = EditorStyle::from_visuals(&visuals);
            let ratio = contrast_on(s.note_selected, s.note_fill);
            assert!(ratio >= 3.0, "{theme}: the outline is {ratio:.2}:1");
        }
    }

    /// The control for the contrast tests: the palette 0.4.3 painted, a
    /// charcoal box under light visuals' dark title text, fails the bar.
    #[test]
    fn the_old_charcoal_box_fails_the_bar_under_light_visuals() {
        let light = egui::Visuals::light();
        let old_box = Color32::from_gray(32);
        assert!(contrast_on(light.strong_text_color(), old_box) < 2.0);
    }

    /// The default follows the theme: a light theme gets light grounds and
    /// boxes, and the ok green is picked for the window it sits on.
    #[test]
    fn from_visuals_follows_the_themes_grounds_and_picks_the_green_for_the_window() {
        let dark = EditorStyle::from_visuals(&egui::Visuals::dark());
        let light = EditorStyle::from_visuals(&egui::Visuals::light());
        assert_eq!(dark.node_fill, egui::Visuals::dark().window_fill);
        assert_eq!(light.node_fill, egui::Visuals::light().window_fill);
        assert_eq!(
            light.timeline_ground,
            egui::Visuals::light().extreme_bg_color
        );
        assert_eq!((dark.ok, light.ok), (OK_ON_DARK, OK_ON_LIGHT));
        assert_ne!(light.lane, light.lane_alt, "two lanes can be told apart");
    }

    fn ui_run(ctx: &egui::Context, visuals: egui::Visuals) -> EditorStyle {
        ctx.set_visuals(visuals);
        let mut seen = None;
        let _ = ctx.run_ui(egui::RawInput::default(), |root| {
            egui::CentralPanel::default().show(root, |ui| seen = Some(editor_style(ui)));
        });
        seen.expect("the panel ran")
    }

    /// With nothing set the editors follow the `Visuals`; a set style wins
    /// over any theme until it is cleared.
    #[test]
    fn a_set_style_wins_until_it_is_cleared() {
        let ctx = egui::Context::default();
        assert_eq!(
            ui_run(&ctx, egui::Visuals::light()),
            EditorStyle::from_visuals(&egui::Visuals::light())
        );
        set_editor_style(&ctx, distinct_style());
        assert_eq!(ui_run(&ctx, egui::Visuals::light()), distinct_style());
        assert_eq!(ui_run(&ctx, egui::Visuals::dark()), distinct_style());
        clear_editor_style(&ctx);
        assert_eq!(
            ui_run(&ctx, egui::Visuals::dark()),
            EditorStyle::from_visuals(&egui::Visuals::dark())
        );
    }

    #[test]
    fn the_distinct_style_gives_every_role_its_own_colour() {
        let s = distinct_style();
        let all = [
            s.canvas_ground,
            s.node_fill,
            s.node_stroke,
            s.node_title,
            s.node_selected,
            s.node_output,
            s.wire,
            s.wire_active,
            s.port,
            s.ok,
            s.warn,
            s.error,
            s.timeline_ground,
            s.timeline_grid,
            s.ground_text,
            s.lane,
            s.lane_alt,
            s.loop_start,
            s.loop_end,
            s.crossfade_band,
            s.note_fill,
            s.note_selected,
            s.note_text,
            s.release_tail,
            s.waveform_ground,
            s.waveform_zero,
            s.waveform_trace,
            s.playhead,
        ];
        for (i, a) in all.iter().enumerate() {
            assert!(all[i + 1..].iter().all(|b| a != b), "role {i} repeats");
        }
    }

    #[test]
    fn contrast_ratio_spans_one_to_twenty_one() {
        assert!((contrast_ratio(Color32::BLACK, Color32::WHITE) - 21.0).abs() < 0.01);
        assert!((contrast_ratio(Color32::GRAY, Color32::GRAY) - 1.0).abs() < 1e-6);
    }
}
