//! The audition strip: the row a host puts above an editor so an edit can be
//! heard (#57, Overlands #1330).
//!
//! [`audition_strip`] draws Audition and Stop, an Auto toggle, a status chip,
//! a caption saying exactly what is played, the reason when a bake fails,
//! and the waveform of the strip's last bake. It *returns* the
//! [`MonitorRequest`] to write, if there is one, instead of writing it, so it
//! takes no Bevy system parameter and a test can drive it headless.
//!
//! # What is played
//!
//! An [`AuditionSource`] is a patch baked at a sample rate for a length, or a
//! whole sequence at its own rate, and the caption says which ("1.0 s at
//! 22.05 kHz, looped"). A host whose world bakes a slot at particular numbers
//! should audition at those numbers, and can say so with
//! [`AuditionSource::with_note`]: an audition at other numbers is another
//! sound, and a 0.4 Hz sweep that swells over four seconds in the editor is
//! a stutter in a world that loops one second of it.
//!
//! # Auto
//!
//! With Auto on, a committed edit (`committed`, the editors'
//! [`EditorResponse::rebake`](crate::ui::EditorResponse::rebake)) re-bakes a
//! playing audition once the edits have been quiet for [`AUTO_QUIET_SECS`].
//! Auto keeps an audition current; it never starts one, and Stop ends it.
//!
//! Auto starts on for patches, and for sequences everywhere but the web.
//! Measured for Overlands #1330 at Overlands' release profile, wasm timed in
//! V8: a 1 s patch at 22.05 kHz bakes in 1–3 ms and a 4 s one in 5–10 ms, so
//! a patch re-bake costs nothing anywhere. A seeded ambient sequence (five or
//! six instruments over 34 beats) bakes in 120–220 ms native and 200–350 ms
//! on wasm. Native, that runs on a pool thread and costs the frame nothing;
//! on wasm the monitor bakes on the main thread, so a sequence re-bake after
//! every drag would freeze the page that long each time, longer on a slower
//! machine. The quiet period, 0.4 s, is Overlands' own settle before it
//! re-bakes a changed ambient bed: forty times a wasm patch bake, and twice
//! the slowest native sequence bake, so a superseded sequence bake (which
//! cannot be cancelled) has usually finished before the next one starts.
//!
//! # Several strips, one monitor
//!
//! There is one [`AudioMonitor`], and a host can show several strips at once
//! (the `host_window` example shows two). Each [`AuditionState`] remembers
//! what it last asked for, and a strip shows only its own audition: another
//! slot's sound is not this slot's, so its chip says Idle, it draws no
//! waveform, and its Auto does not take the monitor over.

use std::time::Duration;

use bevy_egui::egui;

use super::graph::describe_graph_error;
use super::preview::{AudioMonitor, MonitorRequest, MonitorStatus, fingerprint, waveform};
use crate::patch::{AudioPatch, topo_sort};
use crate::sequence::SequenceRecipe;

/// How long the edits must be quiet before Auto re-bakes, in seconds. See
/// the [module docs](self#auto) for where the number comes from.
pub const AUTO_QUIET_SECS: f64 = 0.4;

/// What an audition plays, and how: a patch at a sample rate for a length,
/// or a whole sequence at its own rate. Built per frame from the host's
/// working copy; it borrows, and clones only when a bake is asked for.
#[derive(Clone, Copy, Debug)]
pub struct AuditionSource<'a> {
    what: What<'a>,
    note: Option<&'a str>,
}

#[derive(Clone, Copy, Debug)]
enum What<'a> {
    Patch {
        patch: &'a AudioPatch,
        sample_rate: u32,
        duration_secs: f32,
    },
    Sequence(&'a SequenceRecipe),
}

impl<'a> AuditionSource<'a> {
    /// `patch` baked at `sample_rate` for `duration_secs`, then looped.
    pub fn patch(patch: &'a AudioPatch, sample_rate: u32, duration_secs: f32) -> Self {
        Self {
            what: What::Patch {
                patch,
                sample_rate,
                duration_secs,
            },
            note: None,
        }
    }

    /// The whole of `recipe`, at its own sample rate, looped.
    pub fn sequence(recipe: &'a SequenceRecipe) -> Self {
        Self {
            what: What::Sequence(recipe),
            note: None,
        }
    }

    /// A few words after the caption, for what the host knows and the crate
    /// does not ("as the world plays it").
    #[must_use]
    pub fn with_note(mut self, note: &'a str) -> Self {
        self.note = Some(note);
        self
    }

    /// What is played, in words: "1.0 s at 22.05 kHz, looped", or "The whole
    /// sequence: 34 beats at 60 BPM, 22.05 kHz, looped", then the note.
    pub fn caption(&self) -> String {
        let mut caption = match self.what {
            What::Patch {
                sample_rate,
                duration_secs,
                ..
            } => format!(
                "{} s at {}, looped",
                seconds(duration_secs),
                kilohertz(sample_rate)
            ),
            What::Sequence(recipe) => format!(
                "The whole sequence: {} beats at {} BPM, {}, looped",
                plain(recipe.duration_beats),
                plain(recipe.bpm),
                kilohertz(recipe.sample_rate)
            ),
        };
        if let Some(note) = self.note {
            caption.push_str(" \u{2014} ");
            caption.push_str(note);
        }
        caption
    }

    /// The request that bakes and plays this source.
    pub fn request(&self) -> MonitorRequest {
        match self.what {
            What::Patch {
                patch,
                sample_rate,
                duration_secs,
            } => MonitorRequest::PlayPatch {
                patch: patch.clone(),
                sample_rate,
                duration_secs,
            },
            What::Sequence(recipe) => MonitorRequest::PlaySequence {
                recipe: recipe.clone(),
            },
        }
    }

    /// Whether Auto starts on for this kind of source (see the module docs).
    fn auto_by_default(&self) -> bool {
        match self.what {
            What::Patch { .. } => true,
            What::Sequence(_) => !cfg!(target_arch = "wasm32"),
        }
    }

    /// Why this source will not bake, naming the nodes, if it will not. A
    /// sequence always bakes: the mixdown skips an instrument it cannot.
    fn fault(&self) -> Option<String> {
        match self.what {
            What::Patch { patch, .. } => topo_sort(&patch.graph)
                .err()
                .map(|e| describe_graph_error(&patch.graph, &e)),
            What::Sequence(_) => None,
        }
    }
}

/// `4.0`, `0.25`: at most two decimals, at least one.
fn seconds(value: f32) -> String {
    let text = trimmed(f64::from(value));
    if text.contains('.') {
        text
    } else {
        format!("{text}.0")
    }
}

/// `22.05 kHz`, `44.1 kHz`, `48 kHz`.
fn kilohertz(sample_rate: u32) -> String {
    format!("{} kHz", trimmed(f64::from(sample_rate) / 1000.0))
}

/// `34`, `34.5`, `60`.
fn plain(value: f32) -> String {
    trimmed(f64::from(value))
}

/// `value` to two decimals with the trailing zeros, and a bare point, cut.
fn trimmed(value: f64) -> String {
    let text = format!("{value:.2}");
    text.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// One strip's state across frames: whether Auto is on, a re-bake waiting
/// for the edits to go quiet, and what the strip last asked the monitor for.
/// Keep one per strip, and pass the same one every frame.
#[derive(Clone, Debug, Default)]
pub struct AuditionState {
    /// `None` until the user toggles Auto; then their choice, for every
    /// kind of source.
    auto: Option<bool>,
    /// When a committed edit's re-bake is due, in egui's clock (seconds).
    due: Option<f64>,
    /// The fingerprint of the last play request this strip made.
    mine: Option<u64>,
}

impl AuditionState {
    /// Start an audition of `source`, as pressing Audition does: the request
    /// to write, which the strip then treats as its own audition. For a
    /// host's own shortcut, or a harness.
    pub fn play(&mut self, source: &AuditionSource<'_>) -> MonitorRequest {
        let request = source.request();
        self.mine = fingerprint(&request);
        self.due = None;
        request
    }

    /// Stop the audition, as pressing Stop does. The waveform of the last
    /// bake stays up.
    pub fn stop(&mut self) -> MonitorRequest {
        self.due = None;
        MonitorRequest::Stop
    }

    /// Whether Auto is on for `source`.
    pub fn auto(&self, source: &AuditionSource<'_>) -> bool {
        self.auto.unwrap_or_else(|| source.auto_by_default())
    }

    /// Turn Auto on or off, for every kind of source.
    pub fn set_auto(&mut self, on: bool) {
        self.auto = Some(on);
        if !on {
            self.due = None;
        }
    }
}

/// Draw the audition strip for `source` and return the request to write to
/// the [`AudioMonitor`], if any.
///
/// - **Audition** bakes `source` and loops it, replacing whatever plays; a
///   bake in flight is cancelled.
/// - **Stop** ends the audition.
/// - **Auto**, when on, re-bakes a playing audition [`AUTO_QUIET_SECS`]
///   after the last frame on which `committed` was true. Pass the editor's
///   [`EditorResponse::rebake`](crate::ui::EditorResponse::rebake).
/// - The **chip** says Idle, Baking with the seconds so far, Playing, Muted
///   (playing, while `muted` says the host's sound is off, so nothing is
///   heard) or Error. An error is spelled out under the row, naming the
///   nodes at fault when the source is a patch.
/// - The **caption** says what is played ([`AuditionSource::caption`]).
/// - The **waveform** of the strip's last bake, once there is one.
///
/// Draw it above a canvas, never after one: see the [module docs](crate::ui).
pub fn audition_strip(
    ui: &mut egui::Ui,
    monitor: &AudioMonitor,
    state: &mut AuditionState,
    source: AuditionSource<'_>,
    committed: bool,
    muted: bool,
) -> Option<MonitorRequest> {
    let now = ui.input(|i| i.time);
    // Whose audition the monitor's status is about: this strip's, another
    // strip's, or nobody's in particular (a host that set the fields).
    let about = monitor.auditioning();
    let status = if about.is_none() || about == state.mine {
        monitor.status.clone()
    } else {
        MonitorStatus::Idle
    };
    let live = about.is_some() && about == state.mine && status != MonitorStatus::Idle;
    let auto = state.auto(&source);
    if !(auto && live) {
        state.due = None;
    } else if committed {
        state.due = Some(now + AUTO_QUIET_SECS);
    }

    let mut asked = None;
    ui.horizontal_wrapped(|ui| {
        let play_hover = if muted {
            "Bake this and loop it, in place of what plays. The app's sound is \
             muted, so it will be silent."
        } else {
            "Bake this and loop it, in place of what plays"
        };
        if ui
            .button("\u{25B6} Audition")
            .on_hover_text(play_hover)
            .clicked()
        {
            asked = Some(state.play(&source));
        }
        if ui
            .add_enabled(
                status != MonitorStatus::Idle,
                egui::Button::new("\u{23F9} Stop"),
            )
            .on_hover_text("Stop the audition")
            .on_disabled_hover_text("Nothing of this slot is playing")
            .clicked()
        {
            asked = Some(state.stop());
        }
        let mut on = auto;
        if ui
            .checkbox(&mut on, "Auto")
            .on_hover_text(auto_hover(&source))
            .changed()
        {
            state.set_auto(on);
        }
        status_chip(ui, &status, monitor.bake_elapsed(), muted);
    });
    ui.label(egui::RichText::new(source.caption()).weak());
    if let MonitorStatus::Error(message) = &status {
        let text = source.fault().unwrap_or_else(|| message.clone());
        let color = ui.visuals().error_fg_color;
        ui.add(egui::Label::new(egui::RichText::new(text).color(color)).wrap());
    }
    let samples = monitor.samples_of();
    if !monitor.last_samples.is_empty() && (samples.is_none() || samples == state.mine) {
        waveform(ui, &monitor.last_samples);
    }

    if asked.is_none()
        && let Some(due) = state.due
    {
        if now >= due {
            asked = Some(state.play(&source));
        } else {
            ui.ctx()
                .request_repaint_after(Duration::from_secs_f64(due - now));
        }
    }
    asked
}

/// The Auto toggle's hover, which says why it starts off where it does.
fn auto_hover(source: &AuditionSource<'_>) -> String {
    let mut text = format!(
        "Re-bake and play again {AUTO_QUIET_SECS} s after your last edit, while \
         this audition plays. Stop ends it."
    );
    if !source.auto_by_default() {
        text.push_str(
            " Off at first here: a sequence bakes on the page's own thread on the \
             web, and freezes it for a moment each time.",
        );
    }
    text
}

/// The status chip: a word in a tinted outline, with a spinner while
/// baking. Every colour is the host theme's.
fn status_chip(ui: &mut egui::Ui, status: &MonitorStatus, elapsed: Option<Duration>, muted: bool) {
    let visuals = ui.visuals();
    let (label, tone, hover) = match status {
        MonitorStatus::Idle => (
            "Idle".to_string(),
            visuals.weak_text_color(),
            "Nothing of this slot is playing",
        ),
        MonitorStatus::Baking => (
            elapsed.map_or_else(
                || "Baking".to_string(),
                |t| format!("Baking {:.1} s", t.as_secs_f32()),
            ),
            visuals.text_color(),
            "Baking; it plays when the bake is done",
        ),
        MonitorStatus::Playing if muted => (
            "Muted".to_string(),
            visuals.warn_fg_color,
            "Playing, but the app's sound is muted, so nothing is heard",
        ),
        MonitorStatus::Playing => (
            "Playing".to_string(),
            visuals.strong_text_color(),
            "Looping the last bake",
        ),
        MonitorStatus::Error(_) => (
            "Error".to_string(),
            visuals.error_fg_color,
            "The last bake failed; the reason is below",
        ),
    };
    let fill = visuals.extreme_bg_color;
    egui::Frame::new()
        .fill(fill)
        .stroke(egui::Stroke::new(1.0, tone))
        .corner_radius(4.0)
        .inner_margin(egui::Margin::symmetric(6, 1))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                if *status == MonitorStatus::Baking {
                    let size = ui.text_style_height(&egui::TextStyle::Body);
                    ui.add(egui::Spinner::new().size(size).color(tone));
                }
                ui.label(egui::RichText::new(label).color(tone));
            });
        })
        .response
        .on_hover_text(hover);
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use bevy_egui::egui::accesskit;

    use super::*;
    use crate::{Connection, GraphNode, NodeGraph, NodeId, NodeKind};

    /// The strip on a headless context, with the clock in the test's hands.
    struct Strip {
        ctx: egui::Context,
        out: egui::FullOutput,
        state: AuditionState,
        now: f64,
    }

    impl Strip {
        fn new() -> Self {
            let ctx = egui::Context::default();
            ctx.enable_accesskit();
            Self {
                ctx,
                out: egui::FullOutput::default(),
                state: AuditionState::default(),
                now: 10.0,
            }
        }

        /// One frame at `self.now`; what the strip asked the monitor for.
        fn frame(
            &mut self,
            monitor: &AudioMonitor,
            source: AuditionSource<'_>,
            committed: bool,
            muted: bool,
            events: Vec<egui::Event>,
        ) -> Option<MonitorRequest> {
            let Self {
                ctx, out, state, ..
            } = self;
            let input = egui::RawInput {
                time: Some(self.now),
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(900.0, 300.0),
                )),
                events,
                ..Default::default()
            };
            let mut asked = None;
            *out = ctx.run_ui(input, |root| {
                egui::CentralPanel::default().show(root, |ui| {
                    asked = audition_strip(ui, monitor, state, source, committed, muted);
                });
            });
            asked
        }

        /// A few quiet frames, so a new layout settles; the last one's ask.
        fn settle(
            &mut self,
            monitor: &AudioMonitor,
            source: AuditionSource<'_>,
            muted: bool,
        ) -> Option<MonitorRequest> {
            let mut asked = None;
            for _ in 0..3 {
                asked = asked.or(self.frame(monitor, source, false, muted, Vec::new()));
            }
            asked
        }

        fn texts(&self) -> Vec<String> {
            fn walk(shape: &egui::Shape, out: &mut Vec<String>) {
                match shape {
                    egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| walk(s, out)),
                    egui::Shape::Text(text) => out.push(text.galley.text().to_owned()),
                    _ => {}
                }
            }
            let mut out = Vec::new();
            for clipped in &self.out.shapes {
                walk(&clipped.shape, &mut out);
            }
            out
        }

        fn paints(&self, needle: &str) -> bool {
            self.texts().iter().any(|t| t.contains(needle))
        }

        /// The AccessKit node of the first widget labelled `prefix…`.
        fn node(&self, prefix: &str) -> (accesskit::NodeId, &accesskit::Node) {
            self.out
                .platform_output
                .accesskit_update
                .iter()
                .flat_map(|update| &update.nodes)
                .find(|(_, n)| n.label().is_some_and(|l| l.starts_with(prefix)))
                .map(|(id, node)| (*id, node))
                .unwrap_or_else(|| panic!("no widget labelled {prefix:?}"))
        }

        fn widget(&self, prefix: &str) -> accesskit::NodeId {
            self.node(prefix).0
        }

        fn disabled(&self, prefix: &str) -> bool {
            self.node(prefix).1.is_disabled()
        }

        /// The event that clicks the widget labelled `prefix…`.
        fn click(&self, prefix: &str) -> Vec<egui::Event> {
            vec![egui::Event::AccessKitActionRequest(
                accesskit::ActionRequest {
                    action: accesskit::Action::Click,
                    target_tree: accesskit::TreeId::ROOT,
                    target_node: self.widget(prefix),
                    data: None,
                },
            )]
        }
    }

    fn gain_from(id: u32, from: &[u32]) -> GraphNode {
        let mut inputs = BTreeMap::new();
        if !from.is_empty() {
            inputs.insert(
                "in".to_string(),
                from.iter()
                    .map(|&up| Connection::from_node(NodeId(up)))
                    .collect(),
            );
        }
        GraphNode {
            id: NodeId(id),
            kind: NodeKind::Gain(Default::default()),
            inputs,
        }
    }

    fn patch(nodes: Vec<GraphNode>, output: u32) -> AudioPatch {
        AudioPatch {
            seed: 0,
            graph: NodeGraph {
                nodes,
                output: NodeId(output),
            },
        }
    }

    fn one_gain() -> AudioPatch {
        patch(vec![gain_from(0, &[])], 0)
    }

    /// `#0` and `#1` feed each other: the bake fails with a cycle.
    fn looped() -> AudioPatch {
        patch(vec![gain_from(0, &[1]), gain_from(1, &[0])], 0)
    }

    /// The strip shows each state of its own audition in a chip, and draws
    /// headless in every one (#57, Overlands #1330 D4). Muted is not a
    /// monitor state: it is a Playing audition in an app whose sound is off,
    /// which used to say "playing" and make no sound (A3).
    #[test]
    fn the_strip_renders_headless_in_every_status() {
        let quiet = one_gain();
        let broken = looped();
        let cases: [(&str, &AudioPatch, Option<MonitorStatus>, bool); 5] = [
            ("Idle", &quiet, None, false),
            ("Baking", &quiet, Some(MonitorStatus::Baking), false),
            ("Playing", &quiet, Some(MonitorStatus::Playing), false),
            ("Muted", &quiet, Some(MonitorStatus::Playing), true),
            (
                "Error",
                &broken,
                Some(MonitorStatus::Error("graph contains a cycle".into())),
                false,
            ),
        ];
        for (chip, patch, status, muted) in cases {
            let source = AuditionSource::patch(patch, 22_050, 1.0);
            let mut strip = Strip::new();
            let mut monitor = AudioMonitor::default();
            if let Some(status) = status {
                let request = strip.state.play(&source);
                monitor.stage(&request, status);
            }
            strip.settle(&monitor, source, muted);
            assert!(
                strip.texts().iter().any(|t| t.starts_with(chip)),
                "{chip}: the chip says so; painted {:?}",
                strip.texts()
            );
            assert!(
                strip.paints("1.0 s at 22.05 kHz, looped"),
                "{chip}: the caption says what is played; painted {:?}",
                strip.texts()
            );
        }
    }

    /// A bake in flight says how long it has been running.
    #[test]
    fn the_baking_chip_counts_the_seconds() {
        let quiet = one_gain();
        let source = AuditionSource::patch(&quiet, 22_050, 1.0);
        let mut strip = Strip::new();
        let request = strip.state.play(&source);
        let mut monitor = AudioMonitor::default();
        monitor.stage(&request, MonitorStatus::Baking);
        monitor.backdate_bake(std::time::Duration::from_millis(1_250));
        strip.settle(&monitor, source, false);
        assert!(
            strip.paints("Baking 1.2 s") || strip.paints("Baking 1.3 s"),
            "painted {:?}",
            strip.texts()
        );
    }

    /// A failed bake names the nodes at fault, not just the kind of fault.
    #[test]
    fn an_error_names_the_nodes_of_the_loop() {
        let broken = looped();
        let source = AuditionSource::patch(&broken, 22_050, 1.0);
        let mut strip = Strip::new();
        let request = strip.state.play(&source);
        let mut monitor = AudioMonitor::default();
        monitor.stage(
            &request,
            MonitorStatus::Error("graph contains a cycle".into()),
        );
        strip.settle(&monitor, source, false);
        let text = strip.texts().join("\n");
        assert!(
            text.contains("#0 Gain") && text.contains("#1 Gain"),
            "painted:\n{text}"
        );
    }

    #[test]
    fn audition_and_stop_ask_the_monitor_for_what_they_say() {
        let quiet = one_gain();
        let source = AuditionSource::patch(&quiet, 22_050, 1.0);
        let mut monitor = AudioMonitor::default();
        let mut strip = Strip::new();
        strip.settle(&monitor, source, false);
        assert!(
            strip.disabled("\u{23F9} Stop"),
            "nothing plays, so there is nothing to stop"
        );

        let events = strip.click("\u{25B6} Audition");
        let request = strip.frame(&monitor, source, false, false, events);
        match &request {
            Some(MonitorRequest::PlayPatch {
                patch,
                sample_rate,
                duration_secs,
            }) => {
                assert_eq!(*patch, quiet);
                assert_eq!((*sample_rate, *duration_secs), (22_050, 1.0));
            }
            other => panic!("Audition asked for {}", describe(other)),
        }
        // The monitor takes the request and plays it.
        monitor.stage(&request.expect("a request"), MonitorStatus::Playing);
        strip.settle(&monitor, source, false);
        assert!(!strip.disabled("\u{23F9} Stop"));
        let events = strip.click("\u{23F9} Stop");
        let asked = strip.frame(&monitor, source, false, false, events);
        assert!(
            matches!(asked, Some(MonitorRequest::Stop)),
            "Stop asked for {}",
            describe(&asked)
        );
    }

    fn describe(request: &Option<MonitorRequest>) -> &'static str {
        match request {
            None => "nothing",
            Some(MonitorRequest::PlayPatch { .. }) => "a patch",
            Some(MonitorRequest::PlaySequence { .. }) => "a sequence",
            Some(MonitorRequest::Stop) => "Stop",
        }
    }

    /// A strip whose audition is playing, after a settle.
    fn playing(source: AuditionSource<'_>) -> (Strip, AudioMonitor) {
        let mut strip = Strip::new();
        let request = strip.state.play(&source);
        let mut monitor = AudioMonitor::default();
        monitor.stage(&request, MonitorStatus::Playing);
        strip.settle(&monitor, source, false);
        (strip, monitor)
    }

    /// D2: with Auto on, a committed edit re-bakes what plays once the edits
    /// have been quiet for [`AUTO_QUIET_SECS`], with the edit in it.
    #[test]
    fn auto_rebakes_a_playing_audition_once_the_edits_go_quiet() {
        let before = one_gain();
        let source = AuditionSource::patch(&before, 22_050, 1.0);
        let (mut strip, monitor) = playing(source);
        assert!(strip.state.auto(&source), "Auto is on for a patch");

        let mut edited = before.clone();
        edited.seed = 99;
        let source = AuditionSource::patch(&edited, 22_050, 1.0);
        assert!(
            strip
                .frame(&monitor, source, true, false, Vec::new())
                .is_none()
        );
        strip.now += AUTO_QUIET_SECS * 0.5;
        assert!(
            strip
                .frame(&monitor, source, false, false, Vec::new())
                .is_none(),
            "not before the quiet period"
        );
        strip.now += AUTO_QUIET_SECS * 0.6;
        match strip.frame(&monitor, source, false, false, Vec::new()) {
            Some(MonitorRequest::PlayPatch { patch, .. }) => assert_eq!(patch.seed, 99),
            other => panic!("Auto asked for {}", describe(&other)),
        }
        strip.now += AUTO_QUIET_SECS;
        assert!(
            strip
                .frame(&monitor, source, false, false, Vec::new())
                .is_none(),
            "one re-bake per burst of edits"
        );
    }

    /// Each commit restarts the quiet period, so a burst of edits bakes once.
    #[test]
    fn a_commit_restarts_the_quiet_period() {
        let quiet = one_gain();
        let source = AuditionSource::patch(&quiet, 22_050, 1.0);
        let (mut strip, monitor) = playing(source);
        strip.frame(&monitor, source, true, false, Vec::new());
        strip.now += AUTO_QUIET_SECS * 0.8;
        assert!(
            strip
                .frame(&monitor, source, true, false, Vec::new())
                .is_none()
        );
        strip.now += AUTO_QUIET_SECS * 0.8;
        assert!(
            strip
                .frame(&monitor, source, false, false, Vec::new())
                .is_none(),
            "quiet since the second commit for less than the period"
        );
        strip.now += AUTO_QUIET_SECS * 0.3;
        assert!(
            strip
                .frame(&monitor, source, false, false, Vec::new())
                .is_some()
        );
    }

    /// Auto keeps an audition current; it never starts one. An edit with
    /// nothing playing, or after Stop, makes no sound.
    #[test]
    fn auto_never_starts_an_audition() {
        let quiet = one_gain();
        let source = AuditionSource::patch(&quiet, 22_050, 1.0);
        let monitor = AudioMonitor::default();
        let mut strip = Strip::new();
        strip.settle(&monitor, source, false);
        strip.frame(&monitor, source, true, false, Vec::new());
        strip.now += AUTO_QUIET_SECS * 2.0;
        assert!(
            strip
                .frame(&monitor, source, false, false, Vec::new())
                .is_none()
        );
    }

    #[test]
    fn auto_off_leaves_a_playing_audition_alone() {
        let quiet = one_gain();
        let source = AuditionSource::patch(&quiet, 22_050, 1.0);
        let (mut strip, monitor) = playing(source);
        strip.state.set_auto(false);
        strip.frame(&monitor, source, true, false, Vec::new());
        strip.now += AUTO_QUIET_SECS * 2.0;
        assert!(
            strip
                .frame(&monitor, source, false, false, Vec::new())
                .is_none()
        );
        assert!(!strip.state.auto(&source));
    }

    /// Patches bake in milliseconds everywhere, so Auto starts on for them.
    /// A sequence bakes for 0.1-0.35 s, and on wasm the monitor bakes on the
    /// main thread: there Auto starts off, so an edit never freezes a frame
    /// the owner did not ask for.
    #[test]
    fn auto_starts_on_for_patches_and_for_sequences_only_off_the_web() {
        let quiet = one_gain();
        let recipe = SequenceRecipe::default();
        let state = AuditionState::default();
        assert!(state.auto(&AuditionSource::patch(&quiet, 22_050, 1.0)));
        assert_eq!(
            state.auto(&AuditionSource::sequence(&recipe)),
            !cfg!(target_arch = "wasm32")
        );
    }

    /// Two strips share one monitor. The one whose audition is not playing
    /// says Idle and draws no waveform, and its Auto does not take the
    /// monitor over: the other slot's sound is not this slot's.
    #[test]
    fn a_strip_shows_only_its_own_audition() {
        let mine = one_gain();
        let theirs = looped();
        let mut other = AuditionState::default();
        let request = other.play(&AuditionSource::patch(&theirs, 44_100, 4.0));
        let mut monitor = AudioMonitor::default();
        monitor.stage(&request, MonitorStatus::Playing);

        let source = AuditionSource::patch(&mine, 22_050, 1.0);
        let mut strip = Strip::new();
        strip.state.play(&source);
        strip.settle(&monitor, source, false);
        assert!(
            strip.texts().iter().any(|t| t.starts_with("Idle")),
            "painted {:?}",
            strip.texts()
        );
        assert!(!strip.paints("no signal"), "no waveform box at all");
        strip.frame(&monitor, source, true, false, Vec::new());
        strip.now += AUTO_QUIET_SECS * 2.0;
        assert!(
            strip
                .frame(&monitor, source, false, false, Vec::new())
                .is_none()
        );
    }

    #[test]
    fn the_caption_says_what_is_played_and_how() {
        let quiet = one_gain();
        assert_eq!(
            AuditionSource::patch(&quiet, 22_050, 1.0).caption(),
            "1.0 s at 22.05 kHz, looped"
        );
        assert_eq!(
            AuditionSource::patch(&quiet, 44_100, 4.0)
                .with_note("as the world plays it")
                .caption(),
            "4.0 s at 44.1 kHz, looped \u{2014} as the world plays it"
        );
        let recipe = SequenceRecipe {
            bpm: 60.0,
            sample_rate: 22_050,
            duration_beats: 34.0,
            ..SequenceRecipe::default()
        };
        assert_eq!(
            AuditionSource::sequence(&recipe).caption(),
            "The whole sequence: 34 beats at 60 BPM, 22.05 kHz, looped"
        );
    }
}
