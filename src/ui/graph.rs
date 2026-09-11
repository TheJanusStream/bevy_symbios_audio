//! Visual node-graph canvas for editing a whole [`AudioPatch`].
//!
//! Hand-rolled on `bevy_egui::egui` primitives — no third-party node-graph
//! crate (those pin their own egui version and lag `bevy_egui`).  Pan and
//! zoom come from [`egui::Scene`], which wraps the canvas content in a layer
//! transform so dragging the background pans and the scroll wheel zooms (and
//! widgets scale with the zoom).
//!
//! # Node positions live outside the schema
//!
//! [`crate::patch::GraphNode`] carries no layout — the wire format is mirrored
//! downstream (Overlands' `Sovereign*` PDS/CBOR types) and must stay clean.
//! Positions live here in [`PatchEditorState`], keyed by [`NodeId`], with a
//! topological auto-layout for any node that doesn't have one yet (patches
//! loaded from JSON or built in code).
//!
//! # How editing maps to the schema
//!
//! - **Move a node:** drag its title bar (scene-space delta → stored position).
//! - **Wire a port:** drag from a node's output dot (right edge, on the
//!   title row) onto another node's input: its dot, or anywhere on its row
//!   in the node's "Inputs" list, where the port is named. That appends a
//!   [`Connection::Node`] to the port (fan-in: a port holds a *list*,
//!   summed at bake time). Each input dot sits on the box's left edge
//!   beside its own row — filled when something drives the port, a ring
//!   when nothing does — and so does a port that is not one of its kind's
//!   own (from JSON). While a wire is dragged it is drawn over the boxes,
//!   the port it would connect to is highlighted in the host's selection
//!   colour, and a tooltip names it ("➡ #2 Lowpass · cutoff_hz"); over no
//!   port, nothing is lit and a release does nothing. Only the box on top
//!   under the pointer can take the wire, so a row another box covers is
//!   out of reach.
//! - **Amounts / constants / deletes:** the "Inputs" section inside each node
//!   box edits each connection's `amount` (or a [`Connection::Constant`]'s
//!   value) and removes connections.
//! - **Add / remove nodes, set output:** the toolbar above the canvas.
//!
//! Structural edits are collected as deferred `Action`s while the node loop
//! holds `&mut patch.graph.nodes`, then applied once the loop's borrow ends —
//! the standard way to keep an immediate-mode graph editor borrow-clean.
//!
//! Validity ([`topo_sort`]) is shown live in the toolbar and the output node
//! gets its own border, so cycles / missing-output / unknown-node are visible
//! the moment they're created. A broken graph is also located: the toolbar
//! names the nodes at fault ("#0 Gain and #1 Gain feed each other in a
//! loop…"), and the canvas outlines their boxes in the error colour — the
//! nodes of a loop, the node holding a wire from a node that is not there,
//! or every node sharing an id (#57).
//!
//! # Colours
//!
//! Everything the canvas paints itself — the ground, the node boxes and
//! their edges and titles, wires, ports, the validity line — takes its colour
//! from the [`EditorStyle`] in effect ([`crate::ui::style`]): the host's, if
//! it set one, else one derived from the `Visuals` the canvas is drawn with.
//! A node box is filled with a surface the theme's text reads on, so the
//! widgets inside it read in a light theme as well as a dark one (#58).

use std::collections::{HashMap, HashSet};

use bevy_egui::egui::{
    self, Align, Color32, Id, Layout, Pos2, Rect, Sense, Stroke, UiBuilder, Vec2,
};

use crate::node::NodeKind;
use crate::oscillator::SineOsc;
use crate::patch::{AudioPatch, Connection, GraphError, GraphNode, NodeGraph, NodeId, topo_sort};

use super::EditorResponse;
use super::evolve::{fresh_rng, mutate_node_kind, mutate_patch, randomize_seed};
use super::io::json_io;
use super::node::{node_kind_editor, node_kind_label};
use super::style::{EditorStyle, editor_style};

const NODE_WIDTH: f32 = 210.0;
const PORT_RADIUS: f32 = 5.0;
/// Opacity of the highlight over the row a dragged wire would connect to:
/// the row's labels are under it and must still read.
const DROP_ROW_ALPHA: f32 = 0.2;
/// Horizontal / vertical spacing of the topological auto-layout grid.
const COL_W: f32 = 280.0;
const ROW_H: f32 = 190.0;
/// How close (scene units) a wire drop must land to an input dot to connect
/// when it is not on the port's row.
const SNAP_DIST: f32 = 26.0;

/// Editor-side state for the patch canvas — node layout and view, kept out of
/// the serialized [`AudioPatch`] so the wire format stays clean.
///
/// Construct with [`Default`]; the canvas fills in any missing node positions
/// via topological auto-layout on first sight.
#[derive(Clone, Debug)]
pub struct PatchEditorState {
    /// Node positions in scene-local coordinates.
    positions: HashMap<NodeId, Pos2>,
    /// The [`egui::Scene`] view rectangle — pan and zoom live here.
    scene_rect: Rect,
    /// Selected node (delete target + highlight).
    selected: Option<NodeId>,
    /// Mutation rate for the Mutate buttons.
    mutate_rate: f32,
    /// Buffer + last error for the JSON import/export section.
    json: super::JsonIoState,
    /// Every input port as the canvas last drew it: its dot and its row.
    ports: Vec<PortGeom>,
    /// Each node's output dot as the canvas last drew it.
    outputs: HashMap<NodeId, Pos2>,
    /// Each node's box as the canvas last drew it, in drawing order: a box
    /// lies over the ones before it.
    boxes: Vec<(NodeId, Rect)>,
}

impl Default for PatchEditorState {
    fn default() -> Self {
        Self {
            positions: HashMap::new(),
            scene_rect: Rect::from_min_size(Pos2::ZERO, Vec2::new(1000.0, 700.0)),
            selected: None,
            mutate_rate: 0.3,
            json: super::JsonIoState::default(),
            ports: Vec::new(),
            outputs: HashMap::new(),
            boxes: Vec::new(),
        }
    }
}

/// One input port as the canvas drew it, in canvas units.
///
/// A port's dot sits on its own row of the node's "Inputs" list, where its
/// name is written, so a wire visibly goes into a named port (#56). Until
/// then the dots were spread evenly down the box's edge, beside whatever
/// row happened to be there, and a port that is not one of its kind's own
/// (from JSON) had a row but no dot.
#[derive(Clone, Debug, PartialEq)]
struct PortGeom {
    node: NodeId,
    port: String,
    /// On the box's left edge, level with the port's name.
    dot: Pos2,
    /// The port's part of the "Inputs" list (its name row and the rows of
    /// its connections) across the box's whole width. A wire dropped
    /// anywhere on it connects to this port.
    row: Rect,
    /// Something drives the port: a wire or a constant.
    connected: bool,
}

/// The port a wire from `from` connects to if it is dropped at `at`: the
/// row under the pointer, or failing that the nearest dot within
/// [`SNAP_DIST`].
///
/// Over a box, only that box's ports count, and only the box on top when
/// boxes overlap (`boxes` is in drawing order): a row another box covers is
/// out of sight, so it is out of reach too. A node's own ports are never a
/// target, since a wire into its own node is refused.
fn drop_target<'a>(
    ports: &'a [PortGeom],
    boxes: &[(NodeId, Rect)],
    from: NodeId,
    at: Pos2,
) -> Option<&'a PortGeom> {
    let under = boxes
        .iter()
        .rev()
        .find(|(_, rect)| rect.contains(at))
        .map(|(node, _)| *node);
    let eligible = || {
        ports
            .iter()
            .filter(move |p| p.node != from && under.is_none_or(|node| p.node == node))
    };
    eligible().find(|p| p.row.contains(at)).or_else(|| {
        eligible()
            .map(|p| (p, p.dot.distance(at)))
            .filter(|(_, d)| *d <= SNAP_DIST)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(p, _)| p)
    })
}

impl PatchEditorState {
    /// Assign a position to every node that lacks one, laying fresh nodes out
    /// in topological columns (sources left, output right).  Nodes the user
    /// has already moved keep their stored position.
    fn ensure_layout(&mut self, patch: &AudioPatch) {
        if patch
            .graph
            .nodes
            .iter()
            .all(|n| self.positions.contains_key(&n.id))
        {
            return;
        }
        let depths = compute_depths(patch);
        let mut row_in_col: HashMap<u32, u32> = HashMap::new();
        for node in &patch.graph.nodes {
            if self.positions.contains_key(&node.id) {
                continue;
            }
            let col = depths.get(&node.id).copied().unwrap_or(0);
            let row = row_in_col.entry(col).or_insert(0);
            let pos = Pos2::new(40.0 + col as f32 * COL_W, 40.0 + *row as f32 * ROW_H);
            self.positions.insert(node.id, pos);
            *row += 1;
        }
    }
}

/// Canonical input port names for a node kind — the dots drawn on the left
/// edge and the rows in the "Inputs" editor.  Matches each node's
/// `ctx.input("…")` reads; `Mix` sums any ports so it's given four generic
/// slots.
fn input_ports(kind: &NodeKind) -> &'static [&'static str] {
    match kind {
        NodeKind::Sine(_) | NodeKind::Square(_) | NodeKind::Sawtooth(_) | NodeKind::Triangle(_) => {
            &["freq", "amplitude"]
        }
        NodeKind::Adsr(_) => &["gate"],
        NodeKind::BiquadLowpass(_) | NodeKind::BiquadHighpass(_) => &["in", "cutoff_hz", "q"],
        NodeKind::BiquadBandpass(_) => &["in", "center_hz", "q"],
        NodeKind::Mix(_) => &["a", "b", "c", "d"],
        NodeKind::Gain(_) => &["in", "gain"],
        NodeKind::Chorus(_) | NodeKind::Reverb(_) => &["in"],
        NodeKind::Silence
        | NodeKind::WhiteNoise(_)
        | NodeKind::PinkNoise(_)
        | NodeKind::BrownNoise(_)
        | NodeKind::Lfo(_)
        | NodeKind::Gate(_) => &[],
        // `NodeKind` is `#[non_exhaustive]` (defined in the `symbios-audio`
        // core crate); a not-yet-known kind contributes no input ports here.
        _ => &[],
    }
}

/// Longest-upstream-chain depth per node (the auto-layout column).  Falls back
/// to index order if the graph doesn't topo-sort (a cycle), so a broken graph
/// still lays out somewhere visible.
fn compute_depths(patch: &AudioPatch) -> HashMap<NodeId, u32> {
    let mut depth: HashMap<NodeId, u32> = HashMap::new();
    match topo_sort(&patch.graph) {
        Ok(order) => {
            for id in order {
                let Some(node) = patch.graph.nodes.iter().find(|n| n.id == id) else {
                    continue;
                };
                let mut d = 0;
                for conns in node.inputs.values() {
                    for c in conns {
                        if let Connection::Node { id: up, .. } = c {
                            d = d.max(depth.get(up).copied().unwrap_or(0) + 1);
                        }
                    }
                }
                depth.insert(id, d);
            }
        }
        Err(_) => {
            for (i, n) in patch.graph.nodes.iter().enumerate() {
                depth.insert(n.id, i as u32);
            }
        }
    }
    depth
}

/// The nodes on the canvas that `err` is about, in the patch's node order:
/// the ones a user has to change to make the graph bake (#57).
///
/// - [`GraphError::UnknownNode`] names a node that is not in the patch, so
///   the box to outline is every node holding a wire from it.
/// - [`GraphError::DuplicateId`]: every node carrying that id.
/// - [`GraphError::Cycle`]: the nodes on a loop — not the ones merely
///   upstream or downstream of it, which `topo_sort` also leaves unsorted.
/// - [`GraphError::MissingOutput`]: none; the output is not a box.
pub(crate) fn nodes_at_fault(graph: &NodeGraph, err: &GraphError) -> Vec<NodeId> {
    match err {
        GraphError::UnknownNode(missing) => graph
            .nodes
            .iter()
            .filter(|n| upstream_of(n).any(|up| up == *missing))
            .map(|n| n.id)
            .collect(),
        GraphError::DuplicateId(id) => graph
            .nodes
            .iter()
            .filter(|n| n.id == *id)
            .map(|n| n.id)
            .collect(),
        GraphError::Cycle => cycle_nodes(graph),
        GraphError::MissingOutput(_) => Vec::new(),
    }
}

/// The ids `node` takes a wire from.
fn upstream_of(node: &GraphNode) -> impl Iterator<Item = NodeId> + '_ {
    node.inputs.values().flatten().filter_map(|c| match c {
        Connection::Node { id, .. } => Some(*id),
        Connection::Constant { .. } => None,
    })
}

/// Every node that can reach itself along the wires: the nodes of the
/// graph's loops, in node order. A depth-first walk downstream from each
/// node; graphs are small enough that the walk from every node is cheap.
fn cycle_nodes(graph: &NodeGraph) -> Vec<NodeId> {
    let mut downstream: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for node in &graph.nodes {
        for up in upstream_of(node) {
            downstream.entry(up).or_default().push(node.id);
        }
    }
    graph
        .nodes
        .iter()
        .map(|n| n.id)
        .filter(|&start| {
            let mut seen = HashSet::new();
            let mut stack: Vec<NodeId> = downstream.get(&start).cloned().unwrap_or_default();
            while let Some(id) = stack.pop() {
                if id == start {
                    return true;
                }
                if seen.insert(id)
                    && let Some(next) = downstream.get(&id)
                {
                    stack.extend(next);
                }
            }
            false
        })
        .collect()
}

/// `"#2 Lowpass"`: a node as the canvas titles it, or `"#2"` when no node
/// has that id.
fn node_name(graph: &NodeGraph, id: NodeId) -> String {
    match graph.nodes.iter().find(|n| n.id == id) {
        Some(node) => format!("#{} {}", id.0, node_kind_label(&node.kind)),
        None => format!("#{}", id.0),
    }
}

/// `err` in words that name the nodes it is about, for the canvas's
/// validity line and the audition strip's error. `GraphError`'s own
/// `Display` ("graph contains a cycle") says what kind of fault it is and
/// leaves the user to find it.
pub(crate) fn describe_graph_error(graph: &NodeGraph, err: &GraphError) -> String {
    match err {
        GraphError::UnknownNode(missing) => match nodes_at_fault(graph, err).first() {
            Some(&holder) => format!(
                "{} is wired from #{}, which is not in the patch",
                node_name(graph, holder),
                missing.0
            ),
            None => format!(
                "A wire comes from #{}, which is not in the patch",
                missing.0
            ),
        },
        GraphError::MissingOutput(id) => format!(
            "The output is #{}, which is not in the patch: pick another in Output",
            id.0
        ),
        GraphError::DuplicateId(id) => format!(
            "{} nodes are numbered #{}; each node needs its own number",
            nodes_at_fault(graph, err).len(),
            id.0
        ),
        GraphError::Cycle => {
            let names: Vec<String> = cycle_nodes(graph)
                .into_iter()
                .map(|id| node_name(graph, id))
                .collect();
            format!(
                "{} feed each other in a loop, which cannot be baked: remove one of \
                 their wires",
                join_names(&names)
            )
        }
    }
}

/// `"a"`, `"a and b"`, `"a, b and c"`; past four names, the first three and
/// a count of the rest.
fn join_names(names: &[String]) -> String {
    const SHOWN: usize = 3;
    match names {
        [] => "Some nodes".to_string(),
        [one] => one.clone(),
        [init @ .., last] if names.len() <= SHOWN + 1 => {
            format!("{} and {last}", init.join(", "))
        }
        _ => format!(
            "{} and {} more",
            names[..SHOWN].join(", "),
            names.len() - SHOWN
        ),
    }
}

/// Remove `target` and every connection that referenced it; reassign the
/// graph output if it pointed at the removed node.
fn delete_node(patch: &mut AudioPatch, target: NodeId) {
    patch.graph.nodes.retain(|n| n.id != target);
    for n in &mut patch.graph.nodes {
        for conns in n.inputs.values_mut() {
            conns.retain(|c| !matches!(c, Connection::Node { id, .. } if *id == target));
        }
        n.inputs.retain(|_, v| !v.is_empty());
    }
    if patch.graph.output == target
        && let Some(first) = patch.graph.nodes.first()
    {
        patch.graph.output = first.id;
    }
}

/// Deferred structural edit, applied after the node-drawing loop releases its
/// borrow on `patch.graph.nodes`.
enum Action {
    Select(NodeId),
    Move(NodeId, Vec2),
    /// Finish a wire drag started at `from`'s output, landing at `at`.
    CompleteWire {
        from: NodeId,
        at: Pos2,
    },
}

/// Draw and edit a whole [`AudioPatch`] as a pannable/zoomable node graph.
///
/// `state` holds layout + view across frames; pass the same instance each
/// frame.  Returns an [`EditorResponse`] whose `rebake` flag is set when an
/// edit is committed (param drag stops, a wire/node/output changes), so the
/// host knows when to re-bake audio.
///
/// # The canvas claims the rest of the `Ui` — draw it last
///
/// Below its toolbar the canvas is an [`egui::Scene`], which allocates all of
/// `ui.available_size_before_wrap()`. Anything that must stay visible — an
/// audition strip, a status line, a waveform — goes **before** this call, and
/// side content goes in panels (`egui::Panel::left` for it, then
/// `egui::CentralPanel` holding the canvas).
///
/// Content drawn *after* the canvas is laid out below the space it was given.
/// In a panel or a fixed area that only hides it. In an [`egui::Window`] it
/// also makes the window grow every frame: the window's `Resize` keeps
/// `desired_size.max(last_content_size)` while nobody is dragging its edge
/// (egui 0.35, `containers/resize.rs`) and never gives height back, so the
/// content is always taller than the window and the window keeps growing
/// until it meets its `constrain_to` rect. The trailing content stays
/// out of sight the whole time, and resizing by hand does not help. Overlands
/// shipped exactly that: an audition row after the canvas, in a pop-out that
/// went from 670 px to the screen's height in 30 frames (Overlands #1327).
///
/// The `host_window` example lays out both editors inside windows the way
/// that works, and `tests::a_window_whose_last_item_is_the_canvas_keeps_its_size`
/// holds this rule and its control.
pub fn audio_patch_canvas(
    ui: &mut egui::Ui,
    patch: &mut AudioPatch,
    state: &mut PatchEditorState,
    id: Id,
) -> EditorResponse {
    let style = editor_style(ui);
    let mut res = EditorResponse::NONE;
    res.merge(toolbar(ui, patch, state, &style));
    res.merge(json_io(ui, patch, &mut state.json, id.with("patch_json")));
    state.ensure_layout(patch);

    // The scene takes the rest of the `Ui`, and draws on a layer over this
    // one: the ground goes under it, where it will be.
    ui.painter()
        .rect_filled(ui.available_rect_before_wrap(), 0.0, style.canvas_ground);
    let mut scene_rect = state.scene_rect;
    let scene = egui::Scene::new().zoom_range(egui::Rangef::new(0.2, 2.0));
    let inner = scene.show(ui, &mut scene_rect, |ui| {
        canvas_contents(ui, patch, state, id, &style)
    });
    state.scene_rect = scene_rect;
    res.merge(inner.inner);
    res
}

/// Fixed toolbar above the canvas: add / delete nodes, pick the output, fit
/// the view, and a live validity readout. Its buttons say what they do in
/// words (#58).
fn toolbar(
    ui: &mut egui::Ui,
    patch: &mut AudioPatch,
    state: &mut PatchEditorState,
    style: &EditorStyle,
) -> EditorResponse {
    let mut res = EditorResponse::NONE;
    ui.horizontal_wrapped(|ui| {
        if ui.button("Add node").clicked() {
            let new_id = NodeId(
                patch
                    .graph
                    .nodes
                    .iter()
                    .map(|n| n.id.0)
                    .max()
                    .map_or(0, |m| m + 1),
            );
            patch.graph.nodes.push(GraphNode {
                id: new_id,
                kind: NodeKind::Sine(SineOsc::default()),
                inputs: Default::default(),
            });
            // Drop it near the centre of the current view so it's visible.
            state.positions.insert(new_id, state.scene_rect.center());
            state.selected = Some(new_id);
            res.changed = true;
            res.rebake = true;
        }

        let can_delete = state.selected.is_some() && patch.graph.nodes.len() > 1;
        if ui
            .add_enabled(can_delete, egui::Button::new("Delete"))
            .on_hover_text("Remove the selected node and any wires into it")
            .clicked()
            && let Some(sel) = state.selected
        {
            delete_node(patch, sel);
            state.positions.remove(&sel);
            state.selected = None;
            res.changed = true;
            res.rebake = true;
        }

        ui.separator();
        ui.label("Output:");
        let ids: Vec<NodeId> = patch.graph.nodes.iter().map(|n| n.id).collect();
        egui::ComboBox::from_id_salt("canvas_output_select")
            .selected_text(format!("#{}", patch.graph.output.0))
            .show_ui(ui, |ui| {
                for nid in ids {
                    if ui
                        .selectable_label(nid == patch.graph.output, format!("#{}", nid.0))
                        .clicked()
                    {
                        patch.graph.output = nid;
                        res.changed = true;
                        res.rebake = true;
                    }
                }
            });

        ui.separator();
        if ui
            .button("Fit view")
            .on_hover_text("Pan and zoom so every node is in view")
            .clicked()
        {
            // A zero-size rect makes Scene auto-fit to the content next frame.
            state.scene_rect = Rect::ZERO;
        }
    });

    ui.horizontal_wrapped(|ui| {
        if ui
            .button("Mutate")
            .on_hover_text("Nudge every node's parameters via symbios-genetics")
            .clicked()
        {
            mutate_patch(patch, &mut fresh_rng(), state.mutate_rate);
            res.changed = true;
            res.rebake = true;
        }
        ui.add(egui::Slider::new(&mut state.mutate_rate, 0.0..=1.0).text("rate"));
        if ui
            .button("Reroll seed")
            .on_hover_text(format!(
                "The seed is {}. Reroll it to re-randomise noise and random LFOs",
                patch.seed
            ))
            .clicked()
        {
            randomize_seed(patch, &mut fresh_rng());
            res.changed = true;
            res.rebake = true;
        }
    });

    // The check and the cross are Overlands' `affordances::CHECK` and
    // `CROSS`, its glyphs for valid and failed.
    match topo_sort(&patch.graph) {
        Ok(order) => {
            ui.colored_label(
                style.ok,
                format!("\u{2714} valid graph \u{2014} {} nodes", order.len()),
            );
        }
        Err(e) => {
            // Named, and outlined on the canvas below (#57).
            ui.colored_label(
                style.error,
                format!("\u{2716} {}", describe_graph_error(&patch.graph, &e)),
            );
        }
    }
    res
}

/// Everything painted inside the [`egui::Scene`] (scene-local coordinates).
fn canvas_contents(
    ui: &mut egui::Ui,
    patch: &mut AudioPatch,
    state: &mut PatchEditorState,
    id: Id,
    style: &EditorStyle,
) -> EditorResponse {
    let mut res = EditorResponse::NONE;
    let mut actions: Vec<Action> = Vec::new();

    // This frame's ports and output dots are drawn from, and kept in, the
    // state.
    state.ports.clear();
    state.outputs.clear();
    state.boxes.clear();
    let mut titles: HashMap<NodeId, &'static str> = HashMap::new();
    // The wire being dragged: from which node, where the pointer is, and the
    // output port's response, which owns the target's tooltip.
    let mut dragging: Option<(NodeId, Pos2, egui::Response)> = None;

    let output_id = patch.graph.output;
    let selected = state.selected;
    let mutate_rate = state.mutate_rate;
    // The boxes a broken graph is broken at, outlined in the host's error
    // colour over the selection and output borders: they are what has to
    // change before anything bakes.
    let at_fault: HashSet<NodeId> = match topo_sort(&patch.graph) {
        Ok(_) => HashSet::new(),
        Err(e) => nodes_at_fault(&patch.graph, &e).into_iter().collect(),
    };
    let error_stroke = Stroke::new(2.0, style.error);

    // Reserve a shape slot up front; we backfill it with the wires after node
    // rects are known, so wires render *behind* the node boxes.
    let painter = ui.painter().clone();
    let wire_idx = painter.add(egui::Shape::Noop);

    // --- nodes ----------------------------------------------------------
    for node in &mut patch.graph.nodes {
        let nid = node.id;
        let pos = state
            .positions
            .get(&nid)
            .copied()
            .unwrap_or(Pos2::new(40.0, 40.0));

        let stroke = if at_fault.contains(&nid) {
            error_stroke
        } else if selected == Some(nid) {
            Stroke::new(2.0, style.node_selected)
        } else if output_id == nid {
            Stroke::new(2.0, style.node_output)
        } else {
            Stroke::new(1.0, style.node_stroke)
        };

        let mut child = ui.new_child(
            UiBuilder::new()
                .max_rect(Rect::from_min_size(pos, Vec2::new(NODE_WIDTH, 10.0)))
                .id_salt(("patch_node", nid.0))
                .layout(Layout::top_down(Align::Min)),
        );
        child.set_width(NODE_WIDTH);

        let frame = egui::Frame::group(child.style())
            .fill(style.node_fill)
            .stroke(stroke);
        let fr = frame.show(&mut child, |ui| {
            ui.set_width(NODE_WIDTH);
            // Title bar: drag to move, click to select, Mutate to mutate
            // this node.
            let title_row = ui.horizontal(|ui| {
                let title = format!("#{}  {}", nid.0, node_kind_label(&node.kind));
                let title_resp = ui.add(
                    egui::Label::new(egui::RichText::new(title).strong().color(style.node_title))
                        .sense(Sense::click_and_drag()),
                );
                if title_resp.dragged() {
                    actions.push(Action::Move(nid, title_resp.drag_delta()));
                }
                if title_resp.clicked() {
                    actions.push(Action::Select(nid));
                }
                if ui
                    .small_button("Mutate")
                    .on_hover_text("Mutate this node")
                    .clicked()
                {
                    mutate_node_kind(&mut node.kind, &mut fresh_rng(), mutate_rate);
                    res.changed = true;
                    res.rebake = true;
                }
            });
            ui.separator();
            res.merge(node_kind_editor(ui, &mut node.kind, Id::new(("nk", nid.0))));
            let (conn_res, rows) = connection_editor(ui, node);
            res.merge(conn_res);
            (title_row.response.rect, rows)
        });

        let rect = fr.response.rect;
        let (title_rect, rows) = fr.inner;
        state.boxes.push((nid, rect));
        titles.insert(nid, node_kind_label(&node.kind));
        // The output leaves from the title row, where the node is named.
        let oa = Pos2::new(rect.right(), title_rect.center().y);
        state.outputs.insert(nid, oa);
        for PortRow {
            port,
            name_row,
            section,
        } in rows
        {
            let connected = node.inputs.get(&port).is_some_and(|c| !c.is_empty());
            state.ports.push(PortGeom {
                node: nid,
                port,
                dot: Pos2::new(rect.left(), name_row.center().y),
                row: Rect::from_x_y_ranges(rect.x_range(), section.y_range()),
                connected,
            });
        }

        // This node's dots, painted before the next box so a box that lies
        // over this one covers them too.
        painter.circle_filled(oa, PORT_RADIUS, style.port);
        for p in state.ports.iter().filter(|p| p.node == nid) {
            if p.connected {
                painter.circle_filled(p.dot, PORT_RADIUS, style.port);
            } else {
                // A ring: nothing drives this port yet.
                painter.circle(
                    p.dot,
                    PORT_RADIUS,
                    style.node_fill,
                    Stroke::new(1.5, style.port),
                );
            }
        }

        // Output port: drag to start a wire.
        let out_resp = ui.interact(
            Rect::from_center_size(oa, Vec2::splat(PORT_RADIUS * 2.5)),
            id.with(("outport", nid.0)),
            Sense::drag(),
        );
        // Named for assistive tech and for scripted input (host_window --drag).
        out_resp.widget_info(|| {
            egui::WidgetInfo::labeled(
                egui::WidgetType::Other,
                true,
                format!("Output of #{} {}", nid.0, node_kind_label(&node.kind)),
            )
        });
        if out_resp.dragged()
            && let Some(p) = out_resp.interact_pointer_pos()
        {
            dragging = Some((nid, p, out_resp.clone()));
        }
        if out_resp.drag_stopped()
            && let Some(p) = out_resp.interact_pointer_pos()
        {
            actions.push(Action::CompleteWire { from: nid, at: p });
        }
    }

    // --- wires (behind nodes via the reserved slot) ---------------------
    let mut wires: Vec<egui::Shape> = Vec::new();
    for node in &patch.graph.nodes {
        for (port, conns) in &node.inputs {
            // Every port with connections has a row, so this finds its dot;
            // the box's left centre is only a last resort.
            let dst = state
                .ports
                .iter()
                .find(|p| p.node == node.id && p.port == *port)
                .map(|p| p.dot)
                .or_else(|| {
                    state
                        .boxes
                        .iter()
                        .find(|(id, _)| *id == node.id)
                        .map(|(_, r)| Pos2::new(r.left(), r.center().y))
                });
            let Some(dst) = dst else { continue };
            for c in conns {
                if let Connection::Node { id: src, .. } = c
                    && let Some(src_pos) = state.outputs.get(src)
                {
                    wires.push(wire_shape(*src_pos, dst, style.wire));
                }
            }
        }
    }
    painter.set(wire_idx, egui::Shape::Vec(wires));

    // --- the wire being dragged, over the boxes, and where it would land --
    if let Some((from, at, out_resp)) = dragging
        && let Some(&src) = state.outputs.get(&from)
    {
        let target = drop_target(&state.ports, &state.boxes, from, at);
        let active = style.wire_active;
        if let Some(target) = target {
            painter.rect(
                target.row,
                3.0,
                active.gamma_multiply(DROP_ROW_ALPHA),
                Stroke::new(1.0, active),
                egui::StrokeKind::Outside,
            );
            painter.circle_filled(target.dot, PORT_RADIUS + 1.5, active);
        }
        // The wire ends on the dot it would connect to, so the user sees the
        // connection before letting go; over no port it follows the pointer.
        let end = target.map_or(at, |t| t.dot);
        painter.add(wire_shape(src, end, active));
        if let Some(target) = target {
            // A tooltip is drawn in screen space, so the name stays legible
            // at any zoom of the canvas.
            let kind = titles.get(&target.node).copied().unwrap_or_default();
            let text = format!(
                "\u{27A1} #{} {kind} \u{00B7} {}",
                target.node.0, target.port
            );
            // Never wrapped: an auto-sized area offers the width of its last
            // pass, so a name longer than the last one would wrap and the
            // box would ratchet narrower as the pointer crosses rows.
            egui::Tooltip::for_widget(&out_resp)
                .at_pointer()
                .show(|ui| ui.add(egui::Label::new(text).extend()));
        }
    }

    // --- apply deferred structural edits -------------------------------
    for action in actions {
        match action {
            Action::Select(nid) => state.selected = Some(nid),
            Action::Move(nid, delta) => {
                *state.positions.entry(nid).or_default() += delta;
            }
            Action::CompleteWire { from, at } => {
                if let Some(target) = drop_target(&state.ports, &state.boxes, from, at)
                    && let Some(n) = patch.graph.nodes.iter_mut().find(|n| n.id == target.node)
                {
                    n.inputs
                        .entry(target.port.clone())
                        .or_default()
                        .push(Connection::from_node(from));
                    res.changed = true;
                    res.rebake = true;
                }
            }
        }
    }

    // Claim the bounding area so Scene's "reset view" can fit the content.
    if let Some(bounds) = state
        .boxes
        .iter()
        .map(|(_, r)| *r)
        .reduce(|a, b| a.union(b))
    {
        ui.allocate_rect(bounds.expand(60.0), Sense::hover());
    }

    res
}

/// Where one port's rows landed in a node box.
struct PortRow {
    port: String,
    /// The row that starts with the port's name.
    name_row: Rect,
    /// The name row and the rows of the port's connections below it.
    section: Rect,
}

/// The "Inputs" section inside a node box: per-connection amount/value editing,
/// per-port "add constant", and per-connection delete. Returns where each
/// port's rows went, so the canvas can put the port's dot beside its name.
fn connection_editor(ui: &mut egui::Ui, node: &mut GraphNode) -> (EditorResponse, Vec<PortRow>) {
    let mut res = EditorResponse::NONE;
    let mut rows = Vec::new();

    // Canonical ports first, then any extra ports already present (e.g. from
    // loaded JSON) so nothing wired is hidden.
    let mut ports: Vec<String> = input_ports(&node.kind)
        .iter()
        .map(|s| s.to_string())
        .collect();
    for k in node.inputs.keys() {
        if !ports.contains(k) {
            ports.push(k.clone());
        }
    }
    if ports.is_empty() {
        return (res, rows);
    }

    ui.separator();
    ui.label(egui::RichText::new("Inputs").weak());

    let mut to_delete: Vec<(String, usize)> = Vec::new();
    let mut to_add_const: Vec<String> = Vec::new();

    for port in &ports {
        let name_row = ui
            .horizontal(|ui| {
                ui.label(format!("{port}:"));
                if ui.small_button("Add constant").clicked() {
                    to_add_const.push(port.clone());
                    res.changed = true;
                    res.rebake = true;
                }
            })
            .response
            .rect;
        let mut section = name_row;
        if let Some(conns) = node.inputs.get_mut(port) {
            for (i, c) in conns.iter_mut().enumerate() {
                let row = ui.horizontal(|ui| {
                    match c {
                        Connection::Node { id, amount } => {
                            ui.label(format!("  \u{2B05} #{}", id.0));
                            let r = ui.add(egui::DragValue::new(amount).speed(0.05).prefix("amt "));
                            res.changed |= r.changed();
                            res.rebake |= r.drag_stopped() || (r.changed() && !r.dragged());
                        }
                        Connection::Constant { value } => {
                            ui.label("  const");
                            let r = ui.add(egui::DragValue::new(value).speed(0.01));
                            res.changed |= r.changed();
                            res.rebake |= r.drag_stopped() || (r.changed() && !r.dragged());
                        }
                    }
                    if ui.small_button("\u{2716}").clicked() {
                        to_delete.push((port.clone(), i));
                        res.changed = true;
                        res.rebake = true;
                    }
                });
                section = section.union(row.response.rect);
            }
        }
        rows.push(PortRow {
            port: port.clone(),
            name_row,
            section,
        });
    }

    // Remove highest indices first so earlier removals don't shift them.
    to_delete.sort_by_key(|entry| std::cmp::Reverse(entry.1));
    for (port, idx) in to_delete {
        if let Some(v) = node.inputs.get_mut(&port) {
            if idx < v.len() {
                v.remove(idx);
            }
            if v.is_empty() {
                node.inputs.remove(&port);
            }
        }
    }
    for port in to_add_const {
        node.inputs
            .entry(port)
            .or_default()
            .push(Connection::constant(0.0));
    }

    (res, rows)
}

/// A cubic-bezier wire from `a` to `b`, sampled to a polyline with horizontal
/// control handles (the classic node-editor S-curve).
fn wire_shape(a: Pos2, b: Pos2, color: Color32) -> egui::Shape {
    let handle = (b.x - a.x).abs().max(40.0) * 0.5;
    let c1 = Pos2::new(a.x + handle, a.y);
    let c2 = Pos2::new(b.x - handle, b.y);
    const SEGMENTS: usize = 18;
    let mut pts = Vec::with_capacity(SEGMENTS + 1);
    for i in 0..=SEGMENTS {
        let t = i as f32 / SEGMENTS as f32;
        pts.push(cubic_bezier(a, c1, c2, b, t));
    }
    egui::Shape::line(pts, Stroke::new(2.0, color))
}

fn cubic_bezier(p0: Pos2, p1: Pos2, p2: Pos2, p3: Pos2, t: f32) -> Pos2 {
    let u = 1.0 - t;
    let (w0, w1, w2, w3) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
    Pos2::new(
        w0 * p0.x + w1 * p1.x + w2 * p2.x + w3 * p3.x,
        w0 * p0.y + w1 * p1.y + w2 * p2.y + w3 * p3.y,
    )
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn node(id: u32, kind: NodeKind) -> GraphNode {
        GraphNode {
            id: NodeId(id),
            kind,
            inputs: BTreeMap::new(),
        }
    }

    fn three_node_patch() -> AudioPatch {
        // 0:sine -> 2:lowpass("in"), 1:lfo -> 2:lowpass("cutoff_hz"); output 2.
        let mut filter = node(2, NodeKind::BiquadLowpass(Default::default()));
        filter
            .inputs
            .insert("in".into(), vec![Connection::from_node(NodeId(0))]);
        filter.inputs.insert(
            "cutoff_hz".into(),
            vec![Connection::modulation(NodeId(1), 500.0)],
        );
        AudioPatch {
            seed: 0,
            graph: crate::patch::NodeGraph {
                nodes: vec![
                    node(0, NodeKind::Sine(SineOsc::default())),
                    node(1, NodeKind::Lfo(Default::default())),
                    filter,
                ],
                output: NodeId(2),
            },
        }
    }

    #[test]
    fn delete_node_prunes_connections_and_reassigns_output() {
        let mut patch = three_node_patch();
        delete_node(&mut patch, NodeId(0));
        // Node 0 gone.
        assert!(patch.graph.nodes.iter().all(|n| n.id != NodeId(0)));
        // The "in" port on the filter referenced node 0 — it's now pruned and,
        // being empty, the port key is gone.
        let filter = patch
            .graph
            .nodes
            .iter()
            .find(|n| n.id == NodeId(2))
            .unwrap();
        assert!(!filter.inputs.contains_key("in"));
        // The cutoff_hz connection (to node 1) survives.
        assert!(filter.inputs.contains_key("cutoff_hz"));
    }

    #[test]
    fn delete_output_node_reassigns_output() {
        let mut patch = three_node_patch();
        delete_node(&mut patch, NodeId(2));
        assert_ne!(patch.graph.output, NodeId(2));
        assert!(patch.graph.nodes.iter().any(|n| n.id == patch.graph.output));
    }

    #[test]
    fn auto_layout_assigns_topological_columns() {
        let patch = three_node_patch();
        let depths = compute_depths(&patch);
        // sine (0) and lfo (1) are sources at column 0; the filter (2) is
        // downstream at column 1.
        assert_eq!(depths[&NodeId(0)], 0);
        assert_eq!(depths[&NodeId(1)], 0);
        assert_eq!(depths[&NodeId(2)], 1);
    }

    #[test]
    fn ensure_layout_only_fills_missing_positions() {
        let patch = three_node_patch();
        let mut state = PatchEditorState::default();
        state.positions.insert(NodeId(0), Pos2::new(123.0, 456.0));
        state.ensure_layout(&patch);
        // Pre-set position untouched; the other two got laid out.
        assert_eq!(state.positions[&NodeId(0)], Pos2::new(123.0, 456.0));
        assert!(state.positions.contains_key(&NodeId(1)));
        assert!(state.positions.contains_key(&NodeId(2)));
    }

    /// A port of `node` whose dot is at `dot` and whose row spans `row`.
    fn port(node: u32, name: &str, dot: Pos2, row: Rect) -> PortGeom {
        PortGeom {
            node: NodeId(node),
            port: name.into(),
            dot,
            row,
            connected: false,
        }
    }

    #[test]
    fn a_row_under_the_pointer_is_the_target_however_far_its_dot_is() {
        let row = Rect::from_min_max(Pos2::new(100.0, 200.0), Pos2::new(310.0, 222.0));
        let ports = [port(5, "q", Pos2::new(100.0, 211.0), row)];
        let boxes = [(NodeId(5), row.expand2(Vec2::new(0.0, 80.0)))];
        let far_right = Pos2::new(300.0, 211.0);
        assert!(far_right.distance(ports[0].dot) > SNAP_DIST);
        assert_eq!(
            drop_target(&ports, &boxes, NodeId(1), far_right).map(|p| p.port.as_str()),
            Some("q")
        );
    }

    #[test]
    fn off_every_row_a_dot_within_the_snap_radius_is_the_target() {
        let row = Rect::from_min_max(Pos2::new(100.0, 200.0), Pos2::new(310.0, 222.0));
        let ports = [port(5, "in", Pos2::new(100.0, 211.0), row)];
        let boxes = [(NodeId(5), row.expand2(Vec2::new(0.0, 80.0)))];
        // Left of the box, just inside the snap radius of the dot.
        let near = Pos2::new(100.0 - SNAP_DIST + 1.0, 211.0);
        assert!(!row.contains(near));
        assert_eq!(
            drop_target(&ports, &boxes, NodeId(1), near).map(|p| p.port.as_str()),
            Some("in")
        );
        assert_eq!(
            drop_target(&ports, &boxes, NodeId(1), Pos2::new(500.0, 500.0)),
            None
        );
    }

    #[test]
    fn only_the_box_on_top_can_be_the_target_and_never_the_wires_own() {
        let row = Rect::from_min_max(Pos2::new(100.0, 200.0), Pos2::new(310.0, 222.0));
        let at = Pos2::new(200.0, 211.0);
        let tall = row.expand2(Vec2::new(0.0, 80.0));
        // Two boxes overlap here: the later one is drawn over the earlier.
        let ports = [
            port(5, "under", Pos2::new(100.0, 211.0), row),
            port(6, "over", Pos2::new(100.0, 211.0), row),
        ];
        let boxes = [(NodeId(5), tall), (NodeId(6), tall)];
        assert_eq!(
            drop_target(&ports, &boxes, NodeId(1), at).map(|p| p.port.as_str()),
            Some("over")
        );
        // Out of node 6 and over its own box: nothing, not the row under it.
        assert_eq!(drop_target(&ports, &boxes, NodeId(6), at), None);

        // Node 7 has no ports and covers node 5's row: the row is out of
        // sight, so it is not a target either.
        let covered = [port(5, "hidden", Pos2::new(100.0, 211.0), row)];
        let boxes = [
            (NodeId(5), tall),
            (NodeId(7), Rect::from_center_size(at, Vec2::splat(60.0))),
        ];
        assert_eq!(drop_target(&covered, &boxes, NodeId(1), at), None);
    }

    /// What the canvas keeps in its state after a frame: every port of every
    /// kind with its dot on the box's left edge and inside its own row.
    #[test]
    fn the_state_keeps_each_ports_dot_inside_its_row() {
        for kind in NodeKind::defaults() {
            let name = node_kind_label(&kind);
            let canvas = Canvas::new(one_node(kind.clone()));
            let state = &canvas.state;
            let names: Vec<&str> = state.ports.iter().map(|p| p.port.as_str()).collect();
            assert_eq!(
                names,
                input_ports(&kind),
                "{name}: one entry per port, in order"
            );
            for p in &state.ports {
                assert!(p.row.contains(p.dot), "{name}.{}: {p:?}", p.port);
                assert_eq!(p.dot.x, p.row.left(), "{name}.{}: on the box edge", p.port);
            }
            assert!(
                state.outputs.contains_key(&NodeId(0)),
                "{name}: an output dot"
            );
        }
    }

    #[test]
    fn input_ports_match_node_read_ports() {
        assert_eq!(
            input_ports(&NodeKind::Sine(SineOsc::default())),
            &["freq", "amplitude"]
        );
        assert_eq!(
            input_ports(&NodeKind::BiquadBandpass(Default::default())),
            &["in", "center_hz", "q"]
        );
        assert!(input_ports(&NodeKind::Lfo(Default::default())).is_empty());
    }

    /// Headless render smoke test: drive the canvas through a real
    /// `egui::Context` for a few frames (no display needed) to catch panics in
    /// the egui path — Scene transforms, `new_child`, the reserved-shape wire
    /// trick, id collisions. egui lays out purely on the CPU.
    #[test]
    fn canvas_renders_headless_without_panicking() {
        let ctx = egui::Context::default();
        let mut patch = three_node_patch();
        let mut state = PatchEditorState::default();
        for _ in 0..3 {
            let _ = ctx.run_ui(egui::RawInput::default(), |root| {
                egui::CentralPanel::default().show(root, |ui| {
                    audio_patch_canvas(ui, &mut patch, &mut state, egui::Id::new("smoke"));
                });
            });
        }
    }

    /// A host's window around the canvas, drawn for `frames` frames: a row
    /// of buttons above the canvas and, when `row_after_canvas`, the same
    /// row again below it. Resizable and constrained to a 1920x1080 screen,
    /// the way a host shows a pop-out editor. Returns the window's height on
    /// every frame and the canvas region's height on the last one.
    fn window_around_the_canvas(frames: usize, row_after_canvas: bool) -> (Vec<f32>, f32) {
        let ctx = egui::Context::default();
        let screen = Rect::from_min_size(Pos2::ZERO, Vec2::new(1920.0, 1080.0));
        let mut patch = three_node_patch();
        let mut state = PatchEditorState::default();
        let mut heights = Vec::with_capacity(frames);
        let mut canvas = 0.0;
        for _ in 0..frames {
            let input = egui::RawInput {
                screen_rect: Some(screen),
                ..Default::default()
            };
            let _ = ctx.run_ui(input, |root| {
                let strip = |ui: &mut egui::Ui| {
                    ui.horizontal(|ui| {
                        let _ = ui.button("Audition");
                        let _ = ui.button("Stop");
                    });
                };
                let shown = egui::Window::new("host")
                    .default_pos(Pos2::new(40.0, 40.0))
                    .default_size(Vec2::new(900.0, 640.0))
                    .constrain_to(screen)
                    .resizable(true)
                    .show(root.ctx(), |ui| {
                        strip(ui);
                        ui.separator();
                        let top = ui.cursor().top();
                        audio_patch_canvas(ui, &mut patch, &mut state, Id::new("host_canvas"));
                        canvas = ui.min_rect().bottom() - top;
                        if row_after_canvas {
                            ui.separator();
                            strip(ui);
                        }
                    });
                if let Some(shown) = shown {
                    heights.push(shown.response.rect.height());
                }
            });
        }
        (heights, canvas)
    }

    /// The canvas takes whatever is left, so as the LAST thing in a window
    /// it fits the window exactly and the window keeps the size it opened
    /// at (Overlands #1327, #54 here).
    #[test]
    fn a_window_whose_last_item_is_the_canvas_keeps_its_size() {
        let (heights, canvas) = window_around_the_canvas(30, false);
        assert_eq!(heights.len(), 30, "the window was shown every frame");
        // From the third frame on: a window's first frame is a sizing pass.
        let settled = heights[2];
        for (frame, h) in heights.iter().enumerate().skip(2) {
            assert!(
                (h - settled).abs() < 0.5,
                "frame {}: the window is {h:.1} px tall, {settled:.1} at frame 3",
                frame + 1
            );
        }
        assert!(
            settled < 700.0,
            "the window opened at 640 and is {settled:.1} px tall"
        );
        assert!(
            canvas > 400.0,
            "the canvas got only {canvas:.1} px of a 640 px window"
        );
    }

    /// The control: the same window with the row moved BELOW the canvas
    /// grows every frame until it hits the screen, which is the defect the
    /// rule above exists for. If egui ever stops growing windows this way,
    /// this test fails, and the docs on [`audio_patch_canvas`] need
    /// rewriting.
    #[test]
    fn a_row_after_the_canvas_grows_the_window_to_its_constraint() {
        let (heights, _) = window_around_the_canvas(30, true);
        let (first, last) = (heights[2], heights[heights.len() - 1]);
        assert!(
            last > first + 300.0,
            "the window was supposed to grow and went {first:.1} -> {last:.1}"
        );
        assert!(
            last >= 1080.0 - 41.0,
            "the window stopped at {last:.1}, short of the 1080 px screen"
        );
    }

    // -----------------------------------------------------------------------
    // Ports on their rows, and the drop target (#56, Overlands #1329)
    // -----------------------------------------------------------------------
    //
    // These read what the canvas showed, not what it meant to show: labels
    // from the AccessKit tree, dots, boxes and wires from the painted shapes,
    // so they held the old layout to the same standard. AccessKit bounds are
    // in canvas units and shapes in screen points, so labels go through the
    // canvas layer's transform.

    use egui::accesskit;
    use egui::emath::TSTransform;

    use crate::ui::style::tests::{AA, distinct_style};
    use crate::ui::style::{EditorStyle, set_editor_style, set_style};
    use crate::ui::test_paint::{
        button_labels, colours, contrast_on, glyph_labels, shapes, text_painted,
    };

    /// The canvas on a headless context, large enough to show a few nodes.
    ///
    /// [`Canvas::new`] sets [`distinct_style`] on the context, so every role
    /// the canvas paints has a colour nothing else does and a test can find
    /// a node box by its fill. [`Canvas::themed`] sets no style, so the
    /// canvas falls back to the theme's.
    struct Canvas {
        ctx: egui::Context,
        patch: AudioPatch,
        state: PatchEditorState,
        out: egui::FullOutput,
    }

    impl Canvas {
        fn new(patch: AudioPatch) -> Self {
            let ctx = egui::Context::default();
            set_editor_style(&ctx, distinct_style());
            Self::on(ctx, patch)
        }

        /// The canvas under `visuals`, with no style set.
        fn themed(patch: AudioPatch, visuals: egui::Visuals) -> Self {
            let ctx = egui::Context::default();
            ctx.set_visuals(visuals);
            Self::on(ctx, patch)
        }

        fn on(ctx: egui::Context, patch: AudioPatch) -> Self {
            ctx.enable_accesskit();
            let mut canvas = Self {
                ctx,
                patch,
                state: PatchEditorState::default(),
                out: egui::FullOutput::default(),
            };
            for _ in 0..3 {
                canvas.frame(Vec::new());
            }
            canvas
        }

        fn frame(&mut self, events: Vec<egui::Event>) -> EditorResponse {
            let Self {
                ctx,
                patch,
                state,
                out,
            } = self;
            let mut res = EditorResponse::NONE;
            let input = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1600.0, 1200.0))),
                events,
                ..Default::default()
            };
            *out = ctx.run_ui(input, |root| {
                egui::CentralPanel::default().show(root, |ui| {
                    res = audio_patch_canvas(ui, patch, state, Id::new("geometry"));
                });
            });
            res
        }

        /// Canvas units to screen points: the canvas is the only transformed
        /// layer here.
        fn to_screen(&self) -> TSTransform {
            let transforms = self.ctx.memory(|m| m.to_global.clone());
            assert_eq!(transforms.len(), 1, "one canvas, one layer transform");
            *transforms.values().next().expect("one transform")
        }

        /// Every label's text and its rect on screen.
        fn labels(&self) -> Vec<(String, Rect)> {
            let to_screen = self.to_screen();
            self.out
                .platform_output
                .accesskit_update
                .iter()
                .flat_map(|update| &update.nodes)
                .filter(|(_, n)| n.role() == accesskit::Role::Label)
                .filter_map(|(_, n)| {
                    let b = n.bounds()?;
                    let rect = Rect::from_min_max(
                        Pos2::new(b.x0 as f32, b.y0 as f32),
                        Pos2::new(b.x1 as f32, b.y1 as f32),
                    );
                    Some((n.value()?.to_owned(), to_screen * rect))
                })
                .collect()
        }

        /// The one label reading exactly `text`.
        fn label(&self, text: &str) -> Rect {
            let found: Vec<Rect> = self
                .labels()
                .into_iter()
                .filter(|(t, _)| t == text)
                .map(|(_, r)| r)
                .collect();
            assert_eq!(
                found.len(),
                1,
                "labels reading {text:?}; all labels: {:?}",
                self.labels()
            );
            found[0]
        }

        fn shapes(&self) -> Vec<egui::Shape> {
            shapes(&self.out)
        }

        /// The style the canvas paints with: the one set on the context, or
        /// the theme's.
        fn style(&self) -> EditorStyle {
            set_style(&self.ctx)
                .unwrap_or_else(|| EditorStyle::from_visuals(&self.ctx.global_style().visuals))
        }

        /// The painted rect of each node box, in drawing order, found by
        /// where the state says the boxes are rather than by their colour.
        fn painted_boxes(&self) -> Vec<egui::epaint::RectShape> {
            let to_screen = self.to_screen();
            let rects: Vec<egui::epaint::RectShape> = self
                .shapes()
                .into_iter()
                .filter_map(|s| match s {
                    egui::Shape::Rect(r) => Some(r),
                    _ => None,
                })
                .collect();
            self.state
                .boxes
                .iter()
                .map(|(id, rect)| {
                    let on_screen = to_screen * *rect;
                    rects
                        .iter()
                        .find(|r| {
                            r.rect.center().distance(on_screen.center()) < 2.0
                                && (r.rect.size() - on_screen.size()).length() < 4.0
                        })
                        .cloned()
                        .unwrap_or_else(|| panic!("no painted rect for node #{}", id.0))
                })
                .collect()
        }

        /// Port dots on screen: circles of the port radius at this zoom.
        fn dots(&self) -> Vec<Pos2> {
            let radius = PORT_RADIUS * self.to_screen().scaling;
            self.shapes()
                .into_iter()
                .filter_map(|s| match s {
                    egui::Shape::Circle(c) if (c.radius - radius).abs() < 0.25 * radius => {
                        Some(c.center)
                    }
                    _ => None,
                })
                .collect()
        }

        /// Node boxes on screen: the frames filled with the style's node
        /// fill, which under [`distinct_style`] nothing else paints.
        fn boxes(&self) -> Vec<Rect> {
            let fill = self.style().node_fill;
            self.shapes()
                .into_iter()
                .filter_map(|s| match s {
                    egui::Shape::Rect(r) if r.fill == fill => Some(r.rect),
                    _ => None,
                })
                .collect()
        }

        /// Every polyline's points: the wires.
        fn wires(&self) -> Vec<Vec<Pos2>> {
            self.shapes()
                .into_iter()
                .filter_map(|s| match s {
                    egui::Shape::Path(p) if p.points.len() > 8 => Some(p.points),
                    _ => None,
                })
                .collect()
        }

        fn painted_text(&self) -> Vec<String> {
            self.shapes()
                .into_iter()
                .filter_map(|s| match s {
                    egui::Shape::Text(t) => Some(t.galley.text().to_owned()),
                    _ => None,
                })
                .collect()
        }

        /// Whether each node box, in drawing order (the patch's node order),
        /// is outlined in the style's error colour.
        fn outlined_in_error(&self) -> Vec<bool> {
            let style = self.style();
            self.shapes()
                .into_iter()
                .filter_map(|s| match s {
                    egui::Shape::Rect(r) if r.fill == style.node_fill => {
                        Some(r.stroke.color == style.error)
                    }
                    _ => None,
                })
                .collect()
        }

        fn buttons(&self) -> Vec<String> {
            button_labels(&self.out)
        }
    }

    fn one_node(kind: NodeKind) -> AudioPatch {
        AudioPatch {
            seed: 0,
            graph: crate::patch::NodeGraph {
                nodes: vec![node(0, kind)],
                output: NodeId(0),
            },
        }
    }

    /// A dot's y inside `row`'s vertical span.
    fn beside(dot: Pos2, row: Rect) -> bool {
        (row.top()..=row.bottom()).contains(&dot.y)
    }

    #[test]
    fn every_input_dot_sits_on_the_box_edge_beside_its_own_row() {
        let mut checked = 0;
        for kind in NodeKind::defaults() {
            let name = node_kind_label(&kind);
            let ports = input_ports(&kind);
            let canvas = Canvas::new(one_node(kind.clone()));
            let boxes = canvas.boxes();
            assert_eq!(boxes.len(), 1, "{name}: one node box");
            let node_box = boxes[0];
            let dots = canvas.dots();
            for port in ports {
                let row = canvas.label(&format!("{port}:"));
                assert!(
                    dots.iter()
                        .any(|d| (d.x - node_box.left()).abs() < 0.5 && beside(*d, row)),
                    "{name}: no dot on the box's left edge beside the {port:?} row \
                     ({row:?}); dots {dots:?}, box {node_box:?}"
                );
                checked += 1;
            }
        }
        assert!(checked >= 20, "only {checked} ports were checked");
    }

    #[test]
    fn the_output_dot_sits_on_the_title_row() {
        for kind in NodeKind::defaults() {
            let name = node_kind_label(&kind);
            let canvas = Canvas::new(one_node(kind.clone()));
            let node_box = canvas.boxes()[0];
            let title = canvas.label(&format!("#0  {name}"));
            assert!(
                canvas
                    .dots()
                    .iter()
                    .any(|d| (d.x - node_box.right()).abs() < 0.5 && beside(*d, title)),
                "{name}: the output dot is not on the title row {title:?}; dots {:?}",
                canvas.dots()
            );
        }
    }

    /// `#0` a sine, `#1` a gain whose `extra` port (not one of Gain's own,
    /// as a patch loaded from JSON can have) is wired from the sine.
    fn sine_into_gain(extra: bool) -> AudioPatch {
        let mut gain = node(1, NodeKind::Gain(Default::default()));
        if extra {
            gain.inputs
                .insert("extra".into(), vec![Connection::from_node(NodeId(0))]);
        }
        AudioPatch {
            seed: 0,
            graph: crate::patch::NodeGraph {
                nodes: vec![node(0, NodeKind::Sine(SineOsc::default())), gain],
                output: NodeId(1),
            },
        }
    }

    #[test]
    fn a_wire_into_an_extra_port_ends_on_that_ports_row() {
        let canvas = Canvas::new(sine_into_gain(true));
        let row = canvas.label("extra:");
        let wires = canvas.wires();
        assert_eq!(wires.len(), 1, "one wire");
        let end = *wires[0].last().expect("a wire has points");
        assert!(
            beside(end, row),
            "the wire ends at {end:?}, not beside the extra row {row:?}"
        );
        assert!(
            canvas.dots().iter().any(|d| d.distance(end) < 0.5),
            "the extra port has a dot where its wire ends"
        );
    }

    /// Press on the sine's output dot, carry the wire to `to` over a few
    /// frames and hold it there; the canvas as it looks mid-drag.
    fn drag_from_the_sine_to(canvas: &mut Canvas, to: Pos2) {
        let node_box = canvas
            .boxes()
            .into_iter()
            .min_by(|a, b| a.left().total_cmp(&b.left()))
            .expect("the sine's box is leftmost");
        let from = canvas
            .dots()
            .into_iter()
            .find(|d| (d.x - node_box.right()).abs() < 0.5)
            .expect("the sine's output dot");
        canvas.frame(vec![
            egui::Event::PointerMoved(from),
            egui::Event::PointerButton {
                pos: from,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ]);
        for step in 1..=4 {
            canvas.frame(vec![egui::Event::PointerMoved(
                from.lerp(to, step as f32 / 4.0),
            )]);
        }
        // Then hold still, as a user does before letting go: the target's
        // tooltip is a new area, and its first frame is a sizing pass.
        canvas.frame(Vec::new());
        canvas.frame(Vec::new());
    }

    fn release(canvas: &mut Canvas, at: Pos2) -> EditorResponse {
        canvas.frame(vec![egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }])
    }

    #[test]
    fn a_wire_dropped_anywhere_on_a_row_connects_to_that_rows_port() {
        let mut canvas = Canvas::new(sine_into_gain(false));
        let row = canvas.label("gain:");
        // Well right of the label: on the row, and far from the port's dot,
        // which is on the box's left edge.
        let to = Pos2::new(row.right() + 60.0, row.center().y);
        let scale = canvas.to_screen().scaling;
        let dot_x = canvas
            .boxes()
            .into_iter()
            .map(|b| b.left())
            .fold(f32::MIN, f32::max);
        assert!(
            (to.x - dot_x) / scale > SNAP_DIST,
            "the drop point must be out of the old snap radius"
        );

        drag_from_the_sine_to(&mut canvas, to);
        assert!(
            canvas
                .painted_text()
                .iter()
                .any(|t| t == "\u{27A1} #1 Gain \u{00B7} gain"),
            "mid-drag, the target port is named; painted: {:?}",
            canvas.painted_text()
        );
        let res = release(&mut canvas, to);

        let gain = &canvas.patch.graph.nodes[1];
        assert_eq!(
            gain.inputs.get("gain"),
            Some(&vec![Connection::from_node(NodeId(0))]),
            "the drop wired the row's port; inputs: {:?}",
            gain.inputs
        );
        assert!(res.changed && res.rebake);
    }

    #[test]
    fn a_wire_dropped_on_no_port_connects_nothing_and_names_nothing() {
        let mut canvas = Canvas::new(sine_into_gain(false));
        let gain_box = canvas
            .boxes()
            .into_iter()
            .max_by(|a, b| a.left().total_cmp(&b.left()))
            .expect("the gain's box");
        // Below the gain's box: no row, no dot.
        let to = Pos2::new(gain_box.center().x, gain_box.bottom() + 80.0);
        drag_from_the_sine_to(&mut canvas, to);
        assert!(
            !canvas
                .painted_text()
                .iter()
                .any(|t| t.starts_with('\u{27A1}')),
            "nothing is named over empty canvas"
        );
        release(&mut canvas, to);
        assert!(canvas.patch.graph.nodes[1].inputs.is_empty());
    }

    /// The canvas must not panic on a structurally invalid graph — the
    /// auto-layout cycle fallback and the red validity banner have to cope.
    #[test]
    fn canvas_renders_invalid_graph_without_panicking() {
        // A two-node cycle: 0 -> 1 -> 0. topo_sort returns Err(Cycle).
        let mut a = node(0, NodeKind::Gain(Default::default()));
        a.inputs
            .insert("in".into(), vec![Connection::from_node(NodeId(1))]);
        let mut b = node(1, NodeKind::Gain(Default::default()));
        b.inputs
            .insert("in".into(), vec![Connection::from_node(NodeId(0))]);
        let mut patch = AudioPatch {
            seed: 0,
            graph: crate::patch::NodeGraph {
                nodes: vec![a, b],
                output: NodeId(0),
            },
        };
        assert!(topo_sort(&patch.graph).is_err());

        let ctx = egui::Context::default();
        let mut state = PatchEditorState::default();
        let _ = ctx.run_ui(egui::RawInput::default(), |root| {
            egui::CentralPanel::default().show(root, |ui| {
                audio_patch_canvas(ui, &mut patch, &mut state, egui::Id::new("smoke_bad"));
            });
        });
    }

    // -----------------------------------------------------------------------
    // Where a broken graph is broken (#57, Overlands #1330 D4)
    // -----------------------------------------------------------------------

    /// A gain numbered `id` whose `in` port sums the outputs of `from`.
    fn gain_from(id: u32, from: &[u32]) -> GraphNode {
        let mut gain = node(id, NodeKind::Gain(Default::default()));
        if !from.is_empty() {
            gain.inputs.insert(
                "in".into(),
                from.iter()
                    .map(|&up| Connection::from_node(NodeId(up)))
                    .collect(),
            );
        }
        gain
    }

    fn graph_of(nodes: Vec<GraphNode>, output: u32) -> AudioPatch {
        AudioPatch {
            seed: 0,
            graph: crate::patch::NodeGraph {
                nodes,
                output: NodeId(output),
            },
        }
    }

    /// `#0` and `#1` feed each other. `#2` feeds the loop from outside it
    /// and `#3` listens to it: both are stuck behind the loop, and neither
    /// is part of it.
    fn loop_with_a_source_and_a_listener() -> AudioPatch {
        graph_of(
            vec![
                gain_from(0, &[1, 2]),
                gain_from(1, &[0]),
                gain_from(2, &[]),
                gain_from(3, &[1]),
            ],
            3,
        )
    }

    #[test]
    fn the_canvas_outlines_the_nodes_of_a_cycle_and_no_others() {
        let patch = loop_with_a_source_and_a_listener();
        assert_eq!(topo_sort(&patch.graph), Err(GraphError::Cycle));
        let canvas = Canvas::new(patch);
        assert_eq!(canvas.outlined_in_error(), [true, true, false, false]);
    }

    #[test]
    fn the_canvas_outlines_the_node_whose_wire_names_a_missing_node() {
        let patch = graph_of(vec![gain_from(0, &[]), gain_from(1, &[0, 9])], 1);
        assert_eq!(
            topo_sort(&patch.graph),
            Err(GraphError::UnknownNode(NodeId(9)))
        );
        let canvas = Canvas::new(patch);
        assert_eq!(canvas.outlined_in_error(), [false, true]);
    }

    #[test]
    fn the_canvas_outlines_every_node_that_shares_an_id() {
        let patch = graph_of(
            vec![gain_from(0, &[]), gain_from(4, &[0]), gain_from(4, &[])],
            4,
        );
        assert_eq!(
            topo_sort(&patch.graph),
            Err(GraphError::DuplicateId(NodeId(4)))
        );
        let canvas = Canvas::new(patch);
        assert_eq!(canvas.outlined_in_error(), [false, true, true]);
    }

    /// The control: a valid graph has no error outline, so the three tests
    /// above cannot pass by outlining everything.
    #[test]
    fn a_valid_graph_outlines_no_node() {
        let canvas = Canvas::new(three_node_patch());
        assert_eq!(canvas.outlined_in_error(), [false, false, false]);
    }

    /// The validity line names the nodes, not just the kind of fault: "graph
    /// contains a cycle" left the user to find the loop themselves.
    #[test]
    fn the_validity_line_names_the_nodes_of_a_cycle() {
        let canvas = Canvas::new(loop_with_a_source_and_a_listener());
        let text = canvas.painted_text().join("\n");
        assert!(
            text.contains("#0 Gain") && text.contains("#1 Gain"),
            "the loop's nodes are named; painted:\n{text}"
        );
    }

    #[test]
    fn the_validity_line_names_the_node_holding_a_dangling_wire() {
        let canvas = Canvas::new(graph_of(vec![gain_from(0, &[]), gain_from(1, &[9])], 1));
        let text = canvas.painted_text().join("\n");
        assert!(
            text.contains("#1 Gain") && text.contains("#9"),
            "the node and the missing one are named; painted:\n{text}"
        );
    }

    // -----------------------------------------------------------------------
    // Colours from the editor style (#58, Overlands #1331 E1)
    // -----------------------------------------------------------------------

    /// E1: under the host's light theme the node boxes stayed charcoal and
    /// the theme drew dark titles on them. Read off the painted frame: the
    /// title's colour against its box's fill, in both of egui's themes, with
    /// no style set so the canvas falls back to the theme's.
    #[test]
    fn a_node_title_reads_on_its_box_in_dark_and_light() {
        for (theme, visuals) in [
            ("dark", egui::Visuals::dark()),
            ("light", egui::Visuals::light()),
        ] {
            let canvas = Canvas::themed(three_node_patch(), visuals);
            let boxes = canvas.painted_boxes();
            for (i, name) in ["#0  Sine", "#1  LFO", "#2  Lowpass"]
                .into_iter()
                .enumerate()
            {
                let (_, title) = text_painted(&canvas.out, name)
                    .unwrap_or_else(|| panic!("{theme}: no title {name:?}"));
                let ratio = contrast_on(title, boxes[i].fill);
                assert!(
                    ratio >= AA,
                    "{theme}: {name:?} is {ratio:.2}:1 on its box ({title:?} on {:?})",
                    boxes[i].fill
                );
            }
        }
    }

    /// A style the host set is what the canvas paints, role by role: the
    /// ground, the boxes and their three edges, the title, the wires, the
    /// ports and the validity line.
    #[test]
    fn a_set_style_is_what_the_canvas_paints() {
        let s = distinct_style();
        let mut canvas = Canvas::new(three_node_patch());
        canvas.state.selected = Some(NodeId(0));
        canvas.frame(Vec::new());

        let painted = colours(&canvas.out);
        for (role, colour) in [
            ("canvas_ground", s.canvas_ground),
            ("wire", s.wire),
            ("port", s.port),
        ] {
            assert!(painted.contains(&colour), "{role} is not painted");
        }
        // #0 selected, #1 plain, #2 the output.
        let edges: Vec<Color32> = canvas
            .painted_boxes()
            .iter()
            .map(|b| b.stroke.color)
            .collect();
        assert_eq!(edges, [s.node_selected, s.node_stroke, s.node_output]);
        assert!(canvas.painted_boxes().iter().all(|b| b.fill == s.node_fill));
        let (_, title) = text_painted(&canvas.out, "#1  LFO").expect("a title");
        assert_eq!(title, s.node_title);
        let (_, valid) =
            text_painted(&canvas.out, "\u{2714} valid graph \u{2014} 3 nodes").expect("the line");
        assert_eq!(valid, s.ok);
    }

    #[test]
    fn a_broken_graph_is_named_and_outlined_in_the_styles_error_colour() {
        let s = distinct_style();
        let canvas = Canvas::new(loop_with_a_source_and_a_listener());
        let line = canvas
            .painted_text()
            .into_iter()
            .find(|t| t.contains("feed each other"))
            .expect("the validity line");
        let (_, colour) = text_painted(&canvas.out, &line).expect("painted");
        assert_eq!(colour, s.error);
        assert!(colours(&canvas.out).contains(&s.error));
        assert_eq!(canvas.outlined_in_error(), [true, true, false, false]);
    }

    /// The wire in the hand and the row it would land on are the active
    /// wire colour, not the theme's selection colours.
    #[test]
    fn a_dragged_wire_and_its_target_are_painted_in_the_active_wire_colour() {
        let s = distinct_style();
        let mut canvas = Canvas::new(sine_into_gain(false));
        assert!(
            !colours(&canvas.out).contains(&s.wire_active),
            "nothing in hand yet"
        );
        let row = canvas.label("gain:");
        drag_from_the_sine_to(&mut canvas, Pos2::new(row.right() + 60.0, row.center().y));
        assert!(colours(&canvas.out).contains(&s.wire_active));
    }

    /// E2: the canvas's buttons say what they do in words, as the host's do.
    /// The one glyph left is the remove cross, the code point Overlands'
    /// `affordances::CROSS` uses for the same act.
    #[test]
    fn the_canvas_buttons_say_what_they_do_in_words() {
        let canvas = Canvas::new(three_node_patch());
        let labels = canvas.buttons();
        for word in [
            "Add node",
            "Delete",
            "Fit view",
            "Mutate",
            "Reroll seed",
            "Add constant",
        ] {
            assert!(
                labels.iter().any(|l| l == word),
                "no {word:?} button; buttons: {labels:?}"
            );
        }
        assert_eq!(
            glyph_labels(&labels, &['\u{2716}']),
            Vec::<String>::new(),
            "buttons labelled with a glyph where the vocabulary says a word"
        );
    }

    /// The control: the glyph check sees a glyph label, so the test above
    /// cannot pass by reading no labels.
    #[test]
    fn the_glyph_check_flags_a_glyph_label_and_passes_the_cross() {
        let labels = vec![
            "\u{1F3B2} Mutate".to_string(),
            "\u{2716}".to_string(),
            "Fit view".to_string(),
        ];
        assert_eq!(glyph_labels(&labels, &['\u{2716}']), ["\u{1F3B2} Mutate"]);
    }
}
