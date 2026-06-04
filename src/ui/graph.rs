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
//! - **Wire a port:** drag from a node's output dot (right edge) onto another
//!   node's input dot (left edge) — appends a [`Connection::Node`] to that
//!   port (fan-in: a port holds a *list*, summed at bake time).
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
//! gets a gold border, so cycles / missing-output / unknown-node are visible
//! the moment they're created.

use std::collections::HashMap;

use bevy_egui::egui::{
    self, Align, Color32, Id, Layout, Pos2, Rect, Sense, Stroke, UiBuilder, Vec2,
};

use crate::node::NodeKind;
use crate::oscillator::SineOsc;
use crate::patch::{AudioPatch, Connection, GraphNode, NodeId, topo_sort};

use super::EditorResponse;
use super::evolve::{fresh_rng, mutate_node_kind, mutate_patch, randomize_seed};
use super::io::json_io;
use super::node::{node_kind_editor, node_kind_label};

const NODE_WIDTH: f32 = 210.0;
const PORT_RADIUS: f32 = 5.0;
/// Horizontal / vertical spacing of the topological auto-layout grid.
const COL_W: f32 = 280.0;
const ROW_H: f32 = 190.0;
/// How close (scene units) a wire drop must land to an input dot to connect.
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
    /// Mutation rate for the "🎲 Mutate" buttons.
    mutate_rate: f32,
    /// Buffer + last error for the JSON import/export section.
    json: super::JsonIoState,
}

impl Default for PatchEditorState {
    fn default() -> Self {
        Self {
            positions: HashMap::new(),
            scene_rect: Rect::from_min_size(Pos2::ZERO, Vec2::new(1000.0, 700.0)),
            selected: None,
            mutate_rate: 0.3,
            json: super::JsonIoState::default(),
        }
    }
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
        NodeKind::Silence
        | NodeKind::WhiteNoise(_)
        | NodeKind::PinkNoise(_)
        | NodeKind::BrownNoise(_)
        | NodeKind::Lfo(_)
        | NodeKind::Gate(_) => &[],
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

/// Closest input anchor to `at` within [`SNAP_DIST`], if any.
fn nearest_input(anchors: &HashMap<(NodeId, String), Pos2>, at: Pos2) -> Option<(NodeId, String)> {
    let mut best: Option<(NodeId, String, f32)> = None;
    for ((id, port), p) in anchors {
        let d = p.distance(at);
        if d <= SNAP_DIST && best.as_ref().is_none_or(|(_, _, bd)| d < *bd) {
            best = Some((*id, port.clone(), d));
        }
    }
    best.map(|(id, port, _)| (id, port))
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
pub fn audio_patch_canvas(
    ui: &mut egui::Ui,
    patch: &mut AudioPatch,
    state: &mut PatchEditorState,
    id: Id,
) -> EditorResponse {
    let mut res = EditorResponse::NONE;
    res.merge(toolbar(ui, patch, state));
    res.merge(json_io(ui, patch, &mut state.json, id.with("patch_json")));
    state.ensure_layout(patch);

    let mut scene_rect = state.scene_rect;
    let scene = egui::Scene::new().zoom_range(egui::Rangef::new(0.2, 2.0));
    let inner = scene.show(ui, &mut scene_rect, |ui| {
        canvas_contents(ui, patch, state, id)
    });
    state.scene_rect = scene_rect;
    res.merge(inner.inner);
    res
}

/// Fixed toolbar above the canvas: add / delete nodes, pick the output, reset
/// the view, and a live validity readout.
fn toolbar(
    ui: &mut egui::Ui,
    patch: &mut AudioPatch,
    state: &mut PatchEditorState,
) -> EditorResponse {
    let mut res = EditorResponse::NONE;
    ui.horizontal_wrapped(|ui| {
        if ui.button("\u{2795} Add node").clicked() {
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
            .add_enabled(can_delete, egui::Button::new("\u{1F5D1} Delete"))
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
        if ui.button("\u{27F2} Reset view").clicked() {
            // A zero-size rect makes Scene auto-fit to the content next frame.
            state.scene_rect = Rect::ZERO;
        }
    });

    ui.horizontal_wrapped(|ui| {
        if ui
            .button("\u{1F3B2} Mutate")
            .on_hover_text("Nudge every node's parameters via symbios-genetics")
            .clicked()
        {
            mutate_patch(patch, &mut fresh_rng(), state.mutate_rate);
            res.changed = true;
            res.rebake = true;
        }
        ui.add(egui::Slider::new(&mut state.mutate_rate, 0.0..=1.0).text("rate"));
        if ui
            .button(format!("\u{1F3B2} seed {}", patch.seed))
            .on_hover_text("Reroll the patch seed (re-randomises noise / random LFOs)")
            .clicked()
        {
            randomize_seed(patch, &mut fresh_rng());
            res.changed = true;
            res.rebake = true;
        }
    });

    match topo_sort(&patch.graph) {
        Ok(order) => {
            ui.colored_label(
                Color32::from_rgb(120, 200, 120),
                format!("\u{2713} valid graph \u{2014} {} nodes", order.len()),
            );
        }
        Err(e) => {
            ui.colored_label(Color32::from_rgb(220, 120, 120), format!("\u{2717} {e}"));
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
) -> EditorResponse {
    let mut res = EditorResponse::NONE;
    let mut actions: Vec<Action> = Vec::new();

    let mut out_anchor: HashMap<NodeId, Pos2> = HashMap::new();
    let mut in_anchor: HashMap<(NodeId, String), Pos2> = HashMap::new();
    let mut node_rects: HashMap<NodeId, Rect> = HashMap::new();
    let mut temp_wire: Option<(Pos2, Pos2)> = None;

    let output_id = patch.graph.output;
    let selected = state.selected;
    let mutate_rate = state.mutate_rate;

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

        let stroke = if selected == Some(nid) {
            Stroke::new(2.0, Color32::from_rgb(90, 160, 250))
        } else if output_id == nid {
            Stroke::new(2.0, Color32::from_rgb(230, 190, 90))
        } else {
            Stroke::new(1.0, Color32::from_gray(110))
        };

        let mut child = ui.new_child(
            UiBuilder::new()
                .max_rect(Rect::from_min_size(pos, Vec2::new(NODE_WIDTH, 10.0)))
                .id_salt(("patch_node", nid.0))
                .layout(Layout::top_down(Align::Min)),
        );
        child.set_width(NODE_WIDTH);

        let frame = egui::Frame::group(child.style())
            .fill(Color32::from_gray(32))
            .stroke(stroke);
        let fr = frame.show(&mut child, |ui| {
            ui.set_width(NODE_WIDTH);
            // Title bar: drag to move, click to select, 🎲 to mutate this node.
            ui.horizontal(|ui| {
                let title = format!("#{}  {}", nid.0, node_kind_label(&node.kind));
                let title_resp = ui.add(
                    egui::Label::new(egui::RichText::new(title).strong())
                        .sense(Sense::click_and_drag()),
                );
                if title_resp.dragged() {
                    actions.push(Action::Move(nid, title_resp.drag_delta()));
                }
                if title_resp.clicked() {
                    actions.push(Action::Select(nid));
                }
                if ui
                    .small_button("\u{1F3B2}")
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
            res.merge(connection_editor(ui, node));
        });

        let rect = fr.response.rect;
        node_rects.insert(nid, rect);
        out_anchor.insert(nid, Pos2::new(rect.right(), rect.center().y));

        let ports = input_ports(&node.kind);
        for (i, port) in ports.iter().enumerate() {
            let t = (i as f32 + 1.0) / (ports.len() as f32 + 1.0);
            let anchor = Pos2::new(rect.left(), rect.top() + t * rect.height());
            in_anchor.insert((nid, (*port).to_string()), anchor);
        }

        // Output port: drag to start a wire.
        let oa = out_anchor[&nid];
        let out_resp = ui.interact(
            Rect::from_center_size(oa, Vec2::splat(PORT_RADIUS * 2.5)),
            id.with(("outport", nid.0)),
            Sense::drag(),
        );
        if out_resp.dragged()
            && let Some(p) = out_resp.interact_pointer_pos()
        {
            temp_wire = Some((oa, p));
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
            let dst = in_anchor
                .get(&(node.id, port.clone()))
                .copied()
                .or_else(|| {
                    node_rects
                        .get(&node.id)
                        .map(|r| Pos2::new(r.left(), r.center().y))
                });
            let Some(dst) = dst else { continue };
            for c in conns {
                if let Connection::Node { id: src, .. } = c
                    && let Some(src_pos) = out_anchor.get(src)
                {
                    wires.push(wire_shape(*src_pos, dst, Color32::from_gray(150)));
                }
            }
        }
    }
    if let Some((a, b)) = temp_wire {
        wires.push(wire_shape(a, b, Color32::from_rgb(90, 160, 250)));
    }
    painter.set(wire_idx, egui::Shape::Vec(wires));

    // --- port dots (on top of wires) -----------------------------------
    for p in out_anchor.values() {
        painter.circle_filled(*p, PORT_RADIUS, Color32::from_rgb(230, 190, 90));
    }
    for p in in_anchor.values() {
        painter.circle_filled(*p, PORT_RADIUS, Color32::from_gray(190));
    }

    // --- apply deferred structural edits -------------------------------
    for action in actions {
        match action {
            Action::Select(nid) => state.selected = Some(nid),
            Action::Move(nid, delta) => {
                *state.positions.entry(nid).or_default() += delta;
            }
            Action::CompleteWire { from, at } => {
                if let Some((to, port)) = nearest_input(&in_anchor, at)
                    && to != from
                    && let Some(n) = patch.graph.nodes.iter_mut().find(|n| n.id == to)
                {
                    n.inputs
                        .entry(port)
                        .or_default()
                        .push(Connection::from_node(from));
                    res.changed = true;
                    res.rebake = true;
                }
            }
        }
    }

    // Claim the bounding area so Scene's "reset view" can fit the content.
    if let Some(bounds) = node_rects.values().copied().reduce(|a, b| a.union(b)) {
        ui.allocate_rect(bounds.expand(60.0), Sense::hover());
    }

    res
}

/// The "Inputs" section inside a node box: per-connection amount/value editing,
/// per-port "add constant", and per-connection delete.
fn connection_editor(ui: &mut egui::Ui, node: &mut GraphNode) -> EditorResponse {
    let mut res = EditorResponse::NONE;

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
        return res;
    }

    ui.separator();
    ui.label(egui::RichText::new("Inputs").weak());

    let mut to_delete: Vec<(String, usize)> = Vec::new();
    let mut to_add_const: Vec<String> = Vec::new();

    for port in &ports {
        ui.horizontal(|ui| {
            ui.label(format!("{port}:"));
            if ui.small_button("\u{FF0B}const").clicked() {
                to_add_const.push(port.clone());
                res.changed = true;
                res.rebake = true;
            }
        });
        if let Some(conns) = node.inputs.get_mut(port) {
            for (i, c) in conns.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    match c {
                        Connection::Node { id, amount } => {
                            ui.label(format!("  \u{2190} #{}", id.0));
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
            }
        }
    }

    // Remove highest indices first so earlier removals don't shift them.
    to_delete.sort_by(|a, b| b.1.cmp(&a.1));
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

    res
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

    #[test]
    fn nearest_input_snaps_only_within_threshold() {
        let mut anchors: HashMap<(NodeId, String), Pos2> = HashMap::new();
        anchors.insert((NodeId(5), "in".into()), Pos2::new(100.0, 100.0));
        // Just inside the snap radius.
        assert_eq!(
            nearest_input(&anchors, Pos2::new(100.0 + SNAP_DIST - 1.0, 100.0)),
            Some((NodeId(5), "in".into()))
        );
        // Well outside it.
        assert_eq!(nearest_input(&anchors, Pos2::new(500.0, 500.0)), None);
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
            let _ = ctx.run(egui::RawInput::default(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    audio_patch_canvas(ui, &mut patch, &mut state, egui::Id::new("smoke"));
                });
            });
        }
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
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                audio_patch_canvas(ui, &mut patch, &mut state, egui::Id::new("smoke_bad"));
            });
        });
    }
}
