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
//! That layout is *measured*: each box's size is kept from the frame it was
//! last drawn in, and a column is stacked by those sizes plus a gap. The
//! first frame a box appears in has no measurement to go on, so it is
//! placed at a guess and the auto-placed boxes are laid out again at the
//! start of the next frame — not at the end of this one, because a
//! position may not change between egui's passes over one frame. Toolbar
//! Tidy re-lays every node. A fixed pitch of 190 units against boxes of
//! 194 and up opened the starter patch with its boxes touching and a
//! loaded patch with them on top of each other (#59, Overlands #1332).
//!
//! # How editing maps to the schema
//!
//! - **Move a node:** drag the number at the left of its title (scene-space
//!   delta → stored position); click the number to select it. The rest of
//!   the title is the kind picker, so a node is named and re-kinded in the
//!   same place.
//! - **Wire a port:** drag from a node's output dot (right edge, on the
//!   title row) onto another node's input: its dot, or anywhere on its row
//!   in the node's "Inputs" list, where the port is named. That appends a
//!   [`Connection::Node`] to the port (fan-in: a port holds a *list*,
//!   summed at bake time). Each input dot sits on the box's left edge
//!   beside its own row — filled when something drives the port, a ring
//!   when nothing does — and so does a port that is not one of its kind's
//!   own (from JSON). While a wire is dragged it is drawn over the boxes,
//!   the port it would connect to is highlighted in the host's selection
//!   colour, and a tooltip names it ("➡ #2 Lowpass · cutoff_hz"). Only
//!   the box on top under the pointer can take the wire, so a row another
//!   box covers is out of reach. Let go over no port and the wire is not
//!   thrown away: it opens the Add menu there and drives what is chosen.
//! - **Touch a wire:** the wires are hit-tested against their curves, a few
//!   points either side ([`WireGeom::distance_to`]). Hovering one lights it
//!   and names both its ends, its port and its amount; a click picks it,
//!   Delete removes it, and picking it opens a small panel at the middle of
//!   its curve naming what it drives, with its amount and a cross — drawn in
//!   screen space so it stays readable at any zoom. A wire is
//!   taken off its port by its *end* — a grab point short of the dot,
//!   since several wires can end on one — and dropped on another port to
//!   re-route it, keeping the amount it carried, or on nothing to
//!   disconnect it.
//! - **Amounts / constants / deletes in the box:** the "Inputs" section
//!   inside each node box edits each connection's `amount` (or a
//!   [`Connection::Constant`]'s value) and removes connections; a constant
//!   has no wire, so that section is the only place to edit one. Each
//!   wire's row names the node it comes from ("from #1 LFO"), so nothing on
//!   the canvas asks the reader to remember what a number stands for.
//! - **Add / remove nodes, set output:** the one toolbar row above the
//!   canvas — Add node, Delete, Tidy, Fit view, the Output picker and a
//!   More menu holding the genetics controls and the JSON box. Add node is
//!   a menu grouped by role, built by walking the generated roster so a
//!   kind added upstream appears in it on the version bump alone; the same
//!   menu opens on a right-click on clear canvas and puts the node there.
//!   Delete acts on whatever is picked, a node or a wire, and says which.
//!
//! Structural edits are collected as deferred `Action`s while the node loop
//! holds `&mut patch.graph.nodes`, then applied once the loop's borrow ends —
//! the standard way to keep an immediate-mode graph editor borrow-clean.
//!
//! Validity ([`topo_sort`]) is shown live at the right of the toolbar and
//! the output node wears an OUT badge in its title, so cycles /
//! missing-output / unknown-node are visible the moment they're created.
//! The same line counts what is *heard* — the nodes the output can be
//! reached from along the wires — because "3 nodes" is true of a patch
//! where two of them bake into nothing. A node nothing hears has its
//! contents dimmed and wears a "not heard" badge, while its frame keeps
//! full strength so it is still visibly a box to edit.
//! The badge is a badge and not a border because the selection is a border
//! — a thicker one — and a node can be both the output and selected. A broken graph is also located: the toolbar
//! names the nodes at fault ("#0 Gain and #1 Gain feed each other in a
//! loop…"), and the canvas outlines their boxes in the error colour — the
//! nodes of a loop, the node holding a wire from a node that is not there,
//! or every node sharing an id (#57).
//!
//! # Colours
//!
//! Everything the canvas paints itself — the ground, its border and grid,
//! the node boxes and their edges, titles and rules, wires, ports, the
//! validity line — takes its colour from the [`EditorStyle`] in effect
//! ([`crate::ui::style`]): the host's, if it set one, else one derived from
//! the `Visuals` the canvas is drawn with.
//! A node box is filled with a surface the theme's text reads on, so the
//! widgets inside it read in a light theme as well as a dark one (#58).

use std::collections::{BTreeMap, HashMap, HashSet};

use bevy_egui::egui::{
    self, Align, Color32, Id, Layout, Pos2, Rect, Sense, Stroke, UiBuilder, Vec2,
    emath::TSTransform,
};

use crate::node::NodeKind;
use crate::patch::{AudioPatch, Connection, GraphError, GraphNode, NodeGraph, NodeId, topo_sort};

use super::EditorResponse;
use super::evolve::{fresh_rng, mutate_node_kind, mutate_patch, randomize_seed};
use super::history::EditHistory;
use super::io::json_io;
use super::node::{
    default_kind_for, kinds_by_group, node_kind_body, node_kind_label, node_kind_picker,
};
use super::style::{EditorStyle, editor_style};

/// The narrowest a node box is laid out at. A box whose content needs more
/// takes more — `egui::Frame::group` sizes to what is in it — so this is a
/// floor that keeps a one-parameter node from being a sliver, not a width.
const NODE_MIN_WIDTH: f32 = 210.0;
const PORT_RADIUS: f32 = 5.0;
/// Opacity of the highlight over the row a dragged wire would connect to:
/// the row's labels are under it and must still read.
const DROP_ROW_ALPHA: f32 = 0.2;
/// Clear space between the boxes the auto-layout places, scene units.
const NODE_GAP: f32 = 28.0;
/// Where the auto-layout's first column begins.
const LAYOUT_ORIGIN: Pos2 = Pos2::new(40.0, 40.0);
/// The size the auto-layout assumes for a box it has never drawn. One
/// frame's worth of guess: the real size is recorded the moment the box is
/// drawn and the auto-placed boxes are laid out again from it.
const UNMEASURED: Vec2 = Vec2::new(NODE_MIN_WIDTH, 210.0);
/// The selected box's edge. Thicker than any other so the selection reads
/// as well as the OUT badge beside it, and a node can wear both (#59).
const SELECTED_STROKE: f32 = 2.5;
/// The canvas grid's pitch, scene units.
const GRID_STEP: f32 = 80.0;
/// How many grid lines are ever drawn: zoomed far out the view covers more
/// ground than there is any use in ruling.
const GRID_MAX_LINES: usize = 240;
/// How close (scene units) a wire drop must land to an input dot to connect
/// when it is not on the port's row.
const SNAP_DIST: f32 = 26.0;
/// How close the pointer must come to a wire to hover it, in *screen*
/// points: a wire is as easy to catch zoomed out as zoomed in.
const WIRE_HIT_DIST: f32 = 6.0;
/// How big a wire's end grab point is, in screen points: the square the
/// pointer takes hold of to move a wire off its port. Big enough to hit
/// without covering the wire's own curve.
const WIRE_GRAB_RADIUS: f32 = 7.0;
/// What a node the output never reaches wears in its title, and how much of
/// its box's opacity is left.
const NOT_HEARD: &str = "not heard";
const NOT_HEARD_OPACITY: f32 = 0.45;

/// Editor-side state for the patch canvas — node layout and view, kept out of
/// the serialized [`AudioPatch`] so the wire format stays clean.
///
/// Construct with [`Default`]; the canvas fills in any missing node positions
/// via topological auto-layout on first sight.
#[derive(Clone, Debug)]
pub struct PatchEditorState {
    /// Node positions in scene-local coordinates.
    positions: HashMap<NodeId, Pos2>,
    /// Each node box's size as the canvas last drew it — the pitch the
    /// auto-layout stacks by.
    sizes: HashMap<NodeId, Vec2>,
    /// The nodes the canvas placed and the user has not moved. Only these
    /// are re-laid out when their measured sizes arrive or a Tidy is asked
    /// for; a box someone dragged somewhere stays there.
    auto: HashSet<NodeId>,
    /// A layout is owed once the boxes it placed have been measured.
    relayout: bool,
    /// Whether the JSON box is open, from the toolbar's More menu.
    show_json: bool,
    /// Undo/redo over the patch (#60). Off in the sequence editor's
    /// embedded canvas, where the recipe's history owns the value.
    history: EditHistory<AudioPatch>,
    /// The canvas took this frame's keyboard.
    owns_keys: bool,
    /// The canvas acted on an Escape this frame — it cleared a selection.
    took_escape: bool,
    /// The [`egui::Scene`] view rectangle — pan and zoom live here.
    scene_rect: Rect,
    /// The canvas layer's scene→screen transform as the last frame set it.
    canvas_to_screen: TSTransform,
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
    /// Every wire as the canvas last drew it, for hit-testing and for
    /// whoever asks ([`Self::wires`]).
    wires: Vec<WireGeom>,
    /// The wire the user picked, while the canvas still draws it.
    selected_wire: Option<WireRef>,
    /// An open Add menu: where it was asked for, and the wire it would
    /// complete.
    add_menu: Option<AddMenu>,
}

/// An open Add menu on the canvas.
#[derive(Clone, Debug)]
struct AddMenu {
    /// Where the node will be put, in canvas units — where the menu was
    /// opened, so the node lands under the pointer that asked for it.
    at: Pos2,
    /// An output the new node's first input is wired from: the menu opened
    /// by dropping a wire on empty canvas finishes the wire.
    from: Option<NodeId>,
    /// Whether the menu has been drawn at least once.
    ///
    /// The press that opens a menu is still this frame's press, and an
    /// `Area` on its first frame does not yet know where it is, so a menu
    /// that took any press outside itself as a dismissal would close on the
    /// very click that asked for it.
    shown: bool,
}

impl Default for PatchEditorState {
    fn default() -> Self {
        Self {
            positions: HashMap::new(),
            sizes: HashMap::new(),
            auto: HashSet::new(),
            relayout: false,
            show_json: false,
            scene_rect: Rect::from_min_size(Pos2::ZERO, Vec2::new(1000.0, 700.0)),
            canvas_to_screen: TSTransform::IDENTITY,
            selected: None,
            mutate_rate: 0.3,
            json: super::JsonIoState::default(),
            history: EditHistory::default(),
            owns_keys: false,
            took_escape: false,
            ports: Vec::new(),
            outputs: HashMap::new(),
            boxes: Vec::new(),
            wires: Vec::new(),
            selected_wire: None,
            add_menu: None,
        }
    }
}

/// One wire as the canvas last drew it, in canvas units (#61, Overlands
/// #1334 B2).
///
/// A wire used to be paint and nothing else: to remove one you found its
/// cross in the destination box's Inputs list. The canvas keeps them so it
/// can hit-test them, and publishes them ([`PatchEditorState::wires`]) so a
/// host can hang something on one and a scripted harness can put the
/// pointer on one (crate #67).
#[derive(Clone, Debug, PartialEq)]
pub struct WireGeom {
    /// The node the wire leaves, by its output.
    pub from: NodeId,
    /// The node it drives.
    pub to: NodeId,
    /// Which of `to`'s input ports it drives.
    pub port: String,
    /// Which of that port's connections it is — a port holds a list, summed
    /// at bake time, so a port can have several wires into it.
    pub index: usize,
    /// The wire's curve, sampled; the `from` end first.
    pub points: Vec<Pos2>,
}

impl WireGeom {
    /// The middle of the curve, where the wire's remove cross sits and
    /// where its amount editor opens.
    pub fn midpoint(&self) -> Pos2 {
        self.points
            .get(self.points.len() / 2)
            .copied()
            .unwrap_or_default()
    }

    /// How far `at` is from the curve, in canvas units.
    ///
    /// Against the curve and not its bounding box: the S-curve between two
    /// boxes bulges well away from the straight line between the dots, and
    /// a box test would take a click in the clear space beside it.
    pub fn distance_to(&self, at: Pos2) -> f32 {
        self.points
            .windows(2)
            .map(|seg| distance_to_segment(at, seg[0], seg[1]))
            .fold(f32::INFINITY, f32::min)
    }

    /// Which connection this wire is, without its geometry.
    fn wire_ref(&self) -> WireRef {
        WireRef {
            to: self.to,
            port: self.port.clone(),
            index: self.index,
        }
    }
}

/// Which connection a wire stands for: a port holds a list, so the index
/// within the port is part of the name.
#[derive(Clone, Debug, PartialEq, Eq)]
struct WireRef {
    to: NodeId,
    port: String,
    index: usize,
}

/// The wire passing nearest `at`, if one passes within `within` canvas
/// units of it.
fn wire_at(wires: &[WireGeom], at: Pos2, within: f32) -> Option<&WireGeom> {
    wires
        .iter()
        .map(|w| (w, w.distance_to(at)))
        .filter(|(_, d)| *d <= within)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(w, _)| w)
}

/// Every node the output can be reached from along the wires — what the
/// patch actually plays (#61, Overlands #1334 B13).
///
/// A walk upstream from the output with a visited set, so a loop among the
/// nodes it passes through is counted once and does not spin: a broken
/// graph is exactly when the reader most needs the canvas to keep drawing.
/// The output node itself is heard even when nothing feeds it.
fn heard_nodes(graph: &NodeGraph) -> HashSet<NodeId> {
    let mut heard = HashSet::new();
    let mut stack = vec![graph.output];
    while let Some(id) = stack.pop() {
        if !heard.insert(id) {
            continue;
        }
        if let Some(node) = graph.nodes.iter().find(|n| n.id == id) {
            stack.extend(upstream_of(node));
        }
    }
    // A wire from a node that is not in the patch names an id nothing has;
    // the walk reached it, and it is not a node that can be heard.
    heard.retain(|id| graph.nodes.iter().any(|n| n.id == *id));
    heard
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
    /// Give every node that lacks a position one, and lay the auto-placed
    /// boxes out together so they read as a grid.
    ///
    /// A node the user has dragged, or one dropped by Add node, is not
    /// auto-placed and is left where it is.
    fn ensure_layout(&mut self, patch: &AudioPatch) {
        let fresh: Vec<NodeId> = patch
            .graph
            .nodes
            .iter()
            .map(|n| n.id)
            .filter(|id| !self.positions.contains_key(id))
            .collect();
        if fresh.is_empty() {
            return;
        }
        self.auto.extend(fresh);
        self.place_auto(patch);
        // These boxes have never been drawn, so they were placed at a
        // guessed size; lay them out again once the real one is known.
        self.relayout = true;
    }

    /// Lay every auto-placed node out again — what Tidy does, and what the
    /// canvas does itself once a box it placed has been measured.
    fn place_auto(&mut self, patch: &AudioPatch) {
        let auto = self.auto.clone();
        self.place(patch, &auto);
    }

    /// Stack `place_me` into topological columns — sources left, output
    /// right — each column as wide as its widest box and each box below the
    /// one before it with [`NODE_GAP`] of clear space.
    ///
    /// The pitch is measured, not fixed. A fixed 190-unit row against boxes
    /// of 194 and up opened the starter patch with the LFO lying over the
    /// sine's amplitude row, and a patch loaded from a record opened with
    /// its boxes on each other (#59, Overlands #1332 B4).
    ///
    /// A node not in `place_me` keeps its position, and a column's cursor
    /// starts below any such box standing in that column's width, so a
    /// freshly loaded patch does not land on one already parked there.
    fn place(&mut self, patch: &AudioPatch, place_me: &HashSet<NodeId>) {
        let depths = compute_depths(patch);
        let mut by_col: BTreeMap<u32, Vec<NodeId>> = BTreeMap::new();
        for n in &patch.graph.nodes {
            if place_me.contains(&n.id) {
                by_col
                    .entry(depths.get(&n.id).copied().unwrap_or(0))
                    .or_default()
                    .push(n.id);
            }
        }
        let parked: Vec<Rect> = patch
            .graph
            .nodes
            .iter()
            .filter(|n| !place_me.contains(&n.id))
            .filter_map(|n| {
                Some(Rect::from_min_size(
                    *self.positions.get(&n.id)?,
                    self.size_of(n.id),
                ))
            })
            .collect();

        let mut x = LAYOUT_ORIGIN.x;
        for ids in by_col.into_values() {
            let width = ids
                .iter()
                .map(|id| self.size_of(*id).x)
                .fold(NODE_MIN_WIDTH, f32::max);
            let mut y = LAYOUT_ORIGIN.y;
            for rect in parked
                .iter()
                .filter(|r| r.right() > x && r.left() < x + width)
            {
                y = y.max(rect.bottom() + NODE_GAP);
            }
            for id in ids {
                self.positions.insert(id, Pos2::new(x, y));
                y += self.size_of(id).y + NODE_GAP;
            }
            x += width + NODE_GAP;
        }
    }

    /// A node box's size as the canvas last drew it, or what to assume
    /// until it has drawn one.
    fn size_of(&self, id: NodeId) -> Vec2 {
        self.sizes.get(&id).copied().unwrap_or(UNMEASURED)
    }

    /// Step the patch back to before the last committed edit. `true` if
    /// anything moved.
    ///
    /// A step is a committed edit — a drag is one, however many frames it
    /// took. The canvas does this itself on Ctrl+Z while it owns the
    /// keyboard; this is for a host that offers undo of its own (a menu
    /// item, a toolbar). In the sequence editor's embedded canvas there is
    /// no history to walk and this returns `false`: the recipe's history
    /// owns an instrument's patch, since the patch is part of the recipe.
    pub fn undo(&mut self, patch: &mut AudioPatch) -> bool {
        self.history.undo(patch)
    }

    /// Step the patch forward again after an [`Self::undo`].
    pub fn redo(&mut self, patch: &mut AudioPatch) -> bool {
        self.history.redo(patch)
    }

    /// Whether there is a committed edit to undo.
    pub fn can_undo(&self) -> bool {
        self.history.can_undo()
    }

    /// Whether there is an undone edit to redo.
    pub fn can_redo(&self) -> bool {
        self.history.can_redo()
    }

    /// Tell the canvas its patch was replaced from outside — a host's undo,
    /// a reload, a re-rolled seed — so the swap becomes a step in the
    /// editor's own history.
    ///
    /// Call it *after* writing the new value. Without it the history's
    /// baseline still describes the value before the swap, and the next
    /// undo would step back over both the outside change and the edit made
    /// since, in one jump. With it, a Ctrl+Z inside the editor returns to
    /// what was there before the outside change — which, for a host that
    /// takes the editor's commits as ordinary edits, lands as a forward
    /// edit on its own history (Overlands #1333 A9).
    ///
    /// Nothing is recorded if the value did not actually change, and
    /// nothing is recorded in the sequence editor's embedded canvas, whose
    /// history belongs to the recipe.
    pub fn note_external_change(&mut self, patch: &AudioPatch) {
        self.history.commit(patch);
    }

    /// Whether the canvas took this frame's keyboard — the pointer was over
    /// it and nothing in it held text focus.
    ///
    /// A host whose own shortcuts read the platform's keys rather than
    /// egui's cannot tell that the canvas has consumed a Ctrl+Z, and would
    /// act on it a second time. Ask this first and stand down (Overlands
    /// #1333).
    pub fn wants_keyboard(&self) -> bool {
        self.owns_keys
    }

    /// Every wire the canvas drew on its last frame, in canvas units.
    ///
    /// [`Self::canvas_to_screen`] takes one to the screen. Empty before the
    /// canvas has been drawn once, and re-measured every frame, so a wire
    /// held across frames should be looked up again rather than kept.
    pub fn wires(&self) -> &[WireGeom] {
        &self.wires
    }

    /// Whether a wire is picked — what Delete removes, and what the amount
    /// editor on the canvas is open on.
    pub fn wire_is_selected(&self) -> bool {
        self.selected_wire.is_some()
    }

    /// The canvas layer's scene→screen transform as of the last frame drawn.
    ///
    /// Everything the canvas remembers about its geometry — a node's
    /// position, a port's dot — is in scene units, which pan and zoom move
    /// under the reader. This is what takes them to screen points, so a
    /// host can hang an overlay on a node, and a scripted harness can press
    /// where a port is (crate #67). [`TSTransform::IDENTITY`] before the
    /// canvas has been drawn once.
    pub fn canvas_to_screen(&self) -> TSTransform {
        self.canvas_to_screen
    }

    /// Whether the canvas acted on an Escape this frame by clearing its
    /// selection.
    ///
    /// An Escape ladder should spend this press on that and close nothing:
    /// one step per press. `false` when there was no selection to clear, so
    /// a ladder loses no rung to a no-op.
    pub fn took_escape(&self) -> bool {
        self.took_escape
    }

    /// Hand this patch's history to whoever owns the value it is part of.
    /// See [`crate::ui::history`].
    pub(crate) fn disable_history(&mut self) {
        self.history.disable();
    }

    /// Whether this canvas keeps a history of its own — false inside the
    /// sequence editor, where the recipe's history owns the patch. Read by
    /// the test that holds that rule.
    #[cfg(test)]
    pub(crate) fn history_is_enabled(&self) -> bool {
        self.history.is_enabled()
    }

    /// Forget everything remembered about nodes the patch no longer has, so
    /// a deleted node cannot hold a column open or block a re-layout.
    fn forget_missing(&mut self, patch: &AudioPatch) {
        let live: HashSet<NodeId> = patch.graph.nodes.iter().map(|n| n.id).collect();
        self.positions.retain(|id, _| live.contains(id));
        self.sizes.retain(|id, _| live.contains(id));
        self.auto.retain(|id| live.contains(id));
        if self.selected.is_some_and(|id| !live.contains(&id)) {
            self.selected = None;
        }
        // A wire is named by the node it drives and which of that port's
        // connections it is, so a removal anywhere in the port's list can
        // leave the selection pointing at a different wire, or at none.
        self.forget_removed_wire(patch);
    }

    /// Drop a wire selection that the patch no longer holds.
    fn forget_removed_wire(&mut self, patch: &AudioPatch) {
        let Some(sel) = &self.selected_wire else {
            return;
        };
        let still_there = patch
            .graph
            .nodes
            .iter()
            .find(|n| n.id == sel.to)
            .and_then(|n| n.inputs.get(&sel.port))
            .is_some_and(|conns| matches!(conns.get(sel.index), Some(Connection::Node { .. })));
        if !still_there {
            self.selected_wire = None;
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

/// The Add menu's items: every kind the picker offers, under the heading of
/// the role it fills. Returns the kind chosen, if one was.
///
/// Built by walking the generated roster and asking
/// [`crate::ui::node::kind_group`] what each label is for, never from a
/// hand-written list of kinds: a kind added upstream appears here on the
/// version bump alone, under "Other" until this build is told its role
/// (#50, and #61 / Overlands #1334 B6).
fn add_menu_items(ui: &mut egui::Ui) -> Option<&'static str> {
    let mut chosen = None;
    for (group, kinds) in kinds_by_group() {
        ui.label(egui::RichText::new(group.title()).weak().small());
        for label in kinds {
            if ui.button(label).clicked() {
                chosen = Some(label);
                ui.close();
            }
        }
    }
    chosen
}

/// Put a node of the kind named `label` at `at` (canvas units), select it,
/// and wire `from`'s output into its first input if there is one to wire.
///
/// Returns the new node's id.
fn add_node(
    patch: &mut AudioPatch,
    state: &mut PatchEditorState,
    label: &str,
    at: Pos2,
    from: Option<NodeId>,
) -> NodeId {
    let new_id = NodeId(
        patch
            .graph
            .nodes
            .iter()
            .map(|n| n.id.0)
            .max()
            .map_or(0, |m| m + 1),
    );
    let kind = default_kind_for(label);
    let mut inputs: BTreeMap<String, Vec<Connection>> = Default::default();
    // A wire dragged onto empty canvas asked for something to drive, so the
    // node arrives already driven — by its first input, which is the one a
    // signal goes into for every kind that has one.
    //
    // Unless the node it came from has gone in the meantime: the menu
    // outlives the frame it was opened in, and a wire from an id nothing
    // has is a graph that will not bake.
    if let Some(from) = from
        && patch.graph.nodes.iter().any(|n| n.id == from)
        && let Some(port) = input_ports(&kind).first()
    {
        inputs.insert((*port).to_string(), vec![Connection::from_node(from)]);
    }
    patch.graph.nodes.push(GraphNode {
        id: new_id,
        kind,
        inputs,
    });
    state.positions.insert(new_id, at);
    state.selected = Some(new_id);
    state.selected_wire = None;
    new_id
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
    /// Remove one wire, named by the connection it stands for.
    RemoveWire(WireRef),
    /// Take a wire off its port and put it on the one under `at`; nowhere
    /// under `at` disconnects it.
    RerouteWire {
        wire: WireRef,
        from: NodeId,
        at: Pos2,
    },
    /// Put a node of this kind at `at`, wired from `from`'s output.
    AddNode {
        label: &'static str,
        at: Pos2,
        from: Option<NodeId>,
    },
    /// The node's context menu, applied after the loop like every other
    /// structural edit.
    MutateNode(NodeId),
    SetOutput(NodeId),
    Duplicate(NodeId),
    Delete(NodeId),
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
    state.forget_missing(patch);
    // Before any widget sees the patch: what an undo goes back to.
    state.history.begin(patch);
    res.merge(toolbar(ui, patch, state, &style));
    if state.show_json {
        res.merge(json_io(ui, patch, &mut state.json, id.with("patch_json")));
    }
    state.ensure_layout(patch);

    // The scene takes the rest of the `Ui`, and draws on a layer over this
    // one: the ground and the border that says where the ground ends go
    // under it, where it will be.
    let ground = ui.available_rect_before_wrap();
    ui.painter().rect_filled(ground, 0.0, style.canvas_ground);
    ui.painter().rect_stroke(
        ground,
        0.0,
        Stroke::new(1.0, style.canvas_edge),
        egui::StrokeKind::Inside,
    );
    let mut scene_rect = state.scene_rect;
    let scene = egui::Scene::new().zoom_range(egui::Rangef::new(0.2, 2.0));
    let inner = scene.show(ui, &mut scene_rect, |ui| {
        canvas_contents(ui, patch, state, id, &style)
    });
    state.scene_rect = scene_rect;
    // `Scene` sets the layer's transform before it draws, so this is the
    // one the content just laid itself out under. Recorded rather than
    // looked up by the reader: `Memory::to_global` is keyed by layer, and a
    // host that draws two canvases (a patch and a sequence's instrument)
    // has no way to tell which entry is whose (crate #67).
    let layer = inner.response.layer_id;
    state.canvas_to_screen = ui
        .ctx()
        .memory(|m| m.to_global.get(&layer).copied())
        .unwrap_or(TSTransform::IDENTITY);
    res.merge(inner.inner);
    // This frame's widget edit, recorded before the keyboard is read so a
    // Ctrl+Z in the same frame steps back over it rather than swallowing it.
    if res.rebake {
        state.history.commit(patch);
    }
    let keys = canvas_keys(ui, patch, state, ground);
    if keys.rebake {
        // An undo or a redo leaves the history's baseline where it put the
        // patch, so this records only a key that *edited* — a delete, a
        // duplicate, a nudge.
        state.history.commit(patch);
    }
    res.merge(keys);
    res
}

/// How far an arrow key moves the selected node, in scene units; Shift
/// makes it one unit, for placing a box exactly.
const NUDGE: f32 = 8.0;

/// The canvas's keyboard, read only while the canvas owns it (#60,
/// Overlands #1333 B12).
///
/// Ownership is the pointer being over the canvas with nothing in it
/// holding text focus — type a name into the JSON box and Delete belongs to
/// the text, not to the selected node. Keys are taken with
/// `consume_key`, so a key the canvas acts on does not also reach a widget
/// behind it; a host reading the platform's keys instead of egui's cannot
/// see that, which is what [`PatchEditorState::wants_keyboard`] is for.
///
/// Order matters: `consume_key` ignores *extra* Shift and Alt, so Ctrl+Z
/// would swallow Ctrl+Shift+Z if undo were matched first. Most specific
/// first, as egui's own docs say.
fn canvas_keys(
    ui: &mut egui::Ui,
    patch: &mut AudioPatch,
    state: &mut PatchEditorState,
    ground: Rect,
) -> EditorResponse {
    use egui::{Key, Modifiers};

    let mut res = EditorResponse::NONE;
    state.took_escape = false;
    let typing = ui.memory(|m| m.focused()).is_some();
    state.owns_keys = !typing && ui.rect_contains_pointer(ground);
    if !state.owns_keys {
        return res;
    }

    let shift_arrow = Modifiers::SHIFT;
    let keys = ui.input_mut(|i| Keys {
        redo: i.consume_key(Modifiers::COMMAND | Modifiers::SHIFT, Key::Z)
            | i.consume_key(Modifiers::COMMAND, Key::Y),
        undo: i.consume_key(Modifiers::COMMAND, Key::Z),
        duplicate: i.consume_key(Modifiers::COMMAND, Key::D),
        delete: i.consume_key(Modifiers::NONE, Key::Delete)
            | i.consume_key(Modifiers::NONE, Key::Backspace),
        fit: i.consume_key(Modifiers::NONE, Key::F),
        escape: i.consume_key(Modifiers::NONE, Key::Escape),
        nudge: [
            (Key::ArrowLeft, Vec2::new(-1.0, 0.0)),
            (Key::ArrowRight, Vec2::new(1.0, 0.0)),
            (Key::ArrowUp, Vec2::new(0.0, -1.0)),
            (Key::ArrowDown, Vec2::new(0.0, 1.0)),
        ]
        .into_iter()
        .filter(|(key, _)| i.consume_key(shift_arrow, *key) || i.consume_key(Modifiers::NONE, *key))
        .map(|(_, dir)| dir)
        .fold(Vec2::ZERO, |a, b| a + b),
        fine: i.modifiers.shift,
    });

    if keys.redo {
        res.rebake |= state.history.redo(patch);
        res.changed |= res.rebake;
        return res;
    }
    if keys.undo {
        res.rebake |= state.history.undo(patch);
        res.changed |= res.rebake;
        return res;
    }
    if keys.fit {
        // A zero-size rect makes Scene auto-fit to the content next frame.
        state.scene_rect = Rect::ZERO;
    }
    // One step per press, most recent first: the Add menu was opened by
    // the last gesture, a picked wire by the last click before that.
    if keys.escape
        && (state.add_menu.take().is_some()
            || state.selected_wire.take().is_some()
            || state.selected.take().is_some())
    {
        state.took_escape = true;
    }
    // Delete acts on whatever the last click picked, and picking a wire
    // clears the node selection, so the two never compete.
    if keys.delete
        && let Some(wire) = state.selected_wire.clone()
        && remove_wire(patch, &wire)
    {
        state.forget_removed_wire(patch);
        res.changed = true;
        res.rebake = true;
    }
    if let Some(sel) = state.selected {
        if keys.duplicate
            && let Some(copy) = duplicate_node(patch, state, sel)
        {
            state.selected = Some(copy);
            res.changed = true;
            res.rebake = true;
        }
        if keys.delete && patch.graph.nodes.len() > 1 {
            delete_node(patch, sel);
            state.forget_missing(patch);
            res.changed = true;
            res.rebake = true;
        }
        if keys.nudge != Vec2::ZERO {
            let step = if keys.fine { 1.0 } else { NUDGE };
            *state.positions.entry(sel).or_default() += keys.nudge * step;
            // A nudged node is placed by hand from now on.
            state.auto.remove(&sel);
            res.changed = true;
            res.rebake = true;
        }
    }
    res
}

/// What [`canvas_keys`] found in one frame's input.
struct Keys {
    undo: bool,
    redo: bool,
    duplicate: bool,
    delete: bool,
    fit: bool,
    escape: bool,
    /// Summed direction of the arrows pressed this frame.
    nudge: Vec2,
    /// Shift is down: nudge by one unit rather than [`NUDGE`].
    fine: bool,
}

/// Copy `target` beside itself, with its parameters and the wires *into*
/// it, and return the copy's id.
///
/// The incoming wires are the point: a filter duplicated without them is a
/// filter you have to re-wire, which is most of the work the duplicate was
/// meant to save. Outgoing wires are not copied — two nodes feeding the
/// same port would double that port, which is a change to the sound rather
/// than a copy of a node.
fn duplicate_node(
    patch: &mut AudioPatch,
    state: &mut PatchEditorState,
    target: NodeId,
) -> Option<NodeId> {
    let source = patch.graph.nodes.iter().find(|n| n.id == target)?;
    let new_id = NodeId(
        patch
            .graph
            .nodes
            .iter()
            .map(|n| n.id.0)
            .max()
            .map_or(0, |m| m + 1),
    );
    let copy = GraphNode {
        id: new_id,
        kind: source.kind.clone(),
        inputs: source.inputs.clone(),
    };
    patch.graph.nodes.push(copy);
    // Beside the original, and placed by hand: the auto-layout would
    // otherwise stack it into a column and the copy would appear somewhere
    // other than where it was made.
    let at = state
        .positions
        .get(&target)
        .copied()
        .unwrap_or(LAYOUT_ORIGIN)
        + Vec2::splat(NODE_GAP);
    state.positions.insert(new_id, at);
    Some(new_id)
}

/// The canvas's grid, drawn inside the scene so it moves with the ground:
/// without it a drag on an empty canvas changed nothing anyone could see,
/// and there was no way to tell how far the view had been carried (#59,
/// Overlands #1332 B16).
///
/// Capped at [`GRID_MAX_LINES`]: zoomed right out the view covers more
/// ground than there is any use in ruling.
fn paint_grid(ui: &egui::Ui, style: &EditorStyle) {
    let view = ui.clip_rect();
    if !view.is_finite() || !view.is_positive() {
        return;
    }
    let painter = ui.painter();
    let stroke = Stroke::new(1.0, style.canvas_grid);
    let first = |v: f32| (v / GRID_STEP).floor() * GRID_STEP;
    let mut drawn = 0;
    let mut x = first(view.left());
    while x <= view.right() && drawn < GRID_MAX_LINES {
        painter.vline(x, view.y_range(), stroke);
        x += GRID_STEP;
        drawn += 1;
    }
    let mut y = first(view.top());
    while y <= view.bottom() && drawn < GRID_MAX_LINES {
        painter.hline(view.x_range(), y, stroke);
        y += GRID_STEP;
        drawn += 1;
    }
}

/// One toolbar row above the canvas: add, delete, tidy and fit, the output
/// picker, a More menu for what is not needed every minute, and the live
/// validity readout at the right.
///
/// It was four rows of chrome before the first node — the buttons, the
/// genetics controls, the validity line and the JSON fold — which is what
/// the canvas is for (#59, Overlands #1332 B9). Its buttons say what they
/// do in words (#58), and a disabled Delete says why through
/// `on_disabled_hover_text`, the only hover egui shows on a disabled
/// widget (#1289).
fn toolbar(
    ui: &mut egui::Ui,
    patch: &mut AudioPatch,
    state: &mut PatchEditorState,
    style: &EditorStyle,
) -> EditorResponse {
    let mut res = EditorResponse::NONE;
    // The check and the cross are Overlands' `affordances::CHECK` and
    // `CROSS`, its glyphs for valid and failed.
    let heard = heard_nodes(&patch.graph).len();
    let (colour, line) = match topo_sort(&patch.graph) {
        // What is there, and what of it the patch plays: "3 nodes" is true
        // of a patch where two of them bake into nothing (#61, Overlands
        // #1334 B13). "3 nodes, 3 heard" reads as an arithmetic puzzle in
        // the one case where there is nothing to look for, so that case
        // says so in words.
        Ok(order) if heard >= order.len() => (
            style.ok,
            format!(
                "\u{2714} valid graph \u{2014} {} nodes, all heard",
                order.len()
            ),
        ),
        Ok(order) => (
            style.ok,
            format!(
                "\u{2714} valid graph \u{2014} {} nodes, {heard} heard",
                order.len()
            ),
        ),
        // Named, and outlined on the canvas below (#57).
        Err(e) => (
            style.error,
            format!("\u{2716} {}", describe_graph_error(&patch.graph, &e)),
        ),
    };
    let chip_width = ui
        .painter()
        .layout_no_wrap(
            line.clone(),
            egui::TextStyle::Body.resolve(ui.style()),
            colour,
        )
        .rect
        .width();
    // Set inside the row, read after it: what was left of the row when the
    // buttons had taken their share.
    let mut room_left = 0.0;
    ui.horizontal_wrapped(|ui| {
        // The kind is chosen where the node is asked for, rather than a
        // Sine dropped in the middle of the view followed by a trip to the
        // title's kind combo (#61, Overlands #1334 B6).
        let mut chosen = None;
        ui.menu_button("Add node", |ui| {
            chosen = add_menu_items(ui);
        })
        .response
        .on_hover_text("Put a new node in the middle of the view");
        if let Some(label) = chosen {
            // Near the centre of the current view so it is visible, and out
            // of `auto`: the user put it there.
            add_node(patch, state, label, state.scene_rect.center(), None);
            res.changed = true;
            res.rebake = true;
        }

        // Delete acts on whatever is picked — a node, or since #61 a wire —
        // so it says which, and says why when it can do neither (#1289).
        let last_node = patch.graph.nodes.len() <= 1;
        let wire = state.selected_wire.clone();
        let node = state.selected.filter(|_| !last_node);
        let (can_delete, hover) = match (&wire, node) {
            (Some(w), _) => (
                true,
                format!(
                    "Remove the wire into {} \u{00B7} {}",
                    node_name(&patch.graph, w.to),
                    w.port
                ),
            ),
            (None, Some(_)) => (
                true,
                "Remove the selected node and any wires into it".to_string(),
            ),
            (None, None) => (false, String::new()),
        };
        let why_not = if last_node && state.selected.is_some() {
            "A patch keeps at least one node"
        } else {
            "Select a node or a wire first: click a node's number, or click a wire"
        };
        if ui
            .add_enabled(can_delete, egui::Button::new("Delete"))
            .on_hover_text(hover)
            .on_disabled_hover_text(why_not)
            .clicked()
        {
            if let Some(w) = wire {
                if remove_wire(patch, &w) {
                    state.forget_removed_wire(patch);
                    res.changed = true;
                    res.rebake = true;
                }
            } else if let Some(sel) = node {
                delete_node(patch, sel);
                state.forget_missing(patch);
                res.changed = true;
                res.rebake = true;
            }
        }

        if ui
            .button("Tidy")
            .on_hover_text("Lay every node out again in columns, sources to output")
            .clicked()
        {
            state.auto = patch.graph.nodes.iter().map(|n| n.id).collect();
            state.place_auto(patch);
        }

        if ui
            .button("Fit view")
            .on_hover_text("Pan and zoom so every node is in view")
            .clicked()
        {
            // A zero-size rect makes Scene auto-fit to the content next frame.
            state.scene_rect = Rect::ZERO;
        }

        ui.separator();
        ui.label("Output:");
        let ids: Vec<NodeId> = patch.graph.nodes.iter().map(|n| n.id).collect();
        egui::ComboBox::from_id_salt("canvas_output_select")
            .selected_text(node_name(&patch.graph, patch.graph.output))
            .show_ui(ui, |ui| {
                for nid in ids {
                    let name = node_name(&patch.graph, nid);
                    if ui
                        .selectable_label(nid == patch.graph.output, name)
                        .clicked()
                    {
                        patch.graph.output = nid;
                        res.changed = true;
                        res.rebake = true;
                    }
                }
            });

        ui.separator();
        res.merge(more_menu(ui, patch, state));

        // Room for the validity line at the right of the row?
        room_left = ui.max_rect().right() - ui.next_widget_position().x;
        if room_left >= chip_width + ui.spacing().item_spacing.x {
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.colored_label(colour, line.clone());
            });
        }
    });
    // Too narrow a row to hold it: under the row, on a line of its own.
    //
    // Neither of the obvious ways works here. A right-to-left layout takes
    // whatever width is left however little that is and draws inside it,
    // which on the sequence editor's side panel (about 480 points) drew the
    // whole line across the Output picker and the More button. Appending it
    // to the wrapping row instead wraps the *text*, which spreads it over
    // the buttons rather than moving it off them.
    if room_left < chip_width + ui.spacing().item_spacing.x {
        ui.colored_label(colour, line);
    }
    res
}

/// The toolbar's More menu: the genetics controls and the JSON box, which
/// are not wanted every minute and cost two of the four rows the toolbar
/// used to be.
fn more_menu(
    ui: &mut egui::Ui,
    patch: &mut AudioPatch,
    state: &mut PatchEditorState,
) -> EditorResponse {
    let mut res = EditorResponse::NONE;
    ui.menu_button("More", |ui| {
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
        ui.separator();
        if ui
            .selectable_label(state.show_json, "Import / Export JSON")
            .on_hover_text("Show the JSON box under the toolbar")
            .clicked()
        {
            state.show_json = !state.show_json;
        }
    });
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

    paint_grid(ui, style);

    // Any layout owed from the last frame, before a box is drawn: a
    // position may not change between egui's passes over one frame. egui
    // runs the frame a second time whenever something in it asks for one —
    // a `Grid` measuring its columns does, and every node body is one — and
    // a box that moved in between left a widget's rect holding a different
    // widget's id, which egui reports pass by pass.
    if state.relayout && state.auto.iter().all(|id| state.sizes.contains_key(id)) {
        state.place_auto(patch);
        state.relayout = false;
    }

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
    // Every node's name, taken before the loop borrows the nodes mutably:
    // a wire row says where it comes from by name, not by number (#59 B3).
    let names: HashMap<NodeId, String> = patch
        .graph
        .nodes
        .iter()
        .map(|n| (n.id, format!("#{} {}", n.id.0, node_kind_label(&n.kind))))
        .collect();
    // The boxes a broken graph is broken at, outlined in the host's error
    // colour over the selection and output borders: they are what has to
    // change before anything bakes.
    let at_fault: HashSet<NodeId> = match topo_sort(&patch.graph) {
        Ok(_) => HashSet::new(),
        Err(e) => nodes_at_fault(&patch.graph, &e).into_iter().collect(),
    };
    let error_stroke = Stroke::new(2.0, style.error);
    // What the patch actually plays. A node the output cannot be reached
    // from bakes into nothing, and said so nowhere (#61, Overlands #1334
    // B13).
    let heard = heard_nodes(&patch.graph);

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
            // Thicker, not just another colour: the output's mark is the
            // OUT badge in its title, so one node can show both (#59 B8).
            Stroke::new(SELECTED_STROKE, style.node_selected)
        } else {
            Stroke::new(1.0, style.node_stroke)
        };

        let mut child = ui.new_child(
            UiBuilder::new()
                .max_rect(Rect::from_min_size(pos, Vec2::new(NODE_MIN_WIDTH, 10.0)))
                .id_salt(("patch_node", nid.0))
                .layout(Layout::top_down(Align::Min)),
        );
        child.set_width(NODE_MIN_WIDTH);

        let frame = egui::Frame::group(child.style())
            .fill(style.node_fill)
            .stroke(stroke);
        // Where the box's rules go. They are painted after the frame is
        // measured: `Ui::separator` is only as wide as the `Ui` it is in,
        // which is the box's minimum width, so on every box that grew past
        // it the rule stopped short of the border.
        let mut rules: Vec<f32> = Vec::new();
        let not_heard = !heard.contains(&nid);
        let fr = frame.show(&mut child, |ui| {
            // A node nothing hears is still a node, and still editable, so
            // it fades rather than disappears — and its frame does not,
            // because it is still a box and still where it was.
            if not_heard {
                ui.set_opacity(NOT_HEARD_OPACITY);
            }
            // Title bar: the number is the grip (drag to move, click to
            // select), the kind beside it is the picker, and the badge says
            // whether the patch plays this node.
            let title_row = ui.horizontal(|ui| {
                let grip = ui
                    .add(
                        egui::Label::new(
                            egui::RichText::new(format!("#{}", nid.0))
                                .strong()
                                .color(style.node_title),
                        )
                        .sense(Sense::click_and_drag()),
                    )
                    .on_hover_text(
                        "Drag to move this node, click to select it, \
                         right-click for what can be done to it",
                    );
                if grip.dragged() {
                    actions.push(Action::Move(nid, grip.drag_delta()));
                }
                if grip.clicked() {
                    actions.push(Action::Select(nid));
                }
                // The per-node actions, on the title rather than on it: a
                // one-click Mutate used to sit a pixel from the drag handle
                // with nothing to undo it (#60, Overlands #1333 B7).
                grip.context_menu(|ui| {
                    for (label, action) in [
                        ("Mutate this node", Action::MutateNode(nid)),
                        ("Set as output", Action::SetOutput(nid)),
                        ("Duplicate", Action::Duplicate(nid)),
                        ("Delete", Action::Delete(nid)),
                    ] {
                        if ui.button(label).clicked() {
                            actions.push(action);
                            ui.close();
                        }
                    }
                });
                res.merge(node_kind_picker(ui, &mut node.kind, Id::new(("nk", nid.0))));
                if output_id == nid {
                    out_badge(ui, style);
                }
                if not_heard {
                    // At full strength inside a dimmed box: the badge is
                    // the one thing in it that says why the rest is faded.
                    ui.scope(|ui| {
                        ui.set_opacity(1.0);
                        ui.add(
                            egui::Label::new(
                                egui::RichText::new(NOT_HEARD)
                                    .small()
                                    .color(style.node_title),
                            )
                            .sense(Sense::hover()),
                        )
                        .on_hover_text(
                            "Nothing plays this node: the output cannot be reached from it. \
                             Wire it towards the output, or make it the output.",
                        );
                    });
                }
            });
            rule(ui, &mut rules);
            res.merge(node_kind_body(ui, &mut node.kind));
            let (conn_res, rows) = connection_editor(ui, node, &names, &mut rules);
            res.merge(conn_res);
            (title_row.response.rect, rows)
        });

        let rect = fr.response.rect;
        let (title_rect, rows) = fr.inner;
        // The rules, now that the box's width is known.
        let inset = f32::from(egui::Frame::group(ui.style()).inner_margin.left);
        for y in rules {
            painter.hline(
                (rect.left() + inset)..=(rect.right() - inset),
                y,
                Stroke::new(1.0, style.node_stroke),
            );
        }
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

    // --- wires: where they run, and what the pointer is doing to one ----
    state.wires = collect_wires(&patch.graph, &state.ports, &state.boxes, &state.outputs);
    state.forget_removed_wire(patch);

    // The canvas is a transformed layer, so the pointer arrives in screen
    // points and everything remembered here is in canvas units.
    let from_global = ui.ctx().layer_transform_from_global(ui.layer_id());
    let zoom = ui
        .ctx()
        .layer_transform_to_global(ui.layer_id())
        .map_or(1.0, |t| t.scaling)
        .max(f32::EPSILON);
    let pointer = ui
        .ctx()
        .pointer_latest_pos()
        .map(|at| from_global.map_or(at, |t| t * at));
    // Whether something is drawn over the canvas where the pointer is: a
    // menu, a tooltip, the amount editor. `Ui::rect_contains_pointer`
    // cannot answer this here — it asks `Context::layer_id_at`, which knows
    // only `Area`s, and a `Scene`'s layer is not one, so it says no
    // wherever the pointer is.
    let covered = ui.ctx().pointer_latest_pos().is_some_and(|screen| {
        ui.ctx()
            .layer_id_at(screen)
            .is_some_and(|over| over.order > ui.layer_id().order)
    });
    // A wire is behind every box, so a pointer over one is not on a wire
    // however close the curve passes; and while a wire is being dragged or
    // a menu is open the pointer is spoken for.
    let free_pointer = pointer.filter(|at| {
        ui.clip_rect().contains(*at)
            && !covered
            && !state.boxes.iter().any(|(_, r)| r.contains(*at))
            && dragging.is_none()
            && state.add_menu.is_none()
    });

    // --- take a wire off its port by its end, to re-route or disconnect --
    let mut rerouting: Option<WireRef> = None;
    for wire in &state.wires {
        let grab = wire_grab(wire);
        let resp = ui.interact(
            Rect::from_center_size(grab, Vec2::splat(2.0 * WIRE_GRAB_RADIUS / zoom)),
            id.with(("wire_end", wire.to.0, wire.port.as_str(), wire.index)),
            Sense::drag(),
        );
        // Named inside the closure: it runs only when something is reading
        // the tree, and this is per wire per frame.
        resp.widget_info(|| {
            egui::WidgetInfo::labeled(
                egui::WidgetType::Other,
                true,
                format!("End of {}", wire_name(wire, &names)),
            )
        });
        // `interact_pointer_pos` is already in this layer's own
        // coordinates — canvas units — as the output port's drag below
        // relies on too. `Context::pointer_latest_pos` is not, which is
        // why the hover below goes through `from_global` and this does
        // not.
        if resp.dragged()
            && let Some(at) = resp.interact_pointer_pos()
        {
            rerouting = Some(wire.wire_ref());
            dragging = Some((wire.from, at, resp.clone()));
        }
        if resp.drag_stopped()
            && let Some(at) = resp.interact_pointer_pos()
        {
            actions.push(Action::RerouteWire {
                wire: wire.wire_ref(),
                from: wire.from,
                at,
            });
        }
    }

    let hovered: Option<WireRef> = free_pointer
        .and_then(|at| wire_at(&state.wires, at, WIRE_HIT_DIST / zoom))
        .map(WireGeom::wire_ref);
    // The wire the cross and the amount editor are on: the picked one, or
    // failing that the one under the pointer.
    let marked: Option<WireRef> = state.selected_wire.clone().or_else(|| hovered.clone());

    // The wire being taken off its port is drawn by the drag overlay
    // below, from its source to the pointer, so it is not also drawn in
    // place: a re-route should look like one wire moving, not two.
    painter.set(
        wire_idx,
        egui::Shape::Vec(
            state
                .wires
                .iter()
                .filter(|w| rerouting.as_ref() != Some(&w.wire_ref()))
                .map(|w| {
                    let lit = marked.as_ref() == Some(&w.wire_ref());
                    egui::Shape::line(
                        w.points.clone(),
                        Stroke::new(
                            if lit { 3.0 } else { 2.0 },
                            if lit { style.wire_active } else { style.wire },
                        ),
                    )
                })
                .collect(),
        ),
    );

    if let Some(pick) = &hovered
        && ui.input(|i| i.pointer.primary_clicked())
    {
        state.selected_wire = Some(pick.clone());
        // One selection at a time: Delete has to act on the thing the last
        // click was aimed at.
        state.selected = None;
    }

    // The hovered wire says what it is, at the pointer — unless it is the
    // picked one, whose amount editor already names it a few points away
    // and which the tooltip would sit on top of.
    //
    // Never wrapped: an auto-sized area offers the width of its last pass,
    // so a longer name than the last would wrap and the box would ratchet
    // narrower as the pointer crossed wires (#1290).
    if let Some(hover) = &hovered
        && state.selected_wire.as_ref() != Some(hover)
        && let Some(wire) = state.wires.iter().find(|w| &w.wire_ref() == hover)
    {
        let text = wire_tooltip(wire, &patch.graph, &names);
        egui::Tooltip::always_open(
            ui.ctx().clone(),
            ui.layer_id(),
            id.with("wire_tooltip"),
            egui::PopupAnchor::Pointer,
        )
        .show(|ui| ui.add(egui::Label::new(text).extend()));
    }

    // --- the picked wire's own panel: what it is, its amount, its cross --
    if let Some(pick) = state.selected_wire.clone()
        && let Some(wire) = state.wires.iter().find(|w| w.wire_ref() == pick)
    {
        let mid = wire.midpoint();
        let at = ui
            .ctx()
            .layer_transform_to_global(ui.layer_id())
            .map_or(mid, |t| t * mid);
        let (edit, remove) = wire_panel(ui, patch, id, &pick, &names, at);
        res.merge(edit);
        if remove {
            actions.push(Action::RemoveWire(pick.clone()));
        }
    }

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
        // A tooltip is drawn in screen space, so the name stays legible
        // at any zoom of the canvas.
        let text = match target {
            Some(target) => {
                let kind = titles.get(&target.node).copied().unwrap_or_default();
                format!(
                    "\u{27A1} #{} {kind} \u{00B7} {}",
                    target.node.0, target.port
                )
            }
            // Over nothing, a release does not throw the wire away: it
            // offers to make what the wire would drive (#61, Overlands
            // #1334 B6).
            None if rerouting.is_none() => "\u{27A1} let go for a new node here".to_string(),
            None => "\u{27A1} let go to disconnect".to_string(),
        };
        // Never wrapped: an auto-sized area offers the width of its last
        // pass, so a name longer than the last one would wrap and the
        // box would ratchet narrower as the pointer crosses rows.
        egui::Tooltip::for_widget(&out_resp)
            .at_pointer()
            .show(|ui| ui.add(egui::Label::new(text).extend()));
    }

    // --- the Add menu, where it was asked for ---------------------------
    // A right-click on clear canvas offers what can be put there. Clear
    // canvas: the pointer is over the ground, over no box, and no wire is
    // being dragged — a right-click on a node opens that node's own menu.
    if free_pointer.is_some()
        && hovered.is_none()
        && ui.input(|i| i.pointer.secondary_clicked())
        && let Some(at) = free_pointer
    {
        state.add_menu = Some(AddMenu {
            at,
            from: None,
            shown: false,
        });
    }
    if let Some(menu) = state.add_menu.clone() {
        match add_menu_area(ui, id, &menu) {
            AddMenuOutcome::Chose(label) => {
                actions.push(Action::AddNode {
                    label,
                    at: menu.at,
                    from: menu.from,
                });
                state.add_menu = None;
            }
            AddMenuOutcome::Dismissed => state.add_menu = None,
            AddMenuOutcome::Open => {
                if let Some(open) = &mut state.add_menu {
                    open.shown = true;
                }
            }
        }
    }

    // --- apply deferred structural edits -------------------------------
    for action in actions {
        match action {
            Action::Select(nid) => {
                state.selected = Some(nid);
                // One selection at a time, both ways round: Delete acts on
                // whatever the last click was aimed at.
                state.selected_wire = None;
            }
            Action::Move(nid, delta) => {
                *state.positions.entry(nid).or_default() += delta;
            }
            Action::CompleteWire { from, at } => {
                match drop_target(&state.ports, &state.boxes, from, at) {
                    Some(target) => {
                        if let Some(n) = patch.graph.nodes.iter_mut().find(|n| n.id == target.node)
                        {
                            n.inputs
                                .entry(target.port.clone())
                                .or_default()
                                .push(Connection::from_node(from));
                            res.changed = true;
                            res.rebake = true;
                        }
                    }
                    // Let go over nothing and the wire is not thrown away:
                    // it asks what should be there (#61, Overlands #1334
                    // B6).
                    None => {
                        state.add_menu = Some(AddMenu {
                            at,
                            from: Some(from),
                            shown: false,
                        })
                    }
                }
            }
            Action::RemoveWire(wire) => {
                if remove_wire(patch, &wire) {
                    state.forget_removed_wire(patch);
                    res.changed = true;
                    res.rebake = true;
                }
            }
            Action::RerouteWire { wire, from, at } => {
                let target = drop_target(&state.ports, &state.boxes, from, at)
                    .map(|t| (t.node, t.port.clone()));
                // Off its old port either way: dropped on a port it moves
                // there, dropped on nothing it is gone.
                let amount = remove_wire_amount(patch, &wire);
                if amount.is_some() {
                    state.forget_removed_wire(patch);
                    res.changed = true;
                    res.rebake = true;
                }
                if let (Some(amount), Some((node, port))) = (amount, target)
                    && let Some(n) = patch.graph.nodes.iter_mut().find(|n| n.id == node)
                {
                    n.inputs
                        .entry(port)
                        .or_default()
                        .push(Connection::Node { id: from, amount });
                }
            }
            Action::AddNode { label, at, from } => {
                add_node(patch, state, label, at, from);
                res.changed = true;
                res.rebake = true;
            }
            Action::MutateNode(nid) => {
                if let Some(n) = patch.graph.nodes.iter_mut().find(|n| n.id == nid) {
                    mutate_node_kind(&mut n.kind, &mut fresh_rng(), mutate_rate);
                    res.changed = true;
                    res.rebake = true;
                }
            }
            Action::SetOutput(nid) => {
                if patch.graph.output != nid {
                    patch.graph.output = nid;
                    res.changed = true;
                    res.rebake = true;
                }
            }
            Action::Duplicate(nid) => {
                if let Some(copy) = duplicate_node(patch, state, nid) {
                    state.selected = Some(copy);
                    res.changed = true;
                    res.rebake = true;
                }
            }
            Action::Delete(nid) => {
                if patch.graph.nodes.len() > 1 {
                    delete_node(patch, nid);
                    state.forget_missing(patch);
                    res.changed = true;
                    res.rebake = true;
                }
            }
        }
    }

    // This frame's box sizes. A box placed before it had ever been drawn
    // was placed at a guessed size, and a box whose kind changed under the
    // title picker is a different size than it was; either way the
    // auto-placed boxes owe a layout, which the next frame opens with
    // (#59 B4).
    for (id, rect) in &state.boxes {
        let size = rect.size();
        let moved = state
            .sizes
            .insert(*id, size)
            .is_none_or(|was| (was - size).length() > 0.5);
        state.relayout |= moved && state.auto.contains(id);
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

/// Every wire the graph asks for, where the canvas drew its ends.
///
/// A port with connections always has a row, so its dot is where the wire
/// ends; the box's left centre is only a last resort, for a frame in which
/// a node has been added but not yet measured.
fn collect_wires(
    graph: &NodeGraph,
    ports: &[PortGeom],
    boxes: &[(NodeId, Rect)],
    outputs: &HashMap<NodeId, Pos2>,
) -> Vec<WireGeom> {
    let mut wires = Vec::new();
    for node in &graph.nodes {
        for (port, conns) in &node.inputs {
            let dst = ports
                .iter()
                .find(|p| p.node == node.id && p.port == *port)
                .map(|p| p.dot)
                .or_else(|| {
                    boxes
                        .iter()
                        .find(|(id, _)| *id == node.id)
                        .map(|(_, r)| Pos2::new(r.left(), r.center().y))
                });
            let Some(dst) = dst else { continue };
            for (index, c) in conns.iter().enumerate() {
                if let Connection::Node { id: src, .. } = c
                    && let Some(src_pos) = outputs.get(src)
                {
                    wires.push(WireGeom {
                        from: *src,
                        to: node.id,
                        port: port.clone(),
                        index,
                        points: wire_points(*src_pos, dst),
                    });
                }
            }
        }
    }
    wires
}

/// Where a wire is taken hold of to move its end: along the curve, short of
/// the port it lands on.
///
/// Short of it, and not on it, because a port's dot is shared — a port holds
/// a list of connections summed at bake time, so several wires can end on
/// one dot, and each needs somewhere of its own to be grabbed.
fn wire_grab(wire: &WireGeom) -> Pos2 {
    let last = wire.points.len().saturating_sub(1);
    wire.points
        .get(last * 4 / 5)
        .copied()
        .unwrap_or_else(|| wire.midpoint())
}

/// `"the wire from #1 LFO to #2 Lowpass \u{00B7} cutoff_hz"`: a wire in
/// words, for a hover text and for assistive tech.
fn wire_name(wire: &WireGeom, names: &HashMap<NodeId, String>) -> String {
    let named = |id: NodeId| {
        names
            .get(&id)
            .cloned()
            .unwrap_or_else(|| format!("#{}", id.0))
    };
    format!(
        "the wire from {} to {} \u{00B7} {}",
        named(wire.from),
        named(wire.to),
        wire.port
    )
}

/// What a hovered wire says: both ends, the port it drives and how much of
/// the signal reaches it.
fn wire_tooltip(wire: &WireGeom, graph: &NodeGraph, names: &HashMap<NodeId, String>) -> String {
    let named = |id: NodeId| {
        names
            .get(&id)
            .cloned()
            .unwrap_or_else(|| format!("#{}", id.0))
    };
    let amount = connection_of(graph, &wire.wire_ref()).and_then(|c| match c {
        Connection::Node { amount, .. } => Some(*amount),
        Connection::Constant { .. } => None,
    });
    match amount {
        Some(amount) => format!(
            "{} \u{27A1} {} \u{00B7} {} \u{00D7} {amount}",
            named(wire.from),
            named(wire.to),
            wire.port
        ),
        None => format!(
            "{} \u{27A1} {} \u{00B7} {}",
            named(wire.from),
            named(wire.to),
            wire.port
        ),
    }
}

/// The connection a [`WireRef`] names, if the patch still holds it.
fn connection_of<'a>(graph: &'a NodeGraph, wire: &WireRef) -> Option<&'a Connection> {
    graph
        .nodes
        .iter()
        .find(|n| n.id == wire.to)?
        .inputs
        .get(&wire.port)?
        .get(wire.index)
}

/// Take the connection a [`WireRef`] names out of the patch. `true` when
/// there was one.
fn remove_wire(patch: &mut AudioPatch, wire: &WireRef) -> bool {
    remove_wire_amount(patch, wire).is_some()
}

/// Take the connection out and hand back the amount it carried, so a
/// re-route can put the same wire on another port rather than a fresh one.
fn remove_wire_amount(patch: &mut AudioPatch, wire: &WireRef) -> Option<f32> {
    let node = patch.graph.nodes.iter_mut().find(|n| n.id == wire.to)?;
    let conns = node.inputs.get_mut(&wire.port)?;
    let amount = match conns.get(wire.index)? {
        Connection::Node { amount, .. } => *amount,
        Connection::Constant { .. } => return None,
    };
    conns.remove(wire.index);
    if conns.is_empty() {
        node.inputs.remove(&wire.port);
    }
    Some(amount)
}

/// The picked wire's own panel, at the middle of its curve: what it drives,
/// its amount, and the cross that removes it (#61, Overlands #1334 B2).
///
/// The amount is edited here rather than found again in the destination
/// box's Inputs list, where the source was a number and the amount a bare
/// "amt". All three together, and not a cross floating on the curve beside
/// the panel: the middle of the curve is exactly where the pointer is when
/// a wire has just been picked, so a separate cross there is hovered the
/// moment the panel opens and its own hover text covers the panel. An
/// amount and a cross side by side is also the pairing the Inputs list
/// already uses.
///
/// An `Area` in screen space: the canvas is a transformed layer, and a
/// panel of numbers that shrank with the zoom would stop being readable at
/// exactly the zoom where a wire is hardest to pick out. Every widget in it
/// is fixed-width for the same reason the tooltips are — an auto-sized area
/// offers its last pass's width, so wrappable content ratchets narrow
/// (#1290).
/// Returns the edit made, and whether the cross was clicked.
fn wire_panel(
    ui: &mut egui::Ui,
    patch: &mut AudioPatch,
    id: Id,
    wire: &WireRef,
    names: &HashMap<NodeId, String>,
    at: Pos2,
) -> (EditorResponse, bool) {
    let mut res = EditorResponse::NONE;
    let mut remove = false;
    let Some(node) = patch.graph.nodes.iter_mut().find(|n| n.id == wire.to) else {
        return (res, remove);
    };
    let Some(Connection::Node { id: from, amount }) = node
        .inputs
        .get_mut(&wire.port)
        .and_then(|conns| conns.get_mut(wire.index))
    else {
        return (res, remove);
    };
    let from = *from;
    let named = |id: NodeId| {
        names
            .get(&id)
            .cloned()
            .unwrap_or_else(|| format!("#{}", id.0))
    };
    let title = format!("{} \u{27A1} {}", named(from), wire.port);
    egui::Area::new(id.with("wire_amount"))
        .order(egui::Order::Foreground)
        .fixed_pos(at + Vec2::new(WIRE_GRAB_RADIUS + 4.0, WIRE_GRAB_RADIUS + 4.0))
        .constrain(true)
        .show(ui.ctx(), |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.vertical(|ui| {
                    ui.add(egui::Label::new(egui::RichText::new(title).small()).extend());
                    ui.horizontal(|ui| {
                        let r = ui.add(egui::DragValue::new(amount).speed(0.05).prefix("amt "));
                        res.changed |= r.changed();
                        res.rebake |= r.drag_stopped() || (r.changed() && !r.dragged());
                        remove |= ui
                            .small_button("\u{2716}")
                            .on_hover_text("Remove this wire")
                            .clicked();
                    });
                });
            });
        });
    (res, remove)
}

/// What happened to an open Add menu this frame.
enum AddMenuOutcome {
    /// Still open.
    Open,
    /// A kind was chosen.
    Chose(&'static str),
    /// Clicked away from, or Escaped.
    Dismissed,
}

/// The Add menu, drawn where it was asked for.
///
/// In screen space, like the amount editor and for the same reason: a menu
/// that shrank with the zoom would be unreadable at the zoom where a node
/// is hardest to place. `at` is in canvas units, so it is taken to the
/// screen here.
fn add_menu_area(ui: &mut egui::Ui, id: Id, menu: &AddMenu) -> AddMenuOutcome {
    let at = ui
        .ctx()
        .layer_transform_to_global(ui.layer_id())
        .map_or(menu.at, |t| t * menu.at);
    let mut chosen = None;
    let area = egui::Area::new(id.with("add_menu"))
        .order(egui::Order::Foreground)
        .fixed_pos(at)
        .constrain(true)
        .show(ui.ctx(), |ui| {
            egui::Frame::menu(ui.style()).show(ui, |ui| {
                ui.vertical(|ui| {
                    if menu.from.is_some() {
                        ui.add(
                            egui::Label::new(
                                egui::RichText::new("Drive a new\u{2026}").small().weak(),
                            )
                            .extend(),
                        );
                    }
                    chosen = add_menu_items(ui);
                });
            });
        });
    if let Some(label) = chosen {
        return AddMenuOutcome::Chose(label);
    }
    // Escape, or a press anywhere but in the menu, puts it away. The press
    // and not the click, so a drag begun outside it closes it too — and not
    // before the menu has been drawn once, or the press that opened it
    // would be the press that closed it.
    let dismissed = menu.shown
        && (ui.input(|i| i.key_pressed(egui::Key::Escape))
            || (ui.input(|i| i.pointer.any_pressed()) && !area.response.contains_pointer()));
    if dismissed {
        AddMenuOutcome::Dismissed
    } else {
        AddMenuOutcome::Open
    }
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
fn connection_editor(
    ui: &mut egui::Ui,
    node: &mut GraphNode,
    names: &HashMap<NodeId, String>,
    rules: &mut Vec<f32>,
) -> (EditorResponse, Vec<PortRow>) {
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

    rule(ui, rules);
    ui.label(egui::RichText::new("Inputs").weak());

    let mut to_delete: Vec<(String, usize)> = Vec::new();
    let mut to_add_const: Vec<String> = Vec::new();

    for port in &ports {
        let name_row = ui
            .horizontal(|ui| {
                ui.label(format!("{port}:"));
                if ui
                    .small_button("Add constant")
                    .on_hover_text(format!("Drive {port} with a fixed value instead of a wire"))
                    .clicked()
                {
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
                            // Named, not pointed at: "from #1 LFO" says
                            // which node without the reader going to look
                            // up what #1 is (#59 B3).
                            let from = names
                                .get(id)
                                .cloned()
                                .unwrap_or_else(|| format!("#{}", id.0));
                            ui.label(format!("from {from}"));
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

/// Note where a rule across the node box goes, and leave room for it.
///
/// The line itself is painted once the frame has been measured: a rule is
/// as wide as the box, and `Ui::separator` can only be as wide as the `Ui`
/// it is in.
fn rule(ui: &mut egui::Ui, at: &mut Vec<f32>) {
    let space = ui.spacing().item_spacing.y;
    ui.add_space(space);
    at.push(ui.cursor().top());
    ui.add_space(space);
}

/// The badge on the title of the node the patch plays.
///
/// A badge and not a border, because the selection is a border: they were
/// one border in two colours, so the output stopped saying it was the
/// output the moment it was clicked (#59, Overlands #1332 B8).
fn out_badge(ui: &mut egui::Ui, style: &EditorStyle) {
    let resp = ui
        .add(
            egui::Label::new(
                egui::RichText::new("OUT")
                    .small()
                    .strong()
                    .color(style.node_output),
            )
            .sense(Sense::hover()),
        )
        .on_hover_text("The patch plays this node");
    ui.painter().rect_stroke(
        resp.rect.expand2(Vec2::new(3.0, 1.0)),
        3.0,
        Stroke::new(1.0, style.node_output),
        egui::StrokeKind::Outside,
    );
}

/// A cubic-bezier wire from `a` to `b`, sampled to a polyline with horizontal
/// control handles (the classic node-editor S-curve).
///
/// An even number of segments, so the sample at the middle of the list is
/// the curve at `t = 0.5` — which is where [`WireGeom::midpoint`] puts the
/// wire's cross.
fn wire_points(a: Pos2, b: Pos2) -> Vec<Pos2> {
    let handle = (b.x - a.x).abs().max(40.0) * 0.5;
    let c1 = Pos2::new(a.x + handle, a.y);
    let c2 = Pos2::new(b.x - handle, b.y);
    const SEGMENTS: usize = 18;
    (0..=SEGMENTS)
        .map(|i| cubic_bezier(a, c1, c2, b, i as f32 / SEGMENTS as f32))
        .collect()
}

fn wire_shape(a: Pos2, b: Pos2, color: Color32) -> egui::Shape {
    egui::Shape::line(wire_points(a, b), Stroke::new(2.0, color))
}

/// The distance from `at` to the segment `a`–`b`.
fn distance_to_segment(at: Pos2, a: Pos2, b: Pos2) -> f32 {
    let seg = b - a;
    let len_sq = seg.length_sq();
    if len_sq <= f32::EPSILON {
        return at.distance(a);
    }
    let t = ((at - a).dot(seg) / len_sq).clamp(0.0, 1.0);
    at.distance(a + seg * t)
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

    use crate::oscillator::SineOsc;
    use crate::ui::node::{KIND_LABELS, KindGroup};
    use crate::ui::style::tests::{AA, distinct_style};
    use crate::ui::style::{EditorStyle, set_editor_style, set_style};
    use crate::ui::test_paint::{
        button_labels, colours, contrast_on, glyph_labels, shapes, text_painted,
    };

    /// How far the driver's clock moves per frame. Long enough that two
    /// still frames outlast egui's tooltip delay.
    const FRAME_DT: f64 = 0.25;

    /// An AccessKit node's bounds as an egui [`Rect`], in whatever space
    /// its layer uses.
    fn accesskit_rect(node: &accesskit::Node) -> Option<Rect> {
        let b = node.bounds()?;
        Some(Rect::from_min_max(
            Pos2::new(b.x0 as f32, b.y0 as f32),
            Pos2::new(b.x1 as f32, b.y1 as f32),
        ))
    }

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
        /// Wall clock for the frames driven so far. egui holds a tooltip
        /// back by `interaction.tooltip_delay`, so a frame clock that never
        /// moves is a frame clock in which no hover text is ever shown.
        time: f64,
        /// The screen the canvas is drawn on.
        screen: Vec2,
    }

    impl Canvas {
        fn new(patch: AudioPatch) -> Self {
            let ctx = egui::Context::default();
            set_editor_style(&ctx, distinct_style());
            Self::on(ctx, patch)
        }

        /// The canvas on a screen `width` points wide: the sequence
        /// editor's side panel gives its patch canvas about 480.
        fn narrow(patch: AudioPatch, width: f32) -> Self {
            let ctx = egui::Context::default();
            set_editor_style(&ctx, distinct_style());
            let mut canvas = Self::on_for(ctx, patch, 0);
            canvas.screen = Vec2::new(width, 800.0);
            for _ in 0..3 {
                canvas.frame(Vec::new());
            }
            canvas
        }

        /// The canvas under `visuals`, with no style set.
        fn themed(patch: AudioPatch, visuals: egui::Visuals) -> Self {
            let ctx = egui::Context::default();
            ctx.set_visuals(visuals);
            Self::on(ctx, patch)
        }

        /// The canvas after exactly `frames` frames. The measured
        /// auto-layout settles in two (#59): the first frame places boxes
        /// at a guessed size and records the real one, the second re-places
        /// them and draws them there.
        fn after(patch: AudioPatch, frames: usize) -> Self {
            let ctx = egui::Context::default();
            set_editor_style(&ctx, distinct_style());
            Self::on_for(ctx, patch, frames)
        }

        fn on(ctx: egui::Context, patch: AudioPatch) -> Self {
            Self::on_for(ctx, patch, 3)
        }

        fn on_for(ctx: egui::Context, patch: AudioPatch, frames: usize) -> Self {
            ctx.enable_accesskit();
            let mut canvas = Self {
                ctx,
                patch,
                state: PatchEditorState::default(),
                out: egui::FullOutput::default(),
                time: 0.0,
                screen: Vec2::new(1600.0, 1200.0),
            };
            for _ in 0..frames {
                canvas.frame(Vec::new());
            }
            canvas
        }

        fn frame(&mut self, events: Vec<egui::Event>) -> EditorResponse {
            self.time += FRAME_DT;
            let Self {
                ctx,
                patch,
                state,
                out,
                time,
                screen,
            } = self;
            let mut res = EditorResponse::NONE;
            let input = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, *screen)),
                events,
                time: Some(*time),
                predicted_dt: FRAME_DT as f32,
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

        /// Every label's text and its rect as egui laid it out, with no
        /// transform applied.
        ///
        /// For a label inside the canvas that is scene-local (what
        /// [`PatchEditorState::boxes`] is in); for the toolbar, which is
        /// drawn in the untransformed base layer, it is the screen rect.
        /// [`Canvas::labels`] is the same list put in screen coordinates,
        /// which is right for the canvas and wrong for the toolbar.
        fn labels_untransformed(&self) -> Vec<(String, Rect)> {
            self.out
                .platform_output
                .accesskit_update
                .iter()
                .flat_map(|update| &update.nodes)
                .filter(|(_, n)| n.role() == accesskit::Role::Label)
                .filter_map(|(_, n)| Some((n.value()?.to_owned(), accesskit_rect(n)?)))
                .collect()
        }

        /// Every widget of `role` in the chrome above the canvas, with its
        /// screen rect: the toolbar's layer carries no transform.
        fn chrome(&self, role: accesskit::Role) -> Vec<(String, Rect)> {
            self.out
                .platform_output
                .accesskit_update
                .iter()
                .flat_map(|update| &update.nodes)
                .filter(|(_, n)| n.role() == role)
                .filter_map(|(_, n)| {
                    let text = n.label().or_else(|| n.value())?.to_owned();
                    Some((text, accesskit_rect(n)?))
                })
                .collect()
        }

        /// The one widget of `role` whose text is exactly `text`.
        fn chrome_rect(&self, role: accesskit::Role, text: &str) -> Rect {
            let found: Vec<Rect> = self
                .chrome(role)
                .into_iter()
                .filter(|(t, _)| t == text)
                .map(|(_, r)| r)
                .collect();
            assert_eq!(
                found.len(),
                1,
                "{role:?} reading {text:?}; all of them: {:?}",
                self.chrome(role)
            );
            found[0]
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

        /// The part of a node box its content may use: the painted frame
        /// inside the padding `egui::Frame::group` puts there, at this
        /// canvas's zoom.
        fn content_area(&self, node_box: Rect) -> Rect {
            let margin = f32::from(
                egui::Frame::group(&self.ctx.global_style())
                    .inner_margin
                    .left,
            );
            node_box.shrink(margin * self.to_screen().scaling)
        }

        /// Every text the frame painted and the rect its glyphs cover, on
        /// screen. A galley's bounding rect is its whole extent, clipped or
        /// not, so text that runs off a node box is visible here.
        fn painted_text_rects(&self) -> Vec<(String, Rect)> {
            self.shapes()
                .into_iter()
                .filter_map(|s| match s {
                    egui::Shape::Text(t) => {
                        Some((t.galley.text().to_owned(), t.visual_bounding_rect()))
                    }
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

        /// Click the widget labelled `label`, wherever it is: an AccessKit
        /// click needs no coordinates, so it reaches a widget inside the
        /// canvas's transformed layer as readily as one in the toolbar.
        fn click(&mut self, label: &str) {
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
            self.frame(Vec::new());
        }

        /// Add a node through the toolbar: the Add menu, then a kind.
        ///
        /// One click dropped a Sine before this step, which is what B6 was
        /// about (#61, Overlands #1334). A Reverb, because no patch these
        /// tests use has one and the kind picker of an existing node wears
        /// its own kind's name in the same tree.
        fn add_a_node(&mut self) {
            self.click("Add node");
            self.click("Reverb");
        }

        /// Hold the pointer at `at` until egui's tooltip delay has passed,
        /// and return the text that appeared while it was there.
        ///
        /// One move, then still frames: egui holds a tooltip back until the
        /// pointer has been *still* for `interaction.tooltip_delay`, and a
        /// `PointerMoved` re-sent at the same position still counts as a
        /// move, so a loop of them shows nothing however long it runs.
        fn hover_text_at(&mut self, at: Pos2) -> Vec<String> {
            let before = self.painted_text();
            self.frame(vec![egui::Event::PointerMoved(at)]);
            for _ in 0..6 {
                self.frame(Vec::new());
            }
            self.painted_text()
                .into_iter()
                .filter(|t| !before.contains(t))
                .collect()
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
            let title = canvas.label("#0");
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

    /// A wire let go over no port names no port, and wires nothing. Since
    /// #61 it is not thrown away either: it offers to make what it would
    /// drive, and the port it would have landed on is still not named.
    #[test]
    fn a_wire_dropped_on_no_port_wires_nothing_and_names_no_port() {
        let mut canvas = Canvas::new(sine_into_gain(false));
        let gain_box = canvas
            .boxes()
            .into_iter()
            .max_by(|a, b| a.left().total_cmp(&b.left()))
            .expect("the gain's box");
        // Below the gain's box: no row, no dot.
        let to = Pos2::new(gain_box.center().x, gain_box.bottom() + 80.0);
        drag_from_the_sine_to(&mut canvas, to);
        let named: Vec<String> = canvas
            .painted_text()
            .into_iter()
            .filter(|t| t.starts_with('\u{27A1}'))
            .collect();
        assert_eq!(
            named,
            vec!["\u{27A1} let go for a new node here".to_string()],
            "over empty canvas the drag named a port"
        );
        release(&mut canvas, to);
        assert!(
            canvas.patch.graph.nodes[1].inputs.is_empty(),
            "the drop wired something"
        );
        assert_eq!(canvas.patch.graph.nodes.len(), 2, "the drop added a node");
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
            // The title is the number (the grip) beside the kind picker,
            // and the number is what the canvas paints itself (#59).
            for (i, name) in ["#0", "#1", "#2"].into_iter().enumerate() {
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
        // #0 selected, #1 and #2 plain: the output is marked by its OUT
        // badge now, not by a border it cannot show while selected (#59).
        let edges: Vec<Color32> = canvas
            .painted_boxes()
            .iter()
            .map(|b| b.stroke.color)
            .collect();
        assert_eq!(edges, [s.node_selected, s.node_stroke, s.node_stroke]);
        assert!(canvas.painted_boxes().iter().all(|b| b.fill == s.node_fill));
        assert_eq!(
            text_painted(&canvas.out, "OUT").map(|(_, c)| c),
            Some(s.node_output),
            "the output's badge"
        );
        let (_, title) = text_painted(&canvas.out, "#1").expect("a title");
        assert_eq!(title, s.node_title);
        let (_, valid) = text_painted(
            &canvas.out,
            "\u{2714} valid graph \u{2014} 3 nodes, all heard",
        )
        .expect("the line");
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
        let mut canvas = Canvas::new(three_node_patch());
        // The genetics controls moved into the More menu (#59 B9), so open
        // it: its buttons are drawn only while it is.
        canvas.click("More");
        let labels = canvas.buttons();
        for word in [
            "Add node",
            "Delete",
            "Tidy",
            "Fit view",
            "More",
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

    // ---- step 6: the node box (#59, Overlands #1332) --------------------

    /// A chain of `n` nodes running through the kinds upstream ships, each
    /// wired into the next: boxes of every height the canvas can meet, all
    /// of them placed by the auto-layout. A patch built from a record's
    /// seed arrives exactly like this — no stored positions, one column per
    /// link.
    fn seeded_size_patch(n: usize) -> AudioPatch {
        let kinds = NodeKind::defaults();
        let mut nodes: Vec<GraphNode> = Vec::new();
        for i in 0..n {
            let kind = kinds[i % kinds.len()].clone();
            let mut gn = node(i as u32, kind);
            if let (Some(prev), Some(port)) =
                (i.checked_sub(1), input_ports(&gn.kind).first().copied())
            {
                gn.inputs.insert(
                    port.to_string(),
                    vec![Connection::from_node(NodeId(prev as u32))],
                );
            }
            nodes.push(gn);
        }
        AudioPatch {
            seed: 7,
            graph: crate::patch::NodeGraph {
                nodes,
                output: NodeId(n as u32 - 1),
            },
        }
    }

    /// The node boxes of `canvas` in the canvas's own coordinates, which is
    /// what the state records and what the labels below are read in.
    fn node_rects(canvas: &Canvas) -> Vec<(NodeId, Rect)> {
        canvas.state.boxes.clone()
    }

    /// B5: a node box writes nothing over its own border, and nothing on it.
    ///
    /// Read from the text the frame painted, not from the AccessKit tree: a
    /// slider's label is part of the slider, so it is not a node of its own
    /// there, and it was exactly the text that overflowed — `Slider::text`
    /// put "Freq (Hz)" to the right of the value, off the edge of a box
    /// fixed at 210 units. A galley's bounding rect is its full extent
    /// whatever the clip rect hid, so an overflow is visible here.
    ///
    /// The bar is the box's content area — the frame inside its own padding
    /// — not the frame rect: "Cutoff (Hz)" did not cross the border, it ran
    /// up to it, and its closing bracket merged with the line.
    #[test]
    fn every_label_in_a_node_box_lies_inside_it() {
        let mut checked = 0;
        for kind in NodeKind::defaults() {
            let name = node_kind_label(&kind);
            let canvas = Canvas::new(one_node(kind.clone()));
            let boxes = canvas.painted_boxes();
            assert_eq!(boxes.len(), 1, "{name}: one node box");
            let node_box = boxes[0].rect;
            let content = canvas.content_area(node_box);
            let mut inside = 0;
            for (text, rect) in canvas.painted_text_rects() {
                if !node_box.contains(rect.min) {
                    continue;
                }
                inside += 1;
                assert!(
                    content.contains_rect(rect),
                    "{name}: {text:?} at {rect:?} is outside the content area \
                     {content:?} of its box {node_box:?}"
                );
            }
            assert!(
                inside >= 2,
                "{name}: only {inside} texts found in the box — the filter found \
                 nothing to check"
            );
            checked += inside;
        }
        assert!(checked >= 40, "only {checked} texts were checked");
    }

    /// B5: a node's parameters read as two columns — every label at one
    /// left edge, its control to the right — where `Slider::text` put the
    /// label *after* the value, so the names ran down a ragged edge set by
    /// how wide each value happened to be and ended 2 units from the
    /// border.
    ///
    /// A grid's labels are `ui.label` calls, so they are the box's
    /// `Role::Label` nodes; the port rows and the title are filtered out by
    /// name.
    #[test]
    fn a_node_body_is_a_column_of_labels_and_a_column_of_controls() {
        let mut checked = 0;
        for kind in NodeKind::defaults() {
            let name = node_kind_label(&kind);
            let canvas = Canvas::new(one_node(kind.clone()));
            let node_box = canvas.state.boxes[0].1;
            let labels: Vec<(String, Rect)> = canvas
                .labels_untransformed()
                .into_iter()
                .filter(|(t, r)| {
                    node_box.contains(r.min)
                        // The title row: the grip, the OUT badge.
                        && !t.starts_with('#')
                        && t != "OUT"
                        // The "Inputs" heading and its port rows, which are
                        // a list and not a grid.
                        && !t.ends_with(':')
                        && t != "Inputs"
                })
                .collect();
            // Silence has no parameters, and says so.
            if labels.len() < 2 {
                continue;
            }
            let left = labels[0].1.left();
            for (text, rect) in &labels {
                assert!(
                    (rect.left() - left).abs() < 1.0,
                    "{name}: {text:?} starts at {} where the label column is at \
                     {left}; labels {labels:?}",
                    rect.left()
                );
                checked += 1;
            }
        }
        assert!(checked >= 30, "only {checked} labels were checked");
    }

    /// B5: a box writes its kind once. The title said "#0 Sawtooth" and the
    /// combo directly under it said "Sawtooth" again — a third of the box's
    /// first two rows spent saying the same word twice.
    #[test]
    fn a_node_box_names_its_kind_once() {
        for kind in NodeKind::defaults() {
            let name = node_kind_label(&kind);
            let canvas = Canvas::new(one_node(kind.clone()));
            let node_box = canvas.painted_boxes()[0].rect;
            // The title row, found by the grip that sits on it. Further
            // down a parameter may fairly carry the kind's own name —
            // Gain's gain, Mix's gain — so only the title is counted.
            let title_row = canvas.label("#0").y_range();
            let said: Vec<String> = canvas
                .painted_text_rects()
                .into_iter()
                .filter(|(t, r)| {
                    node_box.contains(r.min) && title_row.contains(r.center().y) && t.contains(name)
                })
                .map(|(t, _)| t)
                .collect();
            assert_eq!(
                said.len(),
                1,
                "{name} is written twice in its title: {said:?}"
            );
        }
        // And the picker's own "Kind" caption, which cost a column of the
        // title row to say what the row already showed, is gone with it.
        let canvas = Canvas::new(one_node(NodeKind::Lfo(Default::default())));
        assert!(
            !canvas.painted_text().iter().any(|t| t == "Kind"),
            "the kind picker still carries a separate caption"
        );
    }

    /// B4: the boxes the canvas places itself never lie on one another.
    ///
    /// The old layout stacked a column on a fixed 190-unit pitch against
    /// boxes of about 210 and more, so the starter patch opened with the
    /// LFO's box over the sine's "amplitude" row.
    #[test]
    fn no_two_node_boxes_overlap_after_two_frames() {
        let patches: [(&str, AudioPatch); 4] = [
            ("three_node_patch", three_node_patch()),
            ("sine_into_gain", sine_into_gain(false)),
            ("seeded 9", seeded_size_patch(9)),
            ("seeded 17", seeded_size_patch(17)),
        ];
        for (name, patch) in patches {
            let canvas = Canvas::after(patch, 2);
            let rects = node_rects(&canvas);
            for (i, (a_id, a)) in rects.iter().enumerate() {
                for (b_id, b) in &rects[i + 1..] {
                    let overlap = a.intersect(*b);
                    assert!(
                        !overlap.is_positive(),
                        "{name}: #{} {a:?} and #{} {b:?} overlap over {overlap:?}",
                        a_id.0,
                        b_id.0
                    );
                }
            }
        }
    }

    /// The measured layout settles: once the boxes have been drawn once,
    /// a quiet frame moves nothing.
    ///
    /// A box that moves while a frame is being drawn is worse than untidy.
    /// egui runs a frame a second time whenever something in it asks for
    /// one — a `Grid` measuring its columns does, and every node body is a
    /// grid — so a position changed at the end of a pass leaves a widget's
    /// rect holding another widget's id on the next.
    #[test]
    fn the_measured_layout_settles_and_a_quiet_frame_moves_nothing() {
        for (name, patch) in [
            ("three_node_patch", three_node_patch()),
            ("seeded 9", seeded_size_patch(9)),
        ] {
            let mut canvas = Canvas::after(patch, 2);
            let settled = canvas.state.positions.clone();
            for frame in 0..3 {
                canvas.frame(Vec::new());
                assert_eq!(
                    canvas.state.positions, settled,
                    "{name}: quiet frame {frame} moved a box"
                );
                assert!(
                    !canvas.state.relayout,
                    "{name}: quiet frame {frame} left a layout owed"
                );
            }
        }
    }

    /// B3: a node is named by its kind wherever it is referred to — the
    /// Output picker and the row of every wire into a port — not by a bare
    /// number the user has to go and look up.
    #[test]
    fn the_output_picker_and_the_wire_rows_name_the_kind() {
        let canvas = Canvas::new(three_node_patch());
        let texts = canvas.painted_text();
        assert!(
            texts.iter().any(|t| t == "#2 Lowpass"),
            "the Output picker does not name the output's kind; painted {texts:?}"
        );
        for row in ["from #0 Sine", "from #1 LFO"] {
            assert!(
                texts.iter().any(|t| t == row),
                "no wire row reading {row:?}; painted {texts:?}"
            );
        }
        assert!(
            !texts.iter().any(|t| t.contains('\u{2B05}')),
            "a wire row still points with an arrow instead of naming its source"
        );
    }

    /// B10 / #1289: `on_hover_text` never fires on a disabled widget, so a
    /// disabled Delete has to say why through `on_disabled_hover_text`.
    /// Two reasons, because there are two ways to be disabled.
    #[test]
    fn a_disabled_delete_says_why_where_egui_will_show_it() {
        // Nothing selected, three nodes: "select something first".
        let mut canvas = Canvas::new(three_node_patch());
        assert!(canvas.state.selected.is_none(), "nothing is selected yet");
        let reason = hover_text_over(&mut canvas, "Delete");
        assert!(
            reason.iter().any(|t| t.contains("Select a node")),
            "a disabled Delete with nothing selected says {reason:?}"
        );

        // One node, and it is selected: the patch may not go empty.
        let mut canvas = Canvas::new(one_node(NodeKind::Sine(SineOsc::default())));
        canvas.state.selected = Some(NodeId(0));
        canvas.frame(Vec::new());
        let reason = hover_text_over(&mut canvas, "Delete");
        assert!(
            reason.iter().any(|t| t.contains("at least one node")),
            "a disabled Delete on a one-node patch says {reason:?}"
        );
    }

    /// Hold the pointer over the toolbar button labelled `button` and
    /// return the text that appeared. The toolbar is drawn in the base
    /// layer, so its AccessKit rect is already a screen rect.
    fn hover_text_over(canvas: &mut Canvas, button: &str) -> Vec<String> {
        let at = canvas.chrome_rect(accesskit::Role::Button, button).center();
        canvas.hover_text_at(at)
    }

    /// B8: which node the patch plays and which node is selected were one
    /// border in two colours, so the output stopped saying so the moment it
    /// was clicked. They are two marks now and a node can wear both.
    #[test]
    fn the_output_wears_a_badge_and_the_selection_is_a_thicker_stroke() {
        let mut canvas = Canvas::new(three_node_patch());
        assert_eq!(
            canvas.painted_text().iter().filter(|t| *t == "OUT").count(),
            1,
            "exactly one box is badged as the output"
        );

        // Select the output itself: both marks are on it at once.
        canvas.state.selected = Some(canvas.patch.graph.output);
        canvas.frame(Vec::new());
        assert_eq!(
            canvas.painted_text().iter().filter(|t| *t == "OUT").count(),
            1,
            "the badge survives being selected"
        );
        let style = canvas.style();
        let widths: Vec<f32> = canvas
            .painted_boxes()
            .iter()
            .map(|r| r.stroke.width)
            .collect();
        let selected = canvas
            .state
            .boxes
            .iter()
            .position(|(id, _)| Some(*id) == canvas.state.selected)
            .expect("the selected box is drawn");
        for (i, w) in widths.iter().enumerate() {
            if i == selected {
                assert!(
                    *w > widths
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != selected)
                        .map(|(_, w)| *w)
                        .fold(0.0, f32::max),
                    "the selected box's edge is not the thickest: {widths:?}"
                );
            }
        }
        assert_eq!(
            canvas.painted_boxes()[selected].stroke.color,
            style.node_selected,
            "the selected box's edge is the style's selection colour"
        );
    }

    /// B9: one row of chrome above the canvas, not four. The genetics
    /// controls and the JSON box moved into its More menu.
    #[test]
    fn the_toolbar_is_one_row() {
        let canvas = Canvas::new(three_node_patch());
        let rows: Vec<Rect> = ["Add node", "Delete", "Tidy", "Fit view", "More"]
            .into_iter()
            .map(|b| canvas.chrome_rect(accesskit::Role::Button, b))
            .collect();
        let first = rows[0];
        for (name, rect) in ["Add node", "Delete", "Tidy", "Fit view", "More"]
            .into_iter()
            .zip(&rows)
        {
            assert!(
                (rect.center().y - first.center().y).abs() < 1.0,
                "{name} is not on the toolbar's one row: {rect:?} against {first:?}"
            );
        }
        let buttons = canvas.buttons();
        // Every one of these is now a menu item — the genetics controls
        // and the JSON box under More, the per-node Mutate in a node's own
        // context menu (#60) — so none is drawn until it is asked for.
        for gone in [
            "Copy JSON",
            "Load current",
            "Apply",
            "Reroll seed",
            "Mutate this node",
        ] {
            assert!(
                !buttons.iter().any(|l| l == gone),
                "{gone:?} is still on the front row; buttons: {buttons:?}"
            );
        }
    }

    /// B9, and the trap in it: putting the validity readout at the right
    /// of the row must not put it *over* the row.
    ///
    /// A right-to-left layout takes whatever width is left however little
    /// that is, so on the sequence editor's side panel — about 480 points
    /// — the line drew straight across the Output picker and the More
    /// button. The picture showed it; no test did.
    #[test]
    fn the_toolbar_never_draws_the_validity_line_over_itself() {
        for width in [480.0, 560.0, 720.0, 1000.0] {
            let canvas = Canvas::narrow(three_node_patch(), width);
            let chip = canvas
                .painted_text_rects()
                .into_iter()
                .find(|(t, _)| t.starts_with('\u{2714}'))
                .map(|(_, r)| r)
                .expect("the validity line");
            for button in ["Add node", "Delete", "Tidy", "Fit view", "More"] {
                let rect = canvas.chrome_rect(accesskit::Role::Button, button);
                assert!(
                    !chip.intersects(rect),
                    "at {width}: the validity line {chip:?} covers {button} {rect:?}"
                );
            }
            let output = canvas
                .painted_text_rects()
                .into_iter()
                .find(|(t, _)| t == "#2 Lowpass")
                .map(|(_, r)| r)
                .expect("the Output picker");
            assert!(
                !chip.intersects(output),
                "at {width}: the validity line {chip:?} covers the Output picker {output:?}"
            );
        }
    }

    /// B16: the canvas has an edge and a grid, so there is something to see
    /// that it pans and how far it has been panned.
    #[test]
    fn the_canvas_has_an_edge_and_a_grid_in_the_styles_colours() {
        let s = distinct_style();
        let canvas = Canvas::new(three_node_patch());
        let painted = colours(&canvas.out);
        assert!(
            painted.contains(&s.canvas_edge),
            "no edge around the canvas"
        );
        assert!(painted.contains(&s.canvas_grid), "no grid on the canvas");
    }

    // ---- step 7a (#60, Overlands #1333): history and keys ---------------

    /// Move the pointer into the canvas, so the editor owns the keyboard,
    /// and hold it there while `events` are delivered.
    fn keys_over_the_canvas(canvas: &mut Canvas, events: Vec<egui::Event>) {
        let over = canvas.state.boxes[0].1.center();
        let at = canvas.to_screen() * over;
        canvas.frame(vec![egui::Event::PointerMoved(at)]);
        canvas.frame(events);
        canvas.frame(Vec::new());
    }

    /// One key press with modifiers, down then up.
    fn press(key: egui::Key, modifiers: egui::Modifiers) -> Vec<egui::Event> {
        vec![
            egui::Event::Key {
                key,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers,
            },
            egui::Event::Key {
                key,
                physical_key: None,
                pressed: false,
                repeat: false,
                modifiers,
            },
        ]
    }

    /// B7: a committed edit can be taken back. Add node is the simplest
    /// one-click structural edit — before this step nothing on the canvas
    /// could be undone at all.
    #[test]
    fn undo_and_redo_round_trip_an_edit_on_the_canvas() {
        let mut canvas = Canvas::new(three_node_patch());
        assert!(!canvas.state.can_undo(), "nothing has been edited yet");

        canvas.add_a_node();
        assert_eq!(canvas.patch.graph.nodes.len(), 4);
        assert!(canvas.state.can_undo());
        assert!(!canvas.state.can_redo());

        assert!(canvas.state.undo(&mut canvas.patch));
        assert_eq!(canvas.patch.graph.nodes.len(), 3);
        assert!(canvas.state.can_redo());
        assert!(!canvas.state.can_undo());

        assert!(canvas.state.redo(&mut canvas.patch));
        assert_eq!(canvas.patch.graph.nodes.len(), 4);
        assert!(!canvas.state.can_redo());
    }

    /// A deletion is a committed edit like any other, so it comes back.
    /// Delete took a node and every wire into it with nothing to undo it.
    #[test]
    fn undo_brings_back_a_deleted_node_and_its_wires() {
        let mut canvas = Canvas::new(three_node_patch());
        let before = canvas.patch.clone();
        canvas.state.selected = Some(NodeId(0));
        canvas.frame(Vec::new());
        canvas.click("Delete");
        assert!(canvas.patch.graph.nodes.iter().all(|n| n.id != NodeId(0)));

        assert!(canvas.state.undo(&mut canvas.patch));
        assert_eq!(
            canvas.patch, before,
            "the node and the wire into the filter's \"in\" are both back"
        );
    }

    /// B12: the canvas reads the keyboard. Ctrl+Z undoes, Ctrl+Shift+Z and
    /// Ctrl+Y redo, all injected as real key events on a headless context.
    #[test]
    fn ctrl_z_on_the_canvas_undoes_and_ctrl_y_redoes() {
        for redo_with in [
            (
                egui::Key::Z,
                egui::Modifiers::COMMAND.plus(egui::Modifiers::SHIFT),
            ),
            (egui::Key::Y, egui::Modifiers::COMMAND),
        ] {
            let mut canvas = Canvas::new(three_node_patch());
            canvas.add_a_node();
            assert_eq!(canvas.patch.graph.nodes.len(), 4);

            keys_over_the_canvas(&mut canvas, press(egui::Key::Z, egui::Modifiers::COMMAND));
            assert_eq!(
                canvas.patch.graph.nodes.len(),
                3,
                "Ctrl+Z did not undo the added node"
            );

            keys_over_the_canvas(&mut canvas, press(redo_with.0, redo_with.1));
            assert_eq!(
                canvas.patch.graph.nodes.len(),
                4,
                "{:?} did not redo",
                redo_with.0
            );
        }
    }

    /// Delete removes the selection, and Ctrl+D duplicates it — with the
    /// wires *into* it, which is what makes a duplicate worth having.
    #[test]
    fn delete_and_ctrl_d_act_on_the_selection() {
        let mut canvas = Canvas::new(three_node_patch());
        canvas.state.selected = Some(NodeId(2));
        canvas.frame(Vec::new());

        // #2 is the filter, wired from #0 and #1.
        keys_over_the_canvas(&mut canvas, press(egui::Key::D, egui::Modifiers::COMMAND));
        assert_eq!(
            canvas.patch.graph.nodes.len(),
            4,
            "Ctrl+D did not duplicate"
        );
        let copy = canvas
            .patch
            .graph
            .nodes
            .iter()
            .find(|n| n.id != NodeId(2) && matches!(n.kind, NodeKind::BiquadLowpass(_)))
            .expect("a second lowpass");
        assert_eq!(
            copy.inputs.len(),
            2,
            "the copy did not keep the wires into it: {:?}",
            copy.inputs
        );
        let duplicated = copy.id;

        canvas.state.selected = Some(duplicated);
        canvas.frame(Vec::new());
        keys_over_the_canvas(&mut canvas, press(egui::Key::Delete, egui::Modifiers::NONE));
        assert!(
            canvas.patch.graph.nodes.iter().all(|n| n.id != duplicated),
            "Delete did not remove the selection"
        );
    }

    /// Esc clears the selection, and the editor says it took the Escape so a
    /// host's own Escape ladder can stand down for that press (#1333).
    #[test]
    fn escape_clears_the_selection_and_is_reported_to_the_host() {
        let mut canvas = Canvas::new(three_node_patch());
        canvas.state.selected = Some(NodeId(1));
        canvas.frame(Vec::new());
        assert!(!canvas.state.took_escape());

        let over = canvas.state.boxes[0].1.center();
        let at = canvas.to_screen() * over;
        canvas.frame(vec![egui::Event::PointerMoved(at)]);
        canvas.frame(press(egui::Key::Escape, egui::Modifiers::NONE));
        assert_eq!(canvas.state.selected, None, "Escape did not deselect");
        assert!(
            canvas.state.took_escape(),
            "the editor did not report the Escape it acted on"
        );

        // With nothing selected there is nothing to clear, so the Escape is
        // the host's: its ladder must not lose a rung to a no-op.
        canvas.frame(press(egui::Key::Escape, egui::Modifiers::NONE));
        assert!(!canvas.state.took_escape());
    }

    /// The keys are the editor's only while it owns them. With the pointer
    /// away from the canvas a Ctrl+Z belongs to whatever else is on screen.
    #[test]
    fn the_canvas_takes_the_keys_only_while_it_owns_them() {
        let mut canvas = Canvas::new(three_node_patch());
        canvas.add_a_node();
        assert_eq!(canvas.patch.graph.nodes.len(), 4);

        // Pointer far outside the canvas.
        canvas.frame(vec![egui::Event::PointerMoved(Pos2::new(1590.0, 5.0))]);
        assert!(!canvas.state.wants_keyboard());
        canvas.frame(press(egui::Key::Z, egui::Modifiers::COMMAND));
        assert_eq!(
            canvas.patch.graph.nodes.len(),
            4,
            "a Ctrl+Z outside the canvas was taken by it anyway"
        );

        let at = canvas.to_screen() * canvas.state.boxes[0].1.center();
        canvas.frame(vec![egui::Event::PointerMoved(at)]);
        assert!(canvas.state.wants_keyboard());
    }

    /// B7: the per-node actions live in a context menu on the title, where
    /// the die used to sit a pixel from the drag handle.
    #[test]
    fn the_node_title_carries_a_context_menu_of_its_actions() {
        let mut canvas = Canvas::new(three_node_patch());
        // `label` already returns a screen rect.
        let grip = canvas.label("#1");
        canvas.frame(vec![
            egui::Event::PointerMoved(grip.center()),
            egui::Event::PointerButton {
                pos: grip.center(),
                button: egui::PointerButton::Secondary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
            egui::Event::PointerButton {
                pos: grip.center(),
                button: egui::PointerButton::Secondary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ]);
        canvas.frame(Vec::new());
        let buttons = canvas.buttons();
        for item in ["Mutate this node", "Set as output", "Duplicate", "Delete"] {
            assert!(
                buttons.iter().any(|l| l == item),
                "no {item:?} in the node's context menu; buttons {buttons:?}"
            );
        }
    }

    /// B16: what a constant does to a port is said where it is offered, not
    /// left for the reader to guess from "+ const".
    #[test]
    fn add_constant_says_what_a_constant_does_to_the_port() {
        let mut canvas = Canvas::new(sine_into_gain(false));
        // The "Add constant" button on the gain port's own row — there is
        // one per port, so it is found by the row it shares.
        let row = canvas.label("gain:");
        let at = canvas
            .chrome(accesskit::Role::Button)
            .into_iter()
            .filter(|(text, _)| text == "Add constant")
            .map(|(_, rect)| canvas.to_screen() * rect)
            .find(|rect| beside(rect.center(), row))
            .expect("an Add constant button on the gain row")
            .center();
        let shown = canvas.hover_text_at(at);
        assert!(
            shown
                .iter()
                .any(|t| t.contains("fixed value") && t.contains("gain")),
            "hovering Add constant on the gain port says {shown:?}"
        );
    }

    // ---- step 7b (#61, Overlands #1334): wires, the Add menu, heard -----

    /// B13: "3 nodes" is true of a patch where two of them bake into
    /// nothing. What is heard is what the output can be reached from.
    #[test]
    fn only_the_nodes_upstream_of_the_output_are_heard() {
        // 0:sine -> 2:lowpass("in"), 1:lfo -> 2("cutoff_hz"); output 2.
        let patch = three_node_patch();
        let heard = heard_nodes(&patch.graph);
        assert_eq!(heard.len(), 3, "every node reaches the output");

        // Point the output at the LFO instead: the sine and the filter are
        // still in the patch, and nothing plays them.
        let mut orphaned = three_node_patch();
        orphaned.graph.output = NodeId(1);
        let heard = heard_nodes(&orphaned.graph);
        assert_eq!(
            heard,
            HashSet::from([NodeId(1)]),
            "only the output node and what feeds it is heard"
        );
    }

    /// A loop upstream of the output is still heard, and the walk that finds
    /// out must not spin on it: a broken graph is exactly when the reader
    /// most needs the canvas to keep drawing.
    #[test]
    fn a_cycle_upstream_of_the_output_is_heard_without_spinning() {
        let mut patch = three_node_patch();
        // 2 already feeds nothing; wire it back into 0 to make 0 -> 2 -> 0.
        patch.graph.nodes[0]
            .inputs
            .insert("freq_hz".into(), vec![Connection::from_node(NodeId(2))]);
        let heard = heard_nodes(&patch.graph);
        assert_eq!(heard.len(), 3);
    }

    /// A node the output never reaches is dimmed and says so, and the
    /// validity chip counts what is heard rather than what is there.
    #[test]
    fn a_node_that_is_not_heard_is_badged_and_counted() {
        let mut patch = three_node_patch();
        patch.graph.output = NodeId(0);
        let canvas = Canvas::new(patch);
        let painted = canvas.painted_text();
        assert_eq!(
            painted.iter().filter(|t| *t == NOT_HEARD).count(),
            2,
            "the LFO and the filter are not heard; painted {painted:?}"
        );
        assert!(
            text_painted(
                &canvas.out,
                "\u{2714} valid graph \u{2014} 3 nodes, 1 heard"
            )
            .is_some(),
            "the chip says {:?}",
            painted
                .iter()
                .find(|t| t.contains("valid graph"))
                .map(String::as_str)
        );
    }

    /// Every node heard means the chip says so in words rather than
    /// repeating the count: "3 nodes, 3 heard" reads as an arithmetic
    /// puzzle in the one case where there is nothing to look for.
    #[test]
    fn a_patch_whose_nodes_are_all_heard_says_so() {
        let canvas = Canvas::new(three_node_patch());
        assert!(
            text_painted(
                &canvas.out,
                "\u{2714} valid graph \u{2014} 3 nodes, all heard"
            )
            .is_some(),
            "the chip says {:?}",
            canvas
                .painted_text()
                .iter()
                .find(|t| t.contains("valid graph"))
                .map(String::as_str)
        );
    }

    /// B2: the hit test is against the wire's curve, not its bounding box.
    /// The S-curve between two boxes bulges well away from the straight
    /// line between the dots, and a box test would take a click in the gap
    /// beside it.
    #[test]
    fn a_wire_is_hit_only_within_a_few_units_of_its_curve() {
        let a = Pos2::new(0.0, 0.0);
        let b = Pos2::new(200.0, 120.0);
        let wire = WireGeom {
            from: NodeId(0),
            to: NodeId(1),
            port: "in".into(),
            index: 0,
            points: wire_points(a, b),
        };

        // On the curve: both ends and the middle.
        for at in [a, b, wire.midpoint()] {
            assert!(
                wire.distance_to(at) <= 0.5,
                "{at:?} is {} from its own wire",
                wire.distance_to(at)
            );
        }
        // A few units off it is still a hit; the width of a node box is not.
        let just_off = wire.midpoint() + Vec2::new(4.0, 0.0);
        assert!(wire.distance_to(just_off) <= 6.0);
        assert!(wire.distance_to(wire.midpoint() + Vec2::new(60.0, 0.0)) > 6.0);

        // The chord's midpoint is on the curve, but the bounding box holds
        // corners that are nowhere near it.
        let corner = Pos2::new(a.x, b.y);
        assert!(
            wire.distance_to(corner) > 6.0,
            "a corner of the wire's bounding box counted as a hit"
        );

        let wires = [wire.clone()];
        assert_eq!(wire_at(&wires, just_off, 6.0), Some(&wires[0]));
        assert_eq!(wire_at(&wires, corner, 6.0), None);
    }

    /// Of two wires crossing, a click takes the nearer.
    #[test]
    fn the_nearer_of_two_wires_is_the_one_hit() {
        let down = WireGeom {
            from: NodeId(0),
            to: NodeId(2),
            port: "in".into(),
            index: 0,
            points: wire_points(Pos2::new(0.0, 0.0), Pos2::new(200.0, 200.0)),
        };
        let up = WireGeom {
            from: NodeId(1),
            to: NodeId(2),
            port: "cutoff_hz".into(),
            index: 0,
            points: wire_points(Pos2::new(0.0, 200.0), Pos2::new(200.0, 0.0)),
        };
        let wires = [down.clone(), up.clone()];
        // Clear of the crossing, where the two curves are far apart: both
        // run vertically through the middle, so a point beside the crossing
        // is genuinely near both and neither is the answer.
        let quarter = |w: &WireGeom| w.points[w.points.len() / 4];
        assert_eq!(
            wire_at(&wires, quarter(&down), 8.0).map(|w| w.from),
            Some(NodeId(0))
        );
        assert_eq!(
            wire_at(&wires, quarter(&up), 8.0).map(|w| w.from),
            Some(NodeId(1))
        );
        assert!(
            quarter(&down).distance(quarter(&up)) > 16.0,
            "the probe points are not on opposite wires"
        );
    }

    /// The canvas publishes the wires it drew, so a host can hang something
    /// on one and this file's own tests can find one without guessing at a
    /// dot's position (crate #67).
    #[test]
    fn the_canvas_publishes_the_wires_it_drew() {
        let canvas = Canvas::new(three_node_patch());
        let wires = canvas.state.wires();
        assert_eq!(wires.len(), 2, "two wires into the filter");
        let cutoff = wires
            .iter()
            .find(|w| w.port == "cutoff_hz")
            .expect("the LFO's wire");
        assert_eq!((cutoff.from, cutoff.to), (NodeId(1), NodeId(2)));
        // Painted where it says it is: the published points are in scene
        // units and the painted polyline in screen points.
        let painted = canvas.wires();
        let to_screen = canvas.to_screen();
        assert!(
            painted.iter().any(|points| {
                points
                    .first()
                    .is_some_and(|p| p.distance(to_screen * cutoff.points[0]) < 0.5)
            }),
            "no painted wire starts where the published one does"
        );
    }

    /// Hold the pointer over the middle of the wire that drives `port`.
    fn over_the_wire(canvas: &mut Canvas, port: &str) -> Pos2 {
        let wire = canvas
            .state
            .wires()
            .iter()
            .find(|w| w.port == port)
            .unwrap_or_else(|| panic!("no wire into {port:?}"));
        let at = canvas.to_screen() * wire.midpoint();
        canvas.frame(vec![egui::Event::PointerMoved(at)]);
        canvas.frame(Vec::new());
        at
    }

    /// B2: a wire says what it is when the pointer is on it.
    #[test]
    fn a_hovered_wire_names_both_ends_its_port_and_its_amount() {
        let mut canvas = Canvas::new(three_node_patch());
        // The tooltip is up from the first frame the pointer is on the
        // wire, so it is read from what the frame painted rather than
        // through `hover_text_at`, which subtracts what was already there.
        over_the_wire(&mut canvas, "cutoff_hz");
        let shown = canvas.painted_text();
        let text = shown
            .iter()
            .find(|t| t.contains("cutoff_hz") && t.contains('\u{27A1}'))
            .unwrap_or_else(|| panic!("no wire tooltip; the frame showed {shown:?}"));
        for part in ["#1 LFO", "#2 Lowpass", "cutoff_hz", "500"] {
            assert!(
                text.contains(part),
                "the wire's tooltip {text:?} omits {part:?}"
            );
        }
    }

    /// The picked wire's amount is edited on the wire, and the hover
    /// tooltip stands down while it is: the two are a few points apart and
    /// say the same thing, and the tooltip covered the editor's title.
    #[test]
    fn picking_a_wire_opens_its_amount_and_puts_the_tooltip_away() {
        let mut canvas = Canvas::new(three_node_patch());
        let at = over_the_wire(&mut canvas, "cutoff_hz");
        assert!(
            canvas
                .painted_text()
                .iter()
                .any(|t| t.contains("cutoff_hz") && t.contains('\u{27A1}')),
            "the hovered wire had no tooltip to begin with"
        );
        click_at(&mut canvas, at);

        let painted = canvas.painted_text();
        assert!(canvas.state.wire_is_selected());
        // A `DragValue` paints its prefix and its value as two galleys.
        assert!(
            painted.iter().any(|t| t.trim() == "amt") && painted.iter().any(|t| t == "500.00"),
            "the picked wire's amount is not on the canvas; painted {painted:?}"
        );
        assert!(
            painted.iter().any(|t| t == "#1 LFO \u{27A1} cutoff_hz"),
            "the amount editor does not say which wire it is on; painted {painted:?}"
        );
        assert!(
            !painted
                .iter()
                .any(|t| t.contains("#2 Lowpass") && t.contains('\u{27A1}')),
            "the hover tooltip is still up over the amount editor; painted {painted:?}"
        );
    }

    /// The picked wire's cross removes it, and it is the one *in* the
    /// panel: a cross floating at the middle of the curve is under the
    /// pointer the moment the wire is picked, so its hover text covered the
    /// panel that had just opened.
    #[test]
    fn the_crosss_in_the_picked_wires_panel_removes_it() {
        let mut canvas = Canvas::new(three_node_patch());
        let at = over_the_wire(&mut canvas, "cutoff_hz");
        click_at(&mut canvas, at);

        // The panel's own cross: the one on the title's row, not one of the
        // crosses in a node box's Inputs list.
        let title = canvas
            .painted_text_rects()
            .into_iter()
            .find(|(t, _)| t == "#1 LFO \u{27A1} cutoff_hz")
            .map(|(_, r)| r)
            .expect("the panel's title");
        let cross = canvas
            .chrome(accesskit::Role::Button)
            .into_iter()
            .filter(|(t, _)| t == "\u{2716}")
            .map(|(_, r)| r)
            .min_by(|a, b| {
                a.center()
                    .distance(title.center())
                    .total_cmp(&b.center().distance(title.center()))
            })
            .expect("a cross in the panel");
        assert!(
            cross.center().distance(title.center()) < 60.0,
            "the nearest cross to the panel's title is {:?} away",
            cross.center().distance(title.center())
        );

        click_at(&mut canvas, cross.center());
        let filter = canvas
            .patch
            .graph
            .nodes
            .iter()
            .find(|n| n.id == NodeId(2))
            .expect("the filter");
        assert!(
            !filter.inputs.contains_key("cutoff_hz"),
            "the panel's cross did not remove the wire; inputs {:?}",
            filter.inputs
        );
        assert!(!canvas.state.wire_is_selected());
        assert!(canvas.state.can_undo(), "the removal is not on the history");
    }

    /// Click at `at` and let the frame settle.
    fn click_at(canvas: &mut Canvas, at: Pos2) {
        canvas.frame(vec![
            egui::Event::PointerButton {
                pos: at,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
            egui::Event::PointerButton {
                pos: at,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ]);
        canvas.frame(Vec::new());
    }

    /// B2 and B7: a wire is picked up by clicking it and removed with the
    /// key that removes a node, and one undo puts it back — a wire removal
    /// is a step on the history like any other committed edit.
    #[test]
    fn a_wire_is_selected_by_a_click_and_removed_by_delete() {
        let mut canvas = Canvas::new(three_node_patch());
        let at = over_the_wire(&mut canvas, "cutoff_hz");
        click_at(&mut canvas, at);
        assert!(
            canvas.state.wire_is_selected(),
            "a click on the wire did not select it"
        );
        assert_eq!(canvas.state.wires().len(), 2);

        canvas.frame(press(egui::Key::Delete, egui::Modifiers::NONE));
        canvas.frame(Vec::new());
        let filter = canvas
            .patch
            .graph
            .nodes
            .iter()
            .find(|n| n.id == NodeId(2))
            .expect("the filter");
        assert!(
            !filter.inputs.contains_key("cutoff_hz"),
            "Delete left the wire in place"
        );
        assert!(
            !canvas.state.wire_is_selected(),
            "the removed wire is still selected"
        );

        keys_over_the_canvas(&mut canvas, press(egui::Key::Z, egui::Modifiers::COMMAND));
        let filter = canvas
            .patch
            .graph
            .nodes
            .iter()
            .find(|n| n.id == NodeId(2))
            .expect("the filter");
        assert_eq!(
            filter.inputs.get("cutoff_hz").map(Vec::len),
            Some(1),
            "one undo did not bring the wire back"
        );
    }

    /// Deleting a node while a wire is selected must still delete the node:
    /// the selections are separate, and the last one made is the one the
    /// key acts on.
    #[test]
    fn selecting_a_wire_clears_the_node_selection() {
        let mut canvas = Canvas::new(three_node_patch());
        let grip = canvas.label("#1");
        canvas.frame(vec![
            egui::Event::PointerMoved(grip.center()),
            egui::Event::PointerButton {
                pos: grip.center(),
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
            egui::Event::PointerButton {
                pos: grip.center(),
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ]);
        canvas.frame(Vec::new());
        assert_eq!(canvas.state.selected, Some(NodeId(1)));

        let at = over_the_wire(&mut canvas, "cutoff_hz");
        click_at(&mut canvas, at);
        assert!(canvas.state.wire_is_selected());
        assert_eq!(canvas.state.selected, None, "both were selected at once");
    }

    /// Both ways round: picking a node puts a picked wire down, as picking
    /// a wire puts a picked node down. Delete acts on one thing.
    #[test]
    fn picking_a_node_clears_a_picked_wire() {
        let mut canvas = Canvas::new(three_node_patch());
        let at = over_the_wire(&mut canvas, "cutoff_hz");
        click_at(&mut canvas, at);
        assert!(canvas.state.wire_is_selected());

        let grip = canvas.label("#1");
        canvas.frame(vec![egui::Event::PointerMoved(grip.center())]);
        click_at(&mut canvas, grip.center());
        assert_eq!(canvas.state.selected, Some(NodeId(1)));
        assert!(
            !canvas.state.wire_is_selected(),
            "both a node and a wire were picked at once"
        );
    }

    /// The toolbar's Delete acts on whatever is picked, and says which. It
    /// used to be greyed out whenever no *node* was picked, while the
    /// Delete key removed the picked wire — the same word doing two things.
    #[test]
    fn the_toolbar_delete_removes_a_picked_wire_and_says_so() {
        let mut canvas = Canvas::new(three_node_patch());
        let at = over_the_wire(&mut canvas, "cutoff_hz");
        click_at(&mut canvas, at);

        let over = hover_text_over(&mut canvas, "Delete");
        assert!(
            over.iter()
                .any(|t| t.contains("wire") && t.contains("cutoff_hz")),
            "Delete says {over:?} with a wire picked"
        );

        canvas.click("Delete");
        let filter = canvas
            .patch
            .graph
            .nodes
            .iter()
            .find(|n| n.id == NodeId(2))
            .expect("the filter");
        assert!(
            !filter.inputs.contains_key("cutoff_hz"),
            "the wire is still there"
        );
        assert_eq!(
            canvas.patch.graph.nodes.len(),
            3,
            "it removed a node as well"
        );
    }

    /// B12's contract, extended: Escape spends one press per step, so a
    /// selected wire is what it clears first.
    #[test]
    fn escape_clears_a_selected_wire_before_anything_else() {
        let mut canvas = Canvas::new(three_node_patch());
        let at = over_the_wire(&mut canvas, "cutoff_hz");
        click_at(&mut canvas, at);
        canvas.frame(press(egui::Key::Escape, egui::Modifiers::NONE));
        assert!(!canvas.state.wire_is_selected());
        assert!(canvas.state.took_escape(), "the press was not spent");
    }

    /// Take a wire off its port by its end and let it go at `to`.
    fn reroute_the_cutoff_wire(canvas: &mut Canvas, to: Pos2) {
        let grab = {
            let wire = canvas
                .state
                .wires()
                .iter()
                .find(|w| w.port == "cutoff_hz")
                .expect("the LFO's wire");
            canvas.to_screen() * wire_grab(wire)
        };
        canvas.frame(vec![
            egui::Event::PointerMoved(grab),
            egui::Event::PointerButton {
                pos: grab,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ]);
        for i in 1..=4 {
            canvas.frame(vec![egui::Event::PointerMoved(
                grab.lerp(to, i as f32 / 4.0),
            )]);
        }
        canvas.frame(vec![egui::Event::PointerButton {
            pos: to,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }]);
        canvas.frame(Vec::new());
    }

    /// B2: a wire's input end comes off its port and goes on another, and
    /// the amount it carried goes with it — a re-route moves a wire, it
    /// does not make a fresh one at 1.0.
    #[test]
    fn a_wire_dragged_off_its_port_onto_another_re_routes_and_keeps_its_amount() {
        let mut canvas = Canvas::new(three_node_patch());
        let q_row = canvas.label("q:");
        reroute_the_cutoff_wire(
            &mut canvas,
            Pos2::new(q_row.right() + 40.0, q_row.center().y),
        );

        let filter = canvas
            .patch
            .graph
            .nodes
            .iter()
            .find(|n| n.id == NodeId(2))
            .expect("the filter");
        assert!(
            !filter.inputs.contains_key("cutoff_hz"),
            "the wire is still on its old port; inputs {:?}",
            filter.inputs
        );
        assert_eq!(
            filter.inputs.get("q"),
            Some(&vec![Connection::modulation(NodeId(1), 500.0)]),
            "the wire did not land on q with the amount it carried; inputs {:?}",
            filter.inputs
        );
    }

    /// And let go over nothing it is simply off: a re-route that lands
    /// nowhere is a disconnection, not an offer to make a node.
    #[test]
    fn a_wire_dragged_off_its_port_onto_nothing_is_disconnected() {
        let mut canvas = Canvas::new(three_node_patch());
        let clear = canvas
            .state
            .boxes
            .iter()
            .map(|(_, r)| *r)
            .reduce(|a, b| a.union(b))
            .expect("boxes")
            .right_top()
            + Vec2::new(60.0, 20.0);
        let on_screen = canvas.to_screen() * clear;
        reroute_the_cutoff_wire(&mut canvas, on_screen);

        let filter = canvas
            .patch
            .graph
            .nodes
            .iter()
            .find(|n| n.id == NodeId(2))
            .expect("the filter");
        assert!(!filter.inputs.contains_key("cutoff_hz"));
        assert_eq!(
            canvas.patch.graph.nodes.len(),
            3,
            "a re-route that landed nowhere offered to make a node"
        );
        assert!(
            canvas.state.can_undo(),
            "the disconnection is not on the history"
        );
    }

    /// B6: the Add menu is grouped by role, and every group's heading and
    /// every kind is in it.
    #[test]
    fn the_add_menu_offers_every_kind_under_a_heading() {
        let mut canvas = Canvas::new(three_node_patch());
        canvas.click("Add node");
        let painted = canvas.painted_text();
        for label in KIND_LABELS.iter().copied() {
            assert!(
                painted.iter().any(|t| t == label),
                "the Add menu does not offer {label:?}; it showed {painted:?}"
            );
        }
        for group in KindGroup::ALL {
            let wanted = kinds_by_group().iter().any(|(g, _)| *g == group);
            assert_eq!(
                painted.iter().any(|t| t == group.title()),
                wanted,
                "{:?}'s heading is shown when it holds no kinds",
                group
            );
        }
        assert_eq!(
            canvas.patch.graph.nodes.len(),
            3,
            "opening the menu added a node by itself"
        );
    }

    /// B6: and adding one from it is one step, not a Sine followed by a
    /// trip to the kind combo.
    #[test]
    fn choosing_a_kind_from_the_add_menu_adds_that_kind() {
        let mut canvas = Canvas::new(three_node_patch());
        canvas.click("Add node");
        canvas.click("Reverb");
        assert_eq!(canvas.patch.graph.nodes.len(), 4);
        let added = canvas.patch.graph.nodes.last().expect("the new node");
        assert_eq!(node_kind_label(&added.kind), "Reverb");
        assert_eq!(canvas.state.selected, Some(added.id));
    }

    /// B6: the same menu opens where the canvas is right-clicked, and the
    /// node lands there rather than in the middle of the view.
    #[test]
    fn the_add_menu_opens_where_the_canvas_was_right_clicked() {
        let mut canvas = Canvas::new(three_node_patch());
        // Empty canvas, well clear of every box — and high enough on the
        // screen that the whole menu fits below it, since a menu that would
        // run off the bottom is moved up to fit and is then not at the
        // pointer for a good reason of its own.
        let clear = canvas
            .state
            .boxes
            .iter()
            .map(|(_, r)| *r)
            .reduce(|a, b| a.union(b))
            .expect("boxes")
            .right_top()
            + Vec2::new(40.0, 10.0);
        let at = canvas.to_screen() * clear;
        canvas.frame(vec![
            egui::Event::PointerMoved(at),
            egui::Event::PointerButton {
                pos: at,
                button: egui::PointerButton::Secondary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
            egui::Event::PointerButton {
                pos: at,
                button: egui::PointerButton::Secondary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ]);
        canvas.frame(Vec::new());

        // The menu is at the pointer: its first heading is drawn beside the
        // point that was clicked, not in the middle of the view.
        let heading = canvas
            .painted_text_rects()
            .into_iter()
            .find(|(t, _)| t == KindGroup::Sources.title())
            .map(|(_, r)| r)
            .expect("the menu's first heading");
        assert!(
            heading.left() >= at.x - 2.0 && heading.left() <= at.x + 40.0,
            "the menu opened at {:?}, not at the pointer {at:?}",
            heading.left_top()
        );
        assert!(
            heading.top() >= at.y - 2.0 && heading.top() <= at.y + 40.0,
            "the menu opened at {:?}, not at the pointer {at:?}",
            heading.left_top()
        );

        canvas.click("Reverb");
        assert_eq!(canvas.patch.graph.nodes.len(), 4);
        let added = canvas.patch.graph.nodes.last().expect("the new node").id;
        let put = canvas.state.positions[&added];
        assert!(
            put.distance(clear) < 2.0,
            "the node landed at {put:?}, not where the menu was opened {clear:?}"
        );
    }

    /// A right-click on a node opens that node's own menu, not the Add
    /// menu: the canvas background is what offers to add.
    #[test]
    fn a_right_click_on_a_node_does_not_open_the_add_menu() {
        let mut canvas = Canvas::new(three_node_patch());
        let grip = canvas.label("#1");
        canvas.frame(vec![
            egui::Event::PointerMoved(grip.center()),
            egui::Event::PointerButton {
                pos: grip.center(),
                button: egui::PointerButton::Secondary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
            egui::Event::PointerButton {
                pos: grip.center(),
                button: egui::PointerButton::Secondary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ]);
        canvas.frame(Vec::new());
        let buttons = canvas.buttons();
        assert!(buttons.iter().any(|b| b == "Mutate this node"));
        assert!(
            !buttons.iter().any(|b| b == "Reverb"),
            "the Add menu opened over the node's own; buttons {buttons:?}"
        );
    }

    /// B6: a wire dragged onto empty canvas offers to make what it would
    /// drive, and wires it up when one is chosen.
    #[test]
    fn a_wire_dropped_on_empty_canvas_opens_the_add_menu_and_wires_it() {
        let mut canvas = Canvas::new(three_node_patch());
        let clear = canvas
            .state
            .boxes
            .iter()
            .map(|(_, r)| *r)
            .reduce(|a, b| a.union(b))
            .expect("boxes")
            .left_bottom()
            + Vec2::new(30.0, 120.0);
        let on_screen = canvas.to_screen() * clear;
        drag_from_the_sine_to(&mut canvas, on_screen);
        canvas.frame(vec![egui::Event::PointerButton {
            pos: on_screen,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }]);
        canvas.frame(Vec::new());
        assert!(
            canvas.painted_text().iter().any(|t| t == "Lowpass"),
            "dropping a wire on empty canvas offered nothing"
        );

        canvas.click("Lowpass");
        assert_eq!(canvas.patch.graph.nodes.len(), 4);
        let added = canvas.patch.graph.nodes.last().expect("the new node");
        assert_eq!(
            added.inputs.get("in").map(Vec::len),
            Some(1),
            "the new node's first input was not wired from the drag"
        );
    }
}
