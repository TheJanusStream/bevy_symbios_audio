//! Serde round-trip test for [`bevy_symbios_audio::AudioPatch`].
//!
//! Phase 1 ticket #2 acceptance criterion: hand-build a patch, serialize to
//! JSON, deserialize, and compare.  Also exercises the public schema surface
//! end-to-end (Connection variants, named input ports, the graph output
//! reference) so any later refactor that breaks the wire format will fail
//! here.

use std::collections::BTreeMap;

use bevy_symbios_audio::{
    AudioPatch, Connection, GraphNode, NodeGraph, NodeId, NodeKind, topo_sort,
};

fn build_patch() -> AudioPatch {
    let mut node_b_inputs = BTreeMap::new();
    node_b_inputs.insert("carrier".to_string(), Connection::from_node(NodeId(0)));
    node_b_inputs.insert("gain".to_string(), Connection::constant(0.75));

    let mut node_c_inputs = BTreeMap::new();
    node_c_inputs.insert("left".to_string(), Connection::from_node(NodeId(0)));
    node_c_inputs.insert("right".to_string(), Connection::from_node(NodeId(1)));

    AudioPatch {
        seed: 0xC0FF_EE00,
        graph: NodeGraph {
            nodes: vec![
                GraphNode {
                    id: NodeId(0),
                    kind: NodeKind::Silence,
                    inputs: BTreeMap::new(),
                },
                GraphNode {
                    id: NodeId(1),
                    kind: NodeKind::Silence,
                    inputs: node_b_inputs,
                },
                GraphNode {
                    id: NodeId(2),
                    kind: NodeKind::Silence,
                    inputs: node_c_inputs,
                },
            ],
            output: NodeId(2),
        },
    }
}

#[test]
fn audio_patch_round_trips_through_json() {
    let original = build_patch();
    let json = serde_json::to_string(&original).expect("serialize");
    let restored: AudioPatch = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(original, restored);
}

#[test]
fn round_tripped_patch_still_topo_sorts() {
    let original = build_patch();
    let json = serde_json::to_string(&original).unwrap();
    let restored: AudioPatch = serde_json::from_str(&json).unwrap();
    let order = topo_sort(&restored.graph).expect("valid DAG");
    assert_eq!(order, vec![NodeId(0), NodeId(1), NodeId(2)]);
}

#[test]
fn pretty_printed_json_round_trips() {
    let original = build_patch();
    let pretty = serde_json::to_string_pretty(&original).unwrap();
    let restored: AudioPatch = serde_json::from_str(&pretty).unwrap();
    assert_eq!(original, restored);
}

#[test]
fn connection_constant_serialises_with_tag() {
    let conn = Connection::constant(0.5);
    let json = serde_json::to_string(&conn).unwrap();
    assert_eq!(json, r#"{"source":"constant","value":0.5}"#);
}

#[test]
fn connection_node_omits_default_output_on_deserialize() {
    // Schema-evolution check: older patches that don't specify an output
    // port name must still deserialize.
    let json = r#"{"source":"node","id":3}"#;
    let conn: Connection = serde_json::from_str(json).unwrap();
    assert_eq!(conn, Connection::from_node(NodeId(3)));
}
