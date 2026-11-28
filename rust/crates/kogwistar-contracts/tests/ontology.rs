use kogwistar_contracts::{
    OntologyEndpointV1, OntologyPackageV1, compose_ontology_packages, validate_ontology_edge_roles,
    validate_ontology_payload,
};
use serde_json::{Value, json};

const PACKAGE: &str =
    include_str!("../../../../contracts/ontology/fixtures/ontology-package-v1.json");
const BASE: &str = include_str!("../../../../contracts/ontology/fixtures/ontology-base-v1.json");
const DEPENDENT: &str =
    include_str!("../../../../contracts/ontology/fixtures/ontology-dependent-v1.json");
const BROKEN_REFERENCE: &str =
    include_str!("../../../../contracts/ontology/fixtures/ontology-broken-reference-v1.json");

#[test]
// Python: tests/ontology/test_ontology_contracts.py::test_package_digest_is_stable_and_json_round_trips
fn python_ontology_package_fixture_round_trips_with_same_digest() {
    let package = OntologyPackageV1::from_json(PACKAGE).expect("Python fixture parses");
    assert_eq!(
        package.manifest.content_sha256,
        "818fc29ebb90dc3ef6b8f2b120953a9edd34092f5511eceaef5d3d376d9c537c"
    );
    assert_eq!(
        package.content_digest().unwrap(),
        package.manifest.content_sha256
    );

    let reordered: Value = {
        let mut value: Value = serde_json::from_str(PACKAGE).unwrap();
        value["descriptors"].as_array_mut().unwrap().reverse();
        value
    };
    let reordered = OntologyPackageV1::from_value(reordered).unwrap();
    assert_eq!(
        reordered.content_digest().unwrap(),
        package.manifest.content_sha256
    );
}

#[test]
// Python: tests/ontology/test_ontology_contracts.py::test_bad_digest_is_rejected
fn ontology_package_rejects_bad_digest_unknown_fields_and_duplicate_descriptors() {
    let mut bad_digest: Value = serde_json::from_str(PACKAGE).unwrap();
    bad_digest["manifest"]["content_sha256"] = json!("f".repeat(64));
    assert!(OntologyPackageV1::from_value(bad_digest).is_err());

    let mut unknown: Value = serde_json::from_str(PACKAGE).unwrap();
    unknown["unexpected"] = json!(true);
    assert!(OntologyPackageV1::from_value(unknown).is_err());

    let mut duplicate: Value = serde_json::from_str(PACKAGE).unwrap();
    let first = duplicate["descriptors"][0].clone();
    duplicate["descriptors"].as_array_mut().unwrap().push(first);
    assert!(OntologyPackageV1::from_value(duplicate).is_err());
}

#[test]
// Python: tests/ontology/test_ontology_contracts.py::test_exact_imports_are_required_and_shared_dependencies_deduplicate
// Python: tests/ontology/test_ontology_contracts.py::test_composition_is_order_independent_and_validates_payload_and_roles
fn ontology_composition_is_input_order_independent_and_checks_references() {
    let package = OntologyPackageV1::from_json(PACKAGE).unwrap();
    let first = compose_ontology_packages(&[package.clone()], &["interop"]).unwrap();
    let second = compose_ontology_packages(&[package], &[]).unwrap();
    assert_eq!(first.composition_sha256, second.composition_sha256);
    assert!(first.get("interop:Thing").is_some());

    let base = OntologyPackageV1::from_json(BASE).unwrap();
    let dependent = OntologyPackageV1::from_json(DEPENDENT).unwrap();
    let composed =
        compose_ontology_packages(&[dependent.clone(), base.clone()], &["dependent"]).unwrap();
    assert_eq!(composed.package_identities.len(), 2);
    assert!(composed.get("base:name").is_some());
    assert!(compose_ontology_packages(&[dependent], &["dependent"]).is_err());

    let broken = OntologyPackageV1::from_json(BROKEN_REFERENCE).unwrap();
    assert!(compose_ontology_packages(&[broken], &["broken"]).is_err());
}

#[test]
// Python: tests/ontology/test_ontology_contracts.py::test_validation_rejects_missing_required_payload_and_bad_role
fn ontology_payload_validation_enforces_required_type_and_extra_properties() {
    let package = OntologyPackageV1::from_json(PACKAGE).unwrap();
    let view = compose_ontology_packages(&[package], &[]).unwrap();
    validate_ontology_payload(&view, "interop:Thing", &json!({"name": "node"})).unwrap();
    assert!(validate_ontology_payload(&view, "interop:Thing", &json!({})).is_err());
    assert!(validate_ontology_payload(&view, "interop:Thing", &json!({"name": 42})).is_err());
    assert!(
        validate_ontology_payload(
            &view,
            "interop:Thing",
            &json!({"name": "node", "unexpected": true})
        )
        .is_err()
    );
}

#[test]
// Python: tests/ontology/test_ontology_contracts.py::test_composition_is_order_independent_and_validates_payload_and_roles
fn ontology_edge_shape_validates_roles_without_creating_graph_entities() {
    let package = OntologyPackageV1::from_json(PACKAGE).unwrap();
    let view = compose_ontology_packages(&[package], &[]).unwrap();
    let roles = [(
        "source".to_owned(),
        vec![OntologyEndpointV1 {
            target_kind: "node".to_owned(),
            class_id: "interop:Thing".to_owned(),
        }],
    )]
    .into();
    validate_ontology_edge_roles(&view, "interop:link", &roles).unwrap();

    let wrong_kind = [(
        "source".to_owned(),
        vec![OntologyEndpointV1 {
            target_kind: "edge".to_owned(),
            class_id: "interop:Thing".to_owned(),
        }],
    )]
    .into();
    assert!(validate_ontology_edge_roles(&view, "interop:link", &wrong_kind).is_err());
}

#[test]
// Python: tests/ontology/test_ontology_contracts.py::test_edge_roles_can_bind_existing_edges_without_creating_graph_entities
fn ontology_edge_role_can_reference_existing_edge_shape() {
    let package = OntologyPackageV1::from_json(PACKAGE).unwrap();
    let view = compose_ontology_packages(&[package], &[]).unwrap();
    let roles = [
        (
            "source".to_owned(),
            vec![OntologyEndpointV1 {
                target_kind: "node".to_owned(),
                class_id: "interop:Thing".to_owned(),
            }],
        ),
        (
            "related".to_owned(),
            vec![OntologyEndpointV1 {
                target_kind: "edge".to_owned(),
                class_id: "interop:link".to_owned(),
            }],
        ),
    ]
    .into();
    validate_ontology_edge_roles(&view, "interop:link", &roles).unwrap();
}
