use kogwistar_contracts::stable_id;
use kogwistar_store_postgres::{PostgresStoreError, PostgresUnitOfWork};

type AclTargetKey = (String, String, String);

async fn resolve_default_acl_graph_effects(
    uow: &PostgresUnitOfWork<'_>,
    scope: &GraphScope,
    mutations: &[GraphMutation],
    truth_graph: &str,
    owner_id: &str,
    security_scope: Option<&str>,
    effect_id: &str,
) -> Result<(Vec<GraphMutation>, Vec<NewIndexJob>), PostgresStoreError> {
    let mut targets = BTreeSet::<AclTargetKey>::new();
    let mut lock_targets = BTreeSet::<(String, String)>::new();
    for mutation in mutations {
        if mutation.record.metadata.get("entity_type")
            == Some(&Value::String("acl_record".to_owned()))
        {
            continue;
        }
        lock_targets.insert((mutation.entity_kind.clone(), mutation.record.id.clone()));
        targets.insert((
            mutation.entity_kind.clone(),
            mutation.record.id.clone(),
            String::new(),
        ));
        if let Some(mentions) = mutation.payload.get("mentions").and_then(Value::as_array) {
            for (grounding_index, grounding) in mentions.iter().enumerate() {
                let grounding_id = format!("{}::mention::{grounding_index}", mutation.record.id);
                targets.insert(("grounding".to_owned(), mutation.record.id.clone(), grounding_id));
                let Some(spans) = grounding.get("spans").and_then(Value::as_array) else {
                    continue;
                };
                for span_index in 0..spans.len() {
                    let span_id = format!(
                        "{}::mention::{grounding_index}::span::{span_index}",
                        mutation.record.id
                    );
                    targets.insert(("span".to_owned(), mutation.record.id.clone(), span_id));
                }
            }
        }
    }

    for (kind, entity_id) in &lock_targets {
        uow.lock_entity_identity(&scope.namespace, kind, entity_id).await?;
    }
    if targets.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let target_entity_ids = targets
        .iter()
        .map(|(_, entity_id, _)| entity_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let existing = uow
        .graph_acl_records_for_entities(scope, truth_graph, &target_entity_ids)
        .await?;
    let latest = latest_acl_records(existing);
    let mut derived_mutations = Vec::new();
    let mut derived_jobs = Vec::new();

    for mutation in mutations {
        if mutation.record.metadata.get("entity_type")
            == Some(&Value::String("acl_record".to_owned()))
            || mutation.op == "TOMBSTONE"
        {
            continue;
        }
        for (grain, entity_id, target_item_id) in targets.iter().filter(|(_, id, _)| id == &mutation.record.id) {
            let key = (grain.clone(), entity_id.clone(), target_item_id.clone());
            let previous = latest.get(&key);
            if previous.is_some_and(|record| default_acl_matches(record, owner_id, security_scope)) {
                continue;
            }
            let version = previous
                .and_then(|record| record.metadata.get("acl_version"))
                .and_then(Value::as_u64)
                .unwrap_or(0)
                .checked_add(1)
                .ok_or_else(|| PostgresStoreError::Backend("ACL version overflow".to_owned()))?;
            let acl_node = acl_record_mutation(
                scope,
                truth_graph,
                grain,
                entity_id,
                target_item_id,
                version,
                owner_id,
                security_scope,
                effect_id,
            );
            derived_jobs.extend(acl_index_jobs(scope, &acl_node, effect_id));
            derived_mutations.push(acl_node);
            if let Some(previous) = previous {
                let current_id = derived_mutations.last().expect("just appended ACL node").record.id.clone();
                let supersedes = acl_supersedes_mutation(
                    scope,
                    truth_graph,
                    entity_id,
                    grain,
                    target_item_id,
                    &current_id,
                    &previous.id,
                    effect_id,
                );
                derived_jobs.extend(acl_index_jobs(scope, &supersedes, effect_id));
                derived_mutations.push(supersedes);
            }
        }
    }
    Ok((derived_mutations, derived_jobs))
}

fn latest_acl_records(records: Vec<GraphRecord>) -> BTreeMap<AclTargetKey, GraphRecord> {
    let mut latest = BTreeMap::new();
    for record in records {
        let Some(entity_type) = record.metadata.get("entity_type").and_then(Value::as_str) else {
            continue;
        };
        if entity_type != "acl_record" {
            continue;
        }
        let (Some(grain), Some(entity_id)) = (
            record.metadata.get("acl_target_grain").and_then(Value::as_str),
            record.metadata.get("acl_target_entity_id").and_then(Value::as_str),
        ) else {
            continue;
        };
        let target_item_id = record
            .metadata
            .get("acl_target_item_id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let key = (grain.to_owned(), entity_id.to_owned(), target_item_id.to_owned());
        let version = record.metadata.get("acl_version").and_then(Value::as_u64).unwrap_or(0);
        let replace = latest.get(&key).is_none_or(|current: &GraphRecord| {
            let current_version = current.metadata.get("acl_version").and_then(Value::as_u64).unwrap_or(0);
            let current_tombstoned = current.metadata.get("tombstoned").and_then(Value::as_bool).unwrap_or(false);
            let tombstoned = record.metadata.get("tombstoned").and_then(Value::as_bool).unwrap_or(false);
            (current_tombstoned && !tombstoned) || (current_tombstoned == tombstoned && version > current_version)
        });
        if replace {
            latest.insert(key, record);
        }
    }
    latest
}

fn default_acl_matches(record: &GraphRecord, owner_id: &str, security_scope: Option<&str>) -> bool {
    record.metadata.get("acl_mode").and_then(Value::as_str) == Some("private")
        && record.metadata.get("owner_id").and_then(Value::as_str) == Some(owner_id)
        && record.metadata.get("created_by").and_then(Value::as_str) == Some(owner_id)
        && record.metadata.get("security_scope").and_then(Value::as_str) == security_scope
        && record.metadata.get("tombstoned").and_then(Value::as_bool) != Some(true)
}

fn acl_record_mutation(
    scope: &GraphScope,
    truth_graph: &str,
    grain: &str,
    entity_id: &str,
    target_item_id: &str,
    version: u64,
    owner_id: &str,
    security_scope: Option<&str>,
    effect_id: &str,
) -> GraphMutation {
    let node_id = stable_id(
        "acl_record",
        &[
            truth_graph.to_owned(),
            entity_id.to_owned(),
            grain.to_owned(),
            if target_item_id.is_empty() { "-" } else { target_item_id }.to_owned(),
            version.to_string(),
        ],
    )
    .to_string();
    let label = format!("acl:{truth_graph}:{entity_id}:{grain}");
    let summary = format!("private acl for {entity_id} {grain}");
    let metadata = json!({
        "entity_type": "acl_record",
        "acl_truth_graph": truth_graph,
        "acl_target_entity_id": entity_id,
        "acl_target_grain": grain,
        "acl_target_item_id": target_item_id,
        "acl_version": version,
        "acl_mode": "private",
        "created_by": owner_id,
        "owner_id": owner_id,
        "security_scope": security_scope,
        "tombstoned": false,
        "supersedes_version": if version > 1 { Some(version - 1) } else { None },
        "level_from_root": 0,
        "workspace_id": scope.workspace_id,
        "graph_space": scope.graph_space,
    });
    let properties = json!({
        "shared_with_principals": [],
        "shared_with_groups": [],
        "source_ids": [],
        "derivation_type": null,
        "derivation_audit": "{}",
        "target_item_id": target_item_id,
    });
    let mut record_metadata = metadata.as_object().cloned().unwrap_or_default();
    record_metadata.extend(properties.as_object().cloned().unwrap_or_default());
    let span = acl_dummy_span(truth_graph);
    let payload = json!({
        "id": node_id,
        "label": label,
        "type": "entity",
        "summary": summary,
        "doc_id": format!("acl:{truth_graph}"),
        "mentions": [{"spans": [span]}],
        "metadata": metadata,
        "properties": properties,
        "embedding": null,
        "level_from_root": 0,
        "domain_id": null,
        "canonical_entity_id": null,
    });
    let record = GraphRecord {
        id: node_id.clone(),
        document: Some(summary),
        metadata: record_metadata,
        embedding: None,
    };
    GraphMutation {
        scope: scope.clone(),
        table: "gke_nodes".to_owned(),
        entity_kind: "node".to_owned(),
        event_id: stable_event_id(effect_id, &node_id),
        op: "ADD".to_owned(),
        payload,
        record,
        embedding_dim: 0,
    }
}

fn acl_supersedes_mutation(
    scope: &GraphScope,
    truth_graph: &str,
    entity_id: &str,
    grain: &str,
    target_item_id: &str,
    current_id: &str,
    previous_id: &str,
    effect_id: &str,
) -> GraphMutation {
    let edge_id = stable_id("acl_supersedes", &[current_id.to_owned(), previous_id.to_owned()]).to_string();
    let summary = "acl supersedes previous acl state";
    let metadata = json!({
        "entity_type": "acl_supersedes",
        "acl_truth_graph": truth_graph,
        "acl_target_entity_id": entity_id,
        "acl_target_grain": grain,
        "acl_target_item_id": target_item_id,
        "level_from_root": 0,
        "workspace_id": scope.workspace_id,
        "graph_space": scope.graph_space,
    });
    let payload = json!({
        "id": edge_id,
        "label": "acl_supersedes",
        "type": "relationship",
        "summary": summary,
        "relation": "acl_supersedes",
        "source_ids": [current_id],
        "target_ids": [previous_id],
        "source_edge_ids": [],
        "target_edge_ids": [],
        "doc_id": format!("acl:{truth_graph}"),
        "mentions": [{"spans": [acl_dummy_span(truth_graph)]}],
        "metadata": metadata,
        "properties": null,
        "embedding": null,
        "domain_id": null,
    });
    GraphMutation {
        scope: scope.clone(),
        table: "gke_edges".to_owned(),
        entity_kind: "edge".to_owned(),
        event_id: stable_event_id(effect_id, &edge_id),
        op: "ADD".to_owned(),
        payload,
        record: GraphRecord {
            id: edge_id,
            document: Some(summary.to_owned()),
            metadata: metadata.as_object().cloned().unwrap_or_default(),
            embedding: None,
        },
        embedding_dim: 0,
    }
}

fn acl_dummy_span(truth_graph: &str) -> Value {
    json!({
        "collection_page_url": format!("acl/{truth_graph}"),
        "document_page_url": format!("acl/{truth_graph}"),
        "doc_id": format!("acl:{truth_graph}"),
        "insertion_method": "acl_record",
        "page_number": 1,
        "start_char": 0,
        "end_char": 1,
        "excerpt": "a",
        "context_before": "",
        "context_after": "",
        "chunk_id": null,
        "source_cluster_id": null,
        "verification": null,
    })
}

fn stable_event_id(effect_id: &str, entity_id: &str) -> String {
    stable_id("runtime.worker.graph_event", &[effect_id.to_owned(), entity_id.to_owned()]).to_string()
}

fn acl_index_jobs(scope: &GraphScope, mutation: &GraphMutation, effect_id: &str) -> Vec<NewIndexJob> {
    let kinds: &[&str] = if mutation.entity_kind == "edge" {
        &["edge_refs", "edge_endpoints"]
    } else {
        &["node_docs", "node_refs"]
    };
    kinds
        .iter()
        .map(|index_kind| NewIndexJob {
            job_id: stable_id(
                "runtime.worker.index_job",
                &[effect_id.to_owned(), mutation.record.id.clone(), (*index_kind).to_owned()],
            )
            .to_string(),
            namespace: scope.namespace.clone(),
            entity_kind: "node".to_owned(),
            entity_id: mutation.record.id.clone(),
            index_kind: (*index_kind).to_owned(),
            op: "UPSERT".to_owned(),
            payload_json: None,
            max_retries: 3,
        })
        .collect()
}

#[cfg(test)]
mod graph_acl_effects_unit_tests {
    use super::*;

    #[test]
    fn default_acl_projection_uses_host_owner_and_stable_identity() {
        let scope = GraphScope {
            namespace: "tenant-a".to_owned(),
            workspace_id: Some("workspace-a".to_owned()),
            graph_space: Some("conversation".to_owned()),
        };
        let mutation = acl_record_mutation(
            &scope,
            "conversation",
            "node",
            "node-1",
            "",
            1,
            "owner-a",
            Some("tenant-a"),
            "effect-1",
        );
        assert_eq!(mutation.entity_kind, "node");
        assert_eq!(mutation.record.metadata["acl_mode"], "private");
        assert_eq!(mutation.record.metadata["owner_id"], "owner-a");
        assert_eq!(mutation.record.metadata["created_by"], "owner-a");
        assert_eq!(mutation.record.metadata["security_scope"], "tenant-a");
        assert_eq!(mutation.payload["id"], mutation.record.id);
        assert_eq!(mutation.payload["metadata"]["acl_target_entity_id"], "node-1");
        assert_eq!(mutation.event_id, stable_event_id("effect-1", &mutation.record.id));
        assert!(acl_index_jobs(&scope, &mutation, "effect-1")
            .iter()
            .any(|job| job.index_kind == "node_refs"));
        let supersedes = acl_supersedes_mutation(
            &scope,
            "conversation",
            "node-1",
            "node",
            "",
            &mutation.record.id,
            "previous-acl-id",
            "effect-1",
        );
        assert_eq!(supersedes.entity_kind, "edge");
        assert_eq!(supersedes.payload["relation"], "acl_supersedes");
        assert_eq!(supersedes.payload["source_ids"][0], mutation.record.id);
        assert_eq!(supersedes.payload["target_ids"][0], "previous-acl-id");
    }

    #[test]
    fn default_acl_match_is_idempotent_but_stale_policy_is_versioned() {
        let scope = GraphScope {
            namespace: "tenant-a".to_owned(),
            workspace_id: None,
            graph_space: None,
        };
        let current = acl_record_mutation(
            &scope,
            "conversation",
            "node",
            "node-1",
            "",
            2,
            "owner-a",
            None,
            "effect-current",
        )
        .record;
        assert!(default_acl_matches(&current, "owner-a", None));
        assert!(!default_acl_matches(&current, "other-owner", None));

        let older = GraphRecord {
            id: "older-acl".to_owned(),
            document: None,
            metadata: Map::from_iter([
                ("entity_type".to_owned(), json!("acl_record")),
                ("acl_target_entity_id".to_owned(), json!("node-1")),
                ("acl_target_grain".to_owned(), json!("node")),
                ("acl_target_item_id".to_owned(), json!("")),
                ("acl_version".to_owned(), json!(1)),
            ]),
            embedding: None,
        };
        let latest = latest_acl_records(vec![older, current.clone()]);
        assert_eq!(latest.len(), 1);
        assert_eq!(
            latest[&("node".to_owned(), "node-1".to_owned(), String::new())].id,
            current.id
        );
    }
}
