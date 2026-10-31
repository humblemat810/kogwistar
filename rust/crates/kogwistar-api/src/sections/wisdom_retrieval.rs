const WISDOM_SEMANTIC_SEED_EXPAND_ROUTE: &str = "/internal/wisdom/semantic-seed-expand";
const GRAPH_SEMANTIC_SEED_EXPAND_ROUTE: &str = "/internal/graph/semantic-seed-expand";

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SemanticSeedExpandRequest {
    text: String,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    hops: Option<usize>,
    #[serde(default)]
    doc_ids: Option<String>,
}

fn graph_record_document_json(record: &GraphRecord) -> Option<String> {
    let mut document: Value = serde_json::from_str(record.document.as_deref()?).ok()?;
    let object = document.as_object_mut()?;
    object.insert("metadata".to_owned(), json!(record.metadata));
    object.insert("embedding".to_owned(), json!(record.embedding));
    serde_json::to_string(&document).ok()
}

impl PostgresRunApplicationService {
    async fn wisdom_semantic_seed_expand_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        self.semantic_seed_expand_effect(&self.wisdom_store, request, Some("wisdom"), 10, 2)
            .await
    }

    async fn graph_semantic_seed_expand_effect(
        &self,
        request: &ApiEffectRequest,
    ) -> ApiEffectResponse {
        self.semantic_seed_expand_effect(&self.store, request, None, 5, 1)
            .await
    }

    async fn semantic_seed_expand_effect(
        &self,
        store: &PostgresStore,
        request: &ApiEffectRequest,
        required_namespace: Option<&str>,
        default_top_k: usize,
        default_hops: usize,
    ) -> ApiEffectResponse {
        let has_namespace = required_namespace.is_none_or(|required| {
            principal_strings(&request.principal, "ns")
                .iter()
                .any(|namespace| namespace == required || namespace == "*")
        });
        if !matches!(principal_role(&request.principal).as_str(), "ro" | "rw")
            || !has_namespace
            || !capability_allowed(request, &self.capabilities, "read_graph", &["read_graph"])
        {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_CAPABILITY_FORBIDDEN",
                "Semantic graph retrieval requires an authorized read_graph scope",
            );
        }
        let input: SemanticSeedExpandRequest = match serde_json::from_slice(&request.body) {
            Ok(input) => input,
            Err(error) => {
                return effect_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "KOGWISTAR_INVALID_REQUEST",
                    error.to_string(),
                );
            }
        };
        let top_k = input.top_k.unwrap_or(default_top_k);
        let hops = input.hops.unwrap_or(default_hops);
        if input.text.trim().is_empty()
            || input.text.chars().count() > 16_384
            || !(1..=1000).contains(&top_k)
            || hops > 32
        {
            return effect_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "KOGWISTAR_INVALID_REQUEST",
                "text must contain 1 to 16384 characters, top_k must be 1 to 1000, and hops must be at most 32",
            );
        }
        let Some(provider) = self.embedding_provider.as_ref() else {
            return effect_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "KOGWISTAR_EMBEDDING_UNAVAILABLE",
                "Wisdom semantic retrieval requires a configured embedding provider",
            );
        };
        let profile = provider.profile();
        if let Err(error) = profile.validate() {
            return effect_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "KOGWISTAR_EMBEDDING_UNAVAILABLE",
                error.to_string(),
            );
        }
        let query_text = input.text.trim().to_owned();
        let mut vectors = match provider.embed(std::slice::from_ref(&query_text)).await {
            Ok(vectors) => vectors,
            Err(_) => {
                return effect_error(
                    StatusCode::BAD_GATEWAY,
                    "KOGWISTAR_EMBEDDING_FAILED",
                    "Embedding provider failed",
                );
            }
        };
        if vectors.len() != 1 {
            return effect_error(
                StatusCode::BAD_GATEWAY,
                "KOGWISTAR_EMBEDDING_FAILED",
                "Embedding provider returned an unexpected query-vector count",
            );
        }
        let query_embedding = vectors.remove(0);
        if query_embedding.len() != profile.dimension
            || query_embedding.iter().any(|value| !value.is_finite())
        {
            return effect_error(
                StatusCode::BAD_GATEWAY,
                "KOGWISTAR_EMBEDDING_FAILED",
                "Embedding provider returned an invalid query vector",
            );
        }
        let query_embedding = query_embedding
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        if query_embedding.iter().any(|value| !value.is_finite()) {
            return effect_error(
                StatusCode::BAD_GATEWAY,
                "KOGWISTAR_EMBEDDING_FAILED",
                "Embedding query vector exceeds graph storage range",
            );
        }
        let metric = match profile.similarity_metric.to_ascii_lowercase().as_str() {
            "cosine" => DistanceMetric::Cosine,
            "l2" => DistanceMetric::L2,
            "ip" | "inner" | "inner_product" | "innerproduct" => {
                DistanceMetric::InnerProduct
            }
            _ => {
                return effect_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "KOGWISTAR_EMBEDDING_UNAVAILABLE",
                    "Embedding profile has an unsupported similarity metric",
                );
            }
        };
        let scope_value = security_scope_value(&request.principal);
        let optional_scope_component = |key: &str| {
            scope_value[key]
                .as_str()
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
        };
        let scope = GraphScope {
            namespace: scope_value["storage_namespace"]
                .as_str()
                .unwrap_or("default")
                .to_owned(),
            workspace_id: optional_scope_component("workspace"),
            graph_space: optional_scope_component("graph_space"),
        };
        if required_namespace.is_some_and(|required| scope.namespace != required) {
            return effect_error(
                StatusCode::FORBIDDEN,
                "KOGWISTAR_SCOPE_FORBIDDEN",
                "Semantic retrieval is restricted to the required namespace",
            );
        }
        let metadata = input
            .doc_ids
            .as_deref()
            .filter(|doc_id| !doc_id.trim().is_empty())
            .map(|doc_id| BTreeMap::from([("doc_id".to_owned(), json!(doc_id))]))
            .unwrap_or_default();
        let matches = match store
            .graph_projection_vector_query_batch(vec![GraphProjectionVectorQuery {
                scope: scope.clone(),
                table: "gke_nodes".to_owned(),
                query: VectorQuery {
                    embedding: query_embedding,
                    limit: top_k,
                    metadata: MetadataFilter { equals: metadata },
                    metric,
                },
                embedding_dim: profile.dimension,
            }])
            .await
        {
            Ok(mut rows) => rows.pop().unwrap_or_default(),
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_VECTOR_QUERY_ERROR",
                    error.to_string(),
                );
            }
        };
        let candidate_ids = matches
            .iter()
            .map(|matched| matched.record.id.clone())
            .collect::<Vec<_>>();
        let truth_graph = required_namespace.unwrap_or("knowledge");
        let candidate_acl = match store
            .graph_acl_records_for_entities(&scope, truth_graph, &candidate_ids)
            .await
        {
            Ok(records) => records,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_ACL_QUERY_ERROR",
                    error.to_string(),
                );
            }
        };
        let mut candidate_acl_index = GraphAclIndex::default();
        candidate_acl_index.extend_from_records(&candidate_acl, truth_graph);
        let acl_principal = graph_acl_principal_from_claims(
            &request.principal,
            scope_value["security_scope"].as_str(),
        );
        let (visible_seeds, _) = graph_records_visible_by_acl_with_index(
            &matches
                .iter()
                .map(|matched| matched.record.clone())
                .collect::<Vec<_>>(),
            &[],
            &candidate_acl_index,
            &acl_principal,
        );
        let visible_seed_ids = visible_seeds
            .iter()
            .map(|record| record.id.clone())
            .collect::<Vec<_>>();
        let seed_documents = visible_seeds
            .iter()
            .filter_map(graph_record_document_json)
            .collect::<Vec<_>>();

        let snapshot = match store
            .graph_projection_snapshot(vec![
                GraphProjectionRead {
                    scope: scope.clone(),
                    table: "gke_nodes".to_owned(),
                    ids: None,
                    metadata: MetadataFilter::default(),
                    limit: 10_000,
                },
                GraphProjectionRead {
                    scope: scope.clone(),
                    table: "gke_edges".to_owned(),
                    ids: None,
                    metadata: MetadataFilter::default(),
                    limit: 10_000,
                },
            ])
            .await
        {
            Ok(snapshot) if snapshot.len() == 2 => snapshot,
            Ok(_) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    "Wisdom graph snapshot returned an invalid shape",
                );
            }
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_STORE_ERROR",
                    error.to_string(),
                );
            }
        };
        let nodes = snapshot[0].clone();
        let edges = snapshot[1].clone();
        let all_ids = nodes
            .iter()
            .chain(&edges)
            .map(|record| record.id.clone())
            .collect::<Vec<_>>();
        let acl_records = match store
            .graph_acl_records_for_entities(&scope, truth_graph, &all_ids)
            .await
        {
            Ok(records) => records,
            Err(error) => {
                return effect_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "KOGWISTAR_ACL_QUERY_ERROR",
                    error.to_string(),
                );
            }
        };
        let mut acl_index = GraphAclIndex::default();
        acl_index.extend_from_records(&acl_records, truth_graph);
        let (nodes, edges) = graph_records_visible_by_acl_with_index(
            &nodes,
            &edges,
            &acl_index,
            &acl_principal,
        );
        let layers = graph_k_hop_from_records(
            &GraphKHopQuery {
                start_ids: visible_seed_ids,
                k: hops,
                doc_id: input.doc_ids,
            },
            &nodes,
            &edges,
        )
        .into_iter()
        .map(|layer| {
            let node_documents = layer
                .nodes
                .iter()
                .filter_map(|id| nodes.iter().find(|record| &record.id == id))
                .filter_map(graph_record_document_json)
                .collect::<BTreeSet<_>>();
            let edge_documents = layer
                .edges
                .iter()
                .filter_map(|id| edges.iter().find(|record| &record.id == id))
                .filter_map(graph_record_document_json)
                .collect::<BTreeSet<_>>();
            json!({
                "nodes": node_documents.into_iter().collect::<Vec<_>>(),
                "edges": edge_documents.into_iter().collect::<Vec<_>>(),
            })
        })
        .collect::<Vec<_>>();
        json_effect(
            StatusCode::OK,
            json!({"seeds": seed_documents, "layers": layers}),
        )
    }
}
