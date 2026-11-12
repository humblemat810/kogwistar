    #[tokio::test]
    async fn sqlite_chroma_graph_topology_reads_are_scoped_and_acl_filtered() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use jsonwebtoken::{EncodingKey, Header, encode};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tower::ServiceExt;

        struct TestTextEmbedder;
        impl EmbeddingProvider for TestTextEmbedder {
            fn profile(&self) -> &kogwistar_engine::embedding::EmbeddingProfile {
                static PROFILE: std::sync::OnceLock<kogwistar_engine::embedding::EmbeddingProfile> =
                    std::sync::OnceLock::new();
                PROFILE.get_or_init(|| kogwistar_engine::embedding::EmbeddingProfile {
                    provider: "test".to_owned(),
                    model: "sqlite-semantic-test".to_owned(),
                    dimension: 2,
                    similarity_metric: "cosine".to_owned(),
                    endpoint_fingerprint: None,
                    max_sequence_length: None,
                    crop_token_budget: None,
                    tokenizer_fingerprint: None,
                    crop_policy: None,
                    embedding_kind: Some("single_vector".to_owned()),
                    model_revision: None,
                    preprocessing_fingerprint: None,
                    max_image_patches: None,
                })
            }

            fn embed<'a>(
                &'a self,
                texts: &'a [String],
            ) -> kogwistar_engine::embedding::EmbeddingFuture<'a> {
                Box::pin(async move { Ok(texts.iter().map(|_| vec![0.1, 0.2]).collect()) })
            }
        }

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let fake = tokio::spawn(async move {
            let mut observed = Vec::new();
            let mut include_cross_doc_edge = false;
            loop {
                let Ok(Ok((mut stream, _))) =
                    tokio::time::timeout(std::time::Duration::from_secs(1), listener.accept())
                        .await
                else {
                    break;
                };
                let mut bytes = Vec::new();
                let mut chunk = [0_u8; 4096];
                let (split, length) = loop {
                    let count = stream.read(&mut chunk).await.unwrap();
                    assert!(count > 0);
                    bytes.extend_from_slice(&chunk[..count]);
                    if let Some(split) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&bytes[..split]);
                        let length = head.lines().find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        }).unwrap_or(0);
                        if bytes.len() >= split + 4 + length { break (split, length); }
                    }
                };
                let head = String::from_utf8_lossy(&bytes[..split]);
                let line = head.lines().next().unwrap().to_owned();
                let input: Value = serde_json::from_slice(&bytes[split + 4..split + 4 + length]).unwrap();
                let payload = if line.contains("/collections HTTP") {
                    json!({"id": format!("collection-{}", input["name"].as_str().unwrap()), "metadata":{"hnsw:space":"cosine"}})
                } else if line.contains("collection-nodes_index/query") {
                    json!({
                        "ids":[["idx:n2|Lung Cancer|review permit"]],
                        "documents":[["Lung Cancer review permit"]],
                        "metadatas":[[{
                            "_kogwistar_namespace":"scope-a",
                            "index_key":"n2|Lung Cancer|review permit",
                            "target_node_id":"n2",
                            "canonical_title":"Lung Cancer",
                            "provision":"review permit",
                            "doc_id":"doc-a"
                        }]],
                        "embeddings":[[[0.1,0.2]]],
                        "distances":[[0.1]]
                    })
                } else if line.contains("collection-nodes/query") {
                    include_cross_doc_edge = input["where"].to_string().contains("doc-a");
                    let scoped = |document: &str| {
                        let workspace = if document == "doc5" {"workspace-b"} else {"workspace-a"};
                        let original = if document == "doc5" {
                            json!({"workspace_id":workspace,"graph_space":"facts"})
                        } else if document == "doc2" {
                            json!({"workspace_id":workspace,"graph_space":"facts","doc_id":"doc-a","_kogwistar_stage2_ready":true,"_kogwistar_source_revision":1})
                        } else if document == "doc6" {
                            json!({"workspace_id":workspace,"graph_space":"facts","_kogwistar_stage2_ready":true,"_kogwistar_source_revision":4})
                        } else {
                            json!({"workspace_id":workspace,"graph_space":"facts","_kogwistar_stage2_ready":true,"_kogwistar_source_revision":2})
                        };
                        json!({
                            "_kogwistar_namespace":"scope-a",
                            "workspace_id":workspace,
                            "graph_space":"facts",
                            "_kogwistar_original_metadata":original.to_string(),
                            "_test_document": document
                        })
                    };
                    if input["where"].to_string().contains("doc-a") {
                        json!({
                            "ids":[["n2","n6"]],
                            "documents":[[r#"{"id":"n2","label":"Lung Cancer"}"#,r#"{"id":"n6","label":"Project scoped node"}"#]],
                            "metadatas":[[scoped("doc2"),scoped("doc6")]],
                            "embeddings":[[[0.1,0.2],[0.1,0.2]]],
                            "distances":[[0.1,0.02]]
                        })
                    } else {
                        json!({
                            "ids":[["n4","n2","n5","n6","n-missing-canonical"]],
                            "documents":[["doc4","doc2","doc5",r#"{"id":"n6","label":"Project scoped node"}"#,"orphan Chroma projection"]],
                            "metadatas":[[scoped("doc4"),scoped("doc2"),scoped("doc5"),scoped("doc6"),scoped("doc-missing-canonical")]],
                            "embeddings":[[[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2]]],
                            "distances":[[0.03,0.1,0.01,0.02,0.001]]
                        })
                    }
                } else if line.contains("collection-edges/query") {
                    let scoped = |document: &str| json!({
                        "_kogwistar_namespace":"scope-a",
                        "workspace_id":"workspace-a",
                        "graph_space":"facts",
                        "_kogwistar_original_metadata":"{\"workspace_id\":\"workspace-a\",\"graph_space\":\"facts\"}",
                        "_test_document": document
                    });
                    json!({
                        "ids":[["e2","e1","e3","e6"]],
                        "documents":[[
                            r#"{"source_ids":["n1"],"target_ids":["n3"]}"#,
                            r#"{"source_ids":["n1"],"target_ids":["n2"]}"#,
                            r#"{"source_ids":["n1"],"target_ids":["n5"]}"#,
                            r#"{"id":"e6","source_ids":["n1"],"target_ids":["n2"]}"#
                        ]],
                        "metadatas":[[scoped("edge2"),scoped("edge1"),scoped("edge3"),scoped("edge6")]],
                        "embeddings":[[[0.1,0.2],[0.1,0.2],[0.1,0.2],[0.1,0.2]]],
                        "distances":[[0.02,0.1,0.01,0.04]]
                    })
                } else if line.contains("collection-nodes/get")
                    && input["ids"] == json!(["redirect-edge-old"])
                {
                    json!({"ids":[],"documents":[],"metadatas":[],"embeddings":[]})
                } else if line.contains("collection-nodes/get")
                    && input["ids"].as_array().is_some_and(|ids| ids == &[json!("redirect-old")])
                {
                    json!({
                        "ids":["redirect-old"],
                        "documents":[r#"{"id":"redirect-old","label":"redirect source","metadata":{"lifecycle_status":"tombstoned","redirect_to_id":"n2"}}"#],
                        "metadatas":[{"_kogwistar_namespace":"scope-a","workspace_id":"workspace-a","graph_space":"facts","_kogwistar_original_metadata":"{\"workspace_id\":\"workspace-a\",\"graph_space\":\"facts\",\"lifecycle_status\":\"tombstoned\",\"redirect_to_id\":\"n2\"}"}],
                        "embeddings":[null]
                    })
                } else if line.contains("collection-nodes/get")
                    && input["ids"].as_array().is_some_and(|ids| ids == &[json!("n2")])
                {
                    json!({
                        "ids":["n2"],
                        "documents":[r#"{"id":"n2","label":"Lung Cancer"}"#],
                        "metadatas":[{"_kogwistar_namespace":"scope-a","workspace_id":"workspace-a","graph_space":"facts","_kogwistar_original_metadata":"{\"workspace_id\":\"workspace-a\",\"graph_space\":\"facts\",\"doc_ids\":[\"doc-a\"],\"_kogwistar_stage2_ready\":true,\"_kogwistar_source_revision\":2}"}],
                        "embeddings":[null]
                    })
                } else if line.contains("collection-edges/get")
                    && input["ids"] == json!(["redirect-edge-old"])
                {
                    json!({
                        "ids":["redirect-edge-old"],
                        "documents":[r#"{"id":"redirect-edge-old","source_ids":[],"target_ids":[],"metadata":{"lifecycle_status":"tombstoned","redirect_to_id":"e1"}}"#],
                        "metadatas":[{"_kogwistar_namespace":"scope-a","workspace_id":"workspace-a","graph_space":"facts","_kogwistar_original_metadata":"{\"workspace_id\":\"workspace-a\",\"graph_space\":\"facts\",\"lifecycle_status\":\"tombstoned\",\"redirect_to_id\":\"e1\"}"}],
                        "embeddings":[null]
                    })
                } else if line.contains("collection-edges/get")
                    && input["ids"] == json!(["e1"])
                {
                    json!({
                        "ids":["e1"],
                        "documents":[r#"{"id":"e1","source_ids":["n1"],"target_ids":["n2"],"doc_id":"doc-a"}"#],
                        "metadatas":[{"_kogwistar_namespace":"scope-a","workspace_id":"workspace-a","graph_space":"facts","_kogwistar_original_metadata":"{\"workspace_id\":\"workspace-a\",\"graph_space\":\"facts\",\"doc_id\":\"doc-a\"}"}],
                        "embeddings":[null]
                    })
                } else if line.contains("collection-nodes/get") {
                    let metas = [
                        json!({}),
                        json!({"_kogwistar_stage2_ready":true,"_kogwistar_source_revision":1}),
                        json!({}),
                        json!({"_kogwistar_stage2_ready":true,"_kogwistar_source_revision":2}),
                        json!({}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"n1","acl_version":1,"acl_mode":"public"}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"n2","acl_version":1,"acl_mode":"public"}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"n3","acl_version":1,"acl_mode":"private"}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"n4","acl_version":1,"acl_mode":"public"}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"n5","acl_version":1,"acl_mode":"public"}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"edge","acl_target_entity_id":"e1","acl_version":1,"acl_mode":"public"}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"edge","acl_target_entity_id":"e3","acl_version":1,"acl_mode":"public"}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"edge","acl_target_entity_id":"e4","acl_version":1,"acl_mode":"public"}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"grounding","acl_target_entity_id":"n1","acl_target_item_id":"n1::mention::0","acl_version":1,"acl_mode":"public"}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"span","acl_target_entity_id":"n1","acl_target_item_id":"n1::mention::0::span::0","acl_version":1,"acl_mode":"public"}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"grounding","acl_target_entity_id":"n2","acl_target_item_id":"n2::mention::0","acl_version":1,"acl_mode":"public"}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"span","acl_target_entity_id":"n2","acl_target_item_id":"n2::mention::0::span::0","acl_version":1,"acl_mode":"public"}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"grounding","acl_target_entity_id":"e1","acl_target_item_id":"e1::mention::0","acl_version":1,"acl_mode":"public"}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"span","acl_target_entity_id":"e1","acl_target_item_id":"e1::mention::0::span::0","acl_version":1,"acl_mode":"public"}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"conversation","acl_target_grain":"node","acl_target_entity_id":"n3","acl_version":1,"acl_mode":"public"}),
                        json!({"entity_type":"node","label":"Project scoped node"}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"n6","acl_version":1,"acl_mode":"scope","owner_id":"other-owner","security_scope":"project-a"}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"edge","acl_target_entity_id":"e6","acl_version":1,"acl_mode":"scope","owner_id":"other-owner","security_scope":"project-a"}),
                        json!({"entity_type":"node","label":"Evidence protected node"}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"node","acl_target_entity_id":"n7","acl_version":1,"acl_mode":"public"}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"grounding","acl_target_entity_id":"n7","acl_target_item_id":"n7::mention::0","acl_version":1,"acl_mode":"public"}),
                        json!({"entity_type":"acl_record","acl_truth_graph":"knowledge","acl_target_grain":"span","acl_target_entity_id":"n7","acl_target_item_id":"n7::mention::0::span::0","acl_version":1,"acl_mode":"private","owner_id":"other-owner"}),
                    ];
                    let source_doc = |doc_id: &str, with_mention: bool| {
                        if !with_mention {
                            return "{}".to_owned();
                        }
                        json!({"mentions":[{"spans":[{"doc_id":doc_id,"insertion_method":"parser"}]}]}).to_string()
                    };
                    let ids = ["n1","n2","n3","n4","n5","acl-n1","acl-n2","acl-n3","acl-n4","acl-n5","acl-e1","acl-e3","acl-e4","acl-n1-mention","acl-n1-span","acl-n2-mention","acl-n2-span","acl-e1-mention","acl-e1-span","acl-n3-conversation","n6","acl-n6","acl-e6","n7","acl-n7","acl-n7-mention","acl-n7-span"];
                    json!({"ids":ids,
                        "documents":[source_doc("doc-a", true),source_doc("doc-a", true),source_doc("doc-a", false),source_doc("doc-b", false),source_doc("doc-c", false),"{}".to_owned(),"{}".to_owned(),"{}".to_owned(),"{}".to_owned(),"{}".to_owned(),"{}".to_owned(),"{}".to_owned(),"{}".to_owned(),"{}".to_owned(),"{}".to_owned(),"{}".to_owned(),"{}".to_owned(),"{}".to_owned(),"{}".to_owned(),"{}".to_owned(),"{}".to_owned(),"{}".to_owned(),"{}".to_owned(),source_doc("doc-a", true),"{}".to_owned(),"{}".to_owned(),"{}".to_owned()],
                        "metadatas":metas.into_iter().enumerate().map(|(index, m)| {
                            let workspace = if index == 4 || index == 9 {"workspace-b"} else {"workspace-a"};
                            let mut original = m.as_object().unwrap().clone();
                            original.insert("workspace_id".to_owned(), json!(workspace));
                            original.insert("graph_space".to_owned(), json!("facts"));
                            if index < 5 {
                                let doc_id = if index < 3 {"doc-a"} else if index == 3 {"doc-b"} else {"doc-c"};
                                original.insert("doc_ids".to_owned(), json!([doc_id]));
                            }
                            json!({"_kogwistar_namespace":"scope-a","workspace_id":workspace,"graph_space":"facts","_kogwistar_original_metadata":Value::Object(original).to_string()})
                        }).collect::<Vec<_>>(),
                        "embeddings":ids.iter().map(|id| match *id {"n1" | "n2" | "n4" => json!([0.1,0.2]), _ => Value::Null}).collect::<Vec<_>>()})
                } else if include_cross_doc_edge {
                    include_cross_doc_edge = false;
                    json!({"ids":["e1","e4"],
                        "documents":[r#"{"id":"e1","source_ids":["n1"],"target_ids":["n2"],"doc_id":"doc-a","mentions":[{"spans":[{"doc_id":"doc-a","insertion_method":"parser"}]}]}"#,r#"{"id":"e4","source_ids":["n1"],"target_ids":["n2"],"doc_id":"doc-b"}"#],
                        "metadatas":[
                            json!({"_kogwistar_namespace":"scope-a","workspace_id":"workspace-a","graph_space":"facts","_kogwistar_original_metadata":"{\"workspace_id\":\"workspace-a\",\"graph_space\":\"facts\",\"doc_id\":\"doc-a\"}"}),
                            json!({"_kogwistar_namespace":"scope-a","workspace_id":"workspace-a","graph_space":"facts","_kogwistar_original_metadata":"{\"workspace_id\":\"workspace-a\",\"graph_space\":\"facts\",\"doc_id\":\"doc-b\"}"})
                        ],
                        "embeddings":[null,null]})
                } else {
                    json!({"ids":["e1","e6"],
                        "documents":[r#"{"id":"e1","source_ids":["n1"],"target_ids":["n2"],"doc_id":"doc-a","mentions":[{"spans":[{"doc_id":"doc-a","insertion_method":"parser"}]}]}"#,
                            r#"{"id":"e6","source_ids":["n1"],"target_ids":["n2"]}"#],
                        "metadatas":[
                            json!({"_kogwistar_namespace":"scope-a","workspace_id":"workspace-a","graph_space":"facts","_kogwistar_original_metadata":"{\"workspace_id\":\"workspace-a\",\"graph_space\":\"facts\",\"doc_id\":\"doc-a\"}"}),
                            json!({"_kogwistar_namespace":"scope-a","workspace_id":"workspace-a","graph_space":"facts","_kogwistar_original_metadata":"{\"workspace_id\":\"workspace-a\",\"graph_space\":\"facts\"}"})
                        ],
                        "embeddings":[null,null]})
                };
                let body = payload.to_string();
                let response = format!("HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}", body.len(), body);
                stream.write_all(response.as_bytes()).await.unwrap();
                observed.push((line, input));
            }
            observed
        });
        let path = std::env::temp_dir().join(format!(
            "kogwistar-sqlite-chroma-{}.sqlite3",
            uuid::Uuid::new_v4()
        ));
        let search_index_path = std::env::temp_dir().join(format!(
            "kogwistar-sqlite-chroma-search-index-{}.sqlite3",
            uuid::Uuid::new_v4()
        ));
        let mut fts = crate::search_index_fts::open_index(&search_index_path).unwrap();
        crate::search_index_fts::upsert_entries(
            &mut fts,
            &[
                kogwistar_contracts::SearchIndexEntryV1 {
                    node_id: "n2".to_owned(),
                    canonical_title: "Lung Cancer".to_owned(),
                    keywords: vec!["review".to_owned()],
                    aliases: Vec::new(),
                    provision: "review permit".to_owned(),
                    doc_id: Some("doc-a".to_owned()),
                },
                kogwistar_contracts::SearchIndexEntryV1 {
                    node_id: "n6".to_owned(),
                    canonical_title: "Project scoped node".to_owned(),
                    keywords: vec!["review".to_owned()],
                    aliases: Vec::new(),
                    provision: "review permit".to_owned(),
                    doc_id: Some("doc-a".to_owned()),
                },
                kogwistar_contracts::SearchIndexEntryV1 {
                    node_id: "n7".to_owned(),
                    canonical_title: "Evidence protected node".to_owned(),
                    keywords: vec!["review".to_owned()],
                    aliases: Vec::new(),
                    provision: "review permit".to_owned(),
                    doc_id: Some("doc-a".to_owned()),
                },
            ],
        )
        .unwrap();
        drop(fts);
        let service = open_sqlite_application_service(
            &path,
            Some(SqliteChromaGraphConfig {
                endpoint,
                metric: DistanceMetric::Cosine,
                node_collection: "nodes".to_owned(),
                edge_collection: "edges".to_owned(),
            }),
        )
        .await
        .unwrap();
        assert!(!service.supports_mcp_tool("root", "kg_semantic_seed_then_expand_text"));
        let service = service
            .with_search_index_path(&search_index_path)
            .unwrap()
            .with_embedding_provider(Arc::new(TestTextEmbedder));
        let fts = service.search_index_fts.as_ref().unwrap().lock().unwrap();
        let lexical_candidates = crate::search_index_fts::search_fts(&fts, "review", 10).unwrap();
        assert!(
            lexical_candidates.iter().any(|hit| hit.node_id == "n7"),
            "private-span node must be a real lexical candidate: {lexical_candidates:?}"
        );
        drop(fts);
        let graph = service.chroma_graph.as_ref().unwrap();
        let filter = MetadataFilter::default();
        let (nodes, edges) = tokio::try_join!(
            graph.graph_records_of_kind_bounded("scope-a", GraphEntityKind::Node, &filter, 100),
            graph.graph_records_of_kind_bounded("scope-a", GraphEntityKind::Edge, &filter, 100),
        )
        .unwrap();
        assert_eq!(nodes.len(), 27, "decoded node records: {nodes:?}");
        assert_eq!(edges.len(), 2, "decoded edge records: {edges:?}");
        let principal = GraphAclPrincipal { principal_id: "reader-1".to_owned(), ..Default::default() };
        let (visible_nodes, visible_edges) = graph_records_visible_by_acl(&nodes, &edges, "knowledge", &principal);
        assert_eq!(visible_nodes.len(), 4, "ACL-visible nodes: {visible_nodes:?}");
        assert!(
            !visible_nodes.iter().any(|record| record.id == "n7"),
            "node with a private source span must be rejected by the ACL projection"
        );
        assert_eq!(visible_edges.len(), 1, "ACL-visible edges: {visible_edges:?}");
        let (conversation_nodes, _) =
            graph_records_visible_by_acl(&nodes, &edges, "conversation", &principal);
        assert_eq!(
            conversation_nodes.iter().map(|record| record.id.as_str()).collect::<Vec<_>>(),
            ["n3"],
            "the shared Chroma projection applies the selected graph role's ACL truth"
        );
        assert_eq!(
            graph_neighbors_from_records("n1", &visible_nodes, &visible_edges, None).nodes,
            ["n2".to_owned()].into()
        );
        let requested = vec!["n2".to_owned(), "n2".to_owned(), "missing".to_owned()];
        let batched = graph
            .graph_records_by_ids_of_kind("scope-a", GraphEntityKind::Node, &requested)
            .await
            .unwrap();
        assert_eq!(batched.iter().map(|record| record.id.as_str()).collect::<Vec<_>>(), ["n2"]);
        for (event_id, entity_kind, entity_id, op) in [
            ("n2-current", "node", "n2", "ADD"),
            ("n4-old", "node", "n4", "ADD"),
            ("n4-current", "node", "n4", "REPLACE"),
            ("n6-project", "node", "n6", "ADD"),
            ("e6-project", "edge", "e6", "ADD"),
        ] {
            service.store.append_raw_entity_event(
                "scope-a",
                SqliteNewRawEntityEvent {
                    event_id: event_id.to_owned(),
                    entity_kind: entity_kind.to_owned(),
                    entity_id: entity_id.to_owned(),
                    op: op.to_owned(),
                    payload_json: "{}".to_owned(),
                },
            ).unwrap();
        }
        for (entity_id, grain, mode, owner_id, security_scope) in [
            ("n1", "node", "public", None, None),
            ("n2", "node", "public", None, None),
            ("e1", "edge", "public", None, None),
            ("e3", "edge", "public", None, None),
            ("e4", "edge", "public", None, None),
            (
                "n6",
                "node",
                "scope",
                Some("other-owner"),
                Some("project-a"),
            ),
            (
                "e6",
                "edge",
                "scope",
                Some("other-owner"),
                Some("project-a"),
            ),
        ] {
            let acl_id = format!("acl-{entity_id}");
            let mut metadata = json!({
                "entity_type":"acl_record",
                "acl_truth_graph":"knowledge",
                "acl_target_entity_id":entity_id,
                "acl_target_grain":grain,
                "acl_version":1,
                "acl_mode":mode,
                "workspace_id":"workspace-a",
                "graph_space":"facts"
            });
            if let Some(owner_id) = owner_id {
                metadata["owner_id"] = json!(owner_id);
            }
            if let Some(security_scope) = security_scope {
                metadata["security_scope"] = json!(security_scope);
            }
            service.store.append_raw_entity_event(
                "scope-a",
                SqliteNewRawEntityEvent {
                    event_id: format!("canonical-{acl_id}"),
                    entity_kind: "node".to_owned(),
                    entity_id: acl_id.clone(),
                    op: "ADD".to_owned(),
                    payload_json: json!({
                        "id":acl_id,
                        "document":null,
                        "metadata":metadata
                    })
                    .to_string(),
                },
            ).unwrap();
        }
        for (entity_id, grain, item_id) in [
            ("n1", "grounding", "n1::mention::0"),
            ("n1", "span", "n1::mention::0::span::0"),
            ("n2", "grounding", "n2::mention::0"),
            ("n2", "span", "n2::mention::0::span::0"),
            ("e1", "grounding", "e1::mention::0"),
            ("e1", "span", "e1::mention::0::span::0"),
        ] {
            let acl_id = format!("acl-{entity_id}-{grain}");
            let metadata = json!({
                "entity_type":"acl_record",
                "acl_truth_graph":"knowledge",
                "acl_target_entity_id":entity_id,
                "acl_target_grain":grain,
                "acl_target_item_id":item_id,
                "acl_version":1,
                "acl_mode":"public",
                "workspace_id":"workspace-a",
                "graph_space":"facts"
            });
            service.store.append_raw_entity_event(
                "scope-a",
                SqliteNewRawEntityEvent {
                    event_id: format!("canonical-{acl_id}"),
                    entity_kind: "node".to_owned(),
                    entity_id: acl_id.clone(),
                    op: "ADD".to_owned(),
                    payload_json: json!({
                        "id":acl_id,
                        "document":null,
                        "metadata":metadata
                    })
                    .to_string(),
                },
            ).unwrap();
        }
        let conversation_acl_id = "acl-n3-conversation";
        let conversation_acl_metadata = json!({
            "entity_type":"acl_record",
            "acl_truth_graph":"conversation",
            "acl_target_entity_id":"n3",
            "acl_target_grain":"node",
            "acl_version":1,
            "acl_mode":"public",
            "workspace_id":"workspace-a",
            "graph_space":"facts"
        });
        service.store.append_raw_entity_event(
            "scope-a",
            SqliteNewRawEntityEvent {
                event_id: "canonical-acl-n3-conversation".to_owned(),
                entity_kind: "node".to_owned(),
                entity_id: conversation_acl_id.to_owned(),
                op: "ADD".to_owned(),
                payload_json: json!({
                    "id":conversation_acl_id,
                    "document":null,
                    "metadata":conversation_acl_metadata
                })
                .to_string(),
            },
        ).unwrap();
        let graph_read_tools = [
            "kg_neighbors",
            "kg_load_persisted",
            "kg_propose_bruteforce",
            "kg_find_edges",
            "kg_shortest_path",
            "kg_k_hop",
            "kg_semantic_seed_then_expand_text",
            "kg_viz_d3_json",
            "kg_viz_cytoscape_json",
        ];
        assert!(graph_read_tools.iter().all(|name| service.supports_mcp_tool("root", name)));
        assert!(!service.supports_mcp_tool("root", "kg_graph_mutate"));
        assert!(!SqliteRunApplicationService::open(&path).unwrap().supports_mcp_tool("root", "kg_neighbors"));

        let auth = AuthConfig {
            mode: "dev".to_owned(),
            algorithm: Some("HS256".to_owned()),
            key: Some("sqlite-chroma-mcp-secret".to_owned()),
            ..AuthConfig::default()
        };
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let token = |capabilities: &[&str]| {
            encode(
                &Header::new(jsonwebtoken::Algorithm::HS256),
                &json!({
                    "sub":"reader-1", "storage_ns":"scope-a", "workspace_id":"workspace-a",
                    "graph_space":"facts", "role":"ro", "ns":["docs"],
                    "capabilities":capabilities, "exp":now + 300
                }),
                &EncodingKey::from_secret(b"sqlite-chroma-mcp-secret"),
            )
            .unwrap()
        };
        let token_for_namespace = |namespace: &str, capabilities: &[&str]| {
            encode(
                &Header::new(jsonwebtoken::Algorithm::HS256),
                &json!({
                    "sub":"reader-1", "storage_ns":namespace, "workspace_id":"workspace-a",
                    "graph_space":"facts", "role":"ro", "ns":[namespace],
                    "capabilities":capabilities, "exp":now + 300
                }),
                &EncodingKey::from_secret(b"sqlite-chroma-mcp-secret"),
            )
            .unwrap()
        };
        let docs_writer_token = encode(
            &Header::new(jsonwebtoken::Algorithm::HS256),
            &json!({
                "sub":"writer-1", "storage_ns":"docs", "workspace_id":"workspace-a",
                "graph_space":"facts", "role":"rw", "ns":["docs"],
                "capabilities":["read_graph", "write_graph"], "exp":now + 300
            }),
            &EncodingKey::from_secret(b"sqlite-chroma-mcp-secret"),
        )
        .unwrap();
        let token_for_project = |project: &str| {
            encode(
                &Header::new(jsonwebtoken::Algorithm::HS256),
                &json!({
                    "sub":"reader-1", "storage_ns":"scope-a", "workspace_id":"workspace-a",
                    "graph_space":"facts", "project":project,
                    "role":"ro", "ns":["docs"], "capabilities":["read_graph"], "exp":now + 300
                }),
                &EncodingKey::from_secret(b"sqlite-chroma-mcp-secret"),
            )
            .unwrap()
        };
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec![],
                implementation: ImplementationSnapshot::default(),
                auth: auth.clone(),
            },
            Arc::new(service.clone()),
        );
        assert!(service.supports_http_operation("GET", "/api/search_index_hybrid"));
        for (project, truth_graph, should_see_project_records) in [
            ("project-a", "knowledge", true),
            ("project-b", "knowledge", false),
            ("project-a", "conversation", false),
        ] {
            for (kind, target_id) in [("node", "n6"), ("edge", "e6")] {
                let response = app
                    .clone()
                    .oneshot(
                        Request::builder()
                            .method("POST")
                            .uri("/internal/graph/vector-query")
                            .header("content-type", "application/json")
                            .header(
                                "authorization",
                                format!("Bearer {}", token_for_project(project)),
                            )
                            .body(Body::from(
                                json!({
                                    "entity_kind":kind,
                                    "query_embeddings":[[0.1,0.2]],
                                    "metric":"cosine",
                                    "truth_graph":truth_graph,
                                    "n_results":10,
                                    "include":["distances"]
                                })
                                .to_string(),
                            ))
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::OK);
                let result: Value =
                    serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap())
                        .unwrap();
                let ids = result["ids"][0].as_array().unwrap();
                assert_eq!(
                    ids.iter().any(|id| id == target_id),
                    should_see_project_records,
                    "unexpected project ACL result for {kind}, project={project}, truth_graph={truth_graph}: {result}"
                );
            }
        }
        let hybrid = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/search_index_hybrid?q=review&limit=5&resolve_node=true")
                    .header(
                        "authorization",
                        format!("Bearer {}", token(&["read_graph"])),
                    )
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(hybrid.status(), StatusCode::OK);
        let hybrid: Value =
            serde_json::from_slice(&to_bytes(hybrid.into_body(), 65_536).await.unwrap())
                .unwrap();
        let result = hybrid["results"]
            .as_array()
            .unwrap()
            .iter()
            .find(|result| result["node_id"] == "n2")
            .unwrap_or_else(|| panic!("authorized Chroma-index hit missing: {hybrid}"));
        assert_eq!(result["canonical_title"], "Lung Cancer");
        assert!(result["documents"].is_string());
        assert_eq!(result["metadatas"]["workspace_id"], "workspace-a");

        for (project, should_see_n6) in [("project-a", true), ("project-b", false)] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("GET")
                        .uri("/api/search_index_hybrid?q=review&limit=5&resolve_node=true")
                        .header(
                            "authorization",
                            format!("Bearer {}", token_for_project(project)),
                        )
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let result: Value =
                serde_json::from_slice(&to_bytes(response.into_body(), 65_536).await.unwrap())
                    .unwrap();
            let ids = result["results"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|hit| hit["node_id"].as_str())
                .collect::<Vec<_>>();
            assert!(ids.contains(&"n2"), "public hybrid hit was lost: {result}");
            assert_eq!(
                ids.contains(&"n6"),
                should_see_n6,
                "project scope leaked or hid a hybrid hit for {project}: {result}"
            );
            assert!(
                !ids.contains(&"n7"),
                "hybrid search leaked a node whose source span is private: {result}"
            );
        }

        let hybrid_denied = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/search_index_hybrid?q=review")
                    .header("authorization", format!("Bearer {}", token(&[])))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(hybrid_denied.status(), StatusCode::FORBIDDEN);

        let hybrid_other_namespace = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/search_index_hybrid?q=review")
                    .header(
                        "authorization",
                        format!(
                            "Bearer {}",
                            token_for_namespace("other-scope", &["read_graph"])
                        ),
                    )
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            hybrid_other_namespace.status(),
            StatusCode::INTERNAL_SERVER_ERROR,
            "foreign-namespace Chroma rows must fail closed rather than leak indexed metadata"
        );
        let hybrid_other_namespace: Value = serde_json::from_slice(
            &to_bytes(hybrid_other_namespace.into_body(), 65_536)
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(
            !hybrid_other_namespace.to_string().contains("Lung Cancer"),
            "foreign search-index metadata must not be returned: {hybrid_other_namespace}"
        );

        let conversation_visualization = app.clone().oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/viz/d3.json?graph_type=conversation")
                .header("authorization", format!("Bearer {}", token(&["read_graph"])))
                .body(Body::empty())
                .unwrap(),
        ).await.unwrap();
        assert_eq!(conversation_visualization.status(), StatusCode::OK);
        let conversation_visualization: Value = serde_json::from_slice(
            &to_bytes(conversation_visualization.into_body(), 65_536).await.unwrap(),
        ).unwrap();
        assert!(
            conversation_visualization["nodes"].to_string().contains("n3"),
            "selected graph role should use conversation ACL truth: {conversation_visualization}"
        );
        assert!(
            !conversation_visualization["nodes"].to_string().contains("n1"),
            "knowledge-only record must not pass conversation ACL: {conversation_visualization}"
        );
        let listing = app.clone().oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {}", token(&["read_graph"])))
                .body(Body::from(r#"{"jsonrpc":"2.0","id":"list","method":"tools/list"}"#))
                .unwrap(),
        ).await.unwrap();
        assert_eq!(listing.status(), StatusCode::OK);
        let listing: Value = serde_json::from_slice(&to_bytes(listing.into_body(), 65_536).await.unwrap()).unwrap();
        let advertised_tools = listing["result"]["tools"].as_array().unwrap();
        let advertised = advertised_tools.iter()
            .filter_map(|tool| tool["name"].as_str()).collect::<Vec<_>>();
        for ingestion_tool in ["doc_parse", "kg_extract", "store_document"] {
            assert!(
                !advertised.contains(&ingestion_tool),
                "Rust must not advertise Python-owned ingestion tool {ingestion_tool}"
            );
            assert!(
                !rust_mcp_tool_supported("root", ingestion_tool),
                "Rust support matrix must keep Python-owned tool {ingestion_tool} disabled"
            );
        }
        let docs_listing = app.clone().oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {docs_writer_token}"))
                .body(Body::from(r#"{"jsonrpc":"2.0","id":"docs-list","method":"tools/list"}"#))
                .unwrap(),
        ).await.unwrap();
        let docs_listing: Value = serde_json::from_slice(
            &to_bytes(docs_listing.into_body(), 65_536).await.unwrap(),
        ).unwrap();
        let docs_tools = docs_listing["result"]["tools"].to_string();
        for ingestion_tool in ["doc_parse", "kg_extract", "store_document"] {
            assert!(
                !docs_tools.contains(ingestion_tool),
                "Rust docs writer discovery must omit Python-owned tool {ingestion_tool}"
            );
        }
        assert!(advertised.contains(&"propose_vector"));
        assert!(!rust_mcp_tool_supported("root", "doc_parse"));
        let advertised_graph_tools = advertised
            .iter()
            .copied()
            .filter(|name| name.starts_with("kg_"))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            advertised_graph_tools,
            graph_read_tools.into_iter().collect(),
            "SQLite+Chroma must advertise exactly the implemented read-only graph surface"
        );
        let advertised_graph_contracts = advertised_tools
            .iter()
            .filter_map(|tool| {
                let name = tool["name"].as_str()?;
                graph_read_tools
                    .contains(&name)
                    .then(|| (name.to_owned(), tool.clone()))
            })
            .collect::<std::collections::BTreeMap<_, _>>();
        let frozen_tools: Value = serde_json::from_str(FROZEN_MCP_TOOLS_JSON).unwrap();
        let expected_graph_contracts = frozen_tools["surfaces"]["root"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|tool| {
                let name = tool["name"].as_str()?;
                graph_read_tools
                    .contains(&name)
                    .then(|| (name.to_owned(), tool.clone()))
            })
            .collect::<std::collections::BTreeMap<_, _>>();
        assert_eq!(
            advertised_graph_contracts, expected_graph_contracts,
            "Rust tools/list must preserve Python's graph-tool schemas and metadata"
        );

        let wisdom_listing = app.clone().oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token_for_namespace("wisdom", &["read_graph"])),
                )
                .body(Body::from(r#"{"jsonrpc":"2.0","id":"wisdom-list","method":"tools/list"}"#))
                .unwrap(),
        ).await.unwrap();
        assert_eq!(wisdom_listing.status(), StatusCode::OK);
        let wisdom_listing: Value = serde_json::from_slice(
            &to_bytes(wisdom_listing.into_body(), 65_536).await.unwrap(),
        ).unwrap();
        let wisdom_tools = wisdom_listing["result"]["tools"].as_array().unwrap();
        let wisdom_semantic_tool = wisdom_tools.iter()
            .find(|tool| tool["name"] == "wisdom.semantic_seed_then_expand");
        let expected_wisdom_semantic_tool = frozen_tools["surfaces"]["root"]
            .as_array().unwrap().iter()
            .find(|tool| tool["name"] == "wisdom.semantic_seed_then_expand")
            .expect("frozen Python catalog contains Wisdom semantic search");
        assert_eq!(wisdom_semantic_tool, Some(expected_wisdom_semantic_tool));
        assert!(
            !wisdom_tools.iter().any(|tool| tool["name"] == "wisdom.kg_upsert_graph"),
            "read-only Wisdom discovery must not expose graph writes"
        );

        let wisdom_listing_without_read_capability = app.clone().oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header(
                    "authorization",
                    format!("Bearer {}", token_for_namespace("wisdom", &[])),
                )
                .body(Body::from(r#"{"jsonrpc":"2.0","id":"wisdom-list-denied","method":"tools/list"}"#))
                .unwrap(),
        ).await.unwrap();
        assert_eq!(wisdom_listing_without_read_capability.status(), StatusCode::OK);
        let wisdom_listing_without_read_capability: Value = serde_json::from_slice(
            &to_bytes(wisdom_listing_without_read_capability.into_body(), 65_536).await.unwrap(),
        ).unwrap();
        assert!(
            !wisdom_listing_without_read_capability["result"]["tools"]
                .to_string().contains("wisdom.semantic_seed_then_expand"),
            "Wisdom semantic search must be hidden without read_graph"
        );

        for ingestion_tool in ["doc_parse", "kg_extract", "store_document"] {
            let rejected = app.clone().oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {docs_writer_token}"))
                    .body(Body::from(
                        json!({
                            "jsonrpc": "2.0",
                            "id": format!("python-owned-{ingestion_tool}"),
                            "method": "tools/call",
                            "params": {"name": ingestion_tool, "arguments": {}}
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            ).await.unwrap();
            let rejected: Value = serde_json::from_slice(
                &to_bytes(rejected.into_body(), 65_536).await.unwrap(),
            ).unwrap();
            assert_eq!(rejected["error"]["code"], -32602, "{ingestion_tool}: {rejected}");
            assert_eq!(
                rejected["error"]["message"],
                "Tool not advertised by this application service",
                "{ingestion_tool} must fail before any Python/provider fallback"
            );
        }

        let vector_proposal = app.clone().oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {}", token(&["read_graph"])))
                .body(Body::from(
                    r#"{"jsonrpc":"2.0","id":"vector-proposal","method":"tools/call","params":{"name":"propose_vector","arguments":{"inp":{"new_node_ids":["n2"],"top_k":10,"include_edges":false,"max_distance":1.0}}}}"#,
                ))
                .unwrap(),
        ).await.unwrap();
        let vector_proposal: Value = serde_json::from_slice(
            &to_bytes(vector_proposal.into_body(), 65_536).await.unwrap(),
        ).unwrap();
        assert!(vector_proposal.get("error").is_none(), "{vector_proposal}");
        assert!(
            vector_proposal["result"]["pairs"].as_array().unwrap().iter().any(|pair| {
                pair["left_id"] == "n1" && pair["right_id"] == "n2"
            }),
            "public vector candidates must be returned: {vector_proposal}"
        );
        assert!(
            !vector_proposal["result"]["pairs"].to_string().contains("n3"),
            "private vector candidate leaked: {vector_proposal}"
        );

        let vector_proposal_denied = app.clone().oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {}", token(&[])))
                .body(Body::from(
                    r#"{"jsonrpc":"2.0","id":"vector-proposal-denied","method":"tools/call","params":{"name":"propose_vector","arguments":{"inp":{"new_node_ids":["n2"],"include_edges":false}}}}"#,
                ))
                .unwrap(),
        ).await.unwrap();
        let vector_proposal_denied: Value = serde_json::from_slice(
            &to_bytes(vector_proposal_denied.into_body(), 65_536).await.unwrap(),
        ).unwrap();
        assert_eq!(vector_proposal_denied["error"]["code"], -32602);
        assert_eq!(
            vector_proposal_denied["error"]["message"],
            "Tool not advertised by this application service"
        );

        let denied = app.clone().oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {}", token(&[])))
                .body(Body::from(r#"{"jsonrpc":"2.0","id":"denied","method":"tools/call","params":{"name":"kg_neighbors","arguments":{"rid":"n1"}}}"#))
                .unwrap(),
        ).await.unwrap();
        let denied: Value = serde_json::from_slice(&to_bytes(denied.into_body(), 65_536).await.unwrap()).unwrap();
        assert_eq!(denied["error"]["data"]["code"], "KOGWISTAR_CAPABILITY_FORBIDDEN");

        let load_persisted_denied = app.clone().oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {}", token(&[])))
                .body(Body::from(r#"{"jsonrpc":"2.0","id":"load-denied","method":"tools/call","params":{"name":"kg_load_persisted","arguments":{"inp":{"doc_ids":["doc-a"],"sid":false}}}}"#))
                .unwrap(),
        ).await.unwrap();
        let load_persisted_denied: Value = serde_json::from_slice(
            &to_bytes(load_persisted_denied.into_body(), 65_536).await.unwrap(),
        ).unwrap();
        assert_eq!(load_persisted_denied["error"]["code"], -32602);
        assert_eq!(
            load_persisted_denied["error"]["message"],
            "Tool not advertised by this application service"
        );

        let load_contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/kg-load-persisted-mcp-v1.json"
        ))).unwrap();
        let loaded = app.clone().oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {}", token(&["read_graph"])))
                .body(Body::from(json!({
                    "jsonrpc":"2.0","id":"load-persisted","method":"tools/call",
                    "params":{"name":load_contract["operation"],"arguments":load_contract["arguments"]}
                }).to_string()))
                .unwrap(),
        ).await.unwrap();
        let loaded: Value = serde_json::from_slice(
            &to_bytes(loaded.into_body(), 65_536).await.unwrap(),
        ).unwrap();
        assert_eq!(loaded["result"]["node_ids"], load_contract["expected_structured_content"]["node_ids"]);
        assert_eq!(loaded["result"]["edge_ids"], load_contract["expected_structured_content"]["edge_ids"]);

        let loaded_by_method = app.clone().oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {}", token(&["read_graph"])))
                .body(Body::from(json!({
                    "jsonrpc":"2.0","id":"load-persisted-method","method":"tools/call",
                    "params":{"name":"kg_load_persisted","arguments":{"inp":{
                        "doc_ids":["doc-a"],"insertion_method":"parser","sid":false
                    }}}
                }).to_string()))
                .unwrap(),
        ).await.unwrap();
        let loaded_by_method: Value = serde_json::from_slice(
            &to_bytes(loaded_by_method.into_body(), 65_536).await.unwrap(),
        ).unwrap();
        assert_eq!(loaded_by_method["result"]["node_ids"], json!(["n1", "n2"]));
        assert_eq!(loaded_by_method["result"]["edge_ids"], json!(["e1"]));

        let neighbors_contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/kg-neighbors-mcp-v1.json"
        ))).unwrap();
        let mcp_read = app.clone().oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {}", token(&["read_graph"])))
                .body(Body::from(json!({
                    "jsonrpc":"2.0","id":"neighbors","method":"tools/call",
                    "params":{"name":neighbors_contract["operation"],"arguments":neighbors_contract["arguments"]}
                }).to_string()))
                .unwrap(),
        ).await.unwrap();
        assert_eq!(mcp_read.status(), StatusCode::OK);
        let mcp_read: Value = serde_json::from_slice(&to_bytes(mcp_read.into_body(), 65_536).await.unwrap()).unwrap();
        assert_eq!(mcp_read["result"]["nodes"], neighbors_contract["expected_structured_content"]["nodes"]);
        assert_eq!(mcp_read["result"]["edges"], neighbors_contract["expected_structured_content"]["edges"]);

        let find_edges_contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/kg-find-edges-mcp-v1.json"
        ))).unwrap();
        let find_edges_response = app.clone().oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header("content-type", "application/json")
                .header("authorization", format!("Bearer {}", token(&["read_graph"])))
                .body(Body::from(json!({
                    "jsonrpc":"2.0","id":"find-edges","method":"tools/call",
                    "params":{"name":find_edges_contract["operation"],"arguments":find_edges_contract["arguments"]}
                }).to_string()))
                .unwrap(),
        ).await.unwrap();
        assert_eq!(find_edges_response.status(), StatusCode::OK);
        let find_edges_response: Value = serde_json::from_slice(
            &to_bytes(find_edges_response.into_body(), 65_536).await.unwrap(),
        ).unwrap();
        assert_eq!(
            find_edges_response["result"]["edges"],
            find_edges_contract["expected_structured_content"]["edges"]
        );

        let k_hop_contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/kg-k-hop-mcp-v1.json"
        )))
        .unwrap();
        let k_hop_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {}", token(&["read_graph"])))
                    .body(Body::from(
                        json!({
                            "jsonrpc":"2.0","id":"k-hop","method":"tools/call",
                            "params":{"name":k_hop_contract["operation"],"arguments":k_hop_contract["arguments"]}
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(k_hop_response.status(), StatusCode::OK);
        let k_hop_response: Value = serde_json::from_slice(
            &to_bytes(k_hop_response.into_body(), 65_536).await.unwrap(),
        )
        .unwrap();
        assert_eq!(
            k_hop_response["result"]["layers"],
            k_hop_contract["expected_structured_content"]["layers"]
        );

        let redirected_read = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/graph/neighbors")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", token(&["read_graph"])),
                    )
                    .body(Body::from(
                        r#"{"rid":"redirect-old","resolve_mode":"redirect"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(redirected_read.status(), StatusCode::OK);
        let redirected_read: Value = serde_json::from_slice(
            &to_bytes(redirected_read.into_body(), 65_536).await.unwrap(),
        )
        .unwrap();
        assert_eq!(redirected_read["nodes"], json!(["n1"]));
        assert_eq!(redirected_read["edges"], json!(["e1"]));

        let redirected_edge_read = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/graph/neighbors")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", token(&["read_graph"])),
                    )
                    .body(Body::from(
                        r#"{"rid":"redirect-edge-old","resolve_mode":"redirect"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(redirected_edge_read.status(), StatusCode::OK);
        let redirected_edge_read: Value = serde_json::from_slice(
            &to_bytes(redirected_edge_read.into_body(), 65_536)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(redirected_edge_read["nodes"], json!(["n1", "n2"]));
        assert_eq!(redirected_edge_read["edges"], json!([]));

        let semantic_contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/kg-semantic-seed-expand-mcp-v1.json"
        )))
        .unwrap();
        let semantic_read = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", token(&["read_graph"])),
                    )
                    .body(Body::from(
                        json!({
                            "jsonrpc":"2.0","id":"semantic","method":"tools/call",
                            "params":{"name":semantic_contract["operation"],"arguments":semantic_contract["arguments"]}
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(semantic_read.status(), StatusCode::OK);
        let semantic_read: Value = serde_json::from_slice(
            &to_bytes(semantic_read.into_body(), 65_536).await.unwrap(),
        )
        .unwrap();
        assert!(
            semantic_read["result"]["seeds"].as_array().is_some(),
            "semantic retrieval should return a structured seed result: {semantic_read}"
        );
        let semantic_seeds = semantic_read["result"]["seeds"].as_array().unwrap();
        let semantic_seed_ids = semantic_seeds
            .iter()
            .filter_map(|seed| seed.as_str())
            .filter_map(|seed| serde_json::from_str::<Value>(seed).ok())
            .filter_map(|seed| seed["id"].as_str().map(str::to_owned))
            .collect::<Vec<_>>();
        assert_eq!(
            semantic_seed_ids,
            semantic_contract["expected_seed_ids"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect::<Vec<_>>()
        );
        let expanded_edge_ids = semantic_read["result"]["layers"][0]["edges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|edge| {
                serde_json::from_str::<Value>(edge.as_str().unwrap()).unwrap()["id"]
                    .as_str()
                    .unwrap()
                    .to_owned()
            })
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            expanded_edge_ids,
            semantic_contract["expected_first_layer_edge_ids"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        );
        let semantic_denied = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {}", token(&[])))
                    .body(Body::from(
                        json!({
                            "jsonrpc":"2.0","id":"semantic-denied","method":"tools/call",
                            "params":{"name":semantic_contract["operation"],"arguments":semantic_contract["arguments"]}
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let semantic_denied: Value = serde_json::from_slice(
            &to_bytes(semantic_denied.into_body(), 65_536).await.unwrap(),
        )
        .unwrap();
        assert_eq!(semantic_denied["error"]["code"], -32602);
        assert_eq!(
            semantic_denied["error"]["message"],
            "Tool not advertised by this application service"
        );

        let proposal_contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/kg-propose-bruteforce-mcp-v1.json"
        )))
        .unwrap();
        let mcp_proposals = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", token(&["read_graph"])),
                    )
                    .body(Body::from(
                        json!({
                            "jsonrpc":"2.0","id":"proposals","method":"tools/call",
                            "params":{"name":proposal_contract["operation"],"arguments":proposal_contract["arguments"]}
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mcp_proposals.status(), StatusCode::OK);
        let mcp_proposals: Value = serde_json::from_slice(
            &to_bytes(mcp_proposals.into_body(), 65_536).await.unwrap(),
        )
        .unwrap();
        assert_eq!(
            mcp_proposals["result"]["pairs"],
            proposal_contract["expected_structured_content"]["pairs"]
        );
        let mixed_proposals = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", token(&["read_graph"])),
                    )
                    .body(Body::from(
                        json!({
                            "jsonrpc":"2.0","id":"mixed-proposals","method":"tools/call",
                            "params":{"name":"kg_propose_bruteforce","arguments":{"inp":{
                                "pair_kind":"any_any","allowed_docs":["doc-a"]
                            }}}
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let mixed_proposals: Value = serde_json::from_slice(
            &to_bytes(mixed_proposals.into_body(), 65_536).await.unwrap(),
        )
        .unwrap();
        assert_eq!(mixed_proposals["result"]["pairs"].as_array().unwrap().len(), 3);
        assert!(mixed_proposals.to_string().contains("e1"));

        let proposal_denied = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {}", token(&[])))
                    .body(Body::from(
                        json!({
                            "jsonrpc":"2.0","id":"proposal-denied","method":"tools/call",
                            "params":{"name":"kg_propose_bruteforce","arguments":{"inp":{}}}
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let proposal_denied: Value = serde_json::from_slice(
            &to_bytes(proposal_denied.into_body(), 65_536).await.unwrap(),
        )
        .unwrap();
        assert_eq!(proposal_denied["error"]["code"], -32602);
        assert_eq!(
            proposal_denied["error"]["message"],
            "Tool not advertised by this application service"
        );
        assert!(proposal_denied.get("result").is_none());

        for (tool, expected_key) in [
            ("kg_viz_d3_json", "nodes"),
            ("kg_viz_cytoscape_json", "elements"),
        ] {
            let visualization = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/mcp")
                        .header("content-type", "application/json")
                        .header(
                            "authorization",
                            format!("Bearer {}", token(&["read_graph"])),
                        )
                        .body(Body::from(format!(
                            "{{\"jsonrpc\":\"2.0\",\"id\":\"{tool}\",\"method\":\"tools/call\",\"params\":{{\"name\":\"{tool}\",\"arguments\":{{\"doc_id\":null,\"mode\":\"reify\"}}}}}}"
                        )))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(visualization.status(), StatusCode::OK);
            let visualization: Value = serde_json::from_slice(
                &to_bytes(visualization.into_body(), 65_536).await.unwrap(),
            )
            .unwrap();
            let rows = visualization["result"][expected_key].as_array().unwrap();
            assert!(rows.iter().any(|row| row.to_string().contains("n1")));
            assert!(rows.iter().any(|row| row.to_string().contains("n2")));
            assert!(
                !visualization.to_string().contains("n3"),
                "private graph endpoint must not leak through {tool}: {visualization}"
            );
            assert!(
                !visualization.to_string().contains("n5"),
                "cross-workspace graph entity must not leak through {tool}: {visualization}"
            );
        }

        let d3_contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/kg-viz-d3-mcp-v1.json"
        )))
        .unwrap();
        let d3_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", token(&["read_graph"])),
                    )
                    .body(Body::from(
                        json!({
                            "jsonrpc":"2.0","id":"d3-contract","method":"tools/call",
                            "params":{"name":d3_contract["operation"],"arguments":d3_contract["arguments"]}
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let d3_response: Value = serde_json::from_slice(
            &to_bytes(d3_response.into_body(), 65_536).await.unwrap(),
        )
        .unwrap();
        let d3_node_ids = d3_response["result"]["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|node| node["id"].as_str().map(str::to_owned))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            d3_node_ids,
            d3_contract["expected_node_ids"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        );
        let d3_links = d3_response["result"]["links"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|link| {
                Some((
                    link["source"].as_str()?.to_owned(),
                    link["target"].as_str()?.to_owned(),
                    link["role"].as_str()?.to_owned(),
                ))
            })
            .collect::<std::collections::BTreeSet<_>>();
        let expected_d3_links = d3_contract["expected_links"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|link| {
                Some((
                    link["source"].as_str()?.to_owned(),
                    link["target"].as_str()?.to_owned(),
                    link["role"].as_str()?.to_owned(),
                ))
            })
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(d3_links, expected_d3_links);
        let d3_denied = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {}", token(&[])))
                    .body(Body::from(
                        json!({
                            "jsonrpc":"2.0","id":"d3-denied","method":"tools/call",
                            "params":{"name":d3_contract["operation"],"arguments":d3_contract["arguments"]}
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let d3_denied: Value = serde_json::from_slice(
            &to_bytes(d3_denied.into_body(), 65_536).await.unwrap(),
        )
        .unwrap();
        assert_eq!(d3_denied["error"]["code"], -32602);
        assert_eq!(
            d3_denied["error"]["message"],
            "Tool not advertised by this application service"
        );
        let cytoscape_contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/kg-viz-cytoscape-mcp-v1.json"
        )))
        .unwrap();
        let cytoscape_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header(
                        "authorization",
                        format!("Bearer {}", token(&["read_graph"])),
                    )
                    .body(Body::from(
                        json!({
                            "jsonrpc":"2.0","id":"cytoscape-contract","method":"tools/call",
                            "params":{"name":cytoscape_contract["operation"],"arguments":cytoscape_contract["arguments"]}
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let cytoscape_response: Value = serde_json::from_slice(
            &to_bytes(cytoscape_response.into_body(), 65_536).await.unwrap(),
        )
        .unwrap();
        let cytoscape_elements = cytoscape_response["result"]["elements"]
            .as_array()
            .unwrap()
            .iter()
            .map(|element| {
                let data = &element["data"];
                let mut normalized = serde_json::Map::from_iter([
                    ("id".to_owned(), data["id"].clone()),
                    ("classes".to_owned(), element["classes"].clone()),
                ]);
                if data.get("source").is_some() {
                    normalized.insert("source".to_owned(), data["source"].clone());
                    normalized.insert("target".to_owned(), data["target"].clone());
                }
                Value::Object(normalized)
            })
            .collect::<Vec<_>>();
        let mut sorted_cytoscape_elements = cytoscape_elements;
        sorted_cytoscape_elements.sort_by(|left, right| {
            left["id"].as_str().cmp(&right["id"].as_str())
        });
        assert_eq!(
            json!(sorted_cytoscape_elements),
            cytoscape_contract["expected_elements"]
        );
        let cytoscape_denied = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .header("authorization", format!("Bearer {}", token(&[])))
                    .body(Body::from(
                        json!({
                            "jsonrpc":"2.0","id":"cytoscape-denied","method":"tools/call",
                            "params":{"name":cytoscape_contract["operation"],"arguments":cytoscape_contract["arguments"]}
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let cytoscape_denied: Value = serde_json::from_slice(
            &to_bytes(cytoscape_denied.into_body(), 65_536).await.unwrap(),
        )
        .unwrap();
        assert_eq!(cytoscape_denied["error"]["code"], -32602);
        assert_eq!(
            cytoscape_denied["error"]["message"],
            "Tool not advertised by this application service"
        );

        for (path, result_key) in [
            ("/api/viz/d3.json", "nodes"),
            ("/api/viz/cytoscape.json", "elements"),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("GET")
                        .uri(format!("{path}?graph_type=knowledge&mode=reify"))
                        .header(
                            "authorization",
                            format!("Bearer {}", token(&["read_graph"])),
                        )
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{path}");
            let visualization: Value = serde_json::from_slice(
                &to_bytes(response.into_body(), 65_536).await.unwrap(),
            )
            .unwrap();
            let rows = visualization[result_key].as_array().unwrap();
            let serialized = visualization.to_string();
            assert!(serialized.contains("n1"), "{path}: {visualization}");
            assert!(serialized.contains("n2"), "{path}: {visualization}");
            assert!(serialized.contains("e1"), "{path}: {visualization}");
            assert!(!serialized.contains("n3"), "private node leaked: {path}");
            assert!(!serialized.contains("n5"), "cross-workspace node leaked: {path}");
            assert!(
                !serialized.contains("n-missing-canonical"),
                "orphan Chroma projection leaked: {path}"
            );
            assert!(!rows.is_empty(), "{path} returned no graph records");
        }

        let bundle = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/viz/d3.bundle?graph_type=knowledge&mode=reify")
                    .header(
                        "authorization",
                        format!("Bearer {}", token(&["read_graph"])),
                    )
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(bundle.status(), StatusCode::OK);
        assert_eq!(bundle.headers()["content-type"], "text/html; charset=utf-8");
        let bundle = String::from_utf8(
            to_bytes(bundle.into_body(), 1_048_576)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        let embedded = bundle
            .split_once("window.__EMBEDDED_DATA__ = ")
            .and_then(|(_, rest)| rest.split_once(';'))
            .map(|(payload, _)| serde_json::from_str::<Value>(payload).unwrap())
            .expect("SQLite+Chroma D3 bundle embeds the authorized projection");
        let embedded_json = embedded.to_string();
        assert!(embedded_json.contains("n1"));
        assert!(embedded_json.contains("n2"));
        assert!(embedded_json.contains("e1"));
        assert!(!embedded_json.contains("n3"));
        assert!(!embedded_json.contains("n5"));
        assert!(!embedded_json.contains("n-missing-canonical"));

        let request = |path: &str, body: Value, capabilities: Value| ApiEffectRequest {
            contract_version: 1,
            method: "POST".to_owned(),
            path_and_query: path.to_owned(),
            body: serde_json::to_vec(&body).unwrap(),
            principal: json!({"sub":"reader-1","storage_ns":"scope-a","workspace_id":"workspace-a","graph_space":"facts","capabilities":capabilities}),
        };
        let neighbor_body = json!({"rid":"n1","truth_graph":"knowledge"});
        let read_capability = json!(["read_graph"]);
        for route in [
            INTERNAL_GRAPH_NEIGHBORS_ROUTE,
            INTERNAL_GRAPH_VECTOR_QUERY_ROUTE,
            INTERNAL_GRAPH_FIND_EDGES_ROUTE,
            INTERNAL_GRAPH_SHORTEST_PATH_ROUTE,
            INTERNAL_GRAPH_K_HOP_ROUTE,
            INTERNAL_GRAPH_VIZ_D3_ROUTE,
            INTERNAL_GRAPH_VIZ_CYTOSCAPE_ROUTE,
        ] {
            assert!(service.supports_http_operation("POST", route), "route {route}");
        }
        assert_eq!(
            service
                .execute(request(
                    INTERNAL_GRAPH_NEIGHBORS_ROUTE,
                    neighbor_body.clone(),
                    json!([]),
                ))
                .await
                .status,
            StatusCode::FORBIDDEN.as_u16()
        );
        let result = service
            .execute(request(
                INTERNAL_GRAPH_NEIGHBORS_ROUTE,
                neighbor_body,
                read_capability.clone(),
            ))
            .await;
        assert_eq!(result.status, StatusCode::OK.as_u16());
        let result: Value = serde_json::from_slice(&result.body).unwrap();
        assert_eq!(result["nodes"], json!(["n2"]), "response: {result}");
        assert_eq!(result["edges"], json!(["e1"]), "response: {result}");

        let find_edges = service
            .execute(request(
                INTERNAL_GRAPH_FIND_EDGES_ROUTE,
                json!({}),
                read_capability.clone(),
            ))
            .await;
        assert_eq!(find_edges.status, StatusCode::OK.as_u16());
        assert_eq!(serde_json::from_slice::<Value>(&find_edges.body).unwrap()["edges"], json!(["e1"]));

        let shortest_path = service
            .execute(request(
                INTERNAL_GRAPH_SHORTEST_PATH_ROUTE,
                json!({"src_id":"n1","dst_id":"n2"}),
                read_capability.clone(),
            ))
            .await;
        assert_eq!(shortest_path.status, StatusCode::OK.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&shortest_path.body).unwrap()["path"],
            json!(["n1", "e1", "n2"])
        );

        let k_hop = service
            .execute(request(
                INTERNAL_GRAPH_K_HOP_ROUTE,
                json!({"start_ids":["n1"],"k":1}),
                read_capability.clone(),
            ))
            .await;
        assert_eq!(k_hop.status, StatusCode::OK.as_u16());
        assert_eq!(
            serde_json::from_slice::<Value>(&k_hop.body).unwrap()["layers"],
            json!([{"nodes":[],"edges":["e1"]}])
        );

        for (route, key) in [
            (INTERNAL_GRAPH_VIZ_D3_ROUTE, "nodes"),
            (INTERNAL_GRAPH_VIZ_CYTOSCAPE_ROUTE, "elements"),
        ] {
            let output = service
                .execute(request(route, json!({"mode":"reify"}), read_capability.clone()))
                .await;
            assert_eq!(
                output.status,
                StatusCode::OK.as_u16(),
                "visualization route {route} failed: {}",
                String::from_utf8_lossy(&output.body)
            );
            let output: Value = serde_json::from_slice(&output.body).unwrap();
            assert!(output[key].as_array().is_some());
            assert!(!output.to_string().contains("n3"));
        }

        let vector_query = service
            .execute(request(
                INTERNAL_GRAPH_VECTOR_QUERY_ROUTE,
                json!({
                    "entity_kind":"node",
                    "query_embeddings":[[0.1,0.2]],
                    "metric":"cosine",
                    "n_results":2,
                    "include":["documents","distances"]
                }),
                read_capability,
            ))
            .await;
        assert_eq!(vector_query.status, StatusCode::OK.as_u16());
        let vector_query: Value = serde_json::from_slice(&vector_query.body).unwrap();
        assert_eq!(
            vector_query["ids"],
            json!([["n2"]]),
            "the highest-ranked orphan Chroma projection must be removed without a canonical event"
        );
        assert_eq!(vector_query["documents"], json!([["doc2"]]));
        assert_eq!(vector_query["distances"], json!([[0.1]]));

        let edge_vector_query = service
            .execute(request(
                INTERNAL_GRAPH_VECTOR_QUERY_ROUTE,
                json!({
                    "entity_kind":"edge",
                    "query_embeddings":[[0.1,0.2]],
                    "metric":"cosine",
                    "n_results":2,
                    "include":["documents","distances"]
                }),
                json!(["read_graph"]),
            ))
            .await;
        assert_eq!(edge_vector_query.status, StatusCode::OK.as_u16());
        let edge_vector_query: Value = serde_json::from_slice(&edge_vector_query.body).unwrap();
        assert_eq!(edge_vector_query["ids"], json!([["e1"]]));
        assert_eq!(edge_vector_query["documents"], json!([[
            r#"{"source_ids":["n1"],"target_ids":["n2"]}"#
        ]]));
        assert_eq!(edge_vector_query["distances"], json!([[0.1]]));

        let revoke_edge_acl = service
            .execute(ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/internal/graph/mutations".to_owned(),
                body: serde_json::to_vec(&json!({
                    "contract_version":1,
                    "atomic":true,
                    "index_jobs":[],
                    "mutations":[{
                        "scope":{"namespace":"scope-a","workspace_id":"workspace-a","graph_space":"facts"},
                        "table":"gke_nodes",
                        "entity_kind":"node",
                        "event_id":"revoke-chroma-acl-e4-v2",
                        "op":"REPLACE",
                        "payload":{
                            "id":"acl-e4",
                            "document":null,
                            "metadata":{
                                "entity_type":"acl_record",
                                "acl_truth_graph":"knowledge",
                                "acl_target_entity_id":"e4",
                                "acl_target_grain":"edge",
                                "acl_version":2,
                                "acl_mode":"private",
                                "owner_id":"different-owner",
                                "workspace_id":"workspace-a",
                                "graph_space":"facts"
                            }
                        },
                        "record":{
                            "id":"acl-e4",
                            "document":null,
                            "metadata":{
                                "entity_type":"acl_record",
                                "acl_truth_graph":"knowledge",
                                "acl_target_entity_id":"e4",
                                "acl_target_grain":"edge",
                                "acl_version":2,
                                "acl_mode":"private",
                                "owner_id":"different-owner",
                                "workspace_id":"workspace-a",
                                "graph_space":"facts"
                            },
                            "embedding":null
                        },
                        "embedding_dim":0
                    }]
                }))
                .unwrap(),
                principal: json!({
                    "sub":"acl-owner",
                    "role":"rw",
                    "storage_ns":"scope-a",
                    "workspace_id":"workspace-a",
                    "graph_space":"facts",
                    "capabilities":["write_graph"]
                }),
            })
            .await;
        assert_eq!(revoke_edge_acl.status, StatusCode::OK.as_u16());
        let post_revoke_snapshot = service
            .chroma_graph_snapshot(
                &request(INTERNAL_GRAPH_VIZ_D3_ROUTE, json!({}), json!(["read_graph"])),
                "knowledge",
            )
            .await
            .unwrap();
        assert!(
            post_revoke_snapshot.edges().iter().all(|edge| edge.id != "e4"),
            "stale public Chroma ACL must not preserve a revoked edge"
        );

        let observed = fake.await.unwrap();
        assert!(observed.len() >= 20);
        assert!(observed.iter().any(|(line, _)| line.contains("collection-nodes/get")));
        assert!(observed.iter().any(|(line, _)| line.contains("collection-edges/get")));
        assert!(observed.iter().any(|(line, payload)| {
            line.contains("collection-nodes/get") && payload["ids"] == json!(["n2", "missing"])
        }));
        let scoped_reads = observed
            .iter()
            .filter(|(line, payload)| {
                line.contains("/get HTTP") && payload["where"].to_string().contains("workspace_id")
            })
            .collect::<Vec<_>>();
        assert!(scoped_reads.len() >= 12);
        assert!(
            scoped_reads.iter().all(|(_, payload)| {
                payload["where"].to_string().contains("workspace_id")
                    && payload["where"].to_string().contains("graph_space")
            }),
            "scoped Chroma query payloads: {scoped_reads:#?}"
        );
        let vector_request = observed
            .iter()
            .find(|(line, payload)| {
                line.contains("collection-nodes/query") && payload["n_results"] == 2
            })
            .expect("scoped node vector request");
        assert_eq!(vector_request.1["n_results"], 2);
        assert!(vector_request.1["where"].to_string().contains("workspace_id"));
        assert!(vector_request.1["where"].to_string().contains("graph_space"));
        let metadata_only_path = std::env::temp_dir().join(format!(
            "kogwistar-sqlite-search-index-unavailable-{}.sqlite3",
            uuid::Uuid::new_v4()
        ));
        let metadata_only = SqliteRunApplicationService::open(&metadata_only_path).unwrap();
        assert!(!metadata_only.supports_http_operation(
            "GET",
            "/api/search_index_hybrid"
        ));
        let metadata_only_app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: vec![],
                implementation: ImplementationSnapshot::default(),
                auth: auth.clone(),
            },
            Arc::new(metadata_only),
        );
        let unavailable_search = metadata_only_app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/search_index_hybrid?q=review")
                    .header(
                        "authorization",
                        format!("Bearer {}", token(&["read_graph"])),
                    )
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unavailable_search.status(), StatusCode::NOT_IMPLEMENTED);
        let unavailable_body = to_bytes(unavailable_search.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&unavailable_body).unwrap()["code"],
            "KOGWISTAR_API_CAPABILITY_NOT_CUT_OVER"
        );
        let _ = std::fs::remove_file(metadata_only_path);
        drop(service);
        let _ = std::fs::remove_file(path);
    }
