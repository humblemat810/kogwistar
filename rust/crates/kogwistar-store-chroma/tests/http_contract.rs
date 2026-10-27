use kogwistar_store::{
    DistanceMetric, GraphEntityKind, GraphReadStore, GraphRecord, GraphWriteStore, MetadataFilter,
    VectorQuery,
};
use kogwistar_store_chroma::{ChromaGraphStore, ChromaStoreConfig};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    task::JoinHandle,
};

async fn fake_chroma(
    request_count: usize,
    metric: &'static str,
    query_distance: f64,
) -> (String, JoinHandle<Vec<(String, Value)>>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fixture");
    let address = listener.local_addr().expect("fixture address");
    let handle = tokio::spawn(async move {
        let mut observed = Vec::new();
        for _ in 0..request_count {
            let (mut stream, _) = listener.accept().await.expect("accept request");
            let mut bytes = Vec::new();
            let mut chunk = [0_u8; 4096];
            let (header_end, body_size) = loop {
                let count = stream.read(&mut chunk).await.expect("read request");
                assert!(count > 0, "client closed incomplete request");
                bytes.extend_from_slice(&chunk[..count]);
                if let Some(index) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&bytes[..index]);
                    let length = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        })
                        .unwrap_or(0);
                    if bytes.len() >= index + 4 + length {
                        break (index, length);
                    }
                }
            };
            let headers = String::from_utf8_lossy(&bytes[..header_end]);
            let first_line = headers.lines().next().expect("request line").to_owned();
            let body = &bytes[header_end + 4..header_end + 4 + body_size];
            let payload: Value = if body.is_empty() {
                Value::Null
            } else {
                serde_json::from_slice(body).expect("JSON request body")
            };
            let response = if first_line.contains("/collections HTTP") {
                json!({
                    "id": format!("collection-{}", payload["name"].as_str().unwrap_or("legacy")),
                    "metadata": {"hnsw:space": metric}
                })
            } else if first_line.contains("/upsert HTTP") || first_line.contains("/delete HTTP") {
                json!({"updated": 1, "deleted": 1})
            } else if first_line.contains("/query HTTP") {
                query_response(query_distance)
            } else if first_line.contains("/get HTTP") && payload.get("offset").is_some() {
                paged_get_response(
                    payload["offset"].as_u64().expect("page offset") as usize,
                    payload["limit"].as_u64().expect("page limit") as usize,
                )
            } else if first_line.contains("/get HTTP") && payload["ids"].is_array() {
                json!({
                    "ids": [payload["ids"][0]],
                    "documents": ["node body"],
                    "metadatas": [record_metadata()],
                    "embeddings": [[0.1, 0.2]]
                })
            } else {
                get_response()
            };
            let encoded = response.to_string();
            let response_head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                encoded.len()
            );
            stream
                .write_all(response_head.as_bytes())
                .await
                .expect("write headers");
            stream
                .write_all(encoded.as_bytes())
                .await
                .expect("write body");
            observed.push((first_line, payload));
        }
        observed
    });
    (format!("http://{address}"), handle)
}

#[tokio::test]
async fn kind_aware_chroma_collections_keep_same_node_and_edge_id_distinct() {
    let (endpoint, server) = fake_chroma(6, "cosine", 0.25).await;
    let common = ChromaStoreConfig {
        endpoint: endpoint.clone(),
        ..Default::default()
    };
    let store = ChromaGraphStore::connect_entity_collections(
        ChromaStoreConfig {
            collection: "nodes".to_owned(),
            ..common.clone()
        },
        ChromaStoreConfig {
            collection: "edges".to_owned(),
            ..common
        },
    )
    .await
    .expect("connect kind-specific collections");

    let mut node = GraphRecord::new("shared-id");
    node.metadata.insert("kind".to_owned(), json!("node"));
    let mut edge = GraphRecord::new("shared-id");
    edge.metadata.insert("kind".to_owned(), json!("edge"));
    store
        .upsert_graph_record_of_kind("workspace-a", GraphEntityKind::Node, node)
        .await
        .expect("write node collection");
    store
        .upsert_graph_record_of_kind("workspace-a", GraphEntityKind::Edge, edge)
        .await
        .expect("write edge collection");
    assert_eq!(
        store
            .graph_record_of_kind("workspace-a", GraphEntityKind::Node, "shared-id")
            .await
            .expect("read node collection")
            .expect("node exists")
            .id,
        "shared-id"
    );
    assert_eq!(
        store
            .graph_record_of_kind("workspace-a", GraphEntityKind::Edge, "shared-id")
            .await
            .expect("read edge collection")
            .expect("edge exists")
            .id,
        "shared-id"
    );

    let requests = server.await.expect("fixture server");
    assert_eq!(requests.len(), 6);
    assert!(
        requests[2]
            .0
            .contains("/collections/collection-nodes/upsert")
    );
    assert!(
        requests[3]
            .0
            .contains("/collections/collection-edges/upsert")
    );
    assert!(requests[4].0.contains("/collections/collection-nodes/get"));
    assert!(requests[5].0.contains("/collections/collection-edges/get"));
    assert_eq!(requests[2].1["ids"][0], "shared-id");
    assert_eq!(requests[3].1["ids"][0], "shared-id");
}

#[tokio::test]
async fn legacy_single_collection_rejects_kind_aware_reads_without_network_fallback() {
    let (endpoint, server) = fake_chroma(1, "cosine", 0.25).await;
    let store = ChromaGraphStore::connect(ChromaStoreConfig {
        endpoint,
        collection: "legacy-graph".to_owned(),
        ..Default::default()
    })
    .await
    .expect("connect legacy collection");

    assert!(
        store
            .graph_record_of_kind("workspace-a", GraphEntityKind::Node, "shared-id")
            .await
            .is_err()
    );
    let requests = server.await.expect("fixture server");
    assert_eq!(
        requests.len(),
        1,
        "the rejected typed read must not hit Chroma"
    );
}

fn record_metadata() -> Value {
    json!({
        "_kogwistar_namespace": "workspace-a",
        "_kogwistar_original_metadata": "{\"kind\":\"node\"}",
        "kind": "node"
    })
}

fn get_response() -> Value {
    json!({
        "ids": ["node-1"],
        "documents": ["node body"],
        "metadatas": [record_metadata()],
        "embeddings": [[0.1, 0.2]]
    })
}

fn paged_get_response(offset: usize, limit: usize) -> Value {
    let end = (offset + limit).min(501);
    let ids = (offset..end)
        .map(|index| format!("node-{index}"))
        .collect::<Vec<_>>();
    let count = ids.len();
    json!({
        "ids": ids,
        "documents": vec![Value::Null; count],
        "metadatas": vec![record_metadata(); count],
        "embeddings": vec![Value::Null; count]
    })
}

fn query_response(distance: f64) -> Value {
    json!({
        "ids": [["node-1"]],
        "documents": [["node body"]],
        "metadatas": [[record_metadata()]],
        "embeddings": [[[0.1, 0.2]]],
        "distances": [[distance]]
    })
}

#[tokio::test]
async fn chroma_graph_store_uses_scoped_v2_crud_and_vector_routes() {
    let (endpoint, server) = fake_chroma(6, "cosine", 0.25).await;
    let store = ChromaGraphStore::connect(ChromaStoreConfig {
        endpoint,
        collection: "test-graph".to_owned(),
        ..Default::default()
    })
    .await
    .expect("connect Chroma client");

    let mut record = GraphRecord::new("node-1");
    record.document = Some("node body".to_owned());
    record.metadata.insert("kind".to_owned(), json!("node"));
    record.embedding = Some(vec![0.1, 0.2]);
    store
        .upsert_graph_record("workspace-a", record.clone())
        .await
        .expect("upsert");

    let read = store
        .graph_record("workspace-a", "node-1")
        .await
        .expect("read")
        .expect("record");
    assert_eq!(read.id, "node-1");
    assert_eq!(read.metadata["kind"], "node");
    let matches = store
        .vector_query(
            "workspace-a",
            &VectorQuery {
                embedding: vec![0.2, 0.1],
                limit: 1,
                metadata: MetadataFilter::default(),
                metric: DistanceMetric::Cosine,
            },
        )
        .await
        .expect("query");
    assert_eq!(matches[0].record.id, "node-1");
    assert_eq!(matches[0].distance, 0.25);
    assert!(
        store
            .delete_graph_record("workspace-a", "node-1")
            .await
            .expect("delete")
    );

    let requests = server.await.expect("fixture server");
    assert!(
        requests[0]
            .0
            .contains("/api/v2/tenants/default_tenant/databases/default_database/collections")
    );
    assert_eq!(
        requests[1].1["metadatas"][0]["_kogwistar_namespace"],
        "workspace-a"
    );
    assert_eq!(
        requests[2].1["where"]["_kogwistar_namespace"]["$eq"],
        "workspace-a"
    );
    assert_eq!(requests[3].1["n_results"], 1);
    assert!(requests[5].0.contains("/delete HTTP"));
}

#[tokio::test]
async fn chroma_graph_records_reads_all_pages_without_skipping_boundary_rows() {
    let (endpoint, server) = fake_chroma(3, "cosine", 0.25).await;
    let store = ChromaGraphStore::connect(ChromaStoreConfig {
        endpoint,
        collection: "paged-graph".to_owned(),
        ..Default::default()
    })
    .await
    .expect("connect Chroma client");

    let records = store
        .graph_records("workspace-a", &MetadataFilter::default())
        .await
        .expect("read all graph pages");
    assert_eq!(records.len(), 501);
    assert_eq!(
        records.first().map(|record| record.id.as_str()),
        Some("node-0")
    );
    assert_eq!(
        records.get(499).map(|record| record.id.as_str()),
        Some("node-499")
    );
    assert_eq!(
        records.last().map(|record| record.id.as_str()),
        Some("node-500")
    );

    let requests = server.await.expect("fixture server");
    let page_requests = &requests[1..];
    assert_eq!(page_requests.len(), 2);
    assert_eq!(page_requests[0].1["offset"], 0);
    assert_eq!(page_requests[1].1["offset"], 500);
    assert_eq!(page_requests[0].1["limit"], 500);
    assert_eq!(page_requests[1].1["limit"], 500);
}

#[tokio::test]
async fn chroma_http_query_normalizes_all_shared_metric_contract_cases() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../contracts/golden/chroma-vector-distance-semantics-v1.json"
    ))
    .expect("shared Chroma distance semantics contract");
    assert_eq!(fixture["schema_version"], 1);

    for case in fixture["cases"].as_array().expect("fixture cases") {
        let (metric_name, metric) = match case["metric"].as_str().expect("metric") {
            "cosine" => ("cosine", DistanceMetric::Cosine),
            "ip" => ("ip", DistanceMetric::InnerProduct),
            "l2" => ("l2", DistanceMetric::L2),
            other => panic!("unknown metric {other:?}"),
        };
        let chroma_distance = case["chroma_distance"].as_f64().expect("distance");
        let (endpoint, server) = fake_chroma(2, metric_name, chroma_distance).await;
        let store = ChromaGraphStore::connect(ChromaStoreConfig {
            endpoint,
            collection: "metric-contract".to_owned(),
            metric,
            ..Default::default()
        })
        .await
        .expect("connect metric-matched Chroma collection");
        let matches = store
            .vector_query(
                "workspace-a",
                &VectorQuery {
                    embedding: vec![0.2, 0.1],
                    limit: 1,
                    metadata: MetadataFilter::default(),
                    metric,
                },
            )
            .await
            .expect("query metric-matched collection");
        assert_eq!(matches.len(), 1);
        assert_eq!(
            matches[0].distance,
            case["kogwistar_distance"]
                .as_f64()
                .expect("normalized distance")
        );
        let requests = server.await.expect("fixture server");
        assert_eq!(requests.len(), 2);
    }
}
