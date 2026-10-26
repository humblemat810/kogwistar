use kogwistar_store::{
    DistanceMetric, GraphReadStore, GraphRecord, GraphWriteStore, MetadataFilter, VectorQuery,
};
use kogwistar_store_chroma::{ChromaGraphStore, ChromaStoreConfig};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    task::JoinHandle,
};

async fn fake_chroma(request_count: usize) -> (String, JoinHandle<Vec<(String, Value)>>) {
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
                json!({"id": "collection-1", "metadata": {"hnsw:space": "cosine"}})
            } else if first_line.contains("/upsert HTTP") || first_line.contains("/delete HTTP") {
                json!({"updated": 1, "deleted": 1})
            } else if first_line.contains("/query HTTP") {
                query_response()
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

fn query_response() -> Value {
    json!({
        "ids": [["node-1"]],
        "documents": [["node body"]],
        "metadatas": [[record_metadata()]],
        "embeddings": [[[0.1, 0.2]]],
        "distances": [[0.25]]
    })
}

#[tokio::test]
async fn chroma_graph_store_uses_scoped_v2_crud_and_vector_routes() {
    let (endpoint, server) = fake_chroma(6).await;
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
