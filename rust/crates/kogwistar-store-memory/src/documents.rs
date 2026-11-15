//! Volatile source-document projection support. Python parity: in-memory graph backend document lookup/upsert/delete.
use super::{InMemoryStore, namespace};
use kogwistar_store::{
    DocumentAuthorityCapability, DocumentProjectionUpsert, DocumentStore, GraphMutationStore,
    GraphProjectionRead, GraphScope, MetadataFilter, StoreError, StoreResult,
};
use serde_json::Value;

const DOCUMENT_TABLE: &str = "gke_documents";

impl DocumentStore for InMemoryStore {
    fn supports_document_projection_lookup(&self) -> bool {
        true
    }

    fn supports_document_projection_delete(&self) -> bool {
        true
    }

    fn document_projection_ids(
        &self,
        scope: GraphScope,
        ids: Vec<String>,
    ) -> impl std::future::Future<Output = StoreResult<Vec<String>>> + Send {
        async move {
            if ids.is_empty() {
                return Ok(Vec::new());
            }
            let rows = self
                .graph_projection_records(GraphProjectionRead {
                    scope,
                    table: DOCUMENT_TABLE.to_owned(),
                    ids: Some(ids),
                    metadata: MetadataFilter::default(),
                    limit: usize::MAX,
                })
                .await?;
            Ok(rows.into_iter().map(|row| row.id).collect())
        }
    }

    fn upsert_document_projection(
        &self,
        write: DocumentProjectionUpsert,
    ) -> impl std::future::Future<Output = StoreResult<()>> + Send {
        async move {
            if write.scope.namespace.trim().is_empty() || write.record.id.trim().is_empty() {
                return Err(StoreError::Backend {
                    backend: "memory-document".to_owned(),
                    message: "document namespace and ID must be non-empty".to_owned(),
                });
            }
            if write.record.document.is_none() {
                return Err(StoreError::Backend {
                    backend: "memory-document".to_owned(),
                    message: "source document text is required".to_owned(),
                });
            }
            namespace(&write.scope.namespace)?;
            write
                .scope
                .validate_metadata_scope(&write.record.metadata)?;
            if let Some(embedding) = &write.record.embedding {
                if embedding.iter().any(|value| !value.is_finite()) {
                    return Err(StoreError::NonFiniteVector);
                }
                if embedding.len() != write.embedding_dim {
                    return Err(StoreError::VectorDimensionMismatch {
                        expected: write.embedding_dim,
                        actual: embedding.len(),
                    });
                }
            }

            let mut state = self.state.write().expect("in-memory store lock poisoned");
            let projection = state.graph_projections.entry(write.scope).or_default();
            let key = (DOCUMENT_TABLE.to_owned(), write.record.id.clone());
            if !projection.records.contains_key(&key) {
                projection.record_order.push(key.clone());
            }
            projection.records.insert(key, write.record);
            Ok(())
        }
    }

    fn delete_document_projection(
        &self,
        scope: GraphScope,
        document_id: String,
    ) -> impl std::future::Future<Output = StoreResult<bool>> + Send {
        async move {
            namespace(&scope.namespace)?;
            if document_id.trim().is_empty() {
                return Err(StoreError::EmptyRecordId);
            }
            let mut state = self.state.write().expect("in-memory store lock poisoned");
            let Some(projection) = state.graph_projections.get_mut(&scope) else {
                return Ok(false);
            };
            let key = (DOCUMENT_TABLE.to_owned(), document_id.to_owned());
            projection.record_order.retain(|item| item != &key);
            Ok(projection.records.remove(&key).is_some())
        }
    }

    fn delete_document_projections_by_source_id(
        &self,
        scope: GraphScope,
        source_document_id: String,
    ) -> impl std::future::Future<Output = StoreResult<Vec<String>>> + Send {
        async move {
            namespace(&scope.namespace)?;
            if source_document_id.trim().is_empty() {
                return Err(StoreError::EmptyRecordId);
            }
            let mut state = self.state.write().expect("in-memory store lock poisoned");
            let Some(projection) = state.graph_projections.get_mut(&scope) else {
                return Ok(Vec::new());
            };
            let ids = projection
                .records
                .iter()
                .filter(|((table, id), record)| {
                    table == DOCUMENT_TABLE
                        && (id == &source_document_id
                            || record.metadata.get("doc_id").and_then(Value::as_str)
                                == Some(source_document_id.as_str()))
                })
                .map(|((_, id), _)| id.clone())
                .collect::<Vec<_>>();
            for id in &ids {
                let key = (DOCUMENT_TABLE.to_owned(), id.clone());
                projection.records.remove(&key);
                projection.record_order.retain(|item| item != &key);
            }
            Ok(ids)
        }
    }

    fn document_authority_capability(&self) -> DocumentAuthorityCapability {
        DocumentAuthorityCapability::ProjectionUpsert
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kogwistar_store::GraphRecord;
    use std::future::Future;
    use std::pin::pin;
    use std::task::{Context, Poll, Waker};

    fn block_on<T>(future: impl Future<Output = T>) -> T {
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        let mut future = pin!(future);
        loop {
            match future.as_mut().poll(&mut context) {
                Poll::Ready(value) => return value,
                Poll::Pending => std::thread::yield_now(),
            }
        }
    }

    fn scope(namespace: &str, workspace_id: Option<&str>) -> GraphScope {
        GraphScope {
            namespace: namespace.to_owned(),
            workspace_id: workspace_id.map(str::to_owned),
            graph_space: None,
        }
    }

    fn write(scope: GraphScope, id: &str, source_id: &str) -> DocumentProjectionUpsert {
        DocumentProjectionUpsert {
            scope,
            record: GraphRecord {
                id: id.to_owned(),
                document: Some(format!("source text for {id}")),
                metadata: serde_json::Map::from_iter([("doc_id".to_owned(), source_id.into())]),
                embedding: Some(vec![1.0, 0.0]),
            },
            embedding_dim: 2,
        }
    }

    #[test]
    fn document_projection_is_upserted_looked_up_and_deleted_with_exact_scope() {
        let store = InMemoryStore::new();
        let tenant = scope("docs", Some("tenant-a"));
        let other_tenant = scope("docs", Some("tenant-b"));
        assert_eq!(
            store.document_authority_capability(),
            DocumentAuthorityCapability::ProjectionUpsert
        );
        assert!(DocumentStore::supports_document_projection_delete(&store));
        assert!(!DocumentStore::supports_document_rollback(&store));
        block_on(store.upsert_document_projection(write(tenant.clone(), "chunk-1", "source-1")))
            .expect("upsert source projection");
        block_on(store.upsert_document_projection(write(
            other_tenant.clone(),
            "chunk-1",
            "source-1",
        )))
        .expect("same ID in another scope is independent");

        assert_eq!(
            block_on(store.document_projection_ids(tenant.clone(), vec!["chunk-1".into()]))
                .expect("lookup scoped document"),
            ["chunk-1"]
        );
        assert!(
            block_on(store.delete_document_projection(other_tenant, "chunk-1".into()))
                .expect("delete other scope")
        );
        assert_eq!(
            block_on(store.document_projection_ids(tenant.clone(), vec!["chunk-1".into()]))
                .expect("tenant record remains"),
            ["chunk-1"]
        );
        assert!(
            block_on(store.delete_document_projection(tenant.clone(), "chunk-1".into()))
                .expect("delete matching scope")
        );
        assert!(
            !block_on(store.delete_document_projection(tenant, "chunk-1".into()))
                .expect("repeat delete is idempotent")
        );
    }

    #[test]
    fn source_document_delete_matches_record_id_or_doc_id_only_in_scope() {
        let store = InMemoryStore::new();
        let tenant = scope("docs-source", Some("tenant-a"));
        let isolated = scope("docs-source", Some("tenant-b"));
        for (scoped, id, source) in [
            (tenant.clone(), "source-1", "other-source"),
            (tenant.clone(), "chunk-1", "source-1"),
            (tenant.clone(), "chunk-2", "source-1"),
            (isolated, "foreign-chunk", "source-1"),
        ] {
            block_on(store.upsert_document_projection(write(scoped, id, source)))
                .expect("write source projection");
        }

        let deleted = block_on(
            store.delete_document_projections_by_source_id(tenant.clone(), "source-1".into()),
        )
        .expect("delete source document group");
        assert_eq!(deleted, ["chunk-1", "chunk-2", "source-1"]);
        assert_eq!(
            block_on(store.document_projection_ids(
                tenant,
                vec!["source-1".into(), "chunk-1".into(), "chunk-2".into()],
            ))
            .expect("deleted records absent"),
            Vec::<String>::new()
        );
        assert_eq!(
            block_on(store.document_projection_ids(
                scope("docs-source", Some("tenant-b")),
                vec!["foreign-chunk".into()],
            ))
            .expect("other scope is untouched"),
            ["foreign-chunk"]
        );
    }

    #[test]
    fn invalid_document_projection_does_not_create_scope_state() {
        let store = InMemoryStore::new();
        let mut invalid = write(scope("docs-invalid", None), "bad", "bad");
        invalid.record.embedding = Some(vec![f32::NAN]);
        assert!(matches!(
            block_on(store.upsert_document_projection(invalid)),
            Err(StoreError::NonFiniteVector)
        ));
        assert!(
            block_on(
                store.document_projection_ids(scope("docs-invalid", None), vec!["bad".into()],)
            )
            .expect("invalid write leaves backend usable")
            .is_empty()
        );
    }
}
