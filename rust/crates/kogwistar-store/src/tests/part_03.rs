// Document-store capability defaults. Python parity: backend capability
// selection before `RollbackSubsystem.rollback_document` dispatch.

#[derive(Default)]
struct UnsupportedDocumentStore;

impl DocumentStore for UnsupportedDocumentStore {}

fn block_on<T>(future: impl std::future::Future<Output = T>) -> T {
    let waker = std::task::Waker::noop();
    let mut context = std::task::Context::from_waker(waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(value) => return value,
            std::task::Poll::Pending => std::thread::yield_now(),
        }
    }
}

#[test]
fn document_rollback_default_fails_closed_without_claiming_support() {
    let store = UnsupportedDocumentStore;
    assert!(!DocumentStore::supports_document_rollback(&store));

    let result = block_on(DocumentStore::rollback_document(
        &store,
        DocumentRollbackRequest {
            scope: GraphScope {
                namespace: "docs".to_owned(),
                workspace_id: None,
                graph_space: None,
            },
            document_id: "source-a".to_owned(),
            operation_id: "rollback-a".to_owned(),
            deleted_at: "2026-11-16T00:00:00Z".to_owned(),
            page_size: 32,
        },
    ));

    assert!(matches!(
        result,
        Err(StoreError::Backend { backend, message })
            if backend == "document" && message.contains("not supported")
    ));
}
