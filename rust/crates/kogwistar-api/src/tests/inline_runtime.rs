    #[tokio::test]
    async fn inline_runtime_iteration_uses_typed_hook_not_generic_effect_path() {
        use std::future::Future;
        use std::pin::Pin;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        struct TypedOnlyService {
            typed_calls: Arc<AtomicUsize>,
            generic_effect_calls: Arc<AtomicUsize>,
        }

        impl ApplicationService for TypedOnlyService {
            fn execute_inline_native_work_once(
                &self,
            ) -> Pin<Box<dyn Future<Output = Result<bool, String>> + Send + '_>> {
                let typed_calls = Arc::clone(&self.typed_calls);
                Box::pin(async move {
                    typed_calls.fetch_add(1, Ordering::SeqCst);
                    Ok(true)
                })
            }

            fn execute(
                &self,
                _request: ApiEffectRequest,
            ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>> {
                let generic_effect_calls = Arc::clone(&self.generic_effect_calls);
                Box::pin(async move {
                    generic_effect_calls.fetch_add(1, Ordering::SeqCst);
                    ApiEffectResponse {
                        status: StatusCode::OK.as_u16(),
                        content_type: "application/json".to_owned(),
                        body: b"{}".to_vec(),
                    }
                })
            }
        }

        let service = TypedOnlyService {
            typed_calls: Arc::new(AtomicUsize::new(0)),
            generic_effect_calls: Arc::new(AtomicUsize::new(0)),
        };

        assert!(run_one_inline_native_worker_iteration(&service)
            .await
            .unwrap());
        assert_eq!(service.typed_calls.load(Ordering::SeqCst), 1);
        assert_eq!(service.generic_effect_calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn projection_reconciliation_settings_are_opt_in_and_require_one_namespace() {
        use std::time::Duration;

        assert_eq!(parse_projection_reconcile_interval(None).unwrap(), Duration::ZERO);
        assert_eq!(parse_projection_reconcile_interval(Some("0")).unwrap(), Duration::ZERO);
        assert_eq!(parse_projection_reconcile_interval(Some("60")).unwrap(), Duration::from_secs(60));
        assert!(parse_projection_reconcile_interval(Some("-1")).is_err());
        assert!(parse_projection_reconcile_interval(Some("hourly")).is_err());
        assert_eq!(
            parse_projection_reconcile_namespace(Duration::ZERO, None).unwrap(),
            ""
        );
        assert!(parse_projection_reconcile_namespace(Duration::from_secs(5), None).is_err());
        assert!(parse_projection_reconcile_namespace(
            Duration::from_secs(5),
            Some("  ".to_owned())
        )
        .is_err());
        assert_eq!(
            parse_projection_reconcile_namespace(
                Duration::from_secs(5),
                Some("tenant-a".to_owned())
            )
            .unwrap(),
            "tenant-a"
        );
    }

    #[tokio::test]
    async fn inline_rust_embedding_worker_runs_native_projection_reconciliation() {
        use std::future::Future;
        use std::pin::Pin;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        use std::time::Duration;

        struct ReconcileOnlyService {
            calls: Arc<AtomicUsize>,
            embedding_calls: Arc<AtomicUsize>,
        }

        impl ApplicationService for ReconcileOnlyService {
            fn reconcile_eventual_projection_once(
                &self,
                namespace: String,
                page_size: usize,
            ) -> Pin<Box<dyn Future<Output = Result<usize, String>> + Send + '_>> {
                let calls = Arc::clone(&self.calls);
                Box::pin(async move {
                    assert_eq!(namespace, "tenant-a");
                    assert_eq!(page_size, 100);
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(2)
                })
            }

            fn execute_inline_embedding_work_once(
                &self,
                _limit: usize,
                _lease_seconds: i64,
            ) -> Pin<Box<dyn Future<Output = Result<bool, String>> + Send + '_>> {
                let embedding_calls = Arc::clone(&self.embedding_calls);
                Box::pin(async move {
                    embedding_calls.fetch_add(1, Ordering::SeqCst);
                    Ok(false)
                })
            }

            fn execute(
                &self,
                _request: ApiEffectRequest,
            ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>> {
                Box::pin(async {
                    ApiEffectResponse {
                        status: StatusCode::OK.as_u16(),
                        content_type: "application/json".to_owned(),
                        body: b"{}".to_vec(),
                    }
                })
            }
        }

        let calls = Arc::new(AtomicUsize::new(0));
        let embedding_calls = Arc::new(AtomicUsize::new(0));
        let (shutdown, shutdown_rx) = tokio::sync::oneshot::channel();
        let worker = tokio::spawn(run_inline_embedding_worker(
            Arc::new(ReconcileOnlyService {
                calls: Arc::clone(&calls),
                embedding_calls: Arc::clone(&embedding_calls),
            }),
            shutdown_rx,
            false,
            Duration::from_secs(3600),
            "tenant-a".to_owned(),
        ));
        tokio::time::timeout(Duration::from_secs(2), async {
            while calls.load(Ordering::SeqCst) == 0 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("Rust worker performs its initial reconciliation pass");
        shutdown.send(()).unwrap();
        worker.await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(embedding_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn inline_runtime_supervisor_restarts_worker_after_panic() {
        use std::future::Future;
        use std::pin::Pin;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        use std::time::Duration;

        struct PanicOnceService(Arc<AtomicUsize>);

        impl ApplicationService for PanicOnceService {
            fn execute_inline_native_work_once(
                &self,
            ) -> Pin<Box<dyn Future<Output = Result<bool, String>> + Send + '_>> {
                let calls = Arc::clone(&self.0);
                Box::pin(async move {
                    if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                        panic!("injected native worker panic");
                    }
                    Ok(false)
                })
            }

            fn execute(
                &self,
                _request: ApiEffectRequest,
            ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>> {
                Box::pin(async {
                    ApiEffectResponse {
                        status: StatusCode::OK.as_u16(),
                        content_type: "application/json".to_owned(),
                        body: b"{}".to_vec(),
                    }
                })
            }
        }

        let calls = Arc::new(AtomicUsize::new(0));
        let (shutdown, shutdown_rx) = tokio::sync::oneshot::channel();
        let supervisor = tokio::spawn(supervise_inline_native_worker(
            Arc::new(PanicOnceService(Arc::clone(&calls))),
            shutdown_rx,
        ));

        tokio::time::timeout(Duration::from_secs(5), async {
            while calls.load(Ordering::SeqCst) < 2 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("supervisor should restart the worker after panic");

        shutdown.send(()).expect("supervisor should still be running");
        tokio::time::timeout(Duration::from_secs(2), supervisor)
            .await
            .expect("supervisor should stop on shutdown")
            .expect("supervisor task should complete");
        let calls_after_shutdown = calls.load(Ordering::SeqCst);
        assert!(calls_after_shutdown >= 2);
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(calls.load(Ordering::SeqCst), calls_after_shutdown);
    }
