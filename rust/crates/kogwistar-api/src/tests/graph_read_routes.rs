    #[test]
    fn graph_traversal_requests_preserve_defaults_and_fail_closed() {
        let shortest: GraphShortestPathRequest =
            serde_json::from_str(r#"{"src_id":"n1","dst_id":"n2"}"#).unwrap();
        assert_eq!(shortest.max_depth, 8);
        assert!(
            serde_json::from_str::<GraphShortestPathRequest>(
                r#"{"src_id":"n1","dst_id":"n2","extra":true}"#
            )
            .is_err()
        );

        let hop: GraphKHopRequest = serde_json::from_str(r#"{"start_ids":["n1"]}"#).unwrap();
        assert_eq!(hop.k, 1);
        assert!(
            serde_json::from_str::<GraphKHopRequest>(r#"{"start_ids":["n1"],"unknown":true}"#)
                .is_err()
        );
    }

    #[tokio::test]
    async fn graph_neighbor_route_authorizes_before_storage_access() {
        let service = PostgresRunApplicationService::from_dsn(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "graph_neighbor_contract",
        )
        .unwrap();
        let denied = service
            .graph_neighbors_effect(&ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/internal/graph/neighbors".to_owned(),
                body: br#"{"rid":"n1"}"#.to_vec(),
                principal: json!({"sub":"alice","capabilities":[]}),
            })
            .await;
        assert_eq!(denied.status, StatusCode::FORBIDDEN.as_u16());

        let malformed = service
            .graph_neighbors_effect(&ApiEffectRequest {
                contract_version: 1,
                method: "POST".to_owned(),
                path_and_query: "/internal/graph/neighbors".to_owned(),
                body: br#"{"rid":"n1","unknown":true}"#.to_vec(),
                principal: json!({"sub":"alice","capabilities":["read_graph"]}),
            })
            .await;
        assert_eq!(malformed.status, StatusCode::UNPROCESSABLE_ENTITY.as_u16());
    }

    #[tokio::test]
    async fn router_exposes_graph_read_effect_routes() {
        let service = Arc::new(
            PostgresRunApplicationService::from_dsn(
                "postgres://unused:unused@127.0.0.1:1/unused",
                "graph_neighbor_router_contract",
            )
            .unwrap(),
        );
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig {
                    key: Some("route-test-secret".to_owned()),
                    ..AuthConfig::default()
                },
            },
            service,
        );
        for (path, body) in [
            ("/internal/graph/neighbors", r#"{"rid":"n1"}"#),
            (
                "/internal/graph/vector-query",
                r#"{"entity_kind":"node","query_embeddings":[[0.1]],"metric":"cosine"}"#,
            ),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(path)
                        .header("content-type", "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();

            // A 403 proves the route reached application authorization; 404
            // would mean the transport seam is missing.
            let status = response.status();
            let body = to_bytes(response.into_body(), 4096).await.unwrap();
            assert_eq!(status, StatusCode::FORBIDDEN, "body={:?}", body);
        }
    }

    #[tokio::test]
    async fn internal_effect_route_catalog_is_registered_in_transport() {
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(UnavailableApplicationService),
        );
        let unique = INTERNAL_APPLICATION_EFFECT_ROUTES
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        assert_eq!(unique.len(), INTERNAL_APPLICATION_EFFECT_ROUTES.len());

        for path in INTERNAL_APPLICATION_EFFECT_ROUTES {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(*path)
                        .header("content-type", "application/json")
                        .body(Body::from("{}"))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_ne!(
                response.status(),
                StatusCode::NOT_FOUND,
                "internal route is missing from Axum transport: {path}"
            );
            assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
        }
    }

    #[tokio::test]
    async fn router_exposes_document_projection_effect_route() {
        let service = Arc::new(
            PostgresRunApplicationService::from_dsn(
                "postgres://unused:unused@127.0.0.1:1/unused",
                "document_projection_router_contract",
            )
            .unwrap(),
        );
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig {
                    key: Some("route-test-secret".to_owned()),
                    ..AuthConfig::default()
                },
            },
            service,
        );
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(INTERNAL_DOCUMENT_PROJECTION_ROUTE)
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"doc_id":"doc-1","content":"text","doc_type":"text"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        let status = response.status();
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        assert_eq!(status, StatusCode::FORBIDDEN, "body={:?}", body);
    }

    #[tokio::test]
    async fn router_exposes_internal_conversation_transcript_route() {
        let service = Arc::new(
            PostgresRunApplicationService::from_dsn(
                "postgres://unused:unused@127.0.0.1:1/unused",
                "conversation_transcript_router_contract",
            )
            .unwrap(),
        );
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig {
                    key: Some("route-test-secret".to_owned()),
                    ..AuthConfig::default()
                },
            },
            service,
        );
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/internal/conversation/transcript")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"conversation_id":"conversation-1"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }
