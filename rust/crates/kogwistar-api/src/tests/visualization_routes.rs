    #[test]
    fn visualization_shells_embed_packaged_templates_without_jinja_drift() {
        for (path, marker) in [
            ("/viz/cytoscape?doc_id=doc-1&mode=graph", "Cytoscape"),
            ("/viz/d3?doc_id=doc-1&mode=graph", "D3 graph viewer"),
            ("/viz/go?doc_id=doc-1&mode=graph", "GoJS"),
        ] {
            let response = visualization_shell_effect(&ApiEffectRequest {
                contract_version: 1,
                method: "GET".to_owned(),
                path_and_query: path.to_owned(),
                body: Vec::new(),
                principal: json!({}),
            })
            .unwrap();
            assert_eq!(response.status, StatusCode::OK.as_u16());
            assert_eq!(response.content_type, "text/html; charset=utf-8");
            let body = String::from_utf8(response.body).unwrap();
            assert!(body.contains(marker));
            assert!(!body.contains("{{"));
            assert!(!body.contains("{%"));
        }
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/response_policy.rs#visualization_shell_effect; python=kogwistar/server_mcp_with_admin.py#viz_cytoscape; fixture=contracts/golden/visualization-cytoscape-shell-http-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/visualization_routes.rs#cytoscape_shell_matches_shared_python_http_contract; python_test=tests/test_visualization.py#test_cytoscape_shell_route_matches_shared_rust_contract
    #[test]
    fn cytoscape_shell_matches_shared_python_http_contract() {
        let contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/visualization-cytoscape-shell-http-v1.json"
        )))
        .expect("parse shared Cytoscape shell contract");
        let response = visualization_shell_effect(&ApiEffectRequest {
            contract_version: 1,
            method: contract["method"].as_str().unwrap().to_owned(),
            path_and_query: contract["path_and_query"].as_str().unwrap().to_owned(),
            body: Vec::new(),
            principal: json!({}),
        })
        .expect("Cytoscape shell route is handled");
        assert_eq!(response.status, contract["success_status"].as_u64().unwrap() as u16);
        assert_eq!(response.content_type, contract["content_type"].as_str().unwrap());
        let body = String::from_utf8(response.body).unwrap();
        for fragment in contract["required_fragments"].as_array().unwrap() {
            assert!(body.contains(fragment.as_str().unwrap()), "missing {fragment}");
        }
        for fragment in contract["forbidden_fragments"].as_array().unwrap() {
            assert!(!body.contains(fragment.as_str().unwrap()), "unexpected {fragment}");
        }
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/response_policy.rs#visualization_shell_effect; python=kogwistar/server_mcp_with_admin.py#viz_d3; fixture=contracts/golden/visualization-d3-shell-http-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/visualization_routes.rs#d3_shell_matches_shared_python_http_contract; python_test=tests/test_visualization.py#test_d3_shell_route_matches_shared_rust_contract
    #[test]
    fn d3_shell_matches_shared_python_http_contract() {
        let contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/visualization-d3-shell-http-v1.json"
        )))
        .expect("parse shared D3 shell contract");
        let response = visualization_shell_effect(&ApiEffectRequest {
            contract_version: 1,
            method: contract["method"].as_str().unwrap().to_owned(),
            path_and_query: contract["path_and_query"].as_str().unwrap().to_owned(),
            body: Vec::new(),
            principal: json!({}),
        })
        .expect("D3 shell route is handled");
        assert_eq!(response.status, contract["success_status"].as_u64().unwrap() as u16);
        assert_eq!(response.content_type, contract["content_type"].as_str().unwrap());
        let body = String::from_utf8(response.body).unwrap();
        for fragment in contract["required_fragments"].as_array().unwrap() {
            assert!(body.contains(fragment.as_str().unwrap()), "missing {fragment}");
        }
        for fragment in contract["forbidden_fragments"].as_array().unwrap() {
            assert!(!body.contains(fragment.as_str().unwrap()), "unexpected {fragment}");
        }
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/response_policy.rs#visualization_shell_effect; python=kogwistar/server_mcp_with_admin.py#viz_go; fixture=contracts/golden/visualization-go-shell-http-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/visualization_routes.rs#go_shell_matches_shared_python_http_contract; python_test=tests/test_visualization.py#test_go_shell_route_matches_shared_rust_contract
    #[tokio::test]
    async fn go_shell_matches_shared_python_http_contract() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use std::future::Future;
        use std::pin::Pin;
        use tower::ServiceExt;

        struct GoShellService;

        impl ApplicationService for GoShellService {
            fn execute(
                &self,
                request: ApiEffectRequest,
            ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>> {
                Box::pin(async move {
                    visualization_shell_effect(&request)
                        .expect("GoJS shell route is implemented")
                })
            }
        }

        let contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/visualization-go-shell-http-v1.json"
        )))
        .expect("parse shared GoJS shell contract");
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(GoShellService),
        );
        let response = app
            .oneshot(
                Request::builder()
                    .method(contract["method"].as_str().unwrap())
                    .uri(contract["path_and_query"].as_str().unwrap())
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            response.status().as_u16(),
            contract["success_status"].as_u64().unwrap() as u16
        );
        assert_eq!(
            response.headers()["content-type"],
            contract["content_type"].as_str().unwrap()
        );
        let body = String::from_utf8(to_bytes(response.into_body(), 1_000_000).await.unwrap().to_vec())
            .unwrap();
        for fragment in contract["required_fragments"].as_array().unwrap() {
            assert!(body.contains(fragment.as_str().unwrap()), "missing {fragment}");
        }
        for fragment in contract["forbidden_fragments"].as_array().unwrap() {
            assert!(!body.contains(fragment.as_str().unwrap()), "unexpected {fragment}");
        }
    }

    // PYTHON_PARITY_V1: rust=rust/crates/kogwistar-api/src/sections/http_transport.rs#application_effect_handler+rust/crates/kogwistar-api/src/sections/response_policy.rs#d3_bundle_html_effect; python=kogwistar/server_mcp_with_admin.py#viz_d3_bundle; fixture=contracts/golden/visualization-d3-bundle-http-v1.json; rust_test=rust/crates/kogwistar-api/src/tests/visualization_routes.rs#d3_bundle_route_matches_shared_python_contract; python_test=tests/test_visualization.py#test_d3_bundle_route_matches_shared_rust_contract_and_escapes_inline_json
    #[tokio::test]
    async fn d3_bundle_route_matches_shared_python_contract() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use std::future::Future;
        use std::pin::Pin;
        use tower::ServiceExt;

        #[derive(Clone)]
        struct BundlePayloadService(Value);

        impl ApplicationService for BundlePayloadService {
            fn execute(
                &self,
                _request: ApiEffectRequest,
            ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>> {
                let payload = self.0.clone();
                Box::pin(async move {
                    ApiEffectResponse {
                        status: StatusCode::OK.as_u16(),
                        content_type: "application/json".to_owned(),
                        body: serde_json::to_vec(&payload).unwrap(),
                    }
                })
            }
        }

        let contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/visualization-d3-bundle-http-v1.json"
        )))
        .expect("parse shared D3 bundle contract");
        let app = router_with_application(
            ApiState {
                health: snapshot(),
                required_roles: Vec::new(),
                implementation: ImplementationSnapshot::default(),
                auth: AuthConfig::default(),
            },
            Arc::new(BundlePayloadService(contract["payload"].clone())),
        );
        let response = app
            .oneshot(
                Request::builder()
                    .method(contract["method"].as_str().unwrap())
                    .uri(contract["path_and_query"].as_str().unwrap())
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            response.status().as_u16(),
            contract["success_status"].as_u64().unwrap() as u16
        );
        assert_eq!(
            response.headers()["content-type"],
            contract["content_type"].as_str().unwrap()
        );
        let body = String::from_utf8(to_bytes(response.into_body(), 1_000_000).await.unwrap().to_vec())
            .unwrap();
        let embedded = body
            .split_once("window.__EMBEDDED_DATA__ = ")
            .and_then(|(_, rest)| rest.split_once(';'))
            .map(|(payload, _)| serde_json::from_str::<Value>(payload).unwrap())
            .expect("bundle contains a JSON graph payload");
        let mut expected_payload = contract["payload"].clone();
        for field in [
            "implementation_authority",
            "seam_attestation_version",
            "seam_crossings",
        ] {
            expected_payload.as_object_mut().unwrap().remove(field);
        }
        assert_eq!(embedded, expected_payload);
        let bundle_meta = body
            .split_once("window.__BUNDLE_META__ = ")
            .and_then(|(_, rest)| rest.split_once(';'))
            .map(|(metadata, _)| serde_json::from_str::<Value>(metadata).unwrap())
            .expect("bundle metadata is JSON");
        assert_eq!(bundle_meta, contract["expected_bundle_meta"]);
        for fragment in contract["forbidden_fragments"].as_array().unwrap() {
            assert!(!body.contains(fragment.as_str().unwrap()));
        }
        assert!(body.contains("\\u003c/script\\u003e"));
    }

    #[test]
    fn d3_bundle_embeds_data_with_script_safe_json_and_python_metadata() {
        let response = d3_bundle_html_effect(
            "/viz/d3.bundle?graph_type=WoRkFlOw&mode=classic&doc_id=doc-1&insertion_method=parser",
            json!({
                "nodes": [{"id":"n1", "label":"</script><script>alert(1)</script>"}],
                "links": []
            }),
        );
        assert_eq!(response.status, StatusCode::OK.as_u16());
        assert_eq!(response.content_type, "text/html; charset=utf-8");
        let body = String::from_utf8(response.body).unwrap();
        assert!(body.contains("bundle</b>=<code>true</code>"));
        assert!(body.contains("window.__BUNDLE_GRAPH_TYPE__ = \"workflow\";"));
        let bundle_meta = body
            .split_once("window.__BUNDLE_META__ = ")
            .and_then(|(_, rest)| rest.split_once(';'))
            .map(|(meta, _)| serde_json::from_str::<Value>(meta).unwrap())
            .expect("bundle metadata is valid JSON");
        assert_eq!(
            bundle_meta,
            json!({
                "graph_type":"workflow",
                "mode":"classic",
                "insertion_method":"parser",
                "doc_id":"doc-1"
            })
        );
        assert!(body.contains("\\u003c/script\\u003e\\u003cscript\\u003e"));
        assert!(!body.contains("</script><script>alert(1)</script>"));
        assert!(!body.contains("{{"));
        assert!(!body.contains("{%"));
    }
