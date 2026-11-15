#[tokio::test]
async fn mcp_authenticated_call_verifies_once_and_attributes_authorization_stages() {
    use axum::body::{Body, to_bytes};
    use axum::http::{Request, StatusCode, header};
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tower::ServiceExt;

    #[derive(Clone)]
    struct TimedMcpService(Arc<AtomicUsize>);

    impl ApplicationService for TimedMcpService {
        fn supports_mcp_tool_for_principal(
            &self,
            surface: &str,
            name: &str,
            principal: &Value,
        ) -> bool {
            surface == "root"
                && name == "kg_propose_bruteforce"
                && principal["role"] == "rw"
                && principal["capabilities"]
                    .as_array()
                    .is_some_and(|items| items.iter().any(|item| item == "write_graph"))
        }

        fn execute(
            &self,
            _request: ApiEffectRequest,
        ) -> Pin<Box<dyn Future<Output = ApiEffectResponse> + Send + '_>> {
            let calls = Arc::clone(&self.0);
            Box::pin(async move {
                calls.fetch_add(1, Ordering::Relaxed);
                ApiEffectResponse {
                    status: StatusCode::OK.as_u16(),
                    content_type: "application/json".to_owned(),
                    body: br#"{"ok":true}"#.to_vec(),
                }
            })
        }
    }

    fn percentile_ns(values: &mut [u128], percentile: f64) -> u128 {
        values.sort_unstable();
        let index = ((values.len() as f64 * percentile).ceil() as usize)
            .saturating_sub(1)
            .min(values.len() - 1);
        values[index]
    }

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let secret = "mcp-auth-timing-test-secret";
    let auth = AuthConfig {
        mode: "jwt".to_owned(),
        algorithm: Some("HS256".to_owned()),
        key: Some(secret.to_owned()),
        jwks_json: None,
        issuer: None,
        audience: None,
        oidc_providers_json: None,
        auth_db_url: None,
    };
    let token_for_role = |role: &str| {
        encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub": "mcp-auth-timing-user",
                "role": role,
                "ns": ["workflow"],
                "capabilities": ["write_graph"],
                "exp": now + 300,
            }),
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap()
    };
    let calls = Arc::new(AtomicUsize::new(0));
    let sessions = Arc::new(McpSessionRegistry::default());
    let app = router_with_application_and_mcp_sessions(
        ApiState {
            health: snapshot(),
            required_roles: vec!["rw".to_owned()],
            implementation: ImplementationSnapshot::default(),
            auth,
        },
        Arc::new(TimedMcpService(Arc::clone(&calls))),
        sessions,
    );
    let request_body = r#"{
        "jsonrpc":"2.0","id":1,"method":"tools/call",
        "params":{"name":"kg_propose_bruteforce","arguments":{"inp":{}}}
    }"#;
    let authorized_token = token_for_role("rw");
    let mut authentication_ns = Vec::with_capacity(100);
    let mut role_authorization_ns = Vec::with_capacity(100);
    let mut tool_authorization_ns = Vec::with_capacity(100);
    let endpoint_started = Instant::now();

    for _ in 0..100 {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header(header::CONTENT_TYPE, "application/json")
                    .header(
                        header::AUTHORIZATION,
                        format!("Bearer {authorized_token}"),
                    )
                    .body(Body::from(request_body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let timings = response
            .extensions()
            .get::<McpRequestTimings>()
            .expect("test-only stage measurements are attached off-wire")
            .clone();
        assert_eq!(timings.authentication_count, 1);
        assert_eq!(timings.required_role_authorization_count, 1);
        assert_eq!(timings.tool_authorization_ns.len(), 1);
        authentication_ns.push(timings.authentication_ns);
        role_authorization_ns.push(timings.required_role_authorization_ns);
        tool_authorization_ns.push(timings.tool_authorization_ns[0]);
        let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let payload: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["result"]["ok"], true);
    }
    let endpoint_elapsed = endpoint_started.elapsed().as_nanos();

    let unauthorized = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/mcp")
                .header(header::CONTENT_TYPE, "application/json")
                .header(
                    header::AUTHORIZATION,
                    format!("Bearer {}", token_for_role("ro")),
                )
                .body(Body::from(request_body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unauthorized.status(), StatusCode::FORBIDDEN);
    let denied_timings = unauthorized
        .extensions()
        .get::<McpRequestTimings>()
        .expect("denied request retains test-only stage measurements");
    assert_eq!(denied_timings.authentication_count, 1);
    assert_eq!(denied_timings.required_role_authorization_count, 1);
    assert!(denied_timings.tool_authorization_ns.is_empty());
    assert_eq!(calls.load(Ordering::Relaxed), 100);

    println!(
        "mcp_auth_stage_benchmark samples=100 endpoint_mean_us={:.2} authn_p50_p95_ns={}_{} role_authz_p50_p95_ns={}_{} tool_authz_p50_p95_ns={}_{}",
        endpoint_elapsed as f64 / 100.0 / 1000.0,
        percentile_ns(&mut authentication_ns, 0.50),
        percentile_ns(&mut authentication_ns, 0.95),
        percentile_ns(&mut role_authorization_ns, 0.50),
        percentile_ns(&mut role_authorization_ns, 0.95),
        percentile_ns(&mut tool_authorization_ns, 0.50),
        percentile_ns(&mut tool_authorization_ns, 0.95),
    );
}
