    #[test]
    fn delegated_capabilities_are_attenuated_and_reserved_state_wins() {
        let shared = Arc::new(Mutex::new(CapabilityState::default()));
        let principal = json!({
            "sub": "operator",
            "role": "rw",
            "ns": "workflow",
            "capabilities": ["workflow.run.write", "read_graph"]
        });
        let args = json!({
            "workflow_id": "wf",
            "conversation_id": "conv",
            "initial_state": {
                "_rt_requested_capabilities": ["workflow.run.write"],
                "_rt_effective_capabilities": ["service.manage"]
            }
        });
        let delegated = prepare_delegated_run_args(&args, &principal, &shared).unwrap();
        assert_eq!(
            delegated["initial_state"]["_rt_effective_capabilities"],
            json!(["workflow.run.write"])
        );
        assert_eq!(delegated["initial_state"]["_rt_delegated_by"], "operator");
        assert_eq!(
            delegated["initial_state"]["_rt_delegation"]["schema_version"],
            1
        );
        let widening = json!({
            "workflow_id": "wf",
            "conversation_id": "conv",
            "initial_state": {
                "_rt_requested_capabilities": ["service.manage"]
            }
        });
        assert!(prepare_delegated_run_args(&widening, &principal, &shared).is_err());
    }

    #[test]
    fn jwt_auth_verifies_signature_claims_expiry_and_role_lattice() {
        use axum::http::HeaderValue;
        use jsonwebtoken::{EncodingKey, Header, encode};
        use std::time::{SystemTime, UNIX_EPOCH};

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let config = AuthConfig {
            mode: "oidc".to_owned(),
            algorithm: Some("HS256".to_owned()),
            key: Some("test-secret".to_owned()),
            jwks_json: None,
            issuer: Some("issuer".to_owned()),
            audience: Some("audience".to_owned()),
            oidc_providers_json: None,
            auth_db_url: None,
        };
        let token = encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "sub": "user-1",
                "role": "rw",
                "iss": "issuer",
                "aud": "audience",
                "exp": now + 300,
            }),
            &EncodingKey::from_secret(b"test-secret"),
        )
        .unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
        );
        assert_eq!(jwt_roles(&headers, &config).unwrap(), vec!["rw"]);
        assert!(has_required_role(&headers, &["ro".to_owned()], &config));

        let expired = encode(
            &Header::new(Algorithm::HS256),
            &json!({
                "role": "rw",
                "iss": "issuer",
                "aud": "audience",
                "exp": now - 300,
            }),
            &EncodingKey::from_secret(b"test-secret"),
        )
        .unwrap();
        headers.insert(
            "authorization",
            HeaderValue::from_str(&format!("Bearer {expired}")).unwrap(),
        );
        assert!(jwt_roles(&headers, &config).is_err());

        let mut forged_headers = HeaderMap::new();
        forged_headers.insert("x-kogwistar-roles", HeaderValue::from_static("rw"));
        assert!(!has_required_role(
            &forged_headers,
            &["ro".to_owned()],
            &config
        ));
    }

    #[test]
    fn jwks_kid_selection_supports_rotation_and_rejects_unknown_keys() {
        use axum::http::HeaderValue;
        use jsonwebtoken::{EncodingKey, Header, encode};
        use std::time::{SystemTime, UNIX_EPOCH};

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let claims = json!({"sub": "user", "role": "ro", "exp": now + 300});
        let token = |kid: &str, secret: &[u8]| {
            let mut header = Header::new(Algorithm::HS256);
            header.kid = Some(kid.to_owned());
            encode(&header, &claims, &EncodingKey::from_secret(secret)).unwrap()
        };
        let config = |jwks_json: Value| AuthConfig {
            mode: "oidc".to_owned(),
            algorithm: Some("HS256".to_owned()),
            key: None,
            jwks_json: Some(serde_json::to_string(&jwks_json).unwrap()),
            issuer: None,
            audience: None,
            oidc_providers_json: None,
            auth_db_url: None,
        };
        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            HeaderValue::from_str(&format!("Bearer {}", token("new", b"new-secret"))).unwrap(),
        );
        let rotating = config(json!({
            "keys": [
                {"kty": "oct", "kid": "old", "alg": "HS256", "k": "b2xkLXNlY3JldA"},
                {"kty": "oct", "kid": "new", "alg": "HS256", "k": "bmV3LXNlY3JldA"}
            ]
        }));
        assert_eq!(jwt_roles(&headers, &rotating).unwrap(), vec!["ro"]);

        let old_only = config(json!({
            "keys": [
                {"kty": "oct", "kid": "old", "alg": "HS256", "k": "b2xkLXNlY3JldA"}
            ]
        }));
        assert_eq!(
            jwt_roles(&headers, &old_only),
            Err("JWT kid is not trusted")
        );

        headers.insert(
            "authorization",
            HeaderValue::from_str(&format!("Bearer {}", token("new", b"forged-secret"))).unwrap(),
        );
        assert_eq!(jwt_roles(&headers, &rotating), Err("Invalid bearer token"));
    }

    #[test]
    fn any_capability_gate_accepts_one_grant_and_audits_alternatives() {
        let principal = json!({"sub":"reader","role":"ro"});
        let mut state = CapabilityState::default();
        state.revoked.insert(
            "reader".to_owned(),
            BTreeSet::from(["project_view".to_owned()]),
        );

        assert!(audit_capability_any(
            &principal,
            &mut state,
            "read process table",
            &["project_view", "workflow.run.read"],
        ));
        assert_eq!(state.audit_log[0]["outcome"], "allow");
        assert_eq!(
            state.audit_log[0]["required_any"],
            json!(["project_view", "workflow.run.read"])
        );

        state
            .revoked
            .get_mut("reader")
            .unwrap()
            .insert("workflow.run.read".to_owned());
        assert!(!audit_capability_any(
            &principal,
            &mut state,
            "read process table",
            &["project_view", "workflow.run.read"],
        ));
        assert_eq!(state.audit_log[1]["outcome"], "deny");
    }

    #[tokio::test]
    async fn dev_token_me_and_logout_share_verified_jwt_contract() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use jsonwebtoken::{DecodingKey, EncodingKey, Header, Validation, decode, encode};
        use tower::ServiceExt;

        let auth_path = std::env::temp_dir().join(format!(
            "kogwistar-auth-me-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let app = router(ApiState {
            health: snapshot(),
            required_roles: Vec::new(),
            implementation: ImplementationSnapshot::default(),
            auth: AuthConfig {
                mode: "dev".to_owned(),
                algorithm: Some("HS256".to_owned()),
                key: Some("dev-secret".to_owned()),
                jwks_json: None,
                issuer: Some("local".to_owned()),
                audience: None,
                oidc_providers_json: None,
                auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
            },
        });
        let contract_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../contracts/golden/auth-dev-token-v1.json");
        let contract: Value = serde_json::from_slice(
            &std::fs::read(contract_path).expect("read shared dev token contract"),
        )
        .expect("parse shared dev token contract");
        let token_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/auth/dev-token")
                    .header("content-type", "application/json")
                    .body(Body::from(contract["request"].to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(token_response.status(), StatusCode::OK);
        let token_body = to_bytes(token_response.into_body(), 8192).await.unwrap();
        let token_body = serde_json::from_slice::<Value>(&token_body).unwrap();
        assert_eq!(token_body.as_object().unwrap().keys().collect::<Vec<_>>(), vec!["token"]);
        let token = token_body["token"].as_str().unwrap().to_owned();
        let claims = decode::<Value>(
            &token,
            &DecodingKey::from_secret(b"dev-secret"),
            &Validation::new(Algorithm::HS256),
        )
        .unwrap()
        .claims;
        for (field, expected) in contract["stable_claims"].as_object().unwrap() {
            assert_eq!(&claims[field], expected, "claim {field}");
        }
        assert_eq!(
            claims["exp"].as_u64().unwrap() - claims["iat"].as_u64().unwrap(),
            contract["token_lifetime_seconds"].as_u64().unwrap()
        );
        let oidc_app = router(ApiState {
            health: snapshot(),
            required_roles: Vec::new(),
            implementation: ImplementationSnapshot::default(),
            auth: AuthConfig {
                mode: "oidc".to_owned(),
                algorithm: Some("HS256".to_owned()),
                key: Some("dev-secret".to_owned()),
                jwks_json: None,
                issuer: Some("local".to_owned()),
                audience: None,
                oidc_providers_json: None,
                auth_db_url: None,
            },
        });
        let disabled = oidc_app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/auth/dev-token")
                    .header("content-type", "application/json")
                    .body(Body::from(contract["request"].to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(disabled.status(), StatusCode::NOT_FOUND);
        let me = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/auth/me")
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(me.status(), StatusCode::OK);
        let me_body = to_bytes(me.into_body(), 8192).await.unwrap();
        let me_value = serde_json::from_slice::<Value>(&me_body).unwrap();
        assert_eq!(me_value["email"], "dev@example.com");
        assert_eq!(me_value["is_active"], true);
        assert_eq!(me_value["role"], "rw");
        assert_eq!(me_value["ns"], json!(["docs", "workflow"]));

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let legacy_token = encode(
            &Header::default(),
            &json!({
                "sub": "dev@example.com",
                "email": "dev@example.com",
                "role": "rw",
                "ns": ["docs", "workflow"],
                "iat": now,
                "exp": now + 60,
                "iss": "local",
            }),
            &EncodingKey::from_secret(b"dev-secret"),
        )
        .unwrap();
        let legacy_me = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/auth/me")
                    .header("authorization", format!("Bearer {legacy_token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(legacy_me.status(), StatusCode::OK);

        let auth_db = rusqlite::Connection::open(&auth_path).unwrap();
        auth_db
            .execute(
                "UPDATE users SET is_active = 0 WHERE email = ?1",
                ["dev@example.com"],
            )
            .unwrap();
        drop(auth_db);
        let inactive_me = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/auth/me")
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(inactive_me.status(), StatusCode::OK);
        let inactive_body = to_bytes(inactive_me.into_body(), 8192).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&inactive_body).unwrap()["is_active"],
            false
        );

        let forged = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/auth/me")
                    .header("authorization", "Bearer forged")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(forged.status(), StatusCode::UNAUTHORIZED);
        let logout_contract: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../contracts/golden/auth-logout-v1.json"
        )))
        .unwrap();
        let logout = app
            .oneshot(
                Request::builder()
                    .method(logout_contract["method"].as_str().unwrap())
                    .uri(logout_contract["path"].as_str().unwrap())
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            logout.status(),
            StatusCode::from_u16(logout_contract["expected_status"].as_u64().unwrap() as u16)
                .unwrap()
        );
        let logout_body: Value = serde_json::from_slice(
            &to_bytes(logout.into_body(), 65_536).await.unwrap(),
        )
        .unwrap();
        assert_eq!(logout_body, logout_contract["expected_json"]);
        let _ = std::fs::remove_file(auth_path);
    }

    #[test]
    fn auth_return_targets_match_python_allowlist_and_reject_open_redirects() {
        assert_eq!(
            resolve_return_target(Some("/workspace?tab=1"), None, None).unwrap(),
            "/workspace?tab=1"
        );
        assert_eq!(
            resolve_return_target(
                Some("https://chat.example.local/welcome"),
                Some("https://chat.example.local/welcome"),
                Some("/"),
            )
            .unwrap(),
            "https://chat.example.local/welcome"
        );
        assert_eq!(
            resolve_return_target(
                Some("https://attacker.example/"),
                Some("https://chat.example.local/"),
                Some("/"),
            )
            .unwrap_err(),
            "return_to target is not allowed"
        );
        assert_eq!(
            resolve_return_target(Some("//attacker.example/"), None, None).unwrap_err(),
            "Invalid return_to target"
        );
        assert_eq!(
            resolve_return_target(Some("https://user:secret@example.com/"), Some(""), None)
                .unwrap_err(),
            "Invalid return_to target"
        );
    }

    #[tokio::test]
    async fn oidc_login_preserves_provider_pkce_and_cookie_contract() {
        use axum::body::Body;
        use axum::http::Request;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tower::ServiceExt;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let discovery_task = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 2048];
            let size = stream.read(&mut request).await.unwrap();
            assert!(String::from_utf8_lossy(&request[..size]).starts_with("GET /discovery "));
            let body = format!(r#"{{"authorization_endpoint":"http://{address}/authorize"}}"#);
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
        });
        let app = router(ApiState {
            health: snapshot(),
            required_roles: Vec::new(),
            implementation: ImplementationSnapshot::default(),
            auth: AuthConfig {
                mode: "oidc".to_owned(),
                algorithm: None,
                key: None,
                jwks_json: None,
                issuer: None,
                audience: None,
                oidc_providers_json: Some(
                    json!({
                        "default_provider": "test",
                        "providers": {
                            "test": {
                                "name": "test",
                                "discovery_url": format!("http://{address}/discovery"),
                                "redirect_uri": "http://localhost/api/auth/callback",
                                "client_id": "client-1",
                                "client_secret": "secret",
                                "scopes": ["openid", "email", "profile"],
                                "allowed": true
                            }
                        }
                    })
                    .to_string(),
                ),
                auth_db_url: None,
            },
        });
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/auth/login?return_to=%2Fwelcome%3Ftab%3D1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        discovery_task.await.unwrap();
        assert_eq!(response.status(), StatusCode::TEMPORARY_REDIRECT);
        let location = response.headers()[header::LOCATION].to_str().unwrap();
        let authorization_url = url::Url::parse(location).unwrap();
        assert_eq!(authorization_url.path(), "/authorize");
        let query = authorization_url
            .query_pairs()
            .into_owned()
            .collect::<BTreeMap<_, _>>();
        assert_eq!(query["client_id"], "client-1");
        assert_eq!(query["response_type"], "code");
        assert_eq!(query["scope"], "openid email profile");
        assert_eq!(query["redirect_uri"], "http://localhost/api/auth/callback");
        assert_eq!(query["code_challenge_method"], "S256");
        assert_eq!(query["state"].len(), 64);
        assert_eq!(query["nonce"].len(), 64);
        assert!(!query["code_challenge"].contains('='));
        let cookies = response
            .headers()
            .get_all(header::SET_COOKIE)
            .iter()
            .map(|value| value.to_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(cookies.len(), 5);
        for cookie_name in [
            "auth_state=",
            "auth_pkce_verifier=",
            "auth_nonce=",
            "auth_provider=test",
            "auth_return_to=/welcome?tab=1",
        ] {
            assert!(cookies.iter().any(|cookie| {
                cookie.starts_with(cookie_name)
                    && cookie.contains("HttpOnly")
                    && cookie.contains("Max-Age=600")
            }));
        }
    }

    #[tokio::test]
    async fn oidc_login_rejects_production_redirect_override_before_network_io() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use tower::ServiceExt;

        let app = router(ApiState {
            health: snapshot(),
            required_roles: Vec::new(),
            implementation: ImplementationSnapshot::default(),
            auth: AuthConfig {
                mode: "oidc".to_owned(),
                oidc_providers_json: Some(r#"{"providers":{}}"#.to_owned()),
                auth_db_url: None,
                ..AuthConfig::default()
            },
        });
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/auth/login?redirect_uri=https://attacker.example")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&body).unwrap()["detail"],
            "redirect_uri override is only allowed when AUTH_MODE=dev"
        );
    }

    #[tokio::test]
    async fn oidc_callback_validates_rs256_and_persists_identity_before_minting() {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use std::time::{SystemTime, UNIX_EPOCH};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::time::{Duration, timeout};
        use tower::ServiceExt;

        const PRIVATE_KEY: &str = r#"-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDZEMrHkpSKPPCq
DATB6Q7At9YZVbLq4acttd+yM1ffjpEXfaMfpurxcpHOIoZOO4ddHUDqtDj+cPbH
T1CyNwrSl1jHLe1o+qZRkhPuuKS5ferI9vB0w2vsTynB7rqR45SPO3+ViCJDPmsq
x2qsX/pTXC4a9RRkOTXdEYg7t47jDxim822/AIRu2mSjA7GjGLoDpMTFm1/wTotm
DVb6/hy9iaSf5SO8qd8ROHH1vbRAtY1lcc4Sx/P2+Ngm94QEdKV/tCqQAV5yCPTR
2OcM5aDjyY7gyTY0kd9Gkw2He0o49INcdsGQSeIVE75ITC/vqommk4OgI2gCtd3g
MWDWm4dDAgMBAAECggEAPxjpASNkR1zYjm2o8l8XWUD3HO0y8aD/kkOEj43qNMOB
/KSaRuij8eSeap/Rj6sxOYl35eHWkWvv6Fbve6aRYE77UQbSNMprj1mZrrKAu6TV
G27gzehClnIajtOg6yiO9iXS+/oTD5300/4czZemshWhF1f3gfy5YhYnFkjQ4cJo
E1+5FnB8dgQLSk9clD47zVkFySWoSN1H7OOfwojtxfpgr/wN+/wHuJY425gaFc+7
VmQUg5RGtdciVYmwTV0tQIsqcqeM82/f2e554EzAOck+vX1PKhq3Ll8Z9S0rln57
o1HADCHBzHQ2APrr518BT/jYwsl+kGQPKI+IcmaE2QKBgQDwIkxejcjHvbg140s4
PyJ6yq3bthNF8XaXXMoa14lPQtTy316mn+uoNBwL9YFhEm9ByHBJ1aLugv2ZulMj
sC837+VZye6nl8fCiebGP8Guy4vHFgm4/Lk3HOkbr4Zd2Iwj5A9tlF/iMyjbHkdN
vAUCmkyhyJNpYk26EKijH0CG2QKBgQDnaE7Rx0/mXqhuNjAL9PypuVpqsA/3j3ZL
sJP+vrkEL8DoQGeOEFHGHGJlxmK/lgSmHcov34Zt4nhxa04BgpsIcd6C3OCaSJpX
Cw+KKit1CKPm5Bka2L9XtxaDKHSFH67K9pngc+QPlrEQeXH3xcCKxv0T50SdyZQ+
HSfvEWyFewKBgHx3OqBT2zr0sjN0QXvA9a0xupXENQ8uzeo8lSD+kNQ9bsUIVDYH
dA02HUdxlALtnC87pkAO9KmtyabRteAspPzYYkd87C9/83F5Kt2dFFX2eNfTK2zv
yUywtn68JugjotfDkN+aZWyIWefhNNIs32fu9ENzBD0+T81ebxpFy5tZAoGAG5rb
3DaUl3yvRwZ70NFW2sBbwuJh5Txd9kWIQhlqZM91ib81G0NjHekA6/cwjH5O66oe
Fnvpw24CxDTyx0dXSziaPK4wtPb4Qm31WpwRNxLiyoZnYEZ+/O3AZ8EJtV/EMD4e
uSHaEOn/EWILcG1MvMFkK12pV9FWN9quitxfP8UCgYEAvffs3ep6KgW8zGtQxlWt
emV55w55nDC3OYHQpvv41roViMihNU5EhxF0YNjus744wLxTSyWc29YgkSqZGRwK
NGj8qC7iDj7eHst5dr2KCfToUOTBidV7ynv8RZ5LyChcKzOiEDh/EqlEfGt5xYEI
6Qv6B43kPAqwE3JgGUyb4Wo=
-----END PRIVATE KEY-----"#;
        const JWK: &str = r#"{"kty":"RSA","kid":"test-key","alg":"RS256","use":"sig","n":"2RDKx5KUijzwqgwEwekOwLfWGVWy6uGnLbXfsjNX346RF32jH6bq8XKRziKGTjuHXR1A6rQ4_nD2x09QsjcK0pdYxy3taPqmUZIT7rikuX3qyPbwdMNr7E8pwe66keOUjzt_lYgiQz5rKsdqrF_6U1wuGvUUZDk13RGIO7eO4w8YpvNtvwCEbtpkowOxoxi6A6TExZtf8E6LZg1W-v4cvYmkn-UjvKnfEThx9b20QLWNZXHOEsfz9vjYJveEBHSlf7QqkAFecgj00djnDOWg48mO4Mk2NJHfRpMNh3tKOPSDXHbBkEniFRO-SEwv76qJppODoCNoArXd4DFg1puHQw","e":"AQAB"}"#;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let issuer = format!("http://{address}");
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let mut id_header = Header::new(Algorithm::RS256);
        id_header.kid = Some("test-key".to_owned());
        let id_token = encode(
            &id_header,
            &json!({
                "iss": issuer,
                "aud": "client-1",
                "sub": "subject-1",
                "email": "alice@example.com",
                "name": "Alice",
                "nonce": "nonce-1",
                "iat": now,
                "exp": now + 300,
            }),
            &EncodingKey::from_rsa_pem(PRIVATE_KEY.as_bytes()).unwrap(),
        )
        .unwrap();
        let mock_issuer = issuer.clone();
        let mock = tokio::spawn(async move {
            for _ in 0..4 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut request = vec![0_u8; 8192];
                let size = stream.read(&mut request).await.unwrap();
                let request = String::from_utf8_lossy(&request[..size]);
                let (status, content_type, body) = if request.starts_with("GET /discovery ") {
                    (
                        "200 OK",
                        "application/json",
                        json!({
                            "issuer": mock_issuer,
                            "authorization_endpoint": format!("http://{address}/authorize"),
                            "token_endpoint": format!("http://{address}/token"),
                            "jwks_uri": format!("http://{address}/jwks"),
                            "userinfo_endpoint": format!("http://{address}/userinfo"),
                        })
                        .to_string(),
                    )
                } else if request.starts_with("POST /token ") {
                    assert!(request.contains("code=auth-code"));
                    assert!(request.contains("code_verifier=verifier-1"));
                    (
                        "200 OK",
                        "application/json",
                        json!({"access_token": "access-1", "id_token": id_token}).to_string(),
                    )
                } else if request.starts_with("GET /jwks ") {
                    (
                        "200 OK",
                        "application/json",
                        format!(r#"{{"keys":[{JWK}]}}"#),
                    )
                } else if request.starts_with("GET /userinfo ") {
                    assert!(request.contains("authorization: Bearer access-1"));
                    (
                        "200 OK",
                        "application/json",
                        json!({"sub": "subject-1", "email": "alice@example.com", "name": "Alice"})
                            .to_string(),
                    )
                } else {
                    ("404 Not Found", "text/plain", "not found".to_owned())
                };
                stream
                    .write_all(
                        format!(
                            "HTTP/1.1 {status}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                            body.len()
                        )
                        .as_bytes(),
                    )
                    .await
                    .unwrap();
            }
        });
        let auth_path = std::env::temp_dir().join(format!(
            "kogwistar-oidc-callback-{}.sqlite3",
            uuid::Uuid::new_v4()
        ));
        let config = AuthConfig {
            mode: "oidc".to_owned(),
            algorithm: Some("HS256".to_owned()),
            key: Some("app-secret".to_owned()),
            jwks_json: None,
            issuer: Some("local".to_owned()),
            audience: None,
            oidc_providers_json: Some(
                json!({
                    "default_provider": "test",
                    "providers": {
                        "test": {
                            "name": "test",
                            "discovery_url": format!("http://{address}/discovery"),
                            "redirect_uri": "http://localhost/api/auth/callback",
                            "issuer": issuer,
                            "client_id": "client-1",
                            "client_secret": "secret",
                            "required_email": true,
                            "allowed": true,
                        }
                    }
                })
                .to_string(),
            ),
            auth_db_url: Some(format!("sqlite:///{}", auth_path.display())),
        };
        let app = router(ApiState {
            health: snapshot(),
            required_roles: Vec::new(),
            implementation: ImplementationSnapshot::default(),
            auth: config.clone(),
        });
        let response = timeout(
            Duration::from_secs(10),
            app.oneshot(
                Request::builder()
                    .uri("/api/auth/callback?code=auth-code&state=state-1")
                    .header(
                        header::COOKIE,
                        "auth_state=state-1; auth_pkce_verifier=verifier-1; auth_nonce=nonce-1; auth_provider=test; auth_return_to=/workspace?view=1",
                    )
                    .body(Body::empty())
                    .unwrap(),
            ),
        )
        .await
        .expect("OIDC callback exceeded the local test timeout")
        .unwrap();
        if response.status() != StatusCode::TEMPORARY_REDIRECT {
            let status = response.status();
            let body = to_bytes(response.into_body(), 16_384).await.unwrap();
            panic!(
                "OIDC callback failed with {status}: {}",
                String::from_utf8_lossy(&body)
            );
        }
        let location = response.headers()[header::LOCATION].to_str().unwrap();
        assert!(location.starts_with("/workspace?view=1&token="));
        let token = url::Url::parse(&format!("http://localhost{location}"))
            .unwrap()
            .query_pairs()
            .find_map(|(name, value)| (name == "token").then(|| value.into_owned()))
            .unwrap();
        let mut token_headers = HeaderMap::new();
        token_headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
        );
        let app_claims = jwt_claims(&token_headers, &config).unwrap();
        assert_eq!(app_claims["email"], "alice@example.com");
        assert_eq!(app_claims["role"], "ro");
        assert_eq!(app_claims["ns"], "docs");
        assert_eq!(
            response
                .headers()
                .get_all(header::SET_COOKIE)
                .iter()
                .count(),
            5
        );
        let auth_store = SqliteAuthStore::open(&auth_path).unwrap();
        let identity = auth_store
            .external_identity(&issuer, "subject-1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(identity.email.as_deref(), Some("alice@example.com"));
        assert_eq!(
            auth_store
                .auth_user(&identity.user_id)
                .await
                .unwrap()
                .unwrap()
                .display_name
                .as_deref(),
            Some("Alice")
        );
        drop(auth_store);
        std::fs::remove_file(auth_path).unwrap();
        timeout(Duration::from_secs(2), mock)
            .await
            .expect("OIDC mock issuer did not receive its expected requests")
            .unwrap();
    }
