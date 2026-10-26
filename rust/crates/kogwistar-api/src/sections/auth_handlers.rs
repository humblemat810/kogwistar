async fn auth_login_handler(
    State(state): State<Arc<ServerState>>,
    Query(query): Query<LoginQuery>,
) -> Response {
    if state.api.auth.mode.eq_ignore_ascii_case("dev") {
        let ui_url = query
            .redirect_uri
            .or_else(|| std::env::var("UI_URL").ok())
            .unwrap_or_else(|| "/".to_owned());
        let Some(secret) = state
            .api
            .auth
            .key
            .as_deref()
            .filter(|value| !value.is_empty())
        else {
            return auth_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "JWT secret is not configured",
            );
        };
        if state.api.auth.algorithm.as_deref().unwrap_or("HS256") != "HS256" {
            return auth_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Dev token minting requires JWT_ALG=HS256",
            );
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|value| value.as_secs())
            .unwrap_or_default();
        let email =
            std::env::var("DEV_AUTH_EMAIL").unwrap_or_else(|_| "dev@example.com".to_owned());
        let mut claims = json!({
            "sub": email,
            "user_id": std::env::var("DEV_AUTH_SUBJECT").unwrap_or_else(|_| "dev".to_owned()),
            "email": email,
            "name": std::env::var("DEV_AUTH_NAME").unwrap_or_else(|_| "Dev User".to_owned()),
            "role": std::env::var("DEV_AUTH_ROLE").unwrap_or_else(|_| "ro".to_owned()),
            "ns": normalized_string_or_list(
                std::env::var("DEV_AUTH_NS").ok().as_ref().map(|value| json!(value)).as_ref(),
                json!(["docs", "conversation", "workflow", "wisdom"]),
            ),
            "iat": now,
            "exp": now + 4 * 60 * 60,
            "iss": state.api.auth.issuer.as_deref().unwrap_or("local"),
        });
        if let Some(audience) = &state.api.auth.audience {
            claims["aud"] = json!(audience);
        }
        return match encode(
            &Header::new(Algorithm::HS256),
            &claims,
            &EncodingKey::from_secret(secret.as_bytes()),
        ) {
            Ok(token) => redirect_response(format!("{ui_url}?token={token}")),
            Err(_) => auth_error(StatusCode::INTERNAL_SERVER_ERROR, "JWT encoding failed"),
        };
    }

    if query.redirect_uri.is_some() {
        return auth_error(
            StatusCode::BAD_REQUEST,
            "redirect_uri override is only allowed when AUTH_MODE=dev",
        );
    }
    let providers = match oidc_providers(&state.api.auth) {
        Ok(providers) => providers,
        Err(message) => return auth_error(StatusCode::BAD_REQUEST, message),
    };
    let provider_name = query.provider.or(providers.default_provider).or_else(|| {
        (providers.providers.len() == 1)
            .then(|| providers.providers.keys().next().cloned())
            .flatten()
    });
    let Some(provider_name) = provider_name else {
        return auth_error(StatusCode::BAD_REQUEST, "OIDC provider is not configured");
    };
    let Some(provider) = providers
        .providers
        .get(&provider_name)
        .filter(|provider| provider.allowed)
    else {
        return auth_error(
            StatusCode::BAD_REQUEST,
            format!("OIDC provider {provider_name:?} is not configured"),
        );
    };
    let discovery = match reqwest::get(&provider.discovery_url).await {
        Ok(response) => match response.error_for_status() {
            Ok(response) => match response.json::<Value>().await {
                Ok(value) => value,
                Err(_) => {
                    return auth_error(
                        StatusCode::BAD_GATEWAY,
                        "OIDC discovery response is invalid",
                    );
                }
            },
            Err(_) => {
                return auth_error(StatusCode::BAD_GATEWAY, "OIDC discovery request failed");
            }
        },
        Err(_) => return auth_error(StatusCode::BAD_GATEWAY, "OIDC discovery request failed"),
    };
    let Some(endpoint) = discovery["authorization_endpoint"].as_str() else {
        return auth_error(
            StatusCode::BAD_GATEWAY,
            "OIDC discovery response is invalid",
        );
    };
    let state_token = random_token();
    let verifier = random_token();
    let nonce = random_token();
    let challenge = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(Sha256::digest(verifier.as_bytes()));
    let mut authorization_url = match url::Url::parse(endpoint) {
        Ok(url) => url,
        Err(_) => {
            return auth_error(
                StatusCode::BAD_GATEWAY,
                "OIDC authorization endpoint is invalid",
            );
        }
    };
    let scopes = if provider.scopes.is_empty() {
        "openid email profile".to_owned()
    } else {
        provider.scopes.join(" ")
    };
    authorization_url.query_pairs_mut().extend_pairs([
        ("client_id", provider.client_id.as_str()),
        ("response_type", "code"),
        ("scope", scopes.as_str()),
        ("redirect_uri", provider.redirect_uri.as_str()),
        ("state", state_token.as_str()),
        ("code_challenge", challenge.as_str()),
        ("code_challenge_method", "S256"),
        ("nonce", nonce.as_str()),
    ]);
    let mut response = redirect_response(authorization_url.into());
    append_login_cookie(&mut response, "auth_state", &state_token);
    append_login_cookie(&mut response, "auth_pkce_verifier", &verifier);
    append_login_cookie(&mut response, "auth_nonce", &nonce);
    append_login_cookie(&mut response, "auth_provider", &provider_name);
    response
}

async fn auth_callback_handler(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    Query(query): Query<CallbackQuery>,
) -> Response {
    if state.api.auth.mode.eq_ignore_ascii_case("dev") {
        return auth_error(
            StatusCode::BAD_REQUEST,
            "OIDC callback disabled (AUTH_MODE=dev)",
        );
    }
    if let Some(error) = query.error {
        return auth_error(
            StatusCode::BAD_REQUEST,
            format!(
                "OIDC Error: {error} - {}",
                query.error_description.unwrap_or_default()
            ),
        );
    }
    let Some(code) = query.code.filter(|value| !value.is_empty()) else {
        return auth_error(StatusCode::BAD_REQUEST, "Missing authorization code");
    };
    let Some(state_value) = query.state else {
        return auth_error(StatusCode::BAD_REQUEST, "Invalid state");
    };
    if request_cookie(&headers, "auth_state").as_deref() != Some(state_value.as_str()) {
        return auth_error(StatusCode::BAD_REQUEST, "Invalid state");
    }
    let Some(verifier) = request_cookie(&headers, "auth_pkce_verifier") else {
        return auth_error(StatusCode::BAD_REQUEST, "Missing PKCE verifier");
    };
    let Some(nonce) = request_cookie(&headers, "auth_nonce") else {
        return auth_error(StatusCode::BAD_REQUEST, "Missing nonce");
    };
    let Some(provider_name) = request_cookie(&headers, "auth_provider") else {
        return auth_error(StatusCode::BAD_REQUEST, "Missing provider");
    };
    let providers = match oidc_providers(&state.api.auth) {
        Ok(providers) => providers,
        Err(message) => return auth_error(StatusCode::BAD_REQUEST, message),
    };
    let Some(provider) = providers
        .providers
        .get(&provider_name)
        .filter(|provider| provider.allowed)
    else {
        return auth_error(
            StatusCode::BAD_REQUEST,
            format!("OIDC provider {provider_name:?} is not configured"),
        );
    };
    let client = reqwest::Client::new();
    let discovery = match client.get(&provider.discovery_url).send().await {
        Ok(response) => match response.error_for_status() {
            Ok(response) => match response.json::<Value>().await {
                Ok(value) => value,
                Err(_) => {
                    return auth_error(
                        StatusCode::BAD_GATEWAY,
                        "OIDC discovery response is invalid",
                    );
                }
            },
            Err(_) => {
                return auth_error(StatusCode::BAD_GATEWAY, "OIDC discovery request failed");
            }
        },
        Err(_) => return auth_error(StatusCode::BAD_GATEWAY, "OIDC discovery request failed"),
    };
    let Some(token_endpoint) = discovery["token_endpoint"].as_str() else {
        return auth_error(
            StatusCode::BAD_GATEWAY,
            "OIDC discovery response is invalid",
        );
    };
    let mut token_form = vec![
        ("client_id", provider.client_id.as_str()),
        ("grant_type", "authorization_code"),
        ("code", code.as_str()),
        ("redirect_uri", provider.redirect_uri.as_str()),
        ("code_verifier", verifier.as_str()),
    ];
    if !provider.client_secret.is_empty() {
        token_form.push(("client_secret", provider.client_secret.as_str()));
    }
    let tokens = match client.post(token_endpoint).form(&token_form).send().await {
        Ok(response) => match response.error_for_status() {
            Ok(response) => match response.json::<Value>().await {
                Ok(value) => value,
                Err(_) => {
                    return auth_error(StatusCode::BAD_GATEWAY, "OIDC token response is invalid");
                }
            },
            Err(_) => return auth_error(StatusCode::BAD_GATEWAY, "OIDC token exchange failed"),
        },
        Err(_) => return auth_error(StatusCode::BAD_GATEWAY, "OIDC token exchange failed"),
    };
    let Some(id_token) = tokens["id_token"].as_str() else {
        return auth_error(StatusCode::BAD_REQUEST, "Missing id_token");
    };
    let Some(access_token) = tokens["access_token"].as_str() else {
        return auth_error(StatusCode::BAD_REQUEST, "Missing access_token");
    };
    let Some(jwks_uri) = discovery["jwks_uri"].as_str() else {
        return auth_error(
            StatusCode::BAD_GATEWAY,
            "OIDC discovery response is invalid",
        );
    };
    let jwks = match client.get(jwks_uri).send().await {
        Ok(response) => match response.error_for_status() {
            Ok(response) => match response.json::<jsonwebtoken::jwk::JwkSet>().await {
                Ok(value) => value,
                Err(_) => return auth_error(StatusCode::BAD_GATEWAY, "OIDC JWKS is invalid"),
            },
            Err(_) => return auth_error(StatusCode::BAD_GATEWAY, "OIDC JWKS request failed"),
        },
        Err(_) => return auth_error(StatusCode::BAD_GATEWAY, "OIDC JWKS request failed"),
    };
    let token_header = match decode_header(id_token) {
        Ok(header) => header,
        Err(_) => return auth_error(StatusCode::UNAUTHORIZED, "Invalid id_token"),
    };
    if token_header.alg != Algorithm::RS256 {
        return auth_error(StatusCode::UNAUTHORIZED, "Invalid id_token algorithm");
    }
    let Some(kid) = token_header.kid.as_deref() else {
        return auth_error(StatusCode::UNAUTHORIZED, "id_token missing kid");
    };
    let Some(jwk) = jwks.find(kid) else {
        return auth_error(StatusCode::UNAUTHORIZED, "id_token kid is not trusted");
    };
    let key = match DecodingKey::from_jwk(jwk) {
        Ok(key) => key,
        Err(_) => return auth_error(StatusCode::UNAUTHORIZED, "Invalid id_token key"),
    };
    let issuer = provider
        .issuer
        .as_deref()
        .or_else(|| discovery["issuer"].as_str());
    let Some(issuer) = issuer else {
        return auth_error(
            StatusCode::BAD_GATEWAY,
            "OIDC discovery response is invalid",
        );
    };
    let mut validation = Validation::new(Algorithm::RS256);
    validation.set_audience(&[provider.client_id.as_str()]);
    validation.set_issuer(&[issuer]);
    validation.set_required_spec_claims(&["exp", "iat", "iss", "aud", "sub"]);
    validation.leeway = 60;
    let claims = match decode::<Value>(id_token, &key, &validation) {
        Ok(token) => token.claims,
        Err(_) => return auth_error(StatusCode::UNAUTHORIZED, "Invalid id_token"),
    };
    if claims["nonce"].as_str() != Some(nonce.as_str()) {
        return auth_error(StatusCode::UNAUTHORIZED, "Invalid nonce");
    }
    if claims["aud"]
        .as_array()
        .is_some_and(|audiences| audiences.len() > 1)
        && claims["azp"].as_str() != Some(provider.client_id.as_str())
    {
        return auth_error(
            StatusCode::UNAUTHORIZED,
            "id_token azp does not match client_id",
        );
    }
    let Some(userinfo_endpoint) = discovery["userinfo_endpoint"].as_str() else {
        return auth_error(
            StatusCode::BAD_GATEWAY,
            "OIDC discovery response is invalid",
        );
    };
    let userinfo = match client
        .get(userinfo_endpoint)
        .bearer_auth(access_token)
        .send()
        .await
    {
        Ok(response) => match response.error_for_status() {
            Ok(response) => match response.json::<Value>().await {
                Ok(value) => value,
                Err(_) => {
                    return auth_error(StatusCode::BAD_GATEWAY, "OIDC userinfo is invalid");
                }
            },
            Err(_) => return auth_error(StatusCode::BAD_GATEWAY, "OIDC userinfo request failed"),
        },
        Err(_) => return auth_error(StatusCode::BAD_GATEWAY, "OIDC userinfo request failed"),
    };
    let Some(subject) = claims["sub"].as_str() else {
        return auth_error(StatusCode::UNAUTHORIZED, "Invalid id_token");
    };
    if userinfo["sub"].as_str() != Some(subject) {
        return auth_error(
            StatusCode::UNAUTHORIZED,
            "userinfo subject does not match validated id_token subject",
        );
    }
    let id_email = claims["email"].as_str();
    let userinfo_email = userinfo["email"].as_str();
    if id_email.is_some() && userinfo_email.is_some() && id_email != userinfo_email {
        return auth_error(
            StatusCode::UNAUTHORIZED,
            "userinfo email does not match validated id_token email",
        );
    }
    let email = id_email.or(userinfo_email);
    if provider.required_email && email.is_none() {
        return auth_error(StatusCode::BAD_REQUEST, "OIDC identity missing email");
    }
    let Some(email) = email else {
        return auth_error(StatusCode::BAD_REQUEST, "OIDC identity missing email");
    };
    let auth_store = match AuthStore::from_config(&state.api.auth).await {
        Ok(store) => store,
        Err(error) => {
            return auth_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Auth store failed: {error}"),
            );
        }
    };
    let user = match auth_store
        .resolve_external_identity(ResolveExternalIdentity {
            issuer: issuer.to_owned(),
            subject: subject.to_owned(),
            email: email.to_owned(),
            display_name: claims["name"]
                .as_str()
                .or_else(|| userinfo["name"].as_str())
                .map(str::to_owned),
            new_user_id: uuid::Uuid::new_v4().to_string(),
            default_role: "ro".to_owned(),
            default_ns: "docs".to_owned(),
        })
        .await
    {
        Ok(user) => user,
        Err(error) => {
            return auth_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Auth store failed: {error}"),
            );
        }
    };
    let token = match mint_app_token(&state.api.auth, &user) {
        Ok(token) => token,
        Err(message) => return auth_error(StatusCode::INTERNAL_SERVER_ERROR, message),
    };
    let ui_url = std::env::var("UI_URL").unwrap_or_else(|_| "/".to_owned());
    let mut response = redirect_response(format!("{ui_url}?token={token}"));
    delete_login_cookies(&mut response);
    response
}

async fn dev_token_handler(
    State(state): State<Arc<ServerState>>,
    Json(input): Json<Value>,
) -> Response {
    if !state.api.auth.mode.eq_ignore_ascii_case("dev") {
        return auth_error(StatusCode::NOT_FOUND, "Dev token endpoint is disabled");
    }
    let Some(secret) = state
        .api
        .auth
        .key
        .as_deref()
        .filter(|value| !value.is_empty())
    else {
        return auth_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "JWT secret is not configured",
        );
    };
    let algorithm = state.api.auth.algorithm.as_deref().unwrap_or("HS256");
    if algorithm != "HS256" {
        return auth_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Dev token minting requires JWT_ALG=HS256",
        );
    }
    let role = input["role"].as_str().unwrap_or("ro");
    if !matches!(role, "ro" | "rw") {
        return auth_error(StatusCode::BAD_REQUEST, "role must be one of ['ro', 'rw']");
    }
    let username = input["username"].as_str().unwrap_or("dev");
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|value| value.as_secs())
        .unwrap_or_default();
    let mut claims = json!({
        "sub": username,
        "ns": normalized_string_or_list(input.get("ns"), json!("docs")),
        "role": role,
        "iat": now,
        "exp": now + 4 * 60 * 60,
        "iss": state.api.auth.issuer.as_deref().unwrap_or("local"),
    });
    if let Some(capabilities) = input.get("capabilities")
        && !capabilities.is_null()
    {
        claims["capabilities"] = normalized_string_or_list(Some(capabilities), Value::Null);
    }
    if let Some(audience) = &state.api.auth.audience {
        claims["aud"] = json!(audience);
    }
    match encode(
        &Header::new(Algorithm::HS256),
        &claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    ) {
        Ok(token) => Json(json!({"token": token})).into_response(),
        Err(_) => auth_error(StatusCode::INTERNAL_SERVER_ERROR, "JWT encoding failed"),
    }
}

async fn auth_me_handler(State(state): State<Arc<ServerState>>, headers: HeaderMap) -> Response {
    if !state.api.auth.configured() {
        return auth_error(StatusCode::UNAUTHORIZED, "Not authenticated");
    }
    let claims = match jwt_claims(&headers, &state.api.auth) {
        Ok(claims) => claims,
        Err(_) => return auth_error(StatusCode::UNAUTHORIZED, "Not authenticated"),
    };
    let user_id = claims["user_id"]
        .as_str()
        .or_else(|| claims["sub"].as_str())
        .unwrap_or_default();
    if user_id.is_empty() {
        return auth_error(StatusCode::UNAUTHORIZED, "Not authenticated");
    }
    Json(json!({
        "user_id": user_id,
        "email": claims["email"].as_str().or_else(|| claims["sub"].as_str()),
        "display_name": claims["name"],
        "is_active": true,
        "role": claims["role"],
        "ns": claims["ns"],
    }))
    .into_response()
}

async fn auth_logout_handler() -> Json<Value> {
    Json(json!({"ok": true}))
}
