#[derive(Clone)]
pub struct ServerState {
    pub api: ApiState,
    pub application: Arc<dyn ApplicationService>,
}

#[derive(Debug, Serialize)]
struct ApiErrorBody {
    code: &'static str,
    message: &'static str,
}

fn roles_from_headers(headers: &HeaderMap) -> Vec<String> {
    headers
        .get("x-kogwistar-roles")
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .collect()
}

fn jwt_algorithm(value: &str) -> Option<Algorithm> {
    match value {
        "HS256" => Some(Algorithm::HS256),
        "RS256" => Some(Algorithm::RS256),
        _ => None,
    }
}

fn jwt_algorithm_name(algorithm: Algorithm) -> &'static str {
    match algorithm {
        Algorithm::HS256 => "HS256",
        Algorithm::RS256 => "RS256",
        _ => "unsupported",
    }
}

fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    let value = headers.get("authorization")?.to_str().ok()?;
    let (scheme, token) = value.split_once(' ')?;
    scheme
        .eq_ignore_ascii_case("bearer")
        .then_some(token.trim())
}

fn jwt_roles(headers: &HeaderMap, config: &AuthConfig) -> Result<Vec<String>, &'static str> {
    let claims = jwt_claims(headers, config)?;
    let mut roles = Vec::new();
    if let Some(role) = claims.get("role").and_then(Value::as_str) {
        roles.push(role.to_ascii_lowercase());
    }
    if let Some(values) = claims.get("roles").and_then(Value::as_array) {
        roles.extend(
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_ascii_lowercase),
        );
    }
    Ok(roles)
}

fn jwt_claims(headers: &HeaderMap, config: &AuthConfig) -> Result<Value, &'static str> {
    let token = bearer_token(headers).ok_or("Missing bearer token")?;
    let configured_algorithm = jwt_algorithm(config.algorithm.as_deref().unwrap_or("HS256"))
        .ok_or("Unsupported JWT algorithm")?;
    let header = decode_header(token).map_err(|_| "Invalid bearer token")?;
    if header.alg != configured_algorithm {
        return Err("JWT algorithm mismatch");
    }
    let key = if let Some(jwks_json) = config.jwks_json.as_deref() {
        let key_id = header.kid.as_deref().ok_or("JWT kid is required")?;
        let set = serde_json::from_str::<jsonwebtoken::jwk::JwkSet>(jwks_json)
            .map_err(|_| "Invalid JWKS JSON")?;
        let jwk = set.find(key_id).ok_or("JWT kid is not trusted")?;
        if jwk.common.key_algorithm.is_some_and(|key_algorithm| {
            key_algorithm.to_string() != jwt_algorithm_name(configured_algorithm)
        }) {
            return Err("JWT JWK algorithm mismatch");
        }
        DecodingKey::from_jwk(jwk).map_err(|_| "Invalid JWK")?
    } else {
        let key_text = config
            .key
            .as_deref()
            .ok_or("JWT verification key is not configured")?;
        match configured_algorithm {
            Algorithm::HS256 => DecodingKey::from_secret(key_text.as_bytes()),
            Algorithm::RS256 => DecodingKey::from_rsa_pem(key_text.as_bytes())
                .map_err(|_| "Invalid RSA public key")?,
            _ => return Err("Unsupported JWT algorithm"),
        }
    };
    let mut validation = Validation::new(configured_algorithm);
    if let Some(issuer) = &config.issuer {
        validation.set_issuer(&[issuer]);
    }
    if let Some(audience) = &config.audience {
        validation.set_audience(&[audience]);
    } else {
        validation.validate_aud = false;
    }
    Ok(decode::<Value>(token, &key, &validation)
        .map_err(|_| "Invalid bearer token")?
        .claims)
}

fn request_roles(headers: &HeaderMap, config: &AuthConfig) -> Result<Vec<String>, &'static str> {
    if config.configured() {
        jwt_roles(headers, config)
    } else {
        Ok(roles_from_headers(headers))
    }
}

fn request_principal(headers: &HeaderMap, config: &AuthConfig) -> Value {
    if config.configured() {
        return jwt_claims(headers, config).unwrap_or_else(|_| json!({}));
    }
    let roles = roles_from_headers(headers);
    let role = if roles.iter().any(|role| role == "rw") {
        "rw"
    } else {
        "ro"
    };
    json!({"sub": "anonymous", "role": role, "roles": roles, "ns": "workflow"})
}

fn has_required_role(headers: &HeaderMap, required_roles: &[String], config: &AuthConfig) -> bool {
    request_roles(headers, config).is_ok_and(|roles| {
        authorize(&ApiAuthRequest {
            roles,
            required_roles: required_roles.to_vec(),
        })
    })
}

fn forbidden_response() -> Response {
    (
        StatusCode::FORBIDDEN,
        Json(ApiErrorBody {
            code: "KOGWISTAR_AUTH_FORBIDDEN",
            message: "required role missing",
        }),
    )
        .into_response()
}

fn auth_error(status: StatusCode, detail: impl Into<String>) -> Response {
    (status, Json(json!({"detail": detail.into()}))).into_response()
}

fn normalized_string_or_list(value: Option<&Value>, default: Value) -> Value {
    match value {
        None | Some(Value::Null) => default,
        Some(Value::String(raw)) if raw.contains(',') => json!(
            raw.split(',')
                .map(str::trim)
                .filter(|item| !item.is_empty())
                .collect::<Vec<_>>()
        ),
        Some(Value::String(raw)) if raw.is_empty() => default,
        Some(value) => value.clone(),
    }
}

#[derive(Clone, Debug, Deserialize)]
struct OidcProviderConfig {
    #[serde(default, rename = "name")]
    _name: String,
    discovery_url: String,
    redirect_uri: String,
    client_id: String,
    #[serde(default)]
    client_secret: String,
    #[serde(default)]
    issuer: Option<String>,
    #[serde(default)]
    scopes: Vec<String>,
    #[serde(default = "default_true")]
    allowed: bool,
    #[serde(default = "default_true")]
    required_email: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
struct OidcProvidersConfig {
    #[serde(default)]
    default_provider: Option<String>,
    #[serde(default)]
    providers: BTreeMap<String, OidcProviderConfig>,
}

#[derive(Debug, Default, Deserialize)]
struct LoginQuery {
    #[serde(default)]
    redirect_uri: Option<String>,
    #[serde(default)]
    provider: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct CallbackQuery {
    #[serde(default)]
    state: Option<String>,
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    error_description: Option<String>,
}

fn request_cookie(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get_all(header::COOKIE)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(';'))
        .filter_map(|part| part.trim().split_once('='))
        .find_map(|(cookie_name, value)| (cookie_name == name).then(|| value.to_owned()))
}

fn delete_login_cookies(response: &mut Response) {
    for name in [
        "auth_state",
        "auth_pkce_verifier",
        "auth_nonce",
        "auth_provider",
    ] {
        let cookie = format!("{name}=; HttpOnly; Max-Age=0; Path=/; SameSite=Lax");
        if let Ok(value) = HeaderValue::from_str(&cookie) {
            response.headers_mut().append(header::SET_COOKIE, value);
        }
    }
}

fn random_token() -> String {
    format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    )
}

fn oidc_providers(config: &AuthConfig) -> Result<OidcProvidersConfig, &'static str> {
    let raw = config
        .oidc_providers_json
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .ok_or("OIDC provider is not configured")?;
    serde_json::from_str(raw).map_err(|_| "OIDC provider configuration is invalid")
}

#[derive(Clone)]
enum AuthStore {
    Sqlite(SqliteAuthStore),
    Postgres(PostgresAuthStore),
}

impl AuthStore {
    async fn from_config(config: &AuthConfig) -> Result<Self, String> {
        let url = config
            .auth_db_url
            .as_deref()
            .unwrap_or("sqlite:///auth.sqlite");
        if let Some(path) = url.strip_prefix("sqlite:///") {
            return SqliteAuthStore::open(path)
                .map(Self::Sqlite)
                .map_err(|error| error.to_string());
        }
        if url.starts_with("postgresql://") || url.starts_with("postgres://") {
            let store = PostgresAuthStore::from_dsn(url).map_err(|error| error.to_string())?;
            store
                .ensure_schema()
                .await
                .map_err(|error| error.to_string())?;
            return Ok(Self::Postgres(store));
        }
        Err("AUTH_DB_URL backend is not supported".to_owned())
    }

    async fn resolve_external_identity(
        &self,
        request: ResolveExternalIdentity,
    ) -> Result<AuthUser, String> {
        match self {
            Self::Sqlite(store) => store.resolve_external_identity(request).await,
            Self::Postgres(store) => store.resolve_external_identity(request).await,
        }
        .map_err(|error| error.to_string())
    }

    async fn auth_user(&self, user_id: &str) -> Result<Option<AuthUser>, String> {
        match self {
            Self::Sqlite(store) => store.auth_user(user_id).await,
            Self::Postgres(store) => store.auth_user(user_id).await,
        }
        .map_err(|error| error.to_string())
    }
}

fn runtime_principal_from_auth_user(user: &AuthUser) -> Value {
    let namespace = user.global_ns.as_ref().map(|value| json!(value));
    json!({
        "user_id": user.user_id,
        "sub": user.email,
        "role": user.global_role.as_deref().unwrap_or("ro"),
        "ns": normalized_string_or_list(namespace.as_ref(), json!("docs")),
    })
}

fn mint_app_token(config: &AuthConfig, user: &AuthUser) -> Result<String, &'static str> {
    let secret = config
        .key
        .as_deref()
        .filter(|value| !value.is_empty())
        .ok_or("JWT secret is not configured")?;
    if config.algorithm.as_deref().unwrap_or("HS256") != "HS256" {
        return Err("App token minting requires JWT_ALG=HS256");
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|value| value.as_secs())
        .unwrap_or_default();
    let mut claims = json!({
        "sub": user.email,
        "user_id": user.user_id,
        "email": user.email,
        "name": user.display_name,
        "role": user.global_role.as_deref().unwrap_or("ro"),
        "ns": normalized_string_or_list(
            user.global_ns.as_ref().map(|value| json!(value)).as_ref(),
            json!("docs"),
        ),
        "iat": now,
        "exp": now + 4 * 60 * 60,
        "iss": config.issuer.as_deref().unwrap_or("local"),
    });
    if let Some(audience) = &config.audience {
        claims["aud"] = json!(audience);
    }
    encode(
        &Header::new(Algorithm::HS256),
        &claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .map_err(|_| "JWT encoding failed")
}

fn redirect_response(location: String) -> Response {
    let mut response = StatusCode::TEMPORARY_REDIRECT.into_response();
    match HeaderValue::from_str(&location) {
        Ok(value) => {
            response.headers_mut().insert(header::LOCATION, value);
            response
        }
        Err(_) => auth_error(StatusCode::INTERNAL_SERVER_ERROR, "Invalid redirect URL"),
    }
}

fn append_login_cookie(response: &mut Response, name: &str, value: &str) {
    let cookie = format!("{name}={value}; HttpOnly; Max-Age=600; Path=/; SameSite=Lax");
    if let Ok(value) = HeaderValue::from_str(&cookie) {
        response.headers_mut().append(header::SET_COOKIE, value);
    }
}
