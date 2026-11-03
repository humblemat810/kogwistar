#[derive(Clone)]
pub struct ServerState {
    pub api: ApiState,
    pub application: Arc<dyn ApplicationService>,
    pub mcp_sessions: Arc<McpSessionRegistry>,
}

const MCP_SESSION_TTL: Duration = Duration::from_secs(30 * 60);
const MCP_MAX_SESSIONS: usize = 10_000;
const MCP_OUTBOUND_EVENT_CAPACITY: usize = 4;
const MCP_OUTBOUND_HISTORY_MAX_EVENTS: usize = 64;
const MCP_OUTBOUND_HISTORY_MAX_BYTES: usize = 16 * 1024;
const MCP_OUTBOUND_MESSAGE_MAX_BYTES: usize = 4 * 1024;

#[derive(Clone)]
pub(crate) struct McpOutboundEvent {
    sequence: u64,
    message: Value,
    serialized_size: usize,
}

#[derive(Clone)]
struct McpSession {
    owner: Value,
    protocol_version: String,
    last_seen: Instant,
    outbound: tokio::sync::broadcast::Sender<McpOutboundEvent>,
    outbound_history: std::collections::VecDeque<McpOutboundEvent>,
    outbound_history_bytes: usize,
    next_outbound_sequence: u64,
    get_stream_active: bool,
}

#[derive(Default)]
pub struct McpSessionRegistry {
    sessions: Mutex<HashMap<String, McpSession>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum McpSessionError {
    NotFound,
    ProtocolVersionMismatch,
    GetStreamAlreadyActive,
    InvalidLastEventId,
    EventHistoryUnavailable,
    InvalidOutboundMessage,
    OutboundMessageTooLarge,
}

pub(crate) struct McpGetSubscription {
    pub replay: Vec<McpOutboundEvent>,
    pub receiver: tokio::sync::broadcast::Receiver<McpOutboundEvent>,
}

pub(crate) struct McpGetStreamGuard {
    registry: Arc<McpSessionRegistry>,
    session_id: String,
}

impl McpGetStreamGuard {
    pub fn new(registry: Arc<McpSessionRegistry>, session_id: String) -> Self {
        Self {
            registry,
            session_id,
        }
    }
}

impl Drop for McpGetStreamGuard {
    fn drop(&mut self) {
        self.registry.finish_get(&self.session_id);
    }
}

impl McpOutboundEvent {
    pub fn event_id(&self) -> String {
        self.sequence.to_string()
    }

    pub fn message(&self) -> &Value {
        &self.message
    }
}

impl McpSessionRegistry {
    pub fn open(&self, owner: Value, protocol_version: String) -> Option<String> {
        let now = Instant::now();
        let mut sessions = self.sessions.lock().unwrap_or_else(|error| error.into_inner());
        sessions.retain(|_, session| {
            session.get_stream_active || now.duration_since(session.last_seen) < MCP_SESSION_TTL
        });
        if sessions.len() >= MCP_MAX_SESSIONS {
            return None;
        }
        let session_id = uuid::Uuid::new_v4().simple().to_string();
        let (outbound, _) = tokio::sync::broadcast::channel(MCP_OUTBOUND_EVENT_CAPACITY);
        sessions.insert(
            session_id.clone(),
            McpSession {
                owner,
                protocol_version,
                last_seen: now,
                outbound,
                outbound_history: std::collections::VecDeque::new(),
                outbound_history_bytes: 0,
                next_outbound_sequence: 1,
                get_stream_active: false,
            },
        );
        Some(session_id)
    }

    pub fn touch(
        &self,
        session_id: &str,
        owner: &Value,
        protocol_version: Option<&str>,
    ) -> Result<String, McpSessionError> {
        let now = Instant::now();
        let mut sessions = self.sessions.lock().unwrap_or_else(|error| error.into_inner());
        let Some(session) = sessions.get_mut(session_id) else {
            return Err(McpSessionError::NotFound);
        };
        if now.duration_since(session.last_seen) >= MCP_SESSION_TTL && !session.get_stream_active {
            sessions.remove(session_id);
            return Err(McpSessionError::NotFound);
        }
        if &session.owner != owner {
            return Err(McpSessionError::NotFound);
        }
        if protocol_version.is_some_and(|version| version != session.protocol_version) {
            return Err(McpSessionError::ProtocolVersionMismatch);
        }
        session.last_seen = now;
        Ok(session.protocol_version.clone())
    }

    pub fn publish(
        &self,
        session_id: &str,
        owner: &Value,
        protocol_version: Option<&str>,
        message: Value,
    ) -> Result<String, McpSessionError> {
        let serialized = serde_json::to_vec(&message)
            .map_err(|_| McpSessionError::InvalidOutboundMessage)?;
        if serialized.len() > MCP_OUTBOUND_MESSAGE_MAX_BYTES {
            return Err(McpSessionError::OutboundMessageTooLarge);
        }
        if message.get("jsonrpc").and_then(Value::as_str) != Some("2.0")
            || message.get("method").and_then(Value::as_str).is_none()
            || message.get("result").is_some()
            || message.get("error").is_some()
        {
            return Err(McpSessionError::InvalidOutboundMessage);
        }

        let now = Instant::now();
        let mut sessions = self.sessions.lock().unwrap_or_else(|error| error.into_inner());
        let Some(session) = sessions.get_mut(session_id) else {
            return Err(McpSessionError::NotFound);
        };
        if now.duration_since(session.last_seen) >= MCP_SESSION_TTL && !session.get_stream_active {
            sessions.remove(session_id);
            return Err(McpSessionError::NotFound);
        }
        if &session.owner != owner {
            return Err(McpSessionError::NotFound);
        }
        if protocol_version.is_some_and(|version| version != session.protocol_version) {
            return Err(McpSessionError::ProtocolVersionMismatch);
        }
        let event = McpOutboundEvent {
            sequence: session.next_outbound_sequence,
            message,
            serialized_size: serialized.len(),
        };
        session.next_outbound_sequence = session
            .next_outbound_sequence
            .checked_add(1)
            .ok_or(McpSessionError::InvalidOutboundMessage)?;
        session.outbound_history.push_back(event.clone());
        session.outbound_history_bytes += event.serialized_size;
        while session.outbound_history.len() > MCP_OUTBOUND_HISTORY_MAX_EVENTS
            || session.outbound_history_bytes > MCP_OUTBOUND_HISTORY_MAX_BYTES
        {
            if let Some(evicted) = session.outbound_history.pop_front() {
                session.outbound_history_bytes -= evicted.serialized_size;
            }
        }
        session.last_seen = now;
        let event_id = event.event_id();
        let _ = session.outbound.send(event);
        Ok(event_id)
    }

    pub(crate) fn subscribe_get(
        &self,
        session_id: &str,
        owner: &Value,
        protocol_version: Option<&str>,
        last_event_id: Option<&str>,
    ) -> Result<McpGetSubscription, McpSessionError> {
        let last_sequence = last_event_id
            .map(|value| value.parse::<u64>().map_err(|_| McpSessionError::InvalidLastEventId))
            .transpose()?;
        let now = Instant::now();
        let mut sessions = self.sessions.lock().unwrap_or_else(|error| error.into_inner());
        let Some(session) = sessions.get_mut(session_id) else {
            return Err(McpSessionError::NotFound);
        };
        if now.duration_since(session.last_seen) >= MCP_SESSION_TTL && !session.get_stream_active {
            sessions.remove(session_id);
            return Err(McpSessionError::NotFound);
        }
        if &session.owner != owner {
            return Err(McpSessionError::NotFound);
        }
        if protocol_version.is_some_and(|version| version != session.protocol_version) {
            return Err(McpSessionError::ProtocolVersionMismatch);
        }
        if session.get_stream_active {
            return Err(McpSessionError::GetStreamAlreadyActive);
        }
        if let Some(last_sequence) = last_sequence {
            if last_sequence >= session.next_outbound_sequence {
                return Err(McpSessionError::InvalidLastEventId);
            }
            if session
                .outbound_history
                .front()
                .is_some_and(|first| last_sequence.saturating_add(1) < first.sequence)
            {
                return Err(McpSessionError::EventHistoryUnavailable);
            }
        }
        let receiver = session.outbound.subscribe();
        let replay = session
            .outbound_history
            .iter()
            .filter(|event| last_sequence.is_some_and(|last| event.sequence > last))
            .cloned()
            .collect();
        session.get_stream_active = true;
        session.last_seen = now;
        Ok(McpGetSubscription { replay, receiver })
    }

    pub fn finish_get(&self, session_id: &str) {
        if let Some(session) = self
            .sessions
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get_mut(session_id)
        {
            session.get_stream_active = false;
            session.last_seen = Instant::now();
        }
    }

    #[cfg(test)]
    pub(crate) fn has_active_get(&self, session_id: &str) -> bool {
        self.sessions
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get(session_id)
            .is_some_and(|session| session.get_stream_active)
    }

    pub fn close(
        &self,
        session_id: &str,
        owner: &Value,
        protocol_version: Option<&str>,
    ) -> Result<(), McpSessionError> {
        let now = Instant::now();
        let mut sessions = self.sessions.lock().unwrap_or_else(|error| error.into_inner());
        let Some(session) = sessions.get(session_id) else {
            return Err(McpSessionError::NotFound);
        };
        if now.duration_since(session.last_seen) >= MCP_SESSION_TTL && !session.get_stream_active {
            sessions.remove(session_id);
            return Err(McpSessionError::NotFound);
        }
        if &session.owner != owner {
            return Err(McpSessionError::NotFound);
        }
        if protocol_version.is_some_and(|version| version != session.protocol_version) {
            return Err(McpSessionError::ProtocolVersionMismatch);
        }
        sessions.remove(session_id);
        Ok(())
    }
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

fn request_mcp_session_owner(headers: &HeaderMap, config: &AuthConfig) -> Value {
    let principal = request_principal(headers, config);
    if config.configured() {
        json!({
            "client_id": principal.get("client_id").or_else(|| principal.get("azp")),
            "issuer": principal.get("iss"),
            "subject": principal.get("sub"),
        })
    } else {
        principal
    }
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
    return_to: Option<String>,
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
        "auth_return_to",
    ] {
        let cookie = format!("{name}=; HttpOnly; Max-Age=0; Path=/; SameSite=Lax");
        if let Ok(value) = HeaderValue::from_str(&cookie) {
            response.headers_mut().append(header::SET_COOKIE, value);
        }
    }
}

fn resolve_return_target(
    value: Option<&str>,
    allowed_targets: Option<&str>,
    ui_url: Option<&str>,
) -> Result<String, &'static str> {
    let default_target = ui_url
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("/");
    let target = value.unwrap_or(default_target).trim();
    if target.starts_with('/') && !target.starts_with("//") {
        return Ok(target.to_owned());
    }
    let parsed = url::Url::parse(target).map_err(|_| "Invalid return_to target")?;
    if parsed.host_str().is_none() || !parsed.username().is_empty() || parsed.password().is_some() {
        return Err("Invalid return_to target");
    }
    let configured = allowed_targets
        .filter(|value| !value.trim().is_empty())
        .map(|value| value.split(',').map(str::trim).collect::<Vec<_>>())
        .unwrap_or_else(|| vec![default_target]);
    if configured.contains(&target) {
        Ok(target.to_owned())
    } else {
        Err("return_to target is not allowed")
    }
}

fn configured_return_target(value: Option<&str>) -> Result<String, &'static str> {
    resolve_return_target(
        value,
        std::env::var("AUTH_ALLOWED_RETURN_URLS").ok().as_deref(),
        std::env::var("UI_URL").ok().as_deref(),
    )
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
pub(crate) enum AuthStore {
    Sqlite(SqliteAuthStore),
    Postgres(PostgresAuthStore),
}

impl AuthStore {
    pub(crate) async fn from_config(config: &AuthConfig) -> Result<Self, String> {
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

    pub(crate) async fn auth_user(&self, user_id: &str) -> Result<Option<AuthUser>, String> {
        match self {
            Self::Sqlite(store) => store.auth_user(user_id).await,
            Self::Postgres(store) => store.auth_user(user_id).await,
        }
        .map_err(|error| error.to_string())
    }

    async fn workflow_acl_role(
        &self,
        workflow_id: &str,
        user_id: &str,
    ) -> Result<Option<String>, String> {
        match self {
            Self::Sqlite(store) => store.workflow_acl_role(workflow_id, user_id).await,
            Self::Postgres(store) => store.workflow_acl_role(workflow_id, user_id).await,
        }
        .map_err(|error| error.to_string())
    }

    async fn workflow_access_allowed(
        &self,
        workflow_id: &str,
        user_id: &str,
        required_role: &str,
    ) -> Result<bool, String> {
        let Some(role) = self.workflow_acl_role(workflow_id, user_id).await? else {
            return Ok(false);
        };
        Ok(match (role.as_str(), required_role) {
            ("rw", "ro" | "rw") | ("ro", "ro") => true,
            _ => false,
        })
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
