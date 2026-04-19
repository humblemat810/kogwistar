# Slice 3 Memory Visibility

```text
user
  |
  v
agent identity
  |
  +-- owner_agent_id / agent_id
  +-- security_scope
  |
  v
conversation artifact
  |
  +-- memory_context
  |     visibility = private | shared
  |     owner_agent_id
  |     shared_with_agents
  |     security_scope
  |
  +-- pinned_kg_ref
        visibility = private | shared
        owner_agent_id
        shared_with_agents
        security_scope

read path
  |
  v
claims_ctx -> current_agent_id + current_security_scope
  |
  v
policy filter
  |
  +-- private: owner_agent_id must match
  +-- shared : explicit agent/scope grant ok
  |
  v
ContextSources / ConversationService view
  |
  +-- memory_context visible only if pass policy
  +-- pinned_kg_ref visible only if pass policy
```

Rules:
- private = agent-private first
- shared = explicit grant
- scope alone no enough if owner agent exist
- backend no new truth; only metadata + filter
