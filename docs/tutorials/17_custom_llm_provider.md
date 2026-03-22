# Tutorial 17: Custom LLM Providers (Task Registry Style)

In this tutorial, we explore how to extend the `graphrag` engine with custom LLM providers. Instead of a monolithic provider interface, the system uses a **task registry style** where individual LLM tasks are defined as pluggable callables.

## What You Will Build

You will build a provider-backed `LLMTaskSet` and see how the engine consumes it without caring about the underlying client library.

## Why This Matters

Splitting provider behavior into task callables keeps the engine deterministic, testable, and easy to swap between model vendors.

## The `LLMTaskSet` Registry

At the heart of the engine's LLM interaction is the `LLMTaskSet`. During initialization, `GraphKnowledgeEngine` sets up `self.llm_tasks`, which is a collection of several core tasks:

- `extract_graph`: Converts text to nodes and edges.
- `adjudicate_pair`: Decides if two entities or relations are equivalent.
- `answer_with_citations`: Generates answers grounded in retrieved graph data.
- ... and others like `summarize_context`, `filter_candidates`, etc.

The engine doesn't care *how* these tasks are implemented, as long as they follow the [LLMTaskSet](file:///c:/Users/chanh/Documents/graphrag_v2_working_tree/kogwistar/llm_tasks/contracts.py) protocol.

---

## Example 1: Vertex AI via REST API

If you want to use a provider without using a library like LangChain, you can implement a task directly using `httpx`.

```python
import httpx
from kogwistar.llm_tasks import (
    AnswerWithCitationsTaskRequest,
    AnswerWithCitationsTaskResult,
)

def vertex_ai_answer_task(request: AnswerWithCitationsTaskRequest) -> AnswerWithCitationsTaskResult:
    # 1. Handle Google Cloud Auth (helper snippet)
    # Use google-auth to get a temporary access token
    import google.auth
    import google.auth.transport.requests
    credentials, project_id = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)
    
    # 2. Prepare the endpoint and headers
    LOCATION = "us-central1"
    MODEL_ID = "gemini-1.5-pro"
    url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:streamGenerateContent"
    headers = {"Authorization": f"Bearer {credentials.token}"}
    
    # 3. Construct the prompt from the request
    prompt = f"{request.system_prompt}\n\nEvidence:\n{request.evidence}\n\nQuestion: {request.question}"
    
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json"
        }
    }
    
    # 4. Call Vertex AI
    response = httpx.post(url, json=payload, headers=headers, timeout=60.0)
    response.raise_for_status()
    
    # 5. Map response back to the Pydantic schema provided in the request
    data = response.json()
    # The first part contains our structured text
    parsed_json = data[0]["candidates"][0]["content"]["parts"][0]["text"]
    
    return AnswerWithCitationsTaskResult(
        answer_payload=request.response_model.model_validate_json(parsed_json).model_dump(),
        raw=data,
        parsing_error=None
    )
```

---

## Example 2: Ollama via LangChain

You can also mix and match. For example, using Ollama with a local model like `qwen3:4b` via the LangChain integration.

```python
from langchain_ollama import ChatOllama
from kogwistar.llm_tasks import (
    SummarizeContextTaskRequest,
    SummarizeContextTaskResult,
)

def ollama_summarize_task(request: SummarizeContextTaskRequest) -> SummarizeContextTaskResult:
    # 1. Initialize the model with local endpoint and larger context
    ollama_model = ChatOllama(
        model="qwen3:4b", 
        temperature=0.1,
        num_ctx=8192,  # Expand context window for large chunks
        base_url="http://localhost:11434"
    )

    # 2. Use LangChain to invoke the model
    # Notice we wrap the request field 'full_text' into the prompt
    prompt = f"Summarize this conversation segment concisely: {request.full_text}"
    response = ollama_model.invoke(prompt)
    
    return SummarizeContextTaskResult(text=str(response.content))
```

---

## Injecting Your Custom Registry

To use your custom tasks, you use `dataclasses.replace` to create a new task set derived from the defaults (or build one from scratch) and pass it to the `GraphKnowledgeEngine` constructor.

```python
from dataclasses import replace
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.llm_tasks import build_default_llm_tasks

# 1. Start with defaults and override specific tasks
custom_tasks = replace(
    build_default_llm_tasks(),
    answer_with_citations=vertex_ai_answer_task,
    summarize_context=ollama_summarize_task
)

# 2. Initialize the engine
engine = GraphKnowledgeEngine(
    persist_directory="./my_graph_data",
    llm_tasks=custom_tasks  # <--- Injected registry
)
```

By providing your own `LLMTaskSet`, you have complete control over every LLM interaction in the GraphRAG pipeline.

## Run or Inspect

Run the companion script or inspect the task registry code to see how provider-specific behavior is injected.

## Inspect The Result

The result should show a custom provider implementation wired into the same engine entry points used by the default tasks.

## Invariant Demonstrated

The engine depends on task contracts, not on a specific provider SDK or client shape.

## Next Tutorial

Move on to the next tutorial in the sequence to continue the provider and orchestration walkthrough.
