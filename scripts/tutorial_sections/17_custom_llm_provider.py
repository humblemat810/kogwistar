# %% [markdown]
# # 17 Custom LLM provider
# This tutorial demonstrates how to add a custom LLM provider


# %%
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
from langchain_ollama import ChatOllama
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.llm_tasks import (
    LLMTaskSet,
    build_default_llm_tasks,
    AnswerWithCitationsTaskRequest,
    AnswerWithCitationsTaskResult,
    SummarizeContextTaskRequest,
    SummarizeContextTaskResult,
)

def vertex_ai_answer_task(request: AnswerWithCitationsTaskRequest) -> AnswerWithCitationsTaskResult:
    """Detailed boilerplate for a direct REST API call to Vertex AI Gemini."""
    print(f"Vertex AI task called for: {request.question[:50]}...")
    
    # --- BOILERPLATE START ---
    # 1. Prepare Authorization (e.g., using google-auth)
    # import google.auth
    # import google.auth.transport.requests
    # credentials, project_id = google.auth.default()
    # auth_req = google.auth.transport.requests.Request()
    # credentials.refresh(auth_req)
    # access_token = credentials.token
    
    # 2. Define the endpoint
    PROJECT_ID = "your-project-id"
    LOCATION = "us-central1"
    MODEL_ID = "gemini-1.5-pro"
    url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:streamGenerateContent"
    
    # 3. Construct the Gemini REST payload
    # Note: request.system_prompt and request.evidence are combined into the prompt.
    prompt = (
        f"{request.system_prompt}\n\n"
        f"Context Evidence:\n{request.evidence}\n\n"
        f"User Question: {request.question}"
    )
    
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json"
        }
    }
    
    # 4. Execute the call
    # headers = {"Authorization": f"Bearer {access_token}"}
    # with httpx.Client() as client:
    #     response = client.post(url, json=payload, headers=headers)
    #     response.raise_for_status()
    #     response_data = response.json()
    # --- BOILERPLATE END ---

    # For this tutorial script, we simulate the result mapping
    return AnswerWithCitationsTaskResult(
        answer_payload={"text": "Tutorial answer via Vertex REST", "claims": []},
        raw={"status": "mocked_success", "note": "Use the boilerplate above for real calls"},
        parsing_error=None
    )

def ollama_summarize_task(request: SummarizeContextTaskRequest) -> SummarizeContextTaskResult:
    """Detailed boilerplate for using LangChain's ChatOllama with qwen3:4b."""
    print("Ollama summarize task called!")
    
    # --- BOILERPLATE START ---
    # 1. Initialize the model
    # Note: Requires 'ollama' server running and 'langchain-ollama' package.
    # from langchain_ollama import ChatOllama
    # model = ChatOllama(
    #     model="qwen3:4b", 
    #     temperature=0.1,
    #     num_ctx=8192  # Ensure enough context for the full_text
    # )
    
    # 2. Invoke the model
    # response = model.invoke(f"Please summarize this technical detail: {request.full_text}")
    # summary_text = str(response.content)
    # --- BOILERPLATE END ---
    
    # Mocking for local script verification
    return SummarizeContextTaskResult(text="Tutorial summary via Ollama (qwen3:4b)")

def verify_pluggable_registry():
    # 1. Start with the default task set
    tasks = build_default_llm_tasks()
    
    # 2. Inject custom versions using replace (since LLMTaskSet is frozen)
    custom_tasks = replace(
        tasks,
        answer_with_citations=vertex_ai_answer_task,
        summarize_context=ollama_summarize_task
    )
    
    # 3. Initialize the engine with the custom registry
    engine = GraphKnowledgeEngine(
        persist_directory="./tmp/tutorial_custom_provider",
        llm_tasks=custom_tasks
    )
    
    # 4. Confirm the overrides are active
    assert engine.llm_tasks.answer_with_citations == vertex_ai_answer_task
    assert engine.llm_tasks.summarize_context == ollama_summarize_task
    
    print("Success: Custom task registry (Vertex AI REST + Ollama LangChain) active in engine.")

if __name__ == "__main__":
    verify_pluggable_registry()
