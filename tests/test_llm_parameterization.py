import pytest
import os
from kogwistar.llm_tasks import LLMTaskSet

pytestmark = pytest.mark.ci

def test_llm_tasks_fixture_exists(llm_tasks):
    """Verify that the llm_tasks fixture is available and returns the correct type."""
    assert isinstance(llm_tasks, LLMTaskSet)
    assert callable(llm_tasks.extract_graph)
    assert callable(llm_tasks.adjudicate_pair)

def test_llm_provider_name_fixture(llm_provider_name):
    """Verify that the llm_provider_name fixture reflects the CLI option."""
    # Default is gemini
    assert llm_provider_name in ["gemini", "openai", "ollama"]

def test_llm_cache_dir_fixture(llm_cache_dir):
    """Verify that the llm_cache_dir fixture reflects the CLI option."""
    assert os.path.exists(llm_cache_dir)
    assert ".cache" in llm_cache_dir

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY") and not os.getenv("GOOGLE_API_KEY"), 
                    reason="Needs API keys for real provider check (or just testing the fixture wrapper)")
def test_llm_tasks_caching(llm_tasks, llm_cache_dir):
    """
    Verify that llm_tasks are wrapped with caching.
    Note: We don't actually need to call the LLM to verify joblib wrapper existence, 
    but it's a good sanity check.
    """
    # Check if the 'extract_graph' function is a joblib-decorated function
    # joblib.Memory.cache decorated functions have a 'call' attribute or similar depending on version.
    # In recent joblib, it's a 'MemorizedFunc'.
    assert hasattr(llm_tasks.extract_graph, "__wrapped__") or "MemorizedFunc" in str(type(llm_tasks.extract_graph))
