from logging import Logger

from typing import Any

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.outputs.chat_generation import ChatGeneration
    from langchain_core.outputs.llm_result import LLMResult
except ModuleNotFoundError:  # pragma: no cover - optional dependency

    class BaseCallbackHandler:
        def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
            _ = (args, kwargs)

        def on_llm_end(self, *args: Any, **kwargs: Any) -> None:
            _ = (args, kwargs)

    class ChatGeneration:
        pass

    class LLMResult:
        generations: list[list[Any]]


GEMINI_PRO_INPUT_COST_PER_1K_TOKENS = 0.0001
GEMINI_PRO_OUTPUT_COST_PER_1K_TOKENS = 0.0004

# per_k
# storage per hour
cost_table = {
    "gemini-2.0-flash": {
        "input": 0.0001,
        "output": 0.0004,
        "cache": 0.0,
        "storage_per_hour": 0.0,
    },
    "gemini-1.5-pro": {
        "input": 0.001250,
        "output": 0.005,
        "cache": 0.0,
        "storage_per_hour": 0.0,
    },
    "gemini-2.5-flash-preview-04-17": {
        "input": 0.000150,
        "output": 0.0035,
        "cache": 0.0000375,
        "storage_per_hour": 0.0010,
    },
    "gemini-2.5-flash": {
        "input": 0.000300,
        "output": 0.0025,
        "cache": 0.0000375,
        "storage_per_hour": 0.0010,
    },
    "gemini-2.5-pro-preview-03-25": {
        "input": 0.001250,
        "output": 0.0100,
        "cache": 0.00031,
        "storage_per_hour": 0.0045,
    },
    "gemini-2.5-pro": {
        "input": 0.001250,
        "output": 0.0100,
        "cache": 0.00031,
        "storage_per_hour": 0.0045,
    },
    "gemini-2.5-flash-lite": {
        "input": 0.0001,
        "output": 0.0040,
        "cache": 0.00025,
        "storage_per_hour": 0.001,
    },
}
keys = list(cost_table.keys())
for k in keys:
    if k.startswith("models/"):
        pass
    else:
        cost_table["models/" + k] = cost_table[k]


def calculate_gemini_cost(input_tokens, output_tokens, cached_tokens, model_name):
    """Calculates the cost based on Gemini Pro pricing."""
    cost = cost_table.get(
        model_name,
        {
            "input": 1.250 / 1000000,
            "output": 5.0 / 1000000,
            "cache": 0.0,
            "storage_per_hour": 0.0,
        },
    )  # prodential to assume high priced model
    input_cost = ((input_tokens - cached_tokens) / 1000) * cost["input"]
    cache_cost = cached_tokens / 1000 * cost["cache"]
    output_cost = (output_tokens / 1000) * cost["output"]
    total_cost = input_cost + output_cost + cache_cost
    return total_cost


import time


class GeminiCostCallbackHandler(BaseCallbackHandler):
    """A custom callback handler to track Gemini API costs."""

    def __init__(self):
        super().__init__()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.cache_tokens = 0
        self.reasoning_tokens = 0
        self.total_cost = 0.0
        self.usage_history = []
        self.run_start_time = None
        self.run_end_time = None

    def on_llm_start(
        self,
        serialized,
        prompts,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ):
        self.run_start_time = time.time()
        return super().on_llm_start(
            serialized,
            prompts,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called at the end of an LLM call."""
        self.run_end_time = time.time()
        for generation in response.generations:
            # The 'generation' is a list of ChatGeneration or Generation objects
            for gen in generation:
                # Check if the generation object is a ChatGeneration instance
                # and has the 'usage_metadata' attribute.
                if isinstance(gen, ChatGeneration) and hasattr(gen, "message"):
                    message = gen.message
                    if hasattr(message, "usage_metadata"):
                        usage_metadata = message.usage_metadata
                        if usage_metadata is not None:
                            input_tokens = usage_metadata.get("input_tokens", 0)
                            output_tokens = usage_metadata.get("output_tokens", 0)

                            try:
                                cached_tokens = usage_metadata["input_token_details"][
                                    "cache_read"
                                ]
                            except KeyError:
                                cached_tokens = 0
                            output_tokens = usage_metadata.get("output_tokens", 0)
                            try:
                                reasoning_tokens = usage_metadata[
                                    "output_token_details"
                                ]["reasoning"]
                            except KeyError:
                                reasoning_tokens = 0

                            if input_tokens > 0 or output_tokens > 0:
                                cost = calculate_gemini_cost(
                                    input_tokens,
                                    output_tokens,
                                    cached_tokens,
                                    model_name=gen.generation_info["model_name"],
                                )
                                self.total_input_tokens += input_tokens
                                self.total_output_tokens += output_tokens
                                self.reasoning_tokens += reasoning_tokens
                                self.cache_tokens += cached_tokens
                                self.total_cost += cost
                                self.usage_history.append(
                                    {
                                        "model_name": gen.generation_info["model_name"],
                                        "input_tokens": input_tokens,
                                        "output_tokens": output_tokens,
                                        "cached_tokens": cached_tokens,
                                        "reasoning_tokens": reasoning_tokens,
                                        "cost": cost,
                                        "start_time": self.run_start_time,
                                        "end_time": self.run_end_time,
                                    }
                                )

    def reset(self):
        """Resets the counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

    def __repr__(self):
        return (
            f"Total Input Tokens: {self.total_input_tokens}\n"
            f"Total Output Tokens: {self.total_output_tokens}\n"
            f"Total Cost: ${self.total_cost:.8f}"
        )

    def model_dump(self):
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_cost": self.total_cost,
            "usage_history": self.usage_history,
        }


from contextlib import contextmanager


@contextmanager
def get_gemini_callback_cost():
    """A context manager to track Gemini API costs for a block of code."""
    # Create an instance of the handler
    """_summary_

    Yields:
        _type_: _description_
        
    _usage_
    with get_gemini_callback_cost() as cb:
        result = chain.invoke(
            {"city": "Paris"},
            config={"callbacks": [cb]} # Pass the yielded handler to the chain
        )
        print("\n--- Inside Context Manager ---")
        print(result.content)

        # You can access the cost immediately after the call
        print("\n--- Cost After First Call ---")
        print(cb)

    # The state is preserved even after the `with` block exits
    print("\n--- Final Cost from Context Manager ---")
    print(cb)
        
    """
    callback_handler = GeminiCostCallbackHandler()
    try:
        # Yield the handler so it can be used inside the 'with' block
        # and its state can be accessed after the block.
        yield callback_handler
    finally:
        # The code inside the 'with' block has finished.
        # The handler now holds the final cost.
        pass


class PromptCostTokenLogger(BaseCallbackHandler):
    def __init__(self, logger: Logger):
        self.cost_token_logger: Logger = logger
        self.total_input_tokens = 0
        self.total_reasoning_tokens = 0
        self.total_cached_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        # self.cost_token_logger.info(response.response_metadata)
        for g in response.generations:
            for gg in g:
                to_log = {"response_metadata": None, "usage_metadata": None}
                if hasattr(gg.message, "response_metadata"):
                    to_log["response_metadata"] = f"{gg.message.response_metadata}"
                if hasattr(gg.message, "usage_metadata"):
                    to_log["usage_metadata"] = f"{gg.message.usage_metadata}"
                if to_log:
                    self.cost_token_logger.info(str(to_log))
        # example "{'input_tokens': 106170, 'output_tokens': 7652, 'total_tokens': 118594, 'input_token_details': {'cache_read': 106164}, 'output_token_details': {'reasoning': 4772}}"
        for generation in response.generations:
            # The 'generation' is a list of ChatGeneration or Generation objects
            for gen in generation:
                # Check if the generation object is a ChatGeneration instance
                # and has the 'usage_metadata' attribute.
                if (
                    isinstance(gen, ChatGeneration)
                    and hasattr(gen.message, "usage_metadata")
                    and gen.message.usage_metadata is not None
                ):
                    usage_metadata = gen.message.usage_metadata

                    input_tokens = usage_metadata.get("input_tokens", 0)
                    try:
                        cached_tokens = usage_metadata["input_token_details"][
                            "cache_read"
                        ]
                    except KeyError:
                        cached_tokens = 0
                    output_tokens = usage_metadata.get("output_tokens", 0)
                    try:
                        reasoning_tokens = usage_metadata["output_token_details"][
                            "reasoning"
                        ]
                    except KeyError:
                        reasoning_tokens = 0
                    if input_tokens > 0 or output_tokens > 0:
                        cost = calculate_gemini_cost(
                            input_tokens,
                            output_tokens,
                            cached_tokens,
                            model_name=gen.generation_info["model_name"],
                        )
                        self.total_input_tokens += input_tokens
                        self.total_cached_tokens += cached_tokens
                        self.total_output_tokens += output_tokens
                        self.total_reasoning_tokens += reasoning_tokens
                        self.total_cost += cost
        return super().on_llm_end(
            response, run_id=run_id, parent_run_id=parent_run_id, **kwargs
        )

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):

        return super().on_llm_error(
            error, run_id=run_id, parent_run_id=parent_run_id, **kwargs
        )


class PromptTokenCounter(BaseCallbackHandler):
    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs):
        for prompt in prompts:
            token_count = self.count_tokens(prompt)
            print(f"Prompt: {prompt}")
            print(f"Token count: {token_count}")

    def count_tokens(self, text: str) -> int:
        # Implement your token counting logic here
        # For example, a simple approximation:
        return len(text.split())
