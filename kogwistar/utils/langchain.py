from collections.abc import Mapping, Sequence
from logging import Logger
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.outputs.chat_generation import ChatGeneration
    from langchain_core.outputs.llm_result import LLMResult
else:
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

            def on_llm_error(self, *args: Any, **kwargs: Any) -> None:
                _ = (args, kwargs)

        class ChatGeneration:
            pass

        class LLMResult:
            generations: list[list[Any]]


class _MessageLike(Protocol):
    usage_metadata: Mapping[str, object] | None
    response_metadata: object


class _GenerationLike(Protocol):
    message: _MessageLike
    generation_info: Mapping[str, object]


def _generation_model_name(generation: _GenerationLike) -> str:
    return str(generation.generation_info.get("model_name", "unknown"))


def _usage_int(usage: Mapping[str, object], key: str) -> int:
    value = usage.get(key, 0)
    return int(value) if isinstance(value, (int, float)) else 0


def _nested_usage_int(usage: Mapping[str, object], group: str, key: str) -> int:
    value = usage.get(group)
    if not isinstance(value, Mapping):
        return 0
    return _usage_int(value, key)


def _message_usage(message: object) -> Mapping[str, object] | None:
    value = getattr(message, "usage_metadata", None)
    return value if isinstance(value, Mapping) else None


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


def calculate_gemini_cost(
    input_tokens: int, output_tokens: int, cached_tokens: int, model_name: str
) -> float:
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
    return float(total_cost)


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
        self.usage_history: list[dict[str, object]] = []
        self.run_start_time: float | None = None
        self.run_end_time: float | None = None

    def on_llm_start(
        self,
        serialized: object,
        prompts: Sequence[str],
        *,
        run_id: object,
        parent_run_id: object = None,
        tags: Sequence[str] | None = None,
        metadata: Mapping[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        self.run_start_time = time.time()
        return cast(Any, super()).on_llm_start(
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
                    message = cast(_GenerationLike, gen).message
                    usage_metadata = _message_usage(message)
                    if usage_metadata is not None:
                            input_tokens = _usage_int(usage_metadata, "input_tokens")
                            output_tokens = _usage_int(usage_metadata, "output_tokens")
                            cached_tokens = _nested_usage_int(
                                usage_metadata, "input_token_details", "cache_read"
                            )
                            reasoning_tokens = _nested_usage_int(
                                usage_metadata, "output_token_details", "reasoning"
                            )

                            if input_tokens > 0 or output_tokens > 0:
                                cost = calculate_gemini_cost(
                                    input_tokens,
                                    output_tokens,
                                    cached_tokens,
                                    model_name=_generation_model_name(cast(_GenerationLike, gen)),
                                )
                                self.total_input_tokens += input_tokens
                                self.total_output_tokens += output_tokens
                                self.reasoning_tokens += reasoning_tokens
                                self.cache_tokens += cached_tokens
                                self.total_cost += cost
                                self.usage_history.append(
                                    {
                                        "model_name": _generation_model_name(cast(_GenerationLike, gen)),
                                        "input_tokens": input_tokens,
                                        "output_tokens": output_tokens,
                                        "cached_tokens": cached_tokens,
                                        "reasoning_tokens": reasoning_tokens,
                                        "cost": cost,
                                        "start_time": self.run_start_time,
                                        "end_time": self.run_end_time,
                                    }
                                )

    def reset(self) -> None:
        """Resets the counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

    def __repr__(self) -> str:
        return (
            f"Total Input Tokens: {self.total_input_tokens}\n"
            f"Total Output Tokens: {self.total_output_tokens}\n"
            f"Total Cost: ${self.total_cost:.8f}"
        )

    def model_dump(self) -> dict[str, object]:
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
    def __init__(self, logger: Logger) -> None:
        self.cost_token_logger: Logger = logger
        self.total_input_tokens = 0
        self.total_reasoning_tokens = 0
        self.total_cached_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: object,
        parent_run_id: object = None,
        **kwargs: object,
    ) -> None:
        # self.cost_token_logger.info(response.response_metadata)
        for g in response.generations:
            for gg in g:
                to_log: dict[str, str | None] = {
                    "response_metadata": None,
                    "usage_metadata": None,
                }
                message = cast(_GenerationLike, gg).message
                if hasattr(message, "response_metadata"):
                    to_log["response_metadata"] = f"{message.response_metadata}"
                if hasattr(message, "usage_metadata"):
                    to_log["usage_metadata"] = f"{message.usage_metadata}"
                if to_log:
                    self.cost_token_logger.info(str(to_log))
        # example "{'input_tokens': 106170, 'output_tokens': 7652, 'total_tokens': 118594, 'input_token_details': {'cache_read': 106164}, 'output_token_details': {'reasoning': 4772}}"
        for generation in response.generations:
            # The 'generation' is a list of ChatGeneration or Generation objects
            for gen in generation:
                # Check if the generation object is a ChatGeneration instance
                # and has the 'usage_metadata' attribute.
                if not isinstance(gen, ChatGeneration):
                    continue
                generation_like = cast(_GenerationLike, gen)
                usage_metadata = _message_usage(generation_like.message)
                if usage_metadata is None:
                    continue
                input_tokens = _usage_int(usage_metadata, "input_tokens")
                cached_tokens = _nested_usage_int(
                    usage_metadata, "input_token_details", "cache_read"
                )
                output_tokens = _usage_int(usage_metadata, "output_tokens")
                reasoning_tokens = _nested_usage_int(
                    usage_metadata, "output_token_details", "reasoning"
                )
                if input_tokens > 0 or output_tokens > 0:
                    cost = calculate_gemini_cost(
                        input_tokens,
                        output_tokens,
                        cached_tokens,
                        model_name=_generation_model_name(cast(_GenerationLike, gen)),
                    )
                    self.total_input_tokens += input_tokens
                    self.total_cached_tokens += cached_tokens
                    self.total_output_tokens += output_tokens
                    self.total_reasoning_tokens += reasoning_tokens
                    self.total_cost += cost
        cast(Any, super()).on_llm_end(
            response, run_id=run_id, parent_run_id=parent_run_id, **kwargs
        )

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):

        cast(Any, super()).on_llm_error(
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
