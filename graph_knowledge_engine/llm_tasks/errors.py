from __future__ import annotations


class LLMTaskError(RuntimeError):
    """Base error for LLM task execution failures."""


class MissingTaskError(LLMTaskError):
    """Raised when a required LLM task callable is missing."""


class ProviderDependencyError(LLMTaskError):
    """Raised when an optional provider dependency is required but unavailable."""
