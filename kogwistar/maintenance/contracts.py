"""Reusable callback contracts for maintenance write boundaries."""

from __future__ import annotations

from typing import Protocol, TypeVar

TWriteItem = TypeVar("TWriteItem", contravariant=True)


class BeforeWrite(Protocol[TWriteItem]):
    """Authorize or observe one item immediately before it is written."""

    def __call__(self, item: TWriteItem, /) -> None: ...


__all__ = ["BeforeWrite"]
