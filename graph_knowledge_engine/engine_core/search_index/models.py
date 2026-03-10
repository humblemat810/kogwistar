from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class IndexingItem(BaseModel):
    node_id: str
    canonical_title: str
    keywords: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    provision: str
    doc_id: Optional[str] = None


class AddIndexEntriesInput(BaseModel):
    index: list[IndexingItem]


def make_index_key(node_id: str, canonical_title: str, provision: str) -> str:
    return f"{node_id}|{canonical_title}|{provision}"


def make_index_key_for_item(item: IndexingItem) -> str:
    return make_index_key(
        node_id=item.node_id,
        canonical_title=item.canonical_title,
        provision=item.provision,
    )


def build_embedding_text(item: IndexingItem) -> str:
    keywords = ", ".join(item.keywords or [])
    aliases = ", ".join(item.aliases or [])
    return "\n".join(
        [
            f"title: {item.canonical_title}",
            f"provision: {item.provision}",
            f"keywords: {keywords}",
            f"aliases: {aliases}",
        ]
    )