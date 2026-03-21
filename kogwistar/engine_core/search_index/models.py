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
    lines = [
        "ENTITY SEARCH INDEX ENTRY",
        f"CANONICAL TITLE: {item.canonical_title}",
        f"PROVISION/DESCRIPTION: {item.provision}",
    ]
    if keywords:
        lines.append(f"KEYWORDS: {keywords}")
    if aliases:
        lines.append(f"ALIASES/KNOWN AS: {aliases}")
    if item.doc_id:
        lines.append(f"SOURCE DOCUMENT: {item.doc_id}")
    lines.append(f"TARGET NODE ID: {item.node_id}")
    return "\n".join(lines)
