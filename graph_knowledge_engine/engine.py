from typing import List, Optional, Dict, Any, Tuple
from chromadb import Client
from chromadb.config import Settings
from .models import (
    Node,
    Edge,
    Document,
    Domain,
    ReferenceSession,
    LLMGraphExtraction,
    LLMNode,
    LLMEdge,
    AdjudicationCandidate,
    AdjudicationVerdict,
    LLMMergeAdjudication,
)
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
import json
import os
from dotenv import load_dotenv
import uuid
from joblib import Memory

# Simple on-disk cache dir (optional)
memory = Memory(location=os.path.join(".cache", "my_cache"), verbose=0)


class GraphKnowledgeEngine:
    """High-level orchestration for extracting, storing, and adjudicating knowledge graph data."""

    # ----------------------------
    # Utilities
    # ----------------------------
    @staticmethod
    def chroma_sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Drop keys whose values are None. ChromaDB metadata rejects None values."""
        return {k: v for k, v in metadata.items() if v is not None}

    @staticmethod
    def _json_or_none(obj: Any) -> Optional[str]:
        return json.dumps(obj) if obj is not None else None

    # ----------------------------
    # Init
    # ----------------------------
    def __init__(self, persist_directory: Optional[str] = None):
        load_dotenv()
        self.chroma_client = Client(
            Settings(
                is_persistent=True,
                persist_directory=persist_directory or "./chroma_db",
                anonymized_telemetry=False,
            )
        )
        self.node_collection = self.chroma_client.get_or_create_collection("nodes")
        self.edge_collection = self.chroma_client.get_or_create_collection("edges")
        self.document_collection = self.chroma_client.get_or_create_collection("documents")
        self.domain_collection = self.chroma_client.get_or_create_collection("domains")

        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME_GPT4_1"),
            model_name=os.getenv("OPENAI_MODEL_NAME_GPT4_1"),
            azure_endpoint=os.getenv("OPENAI_DEPLOYMENT_ENDPOINT_GPT4_1"),
            cache=None,
            openai_api_key=os.getenv("OPENAI_API_KEY_GPT4_1"),
            api_version="2024-08-01-preview",
            model_version=os.getenv("OPENAI_DEPLOYMENT_VERSION_GPT4_1"),
            temperature=0.1,
            max_tokens=12000,
            openai_api_type="azure",
        )

    # ----------------------------
    # Chroma adders
    # ----------------------------
    def add_node(self, node: Node):
        self.node_collection.add(
            ids=[node.id],
            documents=[node.model_dump_json()],
            embeddings=[node.embedding] if node.embedding else None,
            metadatas=[
                self.chroma_sanitize_metadata(
                    {
                        "label": node.label,
                        "type": node.type,
                        "summary": node.summary,
                        "domain_id": node.domain_id,
                        "canonical_entity_id": node.canonical_entity_id,
                        "properties": self._json_or_none(node.properties),
                        "references": self._json_or_none(
                            [ref.model_dump() for ref in (node.references or [])]
                        ),
                    }
                )
            ],
        )

    def add_edge(self, edge: Edge):
        self.edge_collection.add(
            ids=[edge.id],
            documents=[edge.model_dump_json()],
            embeddings=[edge.embedding] if edge.embedding else None,
            metadatas=[
                self.chroma_sanitize_metadata(
                    {
                        "relation": edge.relation,
                        "source_ids": self._json_or_none(edge.source_ids),
                        "target_ids": self._json_or_none(edge.target_ids),
                        "type": edge.type,
                        "summary": edge.summary,
                        "domain_id": edge.domain_id,
                        "canonical_entity_id": edge.canonical_entity_id,
                        "properties": self._json_or_none(edge.properties),
                        "references": self._json_or_none(
                            [ref.model_dump() for ref in (edge.references or [])]
                        ),
                    }
                )
            ],
        )

    def add_document(self, document: Document):
        self.document_collection.add(
            ids=[document.id],
            documents=[document.content],
            metadatas=[
                self.chroma_sanitize_metadata(
                    {
                        "type": document.type,
                        "metadata": self._json_or_none(document.metadata),
                        "domain_id": document.domain_id,
                        "processed": document.processed,
                    }
                )
            ],
        )

    def add_domain(self, domain: Domain):
        self.domain_collection.add(
            ids=[domain.id],
            documents=[domain.model_dump_json()],
            metadatas=[
                self.chroma_sanitize_metadata(
                    {
                        "name": domain.name,
                        "description": domain.description,
                    }
                )
            ],
        )

    # ----------------------------
    # Vector queries
    # ----------------------------
    def vector_search_nodes(self, embedding: List[float], top_k: int = 5):
        return self.node_collection.query(query_embeddings=[embedding], n_results=top_k)

    def vector_search_edges(self, embedding: List[float], top_k: int = 5):
        return self.edge_collection.query(query_embeddings=[embedding], n_results=top_k)

    # ----------------------------
    # Ingestion
    # ----------------------------
    def _build_ref_session(self, document: Document) -> ReferenceSession:
        if document.metadata and "collection_page_url" in document.metadata and "document_page_url" in document.metadata:
            return ReferenceSession(
                collection_page_url=document.metadata["collection_page_url"],
                document_page_url=document.metadata["document_page_url"],
            )
        # Fallback: pseudo URLs
        return ReferenceSession(
            collection_page_url=f"document_collection/{document.id}",
            document_page_url=f"document/{document.id}",
        )

    def _extract_graph_with_llm(self, content: str) -> Tuple[Any, Optional[LLMGraphExtraction], Optional[str]]:
        """Call LLM for structured extraction. Returns (raw, parsed, parsing_error)."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert knowledge graph extractor. Given a document, extract entities and relationships as nodes and edges in a hypergraph.\n"
                    "For each node/edge include: label, type ('entity' or 'relationship'), and a concise 'summary'.\n"
                    "Do not include any ID fields. Return only the structured JSON with keys 'nodes' and 'edges'.",
                ),
                ("human", "Document:\n{document}\n\nReturn only the structured JSON as specified."),
            ]
        )
        chain = prompt | self.llm.with_structured_output(LLMGraphExtraction, include_raw=True)
        result = chain.invoke({"document": content})
        # result is a dict when include_raw=True
        raw = result.get("raw") if isinstance(result, dict) else None
        parsed = result.get("parsed") if isinstance(result, dict) else result
        err = result.get("parsing_error") if isinstance(result, dict) else None
        return raw, parsed, err

    def ingest_document_with_llm(self, document: Document):
        """
        Ingest a document using LLM to extract nodes and edges, then store them in ChromaDB.
        """
        self.add_document(document)
        raw, parsed, error = self._extract_graph_with_llm(document.content)
        if error:
            raise ValueError(f"LLM parsing error: {error}")
        if not isinstance(parsed, LLMGraphExtraction):
            # Some LangChain versions may return dicts; coerce if needed
            parsed = LLMGraphExtraction.model_validate(parsed)

        ref_session = self._build_ref_session(document)

        nodes_added = 0
        for llm_node in parsed.nodes:
            node = Node(
                id=str(uuid.uuid1()),
                label=llm_node.label,
                type=llm_node.type,
                summary=llm_node.summary,
                domain_id=llm_node.domain_id,
                properties=llm_node.properties,
                references=llm_node.references or [ref_session],
            )
            self.add_node(node)
            nodes_added += 1

        edges_added = 0
        for llm_edge in parsed.edges:
            edge = Edge(
                id=str(uuid.uuid1()),
                label=llm_edge.label,
                type=llm_edge.type,
                summary=llm_edge.summary,
                domain_id=llm_edge.domain_id,
                properties=llm_edge.properties,
                references=llm_edge.references or [ref_session],
                source_ids=llm_edge.source_ids,
                target_ids=llm_edge.target_ids,
                relation=llm_edge.relation,
            )
            self.add_edge(edge)
            edges_added += 1

        return {"document_id": document.id, "nodes_added": nodes_added, "edges_added": edges_added}

    # ----------------------------
    # Adjudication (LLM-assisted merge decision)
    # ----------------------------
    def adjudicate_merge(self, left: Node, right: Node) -> AdjudicationVerdict:
        """Use the LLM to decide if two nodes are the SAME real-world entity."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a careful adjudicator. Decide if two nodes refer to the SAME real-world entity.\n"
                    "Be conservative: only return true if confident. Return a structured JSON verdict.",
                ),
                ("human", "Left:\n{left}\n\nRight:\n{right}\n\nReturn only the structured JSON."),
            ]
        )
        chain = prompt | self.llm.with_structured_output(LLMMergeAdjudication)
        result: LLMMergeAdjudication = chain.invoke(
            {"left": left.model_dump(), "right": right.model_dump()}
        )
        return result.verdict

    def commit_merge(self, left: Node, right: Node, verdict: AdjudicationVerdict) -> str:
        """
        Apply a positive adjudication by assigning/propagating a canonical_entity_id
        and recording a `same_as` edge with provenance.
        Returns the canonical id used.
        """
        if not verdict.same_entity:
            raise ValueError("Verdict not positive; will not merge.")

        canonical_id = verdict.canonical_entity_id or (left.canonical_entity_id or right.canonical_entity_id)
        if not canonical_id:
            canonical_id = str(uuid.uuid1())

        # Update in-memory nodes (caller is responsible for persisting updates to DB)
        left.canonical_entity_id = canonical_id
        right.canonical_entity_id = canonical_id

        # Record a same_as edge for auditability
        same_as = Edge(
            id=str(uuid.uuid1()),
            label="same_as",
            type="relationship",
            summary=verdict.reason or "Adjudicated same entity",
            domain_id=None,
            relation="same_as",
            source_ids=[left.id],
            target_ids=[right.id],
            properties={"confidence": verdict.confidence},
            references=None,
        )
        self.add_edge(same_as)
        return canonical_id
