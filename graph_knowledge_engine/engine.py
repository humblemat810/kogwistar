from typing import List, Optional, Dict, Any
from chromadb import Client
from chromadb.config import Settings
from .models import Node, Edge, Document, Domain, ReferenceSession
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

import json
import os
from dotenv import load_dotenv

class GraphKnowledgeEngine:
    def chroma_sanitize_metadata(self, metadata):
        return {k: v for k, v in metadata.items() if v is not None}
    def __init__(self, persist_directory: Optional[str] = None):
        self.chroma_client = Client(Settings(
            is_persistent = True,
            persist_directory=persist_directory or "./chroma_db",
            anonymized_telemetry = False
        ))
        load_dotenv() 
        self.node_collection = self.chroma_client.get_or_create_collection("nodes")
        self.edge_collection = self.chroma_client.get_or_create_collection("edges")
        self.document_collection = self.chroma_client.get_or_create_collection("documents")
        self.domain_collection = self.chroma_client.get_or_create_collection("domains")
        self.llm = AzureChatOpenAI(deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME_GPT4_1"),
                                model_name = os.getenv("OPENAI_MODEL_NAME_GPT4_1"),
                                azure_endpoint=os.getenv("OPENAI_DEPLOYMENT_ENDPOINT_GPT4_1"),
                                cache=None,
                                openai_api_key=os.getenv("OPENAI_API_KEY_GPT4_1"),
                                api_version="2024-08-01-preview",
                                model_version=os.getenv("OPENAI_DEPLOYMENT_VERSION_GPT4_1"),
                                temperature=0.1,
                                max_tokens = 12000,
                                openai_api_type="azure",
            )

    def add_node(self, node: Node):
        self.node_collection.add(
            ids=[node.id],
            documents=[node.model_dump_json()],
            embeddings=[node.embedding] if node.embedding else None,
            metadatas=[self.chroma_sanitize_metadata({
                "label": node.label,
                "type": node.type,
                "domain_id": node.domain_id,
                "properties": json.dumps(node.properties) if node.properties is not None else None,
                "references": json.dumps([ref.model_dump() for ref in node.references]) if node.references is not None else None
            })]
        )

    def add_edge(self, edge: Edge):
        self.edge_collection.add(
            ids=[edge.id],
            documents=[edge.model_dump_json()],
            embeddings=[edge.embedding] if edge.embedding else None,
            metadatas=[self.chroma_sanitize_metadata({
                "relation": edge.relation,
                "source_ids": json.dumps(edge.source_ids) if edge.source_ids is not None else None,
                "target_ids": json.dumps(edge.target_ids) if edge.target_ids is not None else None,
                "type": edge.type,
                "domain_id": edge.domain_id,
                "properties": json.dumps(edge.properties) if edge.properties is not None else None,
                "references": json.dumps([ref.model_dump() for ref in edge.references]) if edge.references is not None else None
            })]
        )
    
    def add_document(self, document: Document):
        self.document_collection.add(
            ids=[document.id],
            documents=[document.content],
            metadatas=[self.chroma_sanitize_metadata({
                "type": document.type,
                "metadata": json.dumps(document.metadata) if document.metadata is not None else None,
                "domain_id": document.domain_id,
                "processed": document.processed
            })]
        )

    def add_domain(self, domain: Domain):
        self.domain_collection.add(
            ids=[domain.id],
            documents=[domain.json()],
            metadatas=[{
                "name": domain.name,
                "description": domain.description
            }]
        )

    def vector_search_nodes(self, embedding: List[float], top_k: int = 5):
        return self.node_collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )

    def vector_search_edges(self, embedding: List[float], top_k: int = 5):
        return self.edge_collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )

    def ingest_document_with_llm(self, document: Document):
        """
        Ingest a document using LLM (via Langchain) to extract nodes and edges, then store them in ChromaDB.
        Azure OpenAI config is loaded from environment variables via dotenv.
        """
        # Add the document to the collection
        self.add_document(document)

        # Define the output schema for the LLM
        node_schema = ResponseSchema(
            name="nodes",
            description="A list of nodes, each with label, type, domain_id (optional), properties (optional)"
        )
        edge_schema = ResponseSchema(
            name="edges",
            description="A list of edges, each with source_ids, target_ids, relation, type, domain_id (optional), properties (optional)"
        )
        parser = StructuredOutputParser.from_response_schemas([node_schema, edge_schema])

        # Prompt template for LLM
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert knowledge graph extractor. Given a document, extract all entities, ideas, and relationships as nodes and edges in a hypergraph. Output as JSON with 'nodes' and 'edges'."
            ),
            (
                "human",
                "Document:\n{document}\n\n{format_instructions}"
            )
        ])

        # Format the prompt
        format_instructions = parser.get_format_instructions()
        chain = prompt | self.llm | parser

        # Run the chain
        result = chain.invoke({
            "document": document.content,
            "format_instructions": format_instructions
        })

        # Parse and store nodes and edges
        nodes = result.get("nodes", [])
        edges = result.get("edges", [])

        # Build a ReferenceSession for this document
        ref_session = None
        if document.metadata and "collection_page_url" in document.metadata and "document_page_url" in document.metadata:
            ref_session = ReferenceSession(
                collection_page_url=document.metadata["collection_page_url"],
                document_page_url=document.metadata["document_page_url"]
            )
        else:
            # Fallback: use document.id as a pseudo-URL
            ref_session = ReferenceSession(
                collection_page_url=f"document_collection/{document.id}",
                document_page_url=f"document/{document.id}"
            )

        for node_data in nodes:
            if "references" not in node_data or not node_data["references"]:
                node_data["references"] = [ref_session]
            node = Node(**node_data)
            self.add_node(node)

        for edge_data in edges:
            if "references" not in edge_data or not edge_data["references"]:
                edge_data["references"] = [ref_session]
            edge = Edge(**edge_data)
            self.add_edge(edge)

        return {
            "document_id": document.id,
            "nodes_added": len(nodes),
            "edges_added": len(edges)
        }
