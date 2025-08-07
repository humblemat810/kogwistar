from typing import List, Optional, Dict, Any
from chromadb import Client
from chromadb.config import Settings
from .models import Node, Edge, Document, Domain

from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

import json
import os
from dotenv import load_dotenv

class GraphKnowledgeEngine:
    def __init__(self, persist_directory: Optional[str] = None):
        self.chroma_client = Client(Settings(
            persist_directory=persist_directory or "./chroma_db"
        ))
        self.node_collection = self.chroma_client.get_or_create_collection("nodes")
        self.edge_collection = self.chroma_client.get_or_create_collection("edges")
        self.document_collection = self.chroma_client.get_or_create_collection("documents")
        self.domain_collection = self.chroma_client.get_or_create_collection("domains")
        load_dotenv()  # Load environment variables from .env

    def add_node(self, node: Node):
        self.node_collection.add(
            ids=[node.id],
            documents=[node.json()],
            embeddings=[node.embedding] if node.embedding else None,
            metadatas=[{
                "label": node.label,
                "type": node.type,
                "domain_id": node.domain_id,
                "properties": json.dumps(node.properties) if node.properties is not None else None
            }]
        )

    def add_edge(self, edge: Edge):
        self.edge_collection.add(
            ids=[edge.id],
            documents=[edge.json()],
            embeddings=[edge.embedding] if edge.embedding else None,
            metadatas=[{
                "relation": edge.relation,
                "source_ids": json.dumps(edge.source_ids) if edge.source_ids is not None else None,
                "target_ids": json.dumps(edge.target_ids) if edge.target_ids is not None else None,
                "type": edge.type,
                "domain_id": edge.domain_id,
                "properties": json.dumps(edge.properties) if edge.properties is not None else None
            }]
        )

    def add_document(self, document: Document):
        self.document_collection.add(
            ids=[document.id],
            documents=[document.content],
            metadatas=[{
                "type": document.type,
                "metadata": json.dumps(document.metadata) if document.metadata is not None else None,
                "domain_id": document.domain_id,
                "processed": document.processed
            }]
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

        # Load Azure OpenAI config from environment
        deployment_name = os.getenv("OPENAI_DEPLOYMENT_NAME_GPT4_1")
        api_key = os.getenv("OPENAI_API_KEY_GPT4_1")
        api_base = os.getenv("OPENAI_DEPLOYMENT_ENDPOINT_GPT4_1")
        api_version = os.getenv("OPENAI_DEPLOYMENT_VERSION_GPT4_1", "2024-12-01-preview")

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

        # Prepare the LLM
        llm = AzureChatOpenAI(
            deployment_name=deployment_name,
            openai_api_key=api_key,
            openai_api_base=api_base,
            openai_api_version=api_version,
            temperature=0.0,
        )

        # Format the prompt
        format_instructions = parser.get_format_instructions()
        chain = prompt | llm | parser

        # Run the chain
        result = chain.invoke({
            "document": document.content,
            "format_instructions": format_instructions
        })

        # Parse and store nodes and edges
        nodes = result.get("nodes", [])
        edges = result.get("edges", [])

        for node_data in nodes:
            node = Node(**node_data)
            self.add_node(node)

        for edge_data in edges:
            edge = Edge(**edge_data)
            self.add_edge(edge)

        return {
            "document_id": document.id,
            "nodes_added": len(nodes),
            "edges_added": len(edges)
        }
