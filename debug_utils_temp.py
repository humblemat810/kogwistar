from chromadb import Client
from chromadb.config import Settings
persist_directory = None
chroma_client = Client(
            Settings(
                is_persistent=True,
                persist_directory=persist_directory or "./chroma_db",  # "./chroma_db", "doc_chroma"
                anonymized_telemetry=False,
            )
        )


node_collection = chroma_client.get_or_create_collection(
    "nodes"
)
edge_collection = chroma_client.get_or_create_collection(
    "edges"
)
edge_endpoints_collection = chroma_client.get_or_create_collection("edge_endpoints")
document_collection = chroma_client.get_or_create_collection("documents")
domain_collection = chroma_client.get_or_create_collection("domains")
node_docs_collection = chroma_client.get_or_create_collection("node_docs")
embedding_exclude_field = ['references', 'doc_ids', 'doc_id']
embedding_keep_field = ["label", "summary"]
pass