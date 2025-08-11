import chromadb

# --- Connect to Chroma ---
client = chromadb.Client()

# --- Create or get collection ---
collection = client.get_or_create_collection(name="my_collection")

# --- Insert an initial document ---
collection.add(
    ids=["doc1"],
    documents=["This is a test document"],
    metadatas=[{"author": "Alice", "version": 1}]
)

# --- Simulated incoming metadata ---
new_metadata = {"version": 2, "reviewed": True}

# --- Fetch current metadata for the doc ---
current = collection.get(ids=["doc1"], include=["metadatas"])
current_meta = current["metadatas"][0] or {}

# --- Decide if replacement or true update ---
if set(new_metadata.items()) <= set(current_meta.items()):
    print("No update needed — metadata already matches or is a subset.")
elif set(new_metadata.keys()) == set(current_meta.keys()):
    print("Replacing metadata entirely.")
    collection.update(
        ids=["doc1"],
        metadatas=[new_metadata]
    )
else:
    print("Merging new keys into metadata.")
    merged_meta = {**current_meta, **new_metadata}
    collection.update(
        ids=["doc1"],
        metadatas=[merged_meta]
    )

# --- Verify ---
print("Final metadata:", collection.get(ids=["doc1"], include=["metadatas"])["metadatas"][0])