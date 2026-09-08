from pathlib import Path
import json

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"


# Files and database settings
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"
COLLECTION_NAME = "enterprise_documents"

# Embedding model
MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks():
    """Load chunks from chunks.jsonl."""

    chunks = []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        for line in file:
            chunks.append(json.loads(line))

    return chunks


def main():

    print("Loading chunks...")

    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks.")

    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded.")

    texts = [chunk["text"] for chunk in chunks]

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    print("Embeddings created.")

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    print("Creating Qdrant database...")

    client = QdrantClient(
        path=str(VECTORSTORE_DIR)
    )

    # Create collection
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=embeddings.shape[1],
            distance=Distance.COSINE
        )
    )

    points = []

    for index, (chunk, embedding) in enumerate(
        zip(chunks, embeddings)
    ):

        points.append(
            PointStruct(
                id=index,
                vector=embedding.tolist(),
                payload=chunk
            )
        )

    print("Uploading vectors to Qdrant...")

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print("\n--------------------------------")
    print("Embedding pipeline completed!")
    print(f"Chunks stored: {len(points)}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Vector database: {VECTORSTORE_DIR}")
    print("--------------------------------")


if __name__ == "__main__":
    main()