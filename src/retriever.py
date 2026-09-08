from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

# Settings
COLLECTION_NAME = "enterprise_documents"
MODEL_NAME = "all-MiniLM-L6-v2"


def search_documents(query, top_k=5):
    """Search the vector database for relevant chunks."""

    print(f"\nSearching for: {query}\n")

    # Load embedding model
    model = SentenceTransformer(MODEL_NAME)

    # Convert question into an embedding
    query_embedding = model.encode(query).tolist()

    # Connect to local Qdrant database
    client = QdrantClient(
        path=str(VECTORSTORE_DIR)
    )

    # Search for similar chunks
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        with_payload=True
    ).points

    return results


def main():

    query = input("Enter your question: ")

    results = search_documents(query)

    print("=" * 60)
    print("RETRIEVAL RESULTS")
    print("=" * 60)

    for rank, result in enumerate(results, start=1):

        payload = result.payload

        print(f"\nResult {rank}")
        print(f"Score: {result.score:.4f}")
        print(f"Document: {payload.get('document_name')}")
        print(f"Page: {payload.get('page_number')}")
        print(f"Section: {payload.get('section')}")
        print(f"Chunk ID: {payload.get('chunk_id')}")
        print(f"Text: {payload.get('text')}")


if __name__ == "__main__":
    main()