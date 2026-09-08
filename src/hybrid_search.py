from pathlib import Path
import json
import re

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"

COLLECTION_NAME = "enterprise_documents"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks():
    chunks = []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        for line in file:
            chunks.append(json.loads(line))

    return chunks


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def build_bm25(chunks):
    tokenized_documents = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_documents)


def semantic_search(query, model, client, top_k=10):

    query_embedding = model.encode(query).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        with_payload=True
    ).points

    return results


def bm25_search(query, chunks, bm25, top_k=10):

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True
    )[:top_k]

    return [
        {
            "index": index,
            "score": float(scores[index]),
            "chunk": chunks[index]
        }
        for index in ranked_indexes
    ]


def reciprocal_rank_fusion(
    semantic_results,
    bm25_results,
    chunks,
    k=60
):
    """Combine rankings using Reciprocal Rank Fusion."""

    scores = {}

    # Semantic ranking
    for rank, result in enumerate(semantic_results, start=1):

        chunk_id = result.payload["chunk_id"]

        scores.setdefault(
            chunk_id,
            {
                "score": 0,
                "chunk": result.payload
            }
        )

        scores[chunk_id]["score"] += 1 / (k + rank)

    # BM25 ranking
    for rank, result in enumerate(bm25_results, start=1):

        chunk_id = result["chunk"]["chunk_id"]

        scores.setdefault(
            chunk_id,
            {
                "score": 0,
                "chunk": result["chunk"]
            }
        )

        scores[chunk_id]["score"] += 1 / (k + rank)

    # Sort by combined RRF score
    ranked_results = sorted(
        scores.values(),
        key=lambda item: item["score"],
        reverse=True
    )

    return ranked_results


def main():

    print("Loading chunks...")

    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks.")

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    client = QdrantClient(
        path=str(VECTORSTORE_DIR)
    )

    print("Building BM25 index...")

    bm25 = build_bm25(chunks)

    query = input("\nEnter your question: ")

    print("\nRunning semantic search...")
    semantic_results = semantic_search(
        query,
        model,
        client,
        top_k=10
    )

    print("Running BM25 search...")
    bm25_results = bm25_search(
        query,
        chunks,
        bm25,
        top_k=10
    )

    print("Combining results using RRF...")

    hybrid_results = reciprocal_rank_fusion(
        semantic_results,
        bm25_results,
        chunks
    )

    print("\n" + "=" * 60)
    print("HYBRID SEARCH RESULTS")
    print("=" * 60)

    for rank, result in enumerate(
        hybrid_results[:5],
        start=1
    ):

        chunk = result["chunk"]

        print(f"\nResult {rank}")
        print(f"RRF Score: {result['score']:.6f}")
        print(f"Document: {chunk.get('document_name')}")
        print(f"Page: {chunk.get('page_number')}")
        print(f"Section: {chunk.get('section')}")
        print(f"Chunk ID: {chunk.get('chunk_id')}")
        print(f"Text: {chunk.get('text')}")


if __name__ == "__main__":
    main()