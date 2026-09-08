from pathlib import Path
import json
import re

from rank_bm25 import BM25Okapi


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"


def load_chunks():
    """Load all chunks from chunks.jsonl."""

    chunks = []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        for line in file:
            chunks.append(json.loads(line))

    return chunks


def tokenize(text):
    """Convert text into lowercase tokens."""

    return re.findall(r"\b\w+\b", text.lower())


def build_bm25(chunks):
    """Create a BM25 search index."""

    tokenized_documents = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_documents)


def search_bm25(query, chunks, bm25, top_k=5):
    """Search chunks using BM25."""

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True
    )[:top_k]

    results = []

    for index in ranked_indexes:
        results.append({
            "score": float(scores[index]),
            "chunk": chunks[index]
        })

    return results


def main():

    print("Loading chunks...")

    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks.")

    print("Building BM25 index...")

    bm25 = build_bm25(chunks)

    print("BM25 index ready.")

    query = input("\nEnter your question: ")

    results = search_bm25(
        query,
        chunks,
        bm25,
        top_k=5
    )

    print("\n" + "=" * 60)
    print("BM25 SEARCH RESULTS")
    print("=" * 60)

    for rank, result in enumerate(results, start=1):

        chunk = result["chunk"]

        print(f"\nResult {rank}")
        print(f"Score: {result['score']:.4f}")
        print(f"Document: {chunk.get('document_name')}")
        print(f"Page: {chunk.get('page_number')}")
        print(f"Section: {chunk.get('section')}")
        print(f"Chunk ID: {chunk.get('chunk_id')}")
        print(f"Text: {chunk.get('text')}")


if __name__ == "__main__":
    main()