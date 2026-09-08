from pathlib import Path

from sentence_transformers import CrossEncoder
from qdrant_client import QdrantClient

from hybrid_search import (
    load_chunks,
    build_bm25,
    semantic_search,
    bm25_search,
    reciprocal_rank_fusion
)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

COLLECTION_NAME = "enterprise_documents"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ============================================================
# RERANK RESULTS
# ============================================================

def rerank_results(
    query,
    hybrid_results,
    top_k=5,
    minimum_results=3
):

    print("\nLoading reranker model...")

    reranker = CrossEncoder(RERANKER_MODEL)

    # --------------------------------------------------------
    # Prepare query-document pairs
    # --------------------------------------------------------

    pairs = []

    for result in hybrid_results:

        text = result["chunk"].get("text", "")

        pairs.append(
            [query, text]
        )

    # --------------------------------------------------------
    # Calculate CrossEncoder scores
    # --------------------------------------------------------

    print("Calculating reranking scores...")

    scores = reranker.predict(pairs)

    reranked = []

    for result, score in zip(
        hybrid_results,
        scores
    ):

        reranked.append(
            {
                "rerank_score": float(score),

                "rrf_score": float(
                    result.get("score", 0)
                ),

                "chunk": result["chunk"]
            }
        )

    # --------------------------------------------------------
    # Sort by relevance
    # --------------------------------------------------------

    reranked.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    # --------------------------------------------------------
    # Remove duplicate chunks
    # --------------------------------------------------------

    unique_results = []

    seen_chunks = set()

    for result in reranked:

        chunk_id = result["chunk"].get(
            "chunk_id"
        )

        if chunk_id in seen_chunks:
            continue

        seen_chunks.add(chunk_id)

        unique_results.append(result)

    # --------------------------------------------------------
    # Keep top results
    # --------------------------------------------------------

    selected = unique_results[:top_k]

    # --------------------------------------------------------
    # Safety fallback
    #
    # Always keep a few results so the LLM has evidence,
    # even when reranker scores are low.
    # --------------------------------------------------------

    if len(selected) < minimum_results:

        selected = unique_results[
            :minimum_results
        ]

    return selected


# ============================================================
# TEST / STANDALONE EXECUTION
# ============================================================

def main():

    print("=" * 70)
    print("              RERANKING TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    print("\nLoading chunks...")

    chunks = load_chunks()

    print(
        f"Loaded {len(chunks)} chunks."
    )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print(
        "\nLoading embedding model..."
    )

    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    # --------------------------------------------------------
    # Connect to Qdrant
    # --------------------------------------------------------

    print(
        "\nConnecting to Qdrant..."
    )

    client = QdrantClient(
        path=str(VECTORSTORE_DIR)
    )

    # --------------------------------------------------------
    # Build BM25
    # --------------------------------------------------------

    print(
        "\nBuilding BM25 index..."
    )

    bm25 = build_bm25(chunks)

    # --------------------------------------------------------
    # User query
    # --------------------------------------------------------

    query = input(
        "\nEnter your question: "
    ).strip()

    if not query:

        print(
            "\nPlease enter a valid question."
        )

        return

    # --------------------------------------------------------
    # Semantic search
    # --------------------------------------------------------

    print(
        "\n[1/3] Running semantic search..."
    )

    semantic_results = semantic_search(
        query,
        embedding_model,
        client,
        top_k=10
    )

    # --------------------------------------------------------
    # BM25 search
    # --------------------------------------------------------

    print(
        "\n[2/3] Running BM25 search..."
    )

    bm25_results = bm25_search(
        query,
        chunks,
        bm25,
        top_k=10
    )

    # --------------------------------------------------------
    # Hybrid search
    # --------------------------------------------------------

    print(
        "\nCombining semantic + BM25..."
    )

    hybrid_results = reciprocal_rank_fusion(
        semantic_results,
        bm25_results,
        chunks
    )

    hybrid_results = hybrid_results[:10]

    print(
        f"Hybrid candidates: "
        f"{len(hybrid_results)}"
    )

    # --------------------------------------------------------
    # Reranking
    # --------------------------------------------------------

    print(
        "\n[3/3] Reranking results..."
    )

    reranked_results = rerank_results(
        query,
        hybrid_results,
        top_k=5
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("              RERANKED RESULTS")
    print("=" * 70)

    for rank, result in enumerate(
        reranked_results,
        start=1
    ):

        chunk = result["chunk"]

        print("\n" + "-" * 70)

        print(
            f"RESULT #{rank}"
        )

        print("-" * 70)

        print(
            f"Document     : "
            f"{chunk.get('document_name')}"
        )

        print(
            f"Page         : "
            f"{chunk.get('page_number')}"
        )

        print(
            f"Section      : "
            f"{chunk.get('section')}"
        )

        print(
            f"Chunk ID     : "
            f"{chunk.get('chunk_id')}"
        )

        print(
            f"RRF Score    : "
            f"{result['rrf_score']:.6f}"
        )

        print(
            f"Rerank Score : "
            f"{result['rerank_score']:.4f}"
        )

        print("\nEvidence:")

        print(
            chunk.get("text")
        )

    print("\n")
    print("=" * 70)
    print("RERANKING COMPLETED SUCCESSFULLY")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()