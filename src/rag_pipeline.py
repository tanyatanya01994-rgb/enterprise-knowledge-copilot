from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from query_rewriter import rewrite_query
from answer_generator import generate_answer

from hybrid_search import (
    load_chunks,
    build_bm25,
    semantic_search,
    bm25_search,
    reciprocal_rank_fusion
)

from reranker import rerank_results
from evidence_verifier import verify_evidence


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

COLLECTION_NAME = "enterprise_documents"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# INITIALIZE RETRIEVAL SYSTEM
# ============================================================

def initialize_retrieval():

    print("=" * 75)
    print("        ENTERPRISE KNOWLEDGE INTELLIGENCE COPILOT")
    print("=" * 75)

    print("\nLoading document chunks...")

    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks.")

    print("\nLoading embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print("Embedding model loaded.")

    print("\nConnecting to Qdrant...")

    client = QdrantClient(
        path=str(VECTORSTORE_DIR)
    )

    print("Qdrant connected.")

    print("\nBuilding BM25 index...")

    bm25 = build_bm25(chunks)

    print("BM25 index ready.")

    return (
        chunks,
        embedding_model,
        client,
        bm25
    )


# ============================================================
# RETRIEVE EVIDENCE
# ============================================================

def retrieve_evidence(
    query,
    conversation_history,
    chunks,
    embedding_model,
    client,
    bm25
):

    # --------------------------------------------------------
    # STEP 1: QUERY REWRITING
    # --------------------------------------------------------

    print("\n[1/4] Rewriting query...")

    rewritten_query = rewrite_query(
        query,
        conversation_history
    )

    print("\nOriginal query:")
    print(query)

    print("\nRewritten search query:")
    print(rewritten_query)


    # --------------------------------------------------------
    # STEP 2: SEMANTIC SEARCH
    # --------------------------------------------------------

    print("\n[2/4] Running semantic search...")

    semantic_results = semantic_search(
        rewritten_query,
        embedding_model,
        client,
        top_k=10
    )

    print(
        f"Semantic search returned "
        f"{len(semantic_results)} results."
    )


    # --------------------------------------------------------
    # STEP 3: BM25 SEARCH
    # --------------------------------------------------------

    print("\n[3/4] Running BM25 search...")

    bm25_results = bm25_search(
        rewritten_query,
        chunks,
        bm25,
        top_k=10
    )

    print(
        f"BM25 search returned "
        f"{len(bm25_results)} results."
    )


    # --------------------------------------------------------
    # STEP 4: HYBRID SEARCH + RRF
    # --------------------------------------------------------

    print("\nCombining semantic + BM25 results...")

    hybrid_results = reciprocal_rank_fusion(
        semantic_results,
        bm25_results,
        chunks
    )

    hybrid_results = hybrid_results[:10]

    print(
        f"Hybrid retrieval candidates: "
        f"{len(hybrid_results)}"
    )


    # --------------------------------------------------------
    # STEP 5: RERANKING
    # --------------------------------------------------------

    print("\n[4/4] Reranking results...")

    reranked_results = rerank_results(
        rewritten_query,
        hybrid_results,
        top_k=5
    )

    print(
        f"Reranked evidence chunks: "
        f"{len(reranked_results)}"
    )

    # --------------------------------------------------------
    # STEP 6: EVIDENCE VERIFICATION
    # --------------------------------------------------------

    print("\n[5/5] Verifying evidence support...")

    verified_results, abstained = verify_evidence(
        rewritten_query,
        reranked_results
    )

    print(
        f"Verified evidence chunks: "
        f"{len(verified_results)}"
    )

    print(
        f"Abstained: "
        f"{'Yes' if abstained else 'No'}"
    )

    return (
        rewritten_query,
        verified_results,
        abstained
    )


# ============================================================
# DISPLAY RETRIEVAL RESULTS
# ============================================================

def display_results(
    original_query,
    rewritten_query,
    results
):

    print("\n")
    print("=" * 75)
    print("                    RETRIEVAL RESULTS")
    print("=" * 75)

    print("\nOriginal Question:")
    print(original_query)

    print("\nRewritten Search Query:")
    print(rewritten_query)

    print("\n" + "-" * 75)
    print("TOP EVIDENCE")
    print("-" * 75)


    for rank, result in enumerate(
        results,
        start=1
    ):

        chunk = result["chunk"]

        print(
            f"\nRESULT #{rank}"
        )

        print("-" * 75)

        print(
            f"Document      : "
            f"{chunk.get('document_name')}"
        )

        print(
            f"Page          : "
            f"{chunk.get('page_number')}"
        )

        print(
            f"Section       : "
            f"{chunk.get('section')}"
        )

        print(
            f"Chunk ID      : "
            f"{chunk.get('chunk_id')}"
        )

        print(
            f"RRF Score     : "
            f"{result.get('rrf_score', 0):.6f}"
        )

        print(
            f"Rerank Score  : "
            f"{result.get('rerank_score', 0):.4f}"
        )

        print("\nEvidence:")

        print(
            chunk.get("text")
        )


    print("\n" + "=" * 75)


# ============================================================
# MAIN RAG PIPELINE
# ============================================================

def main():

    # --------------------------------------------------------
    # INITIALIZE ALL RETRIEVAL COMPONENTS
    # --------------------------------------------------------

    (
        chunks,
        embedding_model,
        client,
        bm25
    ) = initialize_retrieval()


    # --------------------------------------------------------
    # SAMPLE CONVERSATION HISTORY
    # --------------------------------------------------------
    # This demonstrates conversational query rewriting.
    #
    # Previous:
    # User: How many annual leave days do employees get?
    #
    # Follow-up:
    # What about interns?
    #
    # Rewriter should convert it into a standalone query.
    # --------------------------------------------------------

    conversation_history = [

        "User: How many annual leave days do employees get?",

        "Assistant: Eligible full-time employees receive "
        "18 days of annual leave per calendar year."

    ]


    # --------------------------------------------------------
    # GET USER QUESTION
    # --------------------------------------------------------

    query = input(
        "\nEnter your question: "
    ).strip()


    # --------------------------------------------------------
    # VALIDATE QUESTION
    # --------------------------------------------------------

    if not query:

        print(
            "\nPlease enter a valid question."
        )

        return


    # --------------------------------------------------------
    # RUN RETRIEVAL PIPELINE
    # --------------------------------------------------------

    (
        rewritten_query,
        results,
        abstained
    ) = retrieve_evidence(

        query,

        conversation_history,

        chunks,

        embedding_model,

        client,

        bm25
    )


    # --------------------------------------------------------
    # ABSTAIN WHEN NO VERIFIED EVIDENCE SUPPORTS THE QUESTION
    # --------------------------------------------------------

    if abstained:

        print("\n" + "=" * 75)
        print("                         FINAL ANSWER")
        print("=" * 75)

        print("\nQuestion:")
        print(query)

        print("\nAnswer:")
        print(
            "The information is not available in the "
            "provided documents."
        )

        print("\n\nSOURCES")
        print("-" * 50)
        print("No verified supporting sources available.")

        print("\n" + "=" * 75)

        return


    # --------------------------------------------------------
    # DISPLAY VERIFIED EVIDENCE
    # --------------------------------------------------------

    display_results(

        query,

        rewritten_query,

        results

    )


    # --------------------------------------------------------
    # GENERATE GROUNDED ANSWER
    # --------------------------------------------------------

    print(
        "\n[6/6] Generating grounded answer..."
    )

    answer = generate_answer(

        rewritten_query,

        results

    )


    # --------------------------------------------------------
    # DISPLAY FINAL ANSWER
    # --------------------------------------------------------

    print("\n")

    print("=" * 75)

    print(
        "                         FINAL ANSWER"
    )

    print("=" * 75)


    print("\nQuestion:")

    print(query)


    print("\nAnswer:")

    print(answer)


    print("\n" + "=" * 75)

    print(
        "Complete RAG pipeline finished successfully."
    )

    print("=" * 75)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
