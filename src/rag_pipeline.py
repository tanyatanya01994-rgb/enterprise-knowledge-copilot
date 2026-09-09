from pathlib import Path
import re

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from query_rewriter import rewrite_query
from hybrid_search import (
    load_chunks,
    build_bm25,
    semantic_search,
    bm25_search,
    reciprocal_rank_fusion,
    get_available_documents,
)

from reranker import rerank_results
from evidence_verifier import VERIFIED, verify_evidence
from confidence import calculate_confidence


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

COLLECTION_NAME = "enterprise_documents"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Minimum confidence required before generating an answer.
CONFIDENCE_THRESHOLD = 45.0


def is_multi_document_query(query: str) -> bool:
    """Detect questions that explicitly require comparing or joining sources."""
    normalized = " ".join(query.lower().split())
    patterns = (
        r"\bcompare\b.*\b(with|and|to|versus|vs\.? )\b",
        r"\bdifference between\b",
        r"\bhow do\b.*\bdiffer\b",
        r"\bversus\b|\bvs\.?\b",
        r"\brelationship between\b",
        r"\bacross\s+(?:the\s+)?(?:policies|documents)\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def covered_documents(results):
    """Return deterministic source-document coverage for verified evidence."""
    return sorted({
        str(result.get("chunk", {}).get("document_name"))
        for result in results
        if result.get("chunk", {}).get("document_name")
    })


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
        bm25,
    )


# ============================================================
# DOCUMENT FILTER
# ============================================================

def select_document_filter(chunks):

    documents = get_available_documents(
        chunks
    )

    print("\n" + "=" * 75)
    print("DOCUMENT FILTER")
    print("=" * 75)

    print("\n0. All Documents")

    for index, document in enumerate(
        documents,
        start=1
    ):
        print(
            f"{index}. {document}"
        )

    while True:

        choice = input(
            "\nSelect document "
            "(0 for all documents): "
        ).strip()

        if choice == "0":

            print(
                "\nActive document filter: "
                "All Documents"
            )

            return None

        try:

            index = int(choice)

        except ValueError:

            print(
                "\nInvalid selection. "
                "Please enter a number."
            )

            continue

        if 1 <= index <= len(documents):

            selected_document = documents[
                index - 1
            ]

            print(
                "\nActive document filter: "
                f"{selected_document}"
            )

            return selected_document

        print(
            "\nInvalid document number. "
            "Please try again."
        )


# ============================================================
# RETRIEVE AND VERIFY EVIDENCE
# ============================================================

def retrieve_evidence(
    query,
    conversation_history,
    chunks,
    embedding_model,
    client,
    bm25,
    document_name=None,
):

    multi_document_query = is_multi_document_query(query)
    # An explicit sidebar filter is authoritative. With All Documents, a
    # comparison gets a larger candidate pool so relevant policies can survive
    # fusion and reranking together.
    candidate_limit = 15 if multi_document_query and not document_name else 10
    rerank_limit = 8 if multi_document_query and not document_name else 5

    # --------------------------------------------------------
    # STEP 1: QUERY REWRITING
    # --------------------------------------------------------

    print("\n[1/7] Rewriting query...")

    rewritten_query = rewrite_query(
        query,
        conversation_history,
    )

    print("\nOriginal query:")
    print(query)

    print("\nRewritten search query:")
    print(rewritten_query)

    # --------------------------------------------------------
    # STEP 2: SEMANTIC SEARCH
    # --------------------------------------------------------

    print("\n[2/7] Running semantic search...")

    semantic_results = semantic_search(
        rewritten_query,
        embedding_model,
        client,
        top_k=candidate_limit,
        document_name=document_name,
    )

    print(
        f"Semantic search returned "
        f"{len(semantic_results)} results."
    )

    # --------------------------------------------------------
    # STEP 3: BM25 SEARCH
    # --------------------------------------------------------

    print("\n[3/7] Running BM25 search...")

    bm25_results = bm25_search(
        rewritten_query,
        chunks,
        bm25,
        top_k=candidate_limit,
        document_name=document_name,
    )

    print(
        f"BM25 search returned "
        f"{len(bm25_results)} results."
    )

    # --------------------------------------------------------
    # STEP 4: HYBRID SEARCH + RRF
    # --------------------------------------------------------

    print(
        "\n[4/7] Combining semantic + BM25 results..."
    )

    hybrid_results = reciprocal_rank_fusion(
        semantic_results,
        bm25_results,
        chunks,
    )

    hybrid_results = hybrid_results[:candidate_limit]

    print(
        f"Hybrid retrieval candidates: "
        f"{len(hybrid_results)}"
    )

    # --------------------------------------------------------
    # STEP 5: RERANKING
    # --------------------------------------------------------

    print("\n[5/7] Reranking results...")

    reranked_results = rerank_results(
        rewritten_query,
        hybrid_results,
        top_k=rerank_limit,
    )

    print(
        f"Reranked evidence chunks: "
        f"{len(reranked_results)}"
    )

    # --------------------------------------------------------
    # STEP 6: EVIDENCE VERIFICATION
    # --------------------------------------------------------

    print(
        "\n[6/7] Verifying evidence support..."
    )

    verified_results, verification_status = verify_evidence(
        rewritten_query,
        reranked_results,
    )

    print(
        f"Verified evidence chunks: "
        f"{len(verified_results)}"
    )

    # --------------------------------------------------------
    # STEP 7: RETRIEVAL CONFIDENCE SCORING
    # --------------------------------------------------------

    confidence = calculate_confidence(
        reranked_results,
        verified_results,
        verification_status,
        multi_document_query=multi_document_query,
    )

    print(
        f"Retrieval Confidence: "
        f"{confidence['score'] if confidence['score'] is not None else 'Unavailable'} - "
        f"{confidence['level']}"
    )

    # --------------------------------------------------------
    # APPLY CONFIDENCE THRESHOLD
    # --------------------------------------------------------

    confidence_abstained = (
        verification_status == VERIFIED
        and confidence["score"] < CONFIDENCE_THRESHOLD
    )

    coverage_abstained = (
        multi_document_query
        and verification_status == VERIFIED
        and not confidence["coverage_sufficient"]
    )

    final_abstained = (
        verification_status != VERIFIED or confidence_abstained or coverage_abstained
    )

    print(
        f"Confidence Threshold: "
        f"{CONFIDENCE_THRESHOLD:.1f}%"
    )

    print(
        f"Abstained: "
        f"{'Yes' if final_abstained else 'No'}"
    )

    return (
        rewritten_query,
        verified_results,
        final_abstained,
        confidence,
        verification_status,
    )


# ============================================================
# DISPLAY RETRIEVAL RESULTS
# ============================================================

def display_results(
    original_query,
    rewritten_query,
    results,
    confidence,
    document_name=None,
):

    print("\n")
    print("=" * 75)
    print("                    RETRIEVAL RESULTS")
    print("=" * 75)

    print("\nOriginal Question:")
    print(original_query)

    print("\nRewritten Search Query:")
    print(rewritten_query)

    print("\nDocument Filter:")
    print(
        document_name
        if document_name
        else "All Documents"
    )

    print("\nRetrieval Confidence:")
    print(
        f"{confidence['score']:.1f}% - "
        f"{confidence['level']}"
    )

    print(
        f"Verified Evidence: "
        f"{confidence['verified_count']} / "
        f"{confidence['reranked_count']}"
    )

    print("\n" + "-" * 75)
    print("VERIFIED TOP EVIDENCE")
    print("-" * 75)

    for rank, result in enumerate(
        results,
        start=1,
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
        bm25,
    ) = initialize_retrieval()

    # --------------------------------------------------------
    # DOCUMENT FILTER
    # --------------------------------------------------------

    document_name = select_document_filter(
        chunks
    )

    # --------------------------------------------------------
    # SAMPLE CONVERSATION HISTORY
    # --------------------------------------------------------
    #
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
        "18 days of annual leave per calendar year.",

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
        abstained,
        confidence,
        verification_status,
    ) = retrieve_evidence(

        query,

        conversation_history,

        chunks,

        embedding_model,

        client,

        bm25,

        document_name,
    )

    # --------------------------------------------------------
    # ABSTAIN WHEN EVIDENCE IS MISSING
    # OR CONFIDENCE IS LOW
    # --------------------------------------------------------

    if verification_status == "verifier_unavailable":

        print("\nEvidence verification is temporarily unavailable.")
        print("No unverified answer was generated.")
        return

    if abstained:

        print("\n" + "=" * 75)
        print("                         FINAL ANSWER")
        print("=" * 75)

        print("\nQuestion:")
        print(query)

        print("\nDocument Filter:")
        print(
            document_name
            if document_name
            else "All Documents"
        )

        print("\nAnswer:")

        print(
            "The information is not available in the "
            "provided documents with sufficient confidence."
        )

        print(
            f"\nRetrieval Confidence: "
            f"{confidence['score']:.1f}% - "
            f"{confidence['level']}"
        )

        print(
            f"Required Confidence: "
            f"{CONFIDENCE_THRESHOLD:.1f}%"
        )

        print("\n\nSOURCES")
        print("-" * 50)

        print(
            "No verified supporting sources available."
        )

        print("\n" + "=" * 75)

        return

    # --------------------------------------------------------
    # DISPLAY VERIFIED EVIDENCE
    # --------------------------------------------------------

    display_results(

        query,

        rewritten_query,

        results,

        confidence,

        document_name,
    )

    # --------------------------------------------------------
    # GENERATE GROUNDED ANSWER
    # --------------------------------------------------------

    print(
        "\n[7/7] Generating grounded answer..."
    )

    answer = generate_answer(

        rewritten_query,

        results,
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

    print("\nDocument Filter:")

    print(
        document_name
        if document_name
        else "All Documents"
    )

    print("\nAnswer:")

    print(answer)

    print(
        f"\nRetrieval Confidence: "
        f"{confidence['score']:.1f}% - "
        f"{confidence['level']}"
    )

    print(
        f"Verified Evidence Chunks: "
        f"{confidence['verified_count']}"
    )

    # --------------------------------------------------------
    # SOURCE ATTRIBUTION
    # --------------------------------------------------------

    print("\nSOURCES")

    print("-" * 75)

    seen_sources = set()

    for result in results:

        chunk = result["chunk"]

        source = (
            chunk.get("document_name"),
            chunk.get("page_number"),
            chunk.get("section"),
        )

        if source in seen_sources:
            continue

        seen_sources.add(source)

        print(
            f"{len(seen_sources)}. "
            f"{source[0]} | "
            f"Page {source[1]} | "
            f"Section: {source[2]}"
        )

    if not seen_sources:

        print(
            "No verified supporting sources available."
        )

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
