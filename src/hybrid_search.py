from pathlib import Path
import json
import re

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from rank_bm25 import BM25Okapi


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"

COLLECTION_NAME = "enterprise_documents"

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():
    """
    Load processed document chunks from chunks.jsonl.
    """

    chunks = []

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:
            chunks.append(json.loads(line))

    return chunks


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):
    """
    Convert text into lowercase word tokens for BM25.
    """

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


# ============================================================
# BUILD BM25
# ============================================================

def build_bm25(chunks):
    """
    Build a BM25 index over the supplied chunks.
    """

    tokenized_documents = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(
        tokenized_documents
    )


# ============================================================
# METADATA FILTERING
# ============================================================

def filter_chunks_by_document(
    chunks,
    document_name=None
):
    """
    Filter chunks by document name.

    If document_name is None or "All Documents",
    all chunks are returned.

    Example:

        filter_chunks_by_document(
            chunks,
            "Leave_Policy"
        )
    """

    if not document_name:
        return chunks

    if document_name.lower() == "all documents":
        return chunks

    filtered_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("document_name") == document_name
    ]

    return filtered_chunks


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    query,
    model,
    client,
    top_k=10,
    document_name=None
):
    """
    Perform semantic vector search.

    When document_name is provided, Qdrant applies a
    metadata filter so that only chunks belonging to
    that document are considered.
    """

    query_embedding = model.encode(
        query
    ).tolist()

    # --------------------------------------------------------
    # Build Qdrant metadata filter
    # --------------------------------------------------------

    query_filter = None

    if (
        document_name
        and document_name.lower() != "all documents"
    ):

        query_filter = Filter(
            must=[
                FieldCondition(
                    key="document_name",
                    match=MatchValue(
                        value=document_name
                    )
                )
            ]
        )

    # --------------------------------------------------------
    # Run semantic search
    # --------------------------------------------------------

    results = client.query_points(

        collection_name=COLLECTION_NAME,

        query=query_embedding,

        limit=top_k,

        query_filter=query_filter,

        with_payload=True
    ).points

    return results


# ============================================================
# BM25 SEARCH
# ============================================================

def bm25_search(
    query,
    chunks,
    bm25,
    top_k=10,
    document_name=None
):
    """
    Perform BM25 keyword retrieval.

    Metadata filtering is applied before ranking so that
    chunks from other documents cannot enter the result set.
    """

    # --------------------------------------------------------
    # Build allowed chunk indexes
    # --------------------------------------------------------

    allowed_indexes = []

    for index, chunk in enumerate(chunks):

        if (
            not document_name
            or document_name.lower() == "all documents"
            or chunk.get("document_name") == document_name
        ):

            allowed_indexes.append(index)

    # --------------------------------------------------------
    # No matching document
    # --------------------------------------------------------

    if not allowed_indexes:

        return []

    # --------------------------------------------------------
    # Tokenize query
    # --------------------------------------------------------

    query_tokens = tokenize(query)

    # --------------------------------------------------------
    # BM25 scores for all chunks
    # --------------------------------------------------------

    scores = bm25.get_scores(
        query_tokens
    )

    # --------------------------------------------------------
    # Keep only chunks allowed by metadata filter
    # --------------------------------------------------------

    ranked_indexes = sorted(

        allowed_indexes,

        key=lambda index: scores[index],

        reverse=True

    )[:top_k]

    # --------------------------------------------------------
    # Build BM25 result dictionaries
    # --------------------------------------------------------

    return [

        {
            "index": index,
            "score": float(scores[index]),
            "chunk": chunks[index]
        }

        for index in ranked_indexes
    ]


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(
    semantic_results,
    bm25_results,
    chunks,
    k=60
):
    """
    Combine semantic and BM25 rankings using
    Reciprocal Rank Fusion.

    RRF score:

        1 / (k + rank)

    A chunk appearing in both rankings receives
    contributions from both retrieval methods.
    """

    scores = {}

    # --------------------------------------------------------
    # Semantic ranking
    # --------------------------------------------------------

    for rank, result in enumerate(
        semantic_results,
        start=1
    ):

        chunk_id = result.payload["chunk_id"]

        scores.setdefault(

            chunk_id,

            {
                "score": 0,
                "chunk": result.payload
            }
        )

        scores[chunk_id]["score"] += (
            1 / (k + rank)
        )

    # --------------------------------------------------------
    # BM25 ranking
    # --------------------------------------------------------

    for rank, result in enumerate(
        bm25_results,
        start=1
    ):

        chunk_id = result["chunk"]["chunk_id"]

        scores.setdefault(

            chunk_id,

            {
                "score": 0,
                "chunk": result["chunk"]
            }
        )

        scores[chunk_id]["score"] += (
            1 / (k + rank)
        )

    # --------------------------------------------------------
    # Sort by combined RRF score
    # --------------------------------------------------------

    ranked_results = sorted(

        scores.values(),

        key=lambda item: item["score"],

        reverse=True
    )

    return ranked_results


# ============================================================
# AVAILABLE DOCUMENTS
# ============================================================

def get_available_documents(chunks):
    """
    Return unique document names available in the
    knowledge base.

    Used later by the Streamlit document filter.
    """

    documents = sorted(
        {
            chunk.get("document_name")
            for chunk in chunks
            if chunk.get("document_name")
        }
    )

    return documents


# ============================================================
# COMMAND-LINE DEMONSTRATION
# ============================================================

def main():

    print("=" * 60)
    print("ENTERPRISE HYBRID SEARCH")
    print("=" * 60)

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    print("\nLoading chunks...")

    chunks = load_chunks()

    print(
        f"Loaded {len(chunks)} chunks."
    )

    # --------------------------------------------------------
    # Show available documents
    # --------------------------------------------------------

    documents = get_available_documents(
        chunks
    )

    print("\nAvailable documents:")

    print("0. All Documents")

    for index, document in enumerate(
        documents,
        start=1
    ):

        print(
            f"{index}. {document}"
        )

    # --------------------------------------------------------
    # Document filter
    # --------------------------------------------------------

    document_choice = input(
        "\nEnter document number "
        "(0 for all documents): "
    ).strip()

    document_name = None

    if document_choice != "0":

        try:

            document_index = int(
                document_choice
            )

            if (
                1 <= document_index
                <= len(documents)
            ):

                document_name = documents[
                    document_index - 1
                ]

            else:

                print(
                    "\nInvalid document number."
                )

                return

        except ValueError:

            print(
                "\nInvalid document selection."
            )

            return

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print(
        "\nLoading embedding model..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # Connect to Qdrant
    # --------------------------------------------------------

    client = QdrantClient(
        path=str(VECTORSTORE_DIR)
    )

    # --------------------------------------------------------
    # Build BM25
    # --------------------------------------------------------

    print(
        "Building BM25 index..."
    )

    bm25 = build_bm25(
        chunks
    )

    # --------------------------------------------------------
    # Get question
    # --------------------------------------------------------

    query = input(
        "\nEnter your question: "
    ).strip()

    if not query:

        print(
            "\nPlease enter a question."
        )

        return

    # --------------------------------------------------------
    # Display selected filter
    # --------------------------------------------------------

    print(
        "\nDocument filter:"
    )

    print(
        document_name
        if document_name
        else "All Documents"
    )

    # --------------------------------------------------------
    # Semantic search
    # --------------------------------------------------------

    print(
        "\nRunning semantic search..."
    )

    semantic_results = semantic_search(

        query,

        model,

        client,

        top_k=10,

        document_name=document_name
    )

    print(
        f"Semantic search returned "
        f"{len(semantic_results)} results."
    )

    # --------------------------------------------------------
    # BM25 search
    # --------------------------------------------------------

    print(
        "Running BM25 search..."
    )

    bm25_results = bm25_search(

        query,

        chunks,

        bm25,

        top_k=10,

        document_name=document_name
    )

    print(
        f"BM25 search returned "
        f"{len(bm25_results)} results."
    )

    # --------------------------------------------------------
    # Hybrid RRF
    # --------------------------------------------------------

    print(
        "Combining results using RRF..."
    )

    hybrid_results = reciprocal_rank_fusion(

        semantic_results,

        bm25_results,

        chunks
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "HYBRID SEARCH RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"\nActive document filter: "
        f"{document_name or 'All Documents'}"
    )

    for rank, result in enumerate(

        hybrid_results[:5],

        start=1
    ):

        chunk = result["chunk"]

        print(
            f"\nResult {rank}"
        )

        print(
            f"RRF Score: "
            f"{result['score']:.6f}"
        )

        print(
            f"Document: "
            f"{chunk.get('document_name')}"
        )

        print(
            f"Page: "
            f"{chunk.get('page_number')}"
        )

        print(
            f"Section: "
            f"{chunk.get('section')}"
        )

        print(
            f"Chunk ID: "
            f"{chunk.get('chunk_id')}"
        )

        print(
            f"Text: "
            f"{chunk.get('text')}"
        )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()