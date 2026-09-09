from __future__ import annotations

import json
import uuid
from pathlib import Path

from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient

from qdrant_client.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from ingest import extract_pdf
from chunker import split_into_chunks, detect_section


# ============================================================
# PROJECT SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"

COLLECTION_NAME = "enterprise_documents"

MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 120

CHUNK_OVERLAP = 30


# ============================================================
# PROCESS ONE PDF
# ============================================================

def process_uploaded_pdf(pdf_path: Path):
    """
    Extract and chunk a single PDF.

    Uses the existing ingestion and chunking logic so that
    uploaded documents follow the same processing pipeline
    as the original document collection.
    """

    pages = extract_pdf(pdf_path)

    document_name = pdf_path.stem

    chunks = []

    chunk_number = 1

    for page in pages:

        words = page["text"].split()

        page_chunks = split_into_chunks(
            words,
            CHUNK_SIZE,
            CHUNK_OVERLAP,
        )

        for chunk_text in page_chunks:

            chunk = {
                "chunk_id": (
                    f"{document_name}_{chunk_number:04d}"
                ),

                "document_name": document_name,

                "document_type": "HR Policy",

                "year": 2026,

                "page_number": page["page_number"],

                "section": detect_section(
                    page["text"]
                ),

                "text": chunk_text,
            }

            chunks.append(chunk)

            chunk_number += 1

    return chunks


# ============================================================
# UPDATE CHUNKS.JSONL
# ============================================================

def update_chunks_file(new_chunks):
    """
    Update the central chunks.jsonl file.

    Existing documents remain unchanged.

    If the uploaded document already exists,
    its old chunks are replaced.
    """

    existing_chunks = []

    if CHUNKS_FILE.exists():

        with open(
            CHUNKS_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                line = line.strip()

                if line:

                    existing_chunks.append(
                        json.loads(line)
                    )

    new_documents = {
        chunk["document_name"]
        for chunk in new_chunks
    }

    # --------------------------------------------------------
    # Remove old chunks for re-indexed documents
    # --------------------------------------------------------

    existing_chunks = [

        chunk

        for chunk in existing_chunks

        if chunk.get("document_name")
        not in new_documents

    ]

    # --------------------------------------------------------
    # Add new chunks
    # --------------------------------------------------------

    combined_chunks = (
        existing_chunks
        + new_chunks
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        CHUNKS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        for chunk in combined_chunks:

            file.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False,
                )
            )

            file.write("\n")

    return len(combined_chunks)


# ============================================================
# ADD / REPLACE DOCUMENT IN QDRANT
# ============================================================

def add_chunks_to_qdrant(chunks, client=None):
    """
    Add a document to the existing Qdrant collection.

    If the same document already exists, its previous
    vectors are removed first.

    This prevents stale or duplicate document vectors.
    """

    if not chunks:

        return 0

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    document_name = chunks[0]["document_name"]

    print(
        f"Preparing Qdrant index for: "
        f"{document_name}"
    )

    # Streamlit keeps one cached embedded-Qdrant client open. Reusing it is
    # mandatory because a second QdrantClient(path=...) would lock the same
    # local storage folder. A standalone CLI call can still create its own.
    if client is None:
        client = QdrantClient(path=str(VECTORSTORE_DIR))

    # ========================================================
    # REMOVE OLD VECTORS FOR THIS DOCUMENT
    # ========================================================

    print(
        "Removing previous vectors for this document..."
    )

    client.delete(
        collection_name=COLLECTION_NAME,

        points_selector=Filter(

            must=[

                FieldCondition(

                    key="document_name",

                    match=MatchValue(
                        value=document_name
                    ),

                )

            ]

        ),
    )

    # ========================================================
    # LOAD EMBEDDING MODEL
    # ========================================================

    print(
        f"Loading embedding model: "
        f"{MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    # ========================================================
    # CREATE EMBEDDINGS
    # ========================================================

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        f"Creating embeddings for "
        f"{len(texts)} chunks..."
    )

    embeddings = model.encode(
        texts,
        show_progress_bar=False,
    )

    points = []

    # ========================================================
    # BUILD QDRANT POINTS
    # ========================================================

    for index, (
        chunk,
        embedding,
    ) in enumerate(
        zip(chunks, embeddings),
    ):

        points.append(

            PointStruct(

                # Stable IDs prevent replacement from overwriting an
                # unrelated vector after deletions leave ID gaps.
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"])),

                vector=embedding.tolist(),

                payload=chunk,

            )

        )

    # ========================================================
    # UPLOAD VECTORS
    # ========================================================

    print(
        "Uploading vectors to Qdrant..."
    )

    client.upsert(

        collection_name=COLLECTION_NAME,

        points=points,

    )

    print(
        f"Successfully indexed "
        f"{len(points)} vectors."
    )

    return len(points)


# ============================================================
# REINDEX ONE DOCUMENT
# ============================================================

def reindex_document(pdf_path, qdrant_client=None):
    """
    Complete document re-indexing workflow.

    PDF
      ↓
    Extraction
      ↓
    Cleaning
      ↓
    Chunking
      ↓
    chunks.jsonl
      ↓
    Embeddings
      ↓
    Qdrant
    """

    pdf_path = Path(pdf_path)

    # ========================================================
    # VALIDATE FILE
    # ========================================================

    if not pdf_path.exists():

        raise FileNotFoundError(
            f"Document not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":

        raise ValueError(
            "Only PDF documents are currently supported."
        )

    print(
        "\n"
        + "=" * 60
    )

    print(
        f"RE-INDEXING DOCUMENT: "
        f"{pdf_path.name}"
    )

    print(
        "=" * 60
    )

    # ========================================================
    # STEP 1 — EXTRACTION + CHUNKING
    # ========================================================

    print(
        "\n[1/4] Extracting and chunking document..."
    )

    chunks = process_uploaded_pdf(
        pdf_path
    )

    if not chunks:

        raise ValueError(
            "No usable text was extracted "
            "from the document."
        )

    print(
        f"Chunks created: {len(chunks)}"
    )

    # ========================================================
    # STEP 2 — UPDATE CHUNKS.JSONL
    # ========================================================

    print(
        "\n[2/4] Updating chunks.jsonl..."
    )

    total_chunks = update_chunks_file(
        chunks
    )

    print(
        f"Total knowledge-base chunks: "
        f"{total_chunks}"
    )

    # ========================================================
    # STEP 3 — EMBEDDINGS + QDRANT
    # ========================================================

    print(
        "\n[3/4] Creating embeddings and "
        "updating Qdrant..."
    )

    indexed_count = add_chunks_to_qdrant(
        chunks,
        client=qdrant_client,
    )

    print(
        f"New vectors indexed: "
        f"{indexed_count}"
    )

    # ========================================================
    # STEP 4 — COMPLETE
    # ========================================================

    print(
        "\n[4/4] Re-indexing completed."
    )

    print(
        "=" * 60
    )

    return {

        "document_name":
            pdf_path.stem,

        "chunks_created":
            len(chunks),

        "chunks_indexed":
            indexed_count,

        "total_chunks":
            total_chunks,

    }
