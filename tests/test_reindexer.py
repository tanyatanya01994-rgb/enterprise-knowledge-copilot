"""Local, Gemini-free tests for incremental Qdrant re-indexing."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import reindexer


class FakeEmbeddingModel:
    def __init__(self, *_args, **_kwargs):
        pass

    def encode(self, texts, **_kwargs):
        return np.array([[0.1, 0.2] for _ in texts])


class RecordingClient:
    def __init__(self):
        self.deleted = []
        self.upserts = []

    def delete(self, **kwargs):
        self.deleted.append(kwargs)

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)


def sample_chunk(chunk_id="Remote_Work_Security_Policy_0001"):
    return {
        "chunk_id": chunk_id,
        "document_name": "Remote_Work_Security_Policy",
        "document_type": "HR Policy",
        "year": 2026,
        "page_number": 1,
        "section": "Security",
        "text": "Use approved systems for remote work.",
    }


class ReindexerClientReuseTests(unittest.TestCase):
    def test_add_chunks_reuses_supplied_client_without_opening_local_store(self):
        supplied_client = RecordingClient()

        with patch.object(reindexer, "QdrantClient", side_effect=AssertionError("second client opened")), \
             patch.object(reindexer, "SentenceTransformer", FakeEmbeddingModel):
            indexed = reindexer.add_chunks_to_qdrant([sample_chunk()], client=supplied_client)

        self.assertEqual(indexed, 1)
        self.assertEqual(len(supplied_client.deleted), 1)
        self.assertEqual(len(supplied_client.upserts), 1)
        point = supplied_client.upserts[0]["points"][0]
        self.assertIsInstance(point.id, str)
        self.assertEqual(point.payload["document_name"], "Remote_Work_Security_Policy")

    def test_reindex_document_passes_supplied_client_to_vector_update(self):
        supplied_client = RecordingClient()
        chunks = [sample_chunk()]
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "Remote_Work_Security_Policy.pdf"
            pdf_path.write_bytes(b"%PDF-test")
            with patch.object(reindexer, "process_uploaded_pdf", return_value=chunks), \
                 patch.object(reindexer, "update_chunks_file", return_value=32), \
                 patch.object(reindexer, "add_chunks_to_qdrant", return_value=1) as add_chunks:
                result = reindexer.reindex_document(pdf_path, qdrant_client=supplied_client)

        self.assertEqual(result["chunks_indexed"], 1)
        add_chunks.assert_called_once_with(chunks, client=supplied_client)


if __name__ == "__main__":
    unittest.main()
