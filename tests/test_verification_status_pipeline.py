"""Status propagation tests that never call Gemini or Qdrant."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import rag_pipeline


class VerificationStatusPipelineTests(unittest.TestCase):
    def test_verifier_unavailable_propagates_and_stays_fail_closed(self):
        candidate = {"chunk": {"chunk_id": "leave-1", "text": "18 days annual leave"}}
        with patch.object(rag_pipeline, "rewrite_query", return_value="annual leave"), \
             patch.object(rag_pipeline, "semantic_search", return_value=[]), \
             patch.object(rag_pipeline, "bm25_search", return_value=[]), \
             patch.object(rag_pipeline, "reciprocal_rank_fusion", return_value=[candidate]), \
             patch.object(rag_pipeline, "rerank_results", return_value=[candidate]), \
             patch.object(rag_pipeline, "verify_evidence", return_value=([], "verifier_unavailable")):
            _query, verified, abstained, confidence, status = rag_pipeline.retrieve_evidence(
                "How much annual leave?", [], [], object(), object(), object()
            )

        self.assertEqual(status, "verifier_unavailable")
        self.assertEqual(verified, [])
        self.assertTrue(abstained)
        self.assertIsNone(confidence["score"])
        self.assertEqual(confidence["level"], "Unavailable")


if __name__ == "__main__":
    unittest.main()
