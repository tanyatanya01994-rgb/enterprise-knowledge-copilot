"""Local tests for multi-document safety, metadata filtering, and memory."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from confidence import calculate_confidence
from hybrid_search import bm25_search, build_bm25
from query_rewriter import _local_rewrite
from rag_pipeline import is_multi_document_query


def result(chunk_id, document):
    return {"rerank_score": 2.0, "chunk": {"chunk_id": chunk_id, "document_name": document, "text": "annual leave remote work policy"}}


class RagCapabilityTests(unittest.TestCase):
    def test_detects_comparison_patterns_without_document_names(self):
        for query in (
            "Compare annual leave with work from home.",
            "What is the difference between these policies?",
            "How do the two policies differ?",
            "Benefits versus leave provisions",
            "What applies across policies?",
        ):
            self.assertTrue(is_multi_document_query(query), query)
        self.assertFalse(is_multi_document_query("What is the annual leave entitlement?"))

    def test_multi_document_confidence_requires_two_verified_documents(self):
        one = [result("leave-1", "Leave_Policy")]
        insufficient = calculate_confidence(one, one, multi_document_query=True)
        self.assertFalse(insufficient["coverage_sufficient"])
        self.assertLess(insufficient["score"], 45)

        two = one + [result("wfh-1", "Work_From_Home_Policy")]
        sufficient = calculate_confidence(two, two, multi_document_query=True)
        self.assertTrue(sufficient["coverage_sufficient"])
        self.assertEqual(sufficient["covered_documents"], ["Leave_Policy", "Work_From_Home_Policy"])

    def test_document_filter_limits_bm25_candidates(self):
        chunks = [
            {"chunk_id": "leave", "document_name": "Leave", "text": "annual leave entitlement"},
            {"chunk_id": "wfh", "document_name": "WFH", "text": "remote work approval"},
        ]
        results = bm25_search("annual leave", chunks, build_bm25(chunks), document_name="WFH")
        self.assertEqual([item["chunk"]["document_name"] for item in results], ["WFH"])

    def test_local_follow_up_rewrite_is_bounded_and_does_not_add_facts(self):
        history = ["User: How many annual leave days do employees get?"]
        rewritten = _local_rewrite("What about interns?", history)
        self.assertIn("annual leave", rewritten.lower())
        self.assertIn("interns", rewritten.lower())
        self.assertNotIn("18 days", rewritten.lower())


if __name__ == "__main__":
    unittest.main()
