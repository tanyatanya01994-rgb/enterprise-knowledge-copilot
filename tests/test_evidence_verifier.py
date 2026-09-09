"""Unit tests for the fail-closed evidence verifier."""

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import evidence_verifier


class FakeResponse:

    def __init__(self, text):
        self.text = text


class FakeModels:

    def __init__(self, responses):
        self.responses = iter(responses)

    def generate_content(self, **_kwargs):
        response = next(self.responses)

        if isinstance(response, Exception):
            raise response

        return FakeResponse(response)


class FakeClient:

    def __init__(self, responses):
        self.models = FakeModels(responses)


def make_result(chunk_id, text):
    return {
        "rrf_score": 0.02,
        "rerank_score": 1.5,
        "chunk": {
            "chunk_id": chunk_id,
            "document_name": "Leave_Policy",
            "document_type": "HR Policy",
            "year": 2026,
            "page_number": 1,
            "section": "1. Annual Leave",
            "text": text
        }
    }


class EvidenceVerifierTests(unittest.TestCase):

    def setUp(self):
        self.original_client = evidence_verifier.client
        self.results = [
            make_result(
                "Leave_Policy_0001",
                "Eligible full-time employees receive 18 days "
                "of annual leave per calendar year."
            ),
            make_result(
                "Attendance_Policy_0001",
                "Employees must record attendance accurately."
            )
        ]

    def tearDown(self):
        evidence_verifier.client = self.original_client

    def set_responses(self, responses):
        evidence_verifier.client = FakeClient(responses)

    def test_direct_support_returns_original_result(self):
        self.set_responses([
            '{"supported": true, "supporting_chunk_ids": '
            '["Leave_Policy_0001"], "abstain": false}'
        ])

        verified, status = evidence_verifier.verify_evidence(
            "How many annual leave days do employees get?",
            self.results
        )

        self.assertEqual(status, "verified")
        self.assertEqual(len(verified), 1)
        self.assertIs(verified[0], self.results[0])

    def test_unsupported_question_abstains(self):
        self.set_responses([
            '{"supported": false, "supporting_chunk_ids": [], '
            '"abstain": true}'
        ])

        verified, status = evidence_verifier.verify_evidence(
            "What is the company's current stock price?",
            self.results
        )

        self.assertEqual(verified, [])
        self.assertEqual(status, "unsupported")

    def test_invalid_chunk_id_abstains(self):
        self.set_responses([
            '{"supported": true, "supporting_chunk_ids": '
            '["Leave_Policy_9999"], "abstain": false}'
        ])

        verified, status = evidence_verifier.verify_evidence(
            "How many annual leave days do employees get?",
            self.results
        )

        self.assertEqual(verified, [])
        self.assertEqual(status, "unsupported")

    def test_malformed_json_abstains(self):
        self.set_responses(["this is not JSON"])

        verified, status = evidence_verifier.verify_evidence(
            "How many annual leave days do employees get?",
            self.results
        )

        self.assertEqual(verified, [])
        self.assertEqual(status, "unsupported")

    def test_api_failure_is_verifier_unavailable_and_fails_closed(self):
        self.set_responses([
            RuntimeError("first model unavailable"),
            RuntimeError("second model unavailable"),
            RuntimeError("third model unavailable")
        ])

        verified, status = evidence_verifier.verify_evidence(
            "How many annual leave days do employees get?",
            self.results
        )

        self.assertEqual(verified, [])
        self.assertEqual(status, "verifier_unavailable")

    def test_follow_up_query_without_direct_support_abstains(self):
        self.set_responses([
            '{"supported": false, "supporting_chunk_ids": [], '
            '"abstain": true}'
        ])

        verified, status = evidence_verifier.verify_evidence(
            "How many annual leave days do interns get?",
            self.results
        )

        self.assertEqual(verified, [])
        self.assertEqual(status, "unsupported")


if __name__ == "__main__":
    unittest.main()
