from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_pipeline import initialize_retrieval, retrieve_evidence
import evidence_verifier


QUESTIONS_FILE = PROJECT_ROOT / "evaluation" / "questions.json"
RESULTS_FILE = PROJECT_ROOT / "evaluation" / "results.json"


def main():
    print("=" * 70)
    print("ENTERPRISE KNOWLEDGE INTELLIGENCE COPILOT")
    print("AUTOMATED RAG EVALUATION")
    print("=" * 70)

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        questions = json.load(file)

    print(f"\nLoaded {len(questions)} evaluation questions.")

    print("\nInitializing RAG engine...")
    chunks, embedding_model, qdrant_client, bm25 = initialize_retrieval()
    print("RAG engine ready.\n")

    results = []

    for index, item in enumerate(questions, start=1):

        question = item["question"]
        question_type = item["type"]
        history = item.get("history", [])

        print(f"[{index}/{len(questions)}] {question}")

        try:

            (
                rewritten_query,
                verified_results,
                abstained,
                confidence,
                verification_status,
            ) = retrieve_evidence(
                question,
                history,
                chunks,
                embedding_model,
                qdrant_client,
                bm25,
                None,
            )

            # ----------------------------------------------------------
            # IMPORTANT:
            # An abstention does NOT automatically mean the question
            # was unsupported. The verifier/API may also have failed.
            # We record the observed outcome clearly.
            # ----------------------------------------------------------

            if verification_status == "verifier_unavailable":
                outcome = "verifier_api_error"
            elif abstained:
                outcome = "unsupported_abstention"
            else:
                outcome = "answered"

            result = {
                "id": item["id"],
                "type": question_type,
                "question": question,
                "rewritten_query": rewritten_query,
                "verified_evidence": len(verified_results),
                "confidence_score": confidence["score"],
                "confidence_level": confidence["level"],
                "abstained": abstained,
                "verifier_status": verification_status,
                "outcome": outcome,
            }

            results.append(result)

            print(
                f"    Evidence: {len(verified_results)} | "
                f"Confidence: {confidence['score'] if confidence['score'] is not None else 'Unavailable'} | "
                f"Abstained: {abstained} | "
                f"Outcome: {outcome}"
            )

        except Exception as error:

            print(
                f"    ERROR: {type(error).__name__}: {error}"
            )

            results.append(
                {
                    "id": item["id"],
                    "type": question_type,
                    "question": question,
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "outcome": "evaluation_error",
                }
            )

    # ------------------------------------------------------------------
    # Save detailed results
    # ------------------------------------------------------------------

    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------------
    # Evaluation summary
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    successful = [
        result
        for result in results
        if "error" not in result
    ]

    evaluation_errors = [
        result
        for result in results
        if result.get("outcome") == "evaluation_error"
    ]

    answered = [
        result
        for result in successful
        if result.get("outcome") == "answered"
    ]

    abstained = [
        result
        for result in successful
        if result.get("outcome") == "unsupported_abstention"
    ]

    supported_questions = [
        result
        for result in successful
        if result["type"] in {
            "supported",
            "follow_up",
            "multi_document",
        }
    ]

    unsupported_questions = [
        result
        for result in successful
        if result["type"] == "unsupported"
    ]

    supported_answered = [
        result
        for result in supported_questions
        if result.get("outcome") == "answered"
    ]

    unsupported_abstained = [
        result
        for result in unsupported_questions
        if result.get("outcome") == "unsupported_abstention"
    ]

    print(f"\nTotal questions: {len(results)}")
    print(f"Completed evaluations: {len(successful)}")
    print(f"Evaluation errors: {len(evaluation_errors)}")

    print("\n--- Retrieval / Answering ---")

    print(f"Answered with verified evidence: {len(answered)}")
    print(f"Honest abstentions: {len(abstained)}")
    print(f"Verifier/API errors: {sum(1 for result in successful if result.get('outcome') == 'verifier_api_error')}")

    print("\n--- Supported Questions ---")

    print(
        f"Supported questions: {len(supported_questions)}"
    )

    print(
        f"Supported questions answered: "
        f"{len(supported_answered)}"
    )

    if supported_questions:

        supported_answer_rate = (
            len(supported_answered)
            / len(supported_questions)
            * 100
        )

        print(
            f"Supported answer rate: "
            f"{supported_answer_rate:.1f}%"
        )

    print("\n--- Unsupported Questions ---")

    print(
        f"Unsupported questions: "
        f"{len(unsupported_questions)}"
    )

    if unsupported_questions:

        abstention_rate = (
            len(unsupported_abstained)
            / len(unsupported_questions)
            * 100
        )

        print(
            f"Unsupported abstention rate: "
            f"{abstention_rate:.1f}%"
        )

    print("\n--- Confidence ---")

    confidence_values = [
        result["confidence_score"]
        for result in supported_answered
        if isinstance(result.get("confidence_score"), (int, float))
    ]

    if confidence_values:

        average_confidence = (
            sum(confidence_values)
            / len(confidence_values)
        )

        print(
            f"Average confidence on answered supported "
            f"questions: {average_confidence:.1f}%"
        )

    else:

        print(
            "Average confidence: unavailable "
            "(no supported questions were successfully answered)"
        )

    print("\n--- Important Evaluation Note ---")

    if evaluation_errors:

        print(
            "Some questions could not be evaluated because "
            "of runtime/API errors."
        )

    print("Verifier/API failures are recorded separately and are not counted as abstentions.")

    print(
        f"\nDetailed results saved to:\n{RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()
