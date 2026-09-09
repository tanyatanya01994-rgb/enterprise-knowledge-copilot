"""Grounded Gemini answer generation over already verified evidence."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from google import genai

load_dotenv()
MODELS = ("gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.8-flash")


class AnswerGenerationError(RuntimeError):
    """Raised when no model can produce a grounded answer."""


def build_sources(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return distinct deterministic citations for UI or command-line use."""
    seen, sources = set(), []
    for result in evidence:
        chunk = result.get("chunk", {})
        source = (chunk.get("document_name", "Unknown document"), chunk.get("page_number", "?"), chunk.get("section", "Unknown section"))
        if source not in seen:
            seen.add(source)
            sources.append({"document": source[0], "page": source[1], "section": source[2]})
    return sources


def build_evidence_text(evidence: list[dict[str, Any]]) -> str:
    """Serialize only the verified chunks included in the generation prompt."""
    blocks = []
    for index, result in enumerate(evidence, 1):
        chunk = result.get("chunk", {})
        blocks.append(
            f"Evidence {index}\nDocument: {chunk.get('document_name', 'Unknown')}\n"
            f"Page: {chunk.get('page_number', '?')}\nSection: {chunk.get('section', 'Unknown')}\n"
            f"Text: {chunk.get('text', '')}"
        )
    return "\n\n".join(blocks)


def generate_answer(query: str, evidence: list[dict[str, Any]]) -> str:
    """Generate an answer solely from verified evidence; never call Gemini with none."""
    if not evidence:
        raise AnswerGenerationError("No verified evidence was supplied to answer generation.")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise AnswerGenerationError("Gemini is not configured (GEMINI_API_KEY is missing).")

    prompt = f"""You are an enterprise knowledge assistant. Answer using only the verified evidence.
Do not use outside knowledge or make unsupported inferences. If the evidence cannot answer the question, say so.
For comparisons, distinguish each source document. Do not include a Sources heading; the UI renders citations.

Question: {query}

Verified evidence:
{build_evidence_text(evidence)}"""
    client = genai.Client(api_key=api_key)
    errors = []
    for model_name in MODELS:
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            answer = getattr(response, "text", "").strip()
            if answer:
                print(f"[AnswerGenerator] Answer generated with {model_name}.")
                return answer
            errors.append(f"{model_name}: empty response")
        except Exception as error:
            errors.append(f"{model_name}: {type(error).__name__}")
            print(f"[AnswerGenerator] {model_name} unavailable: {type(error).__name__}")
    raise AnswerGenerationError("Gemini answer generation is unavailable. " + "; ".join(errors))
