"""Verify that reranked document chunks support a rewritten user query."""

import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


# Load the same local Gemini configuration style used by the existing modules.
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not set in the .env file."
    )


client = genai.Client(api_key=API_KEY)


# Keep this verifier independent from the existing fallback lists.
MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash"
]


def build_verification_prompt(query, reranked_results):
    """Build one batched request containing only the candidate evidence."""

    evidence_text = ""

    for result in reranked_results:

        chunk = result["chunk"]

        evidence_text += f"""
CHUNK ID: {chunk.get('chunk_id')}
Document: {chunk.get('document_name')}
Page: {chunk.get('page_number')}
Section: {chunk.get('section')}
Text: {chunk.get('text')}
----------------------------------------
"""

    return f"""
You are an evidence verification component for an enterprise document
intelligence system.

Decide whether the candidate evidence directly supports an answer to the
standalone user question. Use only the evidence below. Do not infer missing
facts, use outside knowledge, or rewrite evidence.

Return JSON only. It must have exactly these fields:

{{
  "supported": true or false,
  "supporting_chunk_ids": ["exact chunk IDs from the candidates"],
  "abstain": true or false
}}

Rules:

1. Mark supported true only when one or more candidate chunks directly support
   an answer to the question.
2. Include only chunk IDs that directly support the answer. Exclude chunks that
   are irrelevant, only loosely related, or do not contain the needed facts.
3. If the evidence is insufficient, return supported false, an empty list, and
   abstain true.
4. If supported is true, abstain must be false and the list must be non-empty.
5. If supported is false, abstain must be true and the list must be empty.
6. Never create a chunk ID that is not present in the candidates.

STANDALONE USER QUESTION:
{query}

CANDIDATE EVIDENCE:
{evidence_text}
"""


def parse_verification_response(response_text, valid_chunk_ids):
    """Validate the strict JSON contract and return approved chunk IDs.

    Returning ``None`` means the response is unsafe and must cause abstention.
    """

    try:
        response_data = json.loads(response_text)
    except (TypeError, json.JSONDecodeError):
        return None

    expected_keys = {
        "supported",
        "supporting_chunk_ids",
        "abstain"
    }

    if set(response_data.keys()) != expected_keys:
        return None

    supported = response_data["supported"]
    supporting_chunk_ids = response_data["supporting_chunk_ids"]
    abstain = response_data["abstain"]

    if not isinstance(supported, bool):
        return None

    if not isinstance(abstain, bool):
        return None

    if not isinstance(supporting_chunk_ids, list):
        return None

    if not all(
        isinstance(chunk_id, str)
        for chunk_id in supporting_chunk_ids
    ):
        return None

    # Both possible outcomes have one unambiguous, safe representation.
    if supported is False:
        if abstain is True and not supporting_chunk_ids:
            return []

        return None

    if abstain is True or not supporting_chunk_ids:
        return None

    # Reject the entire result rather than silently accepting invented IDs.
    if not set(supporting_chunk_ids).issubset(valid_chunk_ids):
        return None

    return set(supporting_chunk_ids)


def verify_evidence(query, reranked_results):
    """Return verified original results and an abstention flag.

    The function fails closed: invalid output, an unexpected response, or all
    unavailable models returns no evidence and ``True`` for abstention.
    """

    if not reranked_results:
        return [], True

    valid_chunk_ids = {
        result["chunk"].get("chunk_id")
        for result in reranked_results
    }

    if None in valid_chunk_ids:
        return [], True

    prompt = build_verification_prompt(
        query,
        reranked_results
    )

    for model_name in MODELS:

        print(f"\nTrying verifier model: {model_name}")

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )

        except Exception as error:
            print(
                "Verifier model unavailable: "
                f"{model_name} ({type(error).__name__})"
            )
            continue

        approved_chunk_ids = parse_verification_response(
            getattr(response, "text", None),
            valid_chunk_ids
        )

        # A malformed or unsafe response must abstain immediately.
        if approved_chunk_ids is None:
            print("Verifier response was unsafe. Abstaining.")
            return [], True

        # The model explicitly found no support.
        if not approved_chunk_ids:
            print(f"Evidence verified using: {model_name}")
            return [], True

        # Filter the existing result dictionaries. Never rebuild them from AI.
        verified_results = [
            result
            for result in reranked_results
            if result["chunk"].get("chunk_id")
            in approved_chunk_ids
        ]

        if not verified_results:
            return [], True

        print(f"Evidence verified using: {model_name}")
        return verified_results, False

    print("All verifier models were unavailable. Abstaining.")
    return [], True
