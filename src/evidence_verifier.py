from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

VERIFICATION_TIMEOUT_MS = 120_000

# Gemini models used for evidence verification.
# If the first model is unavailable, the next model is tried.
MODELS = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
]

VERIFIED = "verified"
UNSUPPORTED = "unsupported"
VERIFIER_UNAVAILABLE = "verifier_unavailable"


# ============================================================
# GEMINI CLIENT
# ============================================================

client = None

if API_KEY:
    try:
        client = genai.Client(
            api_key=API_KEY,
            http_options=types.HttpOptions(
                timeout=VERIFICATION_TIMEOUT_MS
            ),
        )

    except Exception as error:
        print(
            "[EvidenceVerifier] Gemini client "
            "initialization failed: "
            f"{type(error).__name__}: {error}"
        )

else:
    print(
        "[EvidenceVerifier] GEMINI_API_KEY was not found. "
        "Local verification fallback will be used."
    )


# ============================================================
# VERIFICATION PROMPT
# ============================================================

def build_verification_prompt(
    query: str,
    candidates: list[dict[str, Any]],
) -> str:
    """
    Build a strict evidence-verification prompt.

    Gemini is asked to return ONLY the chunk IDs that
    directly or strongly support the user's question.
    """

    evidence_blocks = []

    for index, result in enumerate(candidates, start=1):

        chunk = result.get("chunk", {})

        evidence_blocks.append(
            f"""
EVIDENCE {index}

Chunk ID:
{chunk.get("chunk_id", "unknown")}

Document:
{chunk.get("document_name", "unknown")}

Page:
{chunk.get("page_number", "unknown")}

Section:
{chunk.get("section", "unknown")}

Text:
{chunk.get("text", "")}
"""
        )

    evidence_text = "\n".join(evidence_blocks)

    return f"""
You are the evidence verification component of an enterprise
document intelligence RAG system.

Your task is to identify which retrieved evidence chunks
actually support the user's question.

USER QUESTION:
{query}

RETRIEVED EVIDENCE:
{evidence_text}

STRICT RULES:

1. Approve a chunk only when its text directly or strongly
   supports information needed to answer the question.

2. Do not approve a chunk merely because it comes from a
   related document.

3. Do not use outside knowledge.

4. Do not infer facts that are not supported by the evidence.

5. If the retrieved evidence is insufficient, return an
   empty approved_chunk_ids list.

6. You may approve multiple chunks when multiple pieces of
   evidence are required.

7. Return ONLY valid JSON.

8. Use exactly this JSON structure:

{{
  "approved_chunk_ids": [
    "chunk_id_1",
    "chunk_id_2"
  ]
}}

Do not include markdown.
Do not include explanations.
Do not include any additional fields.
"""


# ============================================================
# RESPONSE PARSER
# ============================================================

def parse_verification_response(
    response_text: str,
) -> list[str]:
    """
    Safely parse Gemini's verification response.

    Returns an empty list if the response is invalid.
    """

    if not response_text:
        return []

    text = response_text.strip()

    # Remove accidental markdown code fences.
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    try:
        data = json.loads(text)

    except json.JSONDecodeError:
        print(
            "[EvidenceVerifier] Invalid JSON returned "
            "by Gemini."
        )
        return []

    if not isinstance(data, dict):
        return []

    # Accept the earlier test/CLI schema as well as the current prompt
    # schema.  Both are validated against the supplied candidate IDs later.
    approved = data.get("approved_chunk_ids")
    if approved is None and data.get("supported") is True and not data.get("abstain"):
        approved = data.get("supporting_chunk_ids", [])
    if approved is None:
        approved = []

    if not isinstance(approved, list):
        return []

    return [
        str(chunk_id)
        for chunk_id in approved
        if chunk_id
    ]


# ============================================================
# TOKENIZATION FOR LOCAL FALLBACK
# ============================================================

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "get",
    "gets",
    "how",
    "if",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def tokenize(text: str) -> set[str]:
    """
    Convert text into meaningful lowercase tokens.
    """

    words = re.findall(
        r"[a-zA-Z0-9]+",
        text.lower(),
    )

    return {
        word
        for word in words
        if len(word) >= 3
        and word not in STOPWORDS
    }


# ============================================================
# LOCAL FALLBACK VERIFICATION
# ============================================================

def local_evidence_verification(
    query: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Conservative deterministic fallback.

    This is used only when Gemini verification cannot be
    completed.

    It requires meaningful lexical overlap between the
    question and retrieved evidence.

    This does NOT replace the normal Gemini verifier when
    Gemini is available.
    """

    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    approved = []

    for result in candidates:

        chunk = result.get(
            "chunk",
            {},
        )

        text = str(
            chunk.get(
                "text",
                "",
            )
        )

        chunk_tokens = tokenize(text)

        if not chunk_tokens:
            continue

        overlap = query_tokens.intersection(
            chunk_tokens
        )

        # Number of meaningful query terms that occur
        # in the evidence.
        overlap_count = len(overlap)

        if overlap_count < 2:
            continue

        # Query-term coverage.
        coverage = (
            overlap_count
            / max(len(query_tokens), 1)
        )

        # Require reasonable overlap.
        if coverage >= 0.20:
            approved.append(result)

    # Preserve reranker order and limit the evidence.
    return approved[:5]


# ============================================================
# GEMINI VERIFICATION
# ============================================================

def _gemini_verify(
    query: str,
    candidates: list[dict[str, Any]],
) -> list[str]:
    """
    Ask Gemini to verify which chunks support the query.

    Raises an exception if all configured Gemini models fail.
    """

    if client is None:
        raise RuntimeError(
            "Gemini verifier client is unavailable."
        )

    prompt = build_verification_prompt(
        query,
        candidates,
    )

    last_error = None

    for model_name in MODELS:

        try:

            print(
                "[EvidenceVerifier] Trying model: "
                f"{model_name}"
            )

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )

            response_text = getattr(
                response,
                "text",
                "",
            )

            if not response_text:
                raise RuntimeError(
                    f"{model_name} returned an empty response."
                )

            approved_ids = parse_verification_response(
                response_text
            )

            print(
                "[EvidenceVerifier] Model "
                f"{model_name} approved "
                f"{len(approved_ids)} chunks."
            )

            return approved_ids

        except Exception as error:

            last_error = error

            print(
                "[EvidenceVerifier] Model "
                f"{model_name} failed: "
                f"{type(error).__name__}: {error}"
            )

            continue

    raise RuntimeError(
        "All Gemini evidence-verification models failed. "
        f"Last error: {last_error}"
    )


# ============================================================
# PUBLIC VERIFICATION FUNCTION
# ============================================================

def verify_evidence(
    query: str,
    candidates: list[dict[str, Any]],
):
    """
    Verify retrieved evidence.

    Returns ``(verified_results, status)`` where status is one of
    ``verified``, ``unsupported``, or ``verifier_unavailable``.  The
    unavailable state is deliberately distinct from a verified absence of
    evidence so callers do not make a misleading knowledge-base claim.

    Primary verification:
        Gemini

    Fallback verification:
        Conservative local lexical verification
    """

    # --------------------------------------------------------
    # NO CANDIDATES
    # --------------------------------------------------------

    if not candidates:
        return [], UNSUPPORTED

    # --------------------------------------------------------
    # PRIMARY: GEMINI VERIFICATION
    # --------------------------------------------------------

    try:

        approved_ids = _gemini_verify(
            query,
            candidates,
        )

        approved_set = set(
            approved_ids
        )

        verified = []

        for result in candidates:

            chunk = result.get(
                "chunk",
                {},
            )

            chunk_id = str(
                chunk.get(
                    "chunk_id",
                    "",
                )
            )

            if chunk_id in approved_set:
                verified.append(result)

        print(
            "[EvidenceVerifier] Gemini verification "
            f"approved {len(verified)} / "
            f"{len(candidates)} candidates."
        )

        return (verified, VERIFIED) if verified else ([], UNSUPPORTED)

    except Exception as error:

        print(
            "[EvidenceVerifier] Gemini verification "
            "could not be completed."
        )

        print(
            "[EvidenceVerifier] Reason: "
            f"{type(error).__name__}: {error}"
        )

        # Fail closed.  Lexical overlap is useful for diagnostics but is not
        # proof that evidence supports an answer, and an API outage must not
        # be reported as a successful verified evaluation.
        print("[EvidenceVerifier] Failing closed; no answer will be generated.")
        return [], VERIFIER_UNAVAILABLE
