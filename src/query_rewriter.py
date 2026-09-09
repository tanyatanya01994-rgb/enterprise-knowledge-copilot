"""Conversation-aware query rewriting with a safe local fallback."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence

from dotenv import load_dotenv
from google import genai

load_dotenv()
MODELS = ("gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.8-flash")
MAX_HISTORY_TURNS = 8


def _recent_history(history: Sequence[str] | None) -> list[str]:
    return [str(item).strip() for item in (history or []) if str(item).strip()][-MAX_HISTORY_TURNS:]


def _local_rewrite(query: str, history: Sequence[str]) -> str:
    """Resolve only an explicit follow-up reference; otherwise leave the query intact."""
    query = query.strip()
    if not history or not re.search(r"\b(it|they|them|that|those)\b|^what about\b", query.lower()):
        return query
    previous = [line.split(":", 1)[1].strip() for line in history if line.lower().startswith("user:")]
    if not previous:
        return query
    topic = previous[-1]
    if query.lower().startswith("what about"):
        subject = query[len("what about"):].strip(" ?.")
        return f"{topic} Specifically, how does it apply to {subject}?" if subject else topic
    return f"{query} (Context: {topic})"


def rewrite_query(query: str, conversation_history: Sequence[str] | None = None) -> str:
    """Return a standalone query, falling back deterministically when Gemini is unavailable."""
    query = query.strip()
    history = _recent_history(conversation_history)
    if not history:
        return query
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        prompt = f"""Rewrite the latest enterprise-document question as a standalone retrieval query.
Use the conversation only to resolve references. Do not answer or add facts.
Return only the rewritten query.

Conversation:
{chr(10).join(history)}

Latest question: {query}"""
        client = genai.Client(api_key=api_key)
        for model_name in MODELS:
            try:
                text = getattr(client.models.generate_content(model=model_name, contents=prompt), "text", "").strip()
                if text:
                    print(f"[QueryRewriter] Rewritten with {model_name}.")
                    return text
            except Exception as error:
                print(f"[QueryRewriter] {model_name} unavailable: {type(error).__name__}")
    print("[QueryRewriter] Used bounded local fallback.")
    return _local_rewrite(query, history)
