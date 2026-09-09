

from __future__ import annotations

import re


def sanitize_for_display(value: object) -> str:
    """Remove internal training-source branding from user-visible text."""
    text = str(value)
    text = re.sub(r"NexaCore\s+Technologies", "the organization", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\|?\s*Public\s+Training\s+Dataset\s*", " ", text, flags=re.IGNORECASE)
    return " ".join(text.split())
