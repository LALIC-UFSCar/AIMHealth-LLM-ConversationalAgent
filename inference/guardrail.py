"""Deterministic crisis guardrail for Phase 3 inference."""

from __future__ import annotations

import re
from typing import Dict, List

CRISIS_KEYWORDS = [
    "suicide",
    "suicidal",
    "kill myself",
    "killing myself",
    "end my life",
    "end it all",
    "want to die",
    "wish i was dead",
    "wish i were dead",
    "better off dead",
    "don't want to live",
    "dont want to live",
    "no reason to live",
    "self-harm",
    "self harm",
    "hurt myself",
    "cut myself",
    "overdose",
    "take my own life",
]

CRISIS_PATTERNS = [
    r"\bi\s+(want|wanna|plan|will|might)\s+to\s+(die|kill\s+myself|end\s+my\s+life|end\s+it\s+all)\b",
    r"\b(thinking|thoughts?)\s+(about|of)\s+(suicide|killing\s+myself|ending\s+my\s+life)\b",
    r"\b(no\s+reason\s+to\s+live|life\s+is\s+not\s+worth\s+living)\b",
    r"\b(i\s+am|i'm)\s+suicidal\b",
]


def extract_crisis_signals(text: str) -> List[str]:
    """Return keyword/pattern matches that triggered the crisis detector."""
    if not text:
        return []

    text_lower = text.lower()
    signals: List[str] = []

    for kw in CRISIS_KEYWORDS:
        if kw in text_lower:
            signals.append(f"kw:{kw}")

    for idx, pattern in enumerate(CRISIS_PATTERNS):
        if re.search(pattern, text_lower):
            signals.append(f"re:{idx}")

    return signals


def detect_crisis_risk(text: str) -> Dict[str, object]:
    """Detect crisis risk from user text using deterministic keyword/regex rules."""
    signals = extract_crisis_signals(text)
    is_crisis = len(signals) > 0

    return {
        "is_crisis": is_crisis,
        "risk_level": "IMMEDIATE" if is_crisis else "NO_RISK",
        "signals": signals,
        "detector": "keyword_regex_v1",
    }


def build_crisis_response_988() -> str:
    """Return a fixed empathetic crisis response with 988 resource."""
    return (
        "I hear you, and I am really glad you reached out. "
        "You do not have to go through this alone. "
        "Please contact the 988 Suicide & Crisis Lifeline right now. "
        "You can call or text 988, 24 hours a day, 7 days a week. "
        "They are trained to support you with care and confidentiality."
    )
