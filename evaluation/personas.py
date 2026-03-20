"""Persona loading helpers."""

from __future__ import annotations

import json
from pathlib import Path


def load_personas(personas_file: Path) -> dict[str, str]:
    with personas_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    persona_map: dict[str, str] = {}

    for p in data.get("personas", {}).get("existing", []):
        pid = p.get("id")
        if not pid:
            continue
        demo = p.get("demographics", {})
        profile = p.get("profile", {})
        desc_parts = [
            f"Name: {demo.get('name', pid)}",
            f"Age: {demo.get('age', 'N/A')}",
            f"Education: {demo.get('education', 'N/A')}",
            f"Clinical profile: {profile.get('clinical_profile', 'N/A')}",
            f"Pains: {profile.get('pains', 'N/A')}",
            f"Therapy status: {profile.get('therapy_status', 'N/A')}",
            f"Motivations: {profile.get('motivations', 'N/A')}",
        ]
        persona_map[pid] = "\n".join(desc_parts)

    for p in data.get("personas", {}).get("new", []):
        pid = p.get("id")
        if not pid:
            continue
        demo = p.get("demographics", {})
        summary = p.get("personality_summary", "")
        desc_parts = [
            f"User ID: {demo.get('user_id', pid)}",
            f"Age: {demo.get('age', 'N/A')}",
            f"Employment: {demo.get('employment_status', 'N/A')}",
            f"Education: {demo.get('education', 'N/A')}",
            f"Personality summary: {summary}",
        ]
        persona_map[pid] = "\n".join(desc_parts)

    return persona_map
