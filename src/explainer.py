from typing import Dict, List
import re

from src.classifier import classify_raw, map_to_bias_lense_labels, split_sentences

def explain_emotion(text: str, top_label: str, top_k_sentences: int = 3) -> str:
    
    sents = split_sentences(text)
    if not sents:
        return "Not enough text to explain the emotion."

    scored: List[tuple[str, float]] = []

    for s in sents:
        raw = classify_raw(s)
        mapped = map_to_bias_lense_labels(raw)
        scored.append((s, float(mapped.get(top_label, 0.0))))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = [s for s, sc in scored[:top_k_sentences] if len(s.strip()) > 10]

    if not top:
        return "The emotion signal is weak or spread across the text."

    reasons = {
        "happiness": "positive outcomes, celebration, gratitude, or approval",
        "fear": "threats, uncertainty, risk, or alarming consequences",
        "motivation": "goals, progress, determination, or hopeful framing",
    }
    reason_hint = reasons.get(top_label, "the tone and word choice")

    bullets = "\n".join([f"- {t}" for t in top])

    return (
        f"This article suggests **{top_label}** mainly because it contains language linked to {reason_hint}.\n\n"
        f"Key parts driving this label:\n"
        f"{bullets}\n\n"
        f"These sentences carry the strongest {top_label} signal according to the emotion classifier."
    )
