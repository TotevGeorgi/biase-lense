import json
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from src.article_fetcher import fetch_article_text
from src.summarizer import summarize
from src.classifier import analyze_emotion


@st.cache_data(show_spinner=False, ttl=3600)
def analyze_url(url: str):
    text = fetch_article_text(url)
    summary = summarize(text)

    # Important: analyse the full article text, not only the summary.
    emo = analyze_emotion(text)

    return text, summary, emo


@st.cache_data(show_spinner=False)
def analyze_docs(docs: List[Dict[str, Any]]):
    enriched = []

    for d in docs:
        text = (d.get("text") or "").strip()

        if len(text) < 30:
            continue

        summary = summarize(text)

        # Same here: analyse the full text.
        emo = analyze_emotion(text)

        enriched.append({
            "id": d.get("id"),
            "title": d.get("title", ""),
            "text": text,
            "year": d.get("year", None),
            "summary": summary,

            # New spectrum fields
            "tone_score": float(emo.get("tone_score", 50.0)),
            "tone_label": emo.get("tone_label", "balanced / neutral"),
            "intensity_score": float(emo.get("intensity_score", 0.0)),
            "intensity_label": emo.get("intensity_label", "low"),
            "top_emotions": emo.get("top_emotions", []),
            "evidence_sentences": emo.get("evidence_sentences", []),
        })

    return enriched


@st.cache_data(show_spinner=False)
def load_docs_json(path: str):
    root = Path(__file__).resolve().parents[2]
    p = (root / path).resolve()

    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)