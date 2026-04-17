import streamlit as st
from src.article_fetcher import fetch_article_text
from src.summarizer import summarize
from src.classifier import analyze_emotion

@st.cache_data(show_spinner=False, ttl=3600)
def analyze_url(url: str):
    text = fetch_article_text(url)
    summary = summarize(text)
    emo = analyze_emotion(summary)
    return text, summary, emo

@st.cache_data(show_spinner=False)
def analyze_docs(docs: list[dict]):
    enriched = []
    for d in docs:
        text = (d.get("text") or "").strip()
        if len(text) < 30:
            continue

        summary = summarize(text)
        emo = analyze_emotion(summary)
        scores = emo["scores"]

        enriched.append({
            "id": d.get("id"),
            "title": d.get("title", ""),
            "text": text,
            "year": d.get("year", None),
            "summary": summary,
            "emotion_label": emo["label"],
            "happiness": float(scores.get("happiness", 0.0)),
            "fear": float(scores.get("fear", 0.0)),
            "motivation": float(scores.get("motivation", 0.0)),
        })
    return enriched
