from src.embeddings import build_index, semantic_search
from src.summarizer import summarize
from src.classifier import analyze_emotion


def build_demo_index():
    return build_index(ARTICLES)


def news_emotion_agent(query: str, index):
    """
    1) Semantic search
    2) Summarize each article
    3) Analyse emotional tone of the full article text
    """
    hits = semantic_search(index, query, k=3)
    enriched = []

    for h in hits:
        text = h["text"]
        summary = summarize(text)
        emotion = analyze_emotion(text)

        enriched.append({
            "id": h["id"],
            "title": h["title"],
            "score": h["score"],
            "summary": summary,

            # New emotion fields
            "tone_score": emotion["tone_score"],
            "tone_label": emotion["tone_label"],
            "intensity_score": emotion["intensity_score"],
            "intensity_label": emotion["intensity_label"],
            "top_emotions": emotion["top_emotions"],
            "evidence_sentences": emotion["evidence_sentences"],
        })

    return enriched


if __name__ == "__main__":
    index = build_demo_index()

    user_query = input("Enter your news query: ")
    results = news_emotion_agent(user_query, index)

    for r in results:
        print("=" * 60)
        print(f"Title: {r['title']}")
        print(f"Relevance score: {r['score']:.3f}")
        print(f"Summary: {r['summary']}")
        print(f"Tone: {r['tone_label']} ({r['tone_score']}/100)")
        print(f"Intensity: {r['intensity_label']} ({r['intensity_score']}/100)")
        print(f"Top emotions: {r['top_emotions']}")
        print(f"Evidence: {r['evidence_sentences']}")