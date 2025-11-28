from .embeddings import build_index, semantic_search
from .summarizer import summarize
from .classifier import analyze_emotion

# TEMP dummy data
ARTICLES = [
    {
        "id": 1,
        "title": "Elections bring hope for economic reform",
        "text": "The recent elections have sparked hope among citizens as the new government promises reforms ..."
    },
    {
        "id": 2,
        "title": "New conflict raises fears in the region",
        "text": "Rising tensions between neighbouring countries have created fear among local populations ..."
    },
    {
        "id": 3,
        "title": "Scientists develop breakthrough cancer treatment",
        "text": "A team of scientists announced a breakthrough therapy which could significantly improve survival rates ..."
    },
]


def build_demo_index():
    return build_index(ARTICLES)


def news_emotion_agent(query: str, index):
    """
    1) Semantic search
    2) Summarize each article
    3) Classify emotion of the summary
    """
    hits = semantic_search(index, query, k=3)
    enriched = []

    for h in hits:
        summary = summarize(h["text"])
        emotion = analyze_emotion(summary)

        enriched.append(
            {
                "id": h["id"],
                "title": h["title"],
                "score": h["score"],
                "summary": summary,
                "emotion_label": emotion["label"],
                "emotion_scores": emotion["scores"],
            }
        )

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
        print(f"Emotion: {r['emotion_label']}  {r['emotion_scores']}")
