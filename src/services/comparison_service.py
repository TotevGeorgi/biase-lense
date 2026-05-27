from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

from src.article_fetcher import fetch_article_text
from src.classifier import analyze_emotion
from src.explainer import explain_emotion
from src.summarizer import summarize

from src.embeddings import calculate_text_similarity


def analyse_article(url: str, text: str) -> Dict[str, Any]:
    """
    Analyses one article using the same pipeline as the normal Analyzer page.
    """

    summary = summarize(text)
    emotion_result = analyze_emotion(text)
    explanation = explain_emotion(emotion_result)

    return {
        "url": url,
        "word_count": len(text.split()),
        "summary": summary,
        "emotion": emotion_result,
        "explanation": explanation,
    }


def get_topic_similarity_label(score: float) -> str:
    if score >= 80:
        return "very similar topic"
    if score >= 60:
        return "related topic"
    if score >= 40:
        return "partly related topic"
    return "different topic"


def build_comparison_explanation(
    article_a: Dict[str, Any],
    article_b: Dict[str, Any],
    topic_similarity: float,
) -> str:
    """
    Creates the conclusion shown below the two article cards.
    """

    tone_a = article_a["emotion"]["tone_score"]
    tone_b = article_b["emotion"]["tone_score"]

    tone_label_a = article_a["emotion"]["tone_label"]
    tone_label_b = article_b["emotion"]["tone_label"]

    intensity_a = article_a["emotion"]["intensity_score"]
    intensity_b = article_b["emotion"]["intensity_score"]

    tone_difference = round(tone_b - tone_a, 1)
    intensity_difference = round(intensity_b - intensity_a, 1)

    emotions_a = ", ".join(
        emotion["label"]
        for emotion in article_a["emotion"]["top_emotions"]
    )

    emotions_b = ", ".join(
        emotion["label"]
        for emotion in article_b["emotion"]["top_emotions"]
    )

    if topic_similarity >= 60:
        topic_text = (
            "The two articles appear related enough for their emotional "
            "framing to be compared."
        )
    else:
        topic_text = (
            "The articles may not be closely related enough for a direct "
            "emotional framing comparison to be reliable."
        )

    if abs(tone_difference) < 8:
        tone_text = (
            "Both articles use a similar overall emotional tone."
        )
    elif tone_difference > 0:
        tone_text = (
            f"Article B is {abs(tone_difference)} points more positive "
            f"in emotional tone than Article A."
        )
    else:
        tone_text = (
            f"Article A is {abs(tone_difference)} points more positive "
            f"in emotional tone than Article B."
        )

    if abs(intensity_difference) < 8:
        intensity_text = (
            "Both articles use a similar level of emotional language."
        )
    elif intensity_difference > 0:
        intensity_text = (
            "Article B uses stronger emotional language overall."
        )
    else:
        intensity_text = (
            "Article A uses stronger emotional language overall."
        )

    return (
        f"Topic similarity: **{topic_similarity}/100** "
        f"({get_topic_similarity_label(topic_similarity)}).\n\n"
        f"{topic_text}\n\n"
        f"Article A has a **{tone_label_a}** tone with a score of "
        f"**{tone_a}/100**. Its strongest emotional signals are "
        f"**{emotions_a}**.\n\n"
        f"Article B has a **{tone_label_b}** tone with a score of "
        f"**{tone_b}/100**. Its strongest emotional signals are "
        f"**{emotions_b}**.\n\n"
        f"{tone_text} {intensity_text}"
    )


def compare_urls(url_a: str, url_b: str) -> Dict[str, Any]:
    """
    Fetches two different articles, analyses each one separately,
    then compares their topic and emotional framing.
    """

    url_a = url_a.strip()
    url_b = url_b.strip()

    if not url_a or not url_b:
        raise ValueError("Please provide two article URLs.")

    if url_a == url_b:
        raise ValueError("Please provide two different article URLs.")

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(fetch_article_text, url_a)
        future_b = executor.submit(fetch_article_text, url_b)

        text_a = future_a.result()
        text_b = future_b.result()

    article_a = analyse_article(url_a, text_a)
    article_b = analyse_article(url_b, text_b)

    topic_similarity = calculate_text_similarity(
        article_a["summary"],
        article_b["summary"],
    )

    comparison_explanation = build_comparison_explanation(
        article_a,
        article_b,
        topic_similarity,
    )

    return {
        "article_a": article_a,
        "article_b": article_b,
        "comparison": {
            "topic_similarity_score": topic_similarity,
            "topic_similarity_label": get_topic_similarity_label(
                topic_similarity
            ),
            "tone_difference": round(
                article_b["emotion"]["tone_score"]
                - article_a["emotion"]["tone_score"],
                1,
            ),
            "intensity_difference": round(
                article_b["emotion"]["intensity_score"]
                - article_a["emotion"]["intensity_score"],
                1,
            ),
            "explanation": comparison_explanation,
        },
    }