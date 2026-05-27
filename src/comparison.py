from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer, util

from src.article_fetcher import fetch_article_text
from src.classifier import analyze_emotion
from src.summarizer import summarize


SIMILARITY_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

similarity_model = SentenceTransformer(SIMILARITY_MODEL_NAME)


def build_article_explanation(emotion_result: Dict[str, Any]) -> str:
    """
    Creates the explanation shown under one article card.
    It uses results that were already calculated by analyze_emotion().
    """

    tone_label = emotion_result["tone_label"]
    tone_score = emotion_result["tone_score"]

    emotion_names = [
        emotion["label"]
        for emotion in emotion_result["top_emotions"]
    ]

    if emotion_names:
        emotion_text = ", ".join(emotion_names)
    else:
        emotion_text = "no strong emotional signals"

    evidence_sentences = emotion_result["evidence_sentences"]

    if evidence_sentences:
        bullets = "\n".join(
            f"- {sentence}"
            for sentence in evidence_sentences
        )
    else:
        bullets = "- No strong supporting sentences were detected."

    return (
        f"This article has a **{tone_label}** emotional tone "
        f"with a score of **{tone_score}/100**.\n\n"
        f"Its strongest emotional signals are: **{emotion_text}**.\n\n"
        f"Key sentences influencing this result:\n{bullets}"
    )


def analyse_single_article(url: str, text: str) -> Dict[str, Any]:
    """
    Runs summary and emotional tone analysis for one article.
    """

    summary = summarize(text)
    emotion_result = analyze_emotion(text)

    return {
        "url": url,
        "word_count": len(text.split()),
        "summary": summary,
        "emotion": emotion_result,
        "explanation": build_article_explanation(emotion_result),
    }


def calculate_topic_similarity(summary_a: str, summary_b: str) -> float:
    """
    Compares the meaning of the two summaries.

    The result is useful for estimating whether the articles
    discuss closely related subjects.
    """

    embeddings = similarity_model.encode(
        [summary_a, summary_b],
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

    percentage = max(0.0, min(1.0, similarity)) * 100

    return round(percentage, 1)


def get_topic_similarity_label(score: float) -> str:
    if score >= 80:
        return "very similar topic"
    if score >= 60:
        return "related topic"
    if score >= 40:
        return "partly related topic"
    return "different topic"


def calculate_tone_difference(
    article_a: Dict[str, Any],
    article_b: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calculates emotional tone difference.

    A positive signed_difference means Article B is more positive.
    A negative signed_difference means Article A is more positive.
    """

    score_a = article_a["emotion"]["tone_score"]
    score_b = article_b["emotion"]["tone_score"]

    signed_difference = round(score_b - score_a, 1)
    absolute_difference = round(abs(signed_difference), 1)

    if absolute_difference < 8:
        direction = "similar emotional tone"
    elif signed_difference > 0:
        direction = "article_b_more_positive"
    else:
        direction = "article_a_more_positive"

    return {
        "signed_difference": signed_difference,
        "absolute_difference": absolute_difference,
        "direction": direction,
    }


def calculate_intensity_difference(
    article_a: Dict[str, Any],
    article_b: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compares how emotionally strong the two articles are.
    """

    intensity_a = article_a["emotion"]["intensity_score"]
    intensity_b = article_b["emotion"]["intensity_score"]

    difference = round(intensity_b - intensity_a, 1)
    absolute_difference = round(abs(difference), 1)

    if absolute_difference < 8:
        result = "similar intensity"
    elif difference > 0:
        result = "article_b_more_emotional"
    else:
        result = "article_a_more_emotional"

    return {
        "signed_difference": difference,
        "absolute_difference": absolute_difference,
        "direction": result,
    }


def get_emotion_names(article: Dict[str, Any]) -> List[str]:
    return [
        emotion["label"]
        for emotion in article["emotion"]["top_emotions"]
    ]


def build_comparison_explanation(
    article_a: Dict[str, Any],
    article_b: Dict[str, Any],
    topic_similarity: float,
    tone_difference: Dict[str, Any],
    intensity_difference: Dict[str, Any],
) -> str:
    """
    Creates the final readable comparison shown below both articles.
    """

    similarity_label = get_topic_similarity_label(topic_similarity)

    if topic_similarity >= 80:
        topic_text = (
            "The two articles appear to discuss a very similar subject "
            "or the same event."
        )
    elif topic_similarity >= 60:
        topic_text = (
            "The two articles appear to discuss related subject matter, "
            "although their exact focus may differ."
        )
    else:
        topic_text = (
            "The articles do not appear close enough in topic for a strong "
            "emotional framing comparison."
        )

    score_a = article_a["emotion"]["tone_score"]
    score_b = article_b["emotion"]["tone_score"]

    tone_label_a = article_a["emotion"]["tone_label"]
    tone_label_b = article_b["emotion"]["tone_label"]

    if tone_difference["direction"] == "similar emotional tone":
        tone_text = (
            f"Article A scores {score_a}/100 and Article B scores "
            f"{score_b}/100. Their overall emotional tones are very similar."
        )
    elif tone_difference["direction"] == "article_b_more_positive":
        tone_text = (
            f"Article A is classified as **{tone_label_a}** with a tone score "
            f"of {score_a}/100, while Article B is classified as "
            f"**{tone_label_b}** with a tone score of {score_b}/100. "
            f"Article B is {tone_difference['absolute_difference']} points "
            f"more positive in emotional tone."
        )
    else:
        tone_text = (
            f"Article A is classified as **{tone_label_a}** with a tone score "
            f"of {score_a}/100, while Article B is classified as "
            f"**{tone_label_b}** with a tone score of {score_b}/100. "
            f"Article A is {tone_difference['absolute_difference']} points "
            f"more positive in emotional tone."
        )

    emotions_a = ", ".join(get_emotion_names(article_a))
    emotions_b = ", ".join(get_emotion_names(article_b))

    emotion_text = (
        f"The strongest emotional signals in Article A are **{emotions_a}**. "
        f"The strongest emotional signals in Article B are **{emotions_b}**."
    )

    if intensity_difference["direction"] == "similar intensity":
        intensity_text = (
            "Both articles use a similar level of emotional language."
        )
    elif intensity_difference["direction"] == "article_a_more_emotional":
        intensity_text = (
            f"Article A uses stronger emotional language overall, by "
            f"{intensity_difference['absolute_difference']} intensity points."
        )
    else:
        intensity_text = (
            f"Article B uses stronger emotional language overall, by "
            f"{intensity_difference['absolute_difference']} intensity points."
        )

    return (
        f"Topic similarity: **{topic_similarity}/100**, classified as "
        f"**{similarity_label}**.\n\n"
        f"{topic_text}\n\n"
        f"{tone_text}\n\n"
        f"{emotion_text}\n\n"
        f"{intensity_text}"
    )


def compare_articles(url_a: str, url_b: str) -> Dict[str, Any]:
    """
    Main function for Bias Lense Comparison Mode.

    Process:
    1. Fetch both articles at the same time.
    2. Analyse each article independently.
    3. Compare topic similarity.
    4. Compare emotional tone and intensity.
    5. Produce a final explanation.
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

        try:
            text_a = future_a.result()
        except Exception as exc:
            raise ValueError(
                f"Article A could not be loaded: {exc}"
            ) from exc

        try:
            text_b = future_b.result()
        except Exception as exc:
            raise ValueError(
                f"Article B could not be loaded: {exc}"
            ) from exc

    article_a = analyse_single_article(url_a, text_a)
    article_b = analyse_single_article(url_b, text_b)

    topic_similarity = calculate_topic_similarity(
        article_a["summary"],
        article_b["summary"],
    )

    tone_difference = calculate_tone_difference(
        article_a,
        article_b,
    )

    intensity_difference = calculate_intensity_difference(
        article_a,
        article_b,
    )

    explanation = build_comparison_explanation(
        article_a,
        article_b,
        topic_similarity,
        tone_difference,
        intensity_difference,
    )

    return {
        "article_a": article_a,
        "article_b": article_b,
        "comparison": {
            "topic_similarity_score": topic_similarity,
            "topic_similarity_label": get_topic_similarity_label(
                topic_similarity
            ),
            "tone_difference": tone_difference,
            "intensity_difference": intensity_difference,
            "explanation": explanation,
        },
    }