from typing import Any, Dict


def explain_emotion(
    emotion_result: Dict[str, Any],
    top_k_sentences: int = 3,
) -> str:

    tone_score = emotion_result["tone_score"]
    tone_label = emotion_result["tone_label"]

    intensity_score = emotion_result["intensity_score"]
    intensity_label = emotion_result["intensity_label"]

    top_emotions = emotion_result.get("top_emotions", [])
    evidence_sentences = emotion_result.get("evidence_sentences", [])

    if top_emotions:
        emotion_names = [
            emotion["label"]
            for emotion in top_emotions
        ]
        emotion_text = ", ".join(emotion_names)
    else:
        emotion_text = "no strong emotional signals"

    selected_sentences = evidence_sentences[:top_k_sentences]

    if selected_sentences:
        bullets = "\n".join(
            f"- {sentence}"
            for sentence in selected_sentences
        )
    else:
        bullets = "- No strong supporting sentences were detected."

    return (
        f"This article has a **{tone_label}** emotional tone, "
        f"with a score of **{tone_score}/100**.\n\n"
        f"Its emotional intensity is **{intensity_label}** "
        f"with a score of **{intensity_score}/100**.\n\n"
        f"The strongest emotional signals are: **{emotion_text}**.\n\n"
        f"Key parts influencing this result:\n"
        f"{bullets}"
    )