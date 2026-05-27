import streamlit as st

from src.services.analysis_service import analyze_url
from src.explainer import explain_emotion


def render():
    st.subheader("Analyzer")

    url = st.text_input(
        "Article URL",
        placeholder="https://...",
    )

    column_1, column_2 = st.columns([1, 1])

    run = column_1.button(
        "Analyze",
        use_container_width=True,
    )

    show_text = column_2.checkbox(
        "Show extracted text",
        value=False,
    )

    mode = st.radio(
        "Output mode",
        ["Standard summary", "Explanatory insight"],
        horizontal=True,
    )

    if not run:
        return

    if not url.strip():
        st.error("Paste a URL first.")
        return

    try:
        with st.spinner("Analyzing..."):
            text, summary, emotion_result = analyze_url(url.strip())
    except Exception as error:
        st.error(f"Could not analyse this article: {error}")
        return

    st.subheader("Summary")
    st.write(summary)

    st.subheader("Emotional Tone")

    tone_score = emotion_result["tone_score"]
    tone_label = emotion_result["tone_label"]

    st.write(
        f"**Tone:** {tone_label.title()} "
        f"({tone_score}/100)"
    )

    st.caption(
        "0 = strongly negative, 50 = balanced or neutral, "
        "100 = strongly positive."
    )

    st.progress(
        min(max(float(tone_score) / 100, 0.0), 1.0)
    )

    st.subheader("Emotional Intensity")

    intensity_score = emotion_result["intensity_score"]
    intensity_label = emotion_result["intensity_label"]

    st.write(
        f"**Intensity:** {intensity_label.title()} "
        f"({intensity_score}/100)"
    )

    st.caption(
        "Intensity shows how strongly emotional language appears "
        "in the article, not whether the article is positive or negative."
    )

    st.progress(
        min(max(float(intensity_score) / 100, 0.0), 1.0)
    )

    st.subheader("Strongest Emotional Signals")

    top_emotions = emotion_result.get("top_emotions", [])

    if top_emotions:
        for emotion in top_emotions:
            st.write(
                f"**{emotion['label'].title()}**: "
                f"{emotion['score']}/100"
            )

            st.progress(
                min(max(float(emotion["score"]) / 100, 0.0), 1.0)
            )
    else:
        st.write("No strong emotional signals were detected.")

    if mode == "Explanatory insight":
        st.subheader("Why this result?")

        insight = explain_emotion(
            emotion_result,
            top_k_sentences=3,
        )

        st.write(insight)

    st.subheader("Analysis Values")

    st.json({
    "tone_score": tone_score,
    "tone_label": tone_label,
    "intensity_score": intensity_score,
    "intensity_label": intensity_label,
    "top_emotions": top_emotions,
    "evidence_sentences": emotion_result.get("evidence_sentences", []),
    })

    if show_text:
        st.subheader("Extracted Article Text")
        st.write(text)