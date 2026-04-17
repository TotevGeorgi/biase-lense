import streamlit as st
from src.services.analysis_service import analyze_url
from src.explainer import explain_emotion

def render():
    st.subheader("Analyzer")

    url = st.text_input("Article URL", placeholder="https://...")
    c1, c2 = st.columns([1, 1])
    run = c1.button("Analyze", use_container_width=True)
    show_text = c2.checkbox("Show extracted text", value=False)
    mode = st.radio(
    "Output mode",
    ["Standard summary", "Explanatory insight"],
    horizontal=True
)


    if not run:
        return

    if not url.strip():
        st.error("Paste a URL first.")
        return

    with st.spinner("Analyzing..."):
        text, summary, emo = analyze_url(url.strip())

    st.subheader("Summary")
    st.write(summary)

    st.subheader("Emotion")
    scores = emo["scores"]
    st.write(f"**Top label:** {emo['label']}")

    if mode == "Explanatory insight":
        st.subheader("Why this emotion?")
        insight = explain_emotion(summary, emo["label"], top_k_sentences=3)
        st.write(insight)


    for k, v in scores.items():
        st.write(k)
        st.progress(min(max(float(v), 0.0), 1.0))

    st.subheader("Raw scores")
    st.caption("Closer to 1 the better.")
    st.json({k: round(float(v), 2) for k, v in scores.items()})

    if show_text:
        st.subheader("Extracted article text")
        st.write(text)
