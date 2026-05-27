import streamlit as st

from src.services.comparison_service import compare_urls


def render_article_card(title: str, article: dict):
    st.markdown(f"### {title}")

    st.markdown(f"**Source:** {article['url']}")

    st.markdown("#### Summary")
    st.write(article["summary"])

    emotion = article["emotion"]

    st.markdown("#### Emotional Tone")
    st.write(
        f"**{emotion['tone_label'].title()}** "
        f"({emotion['tone_score']}/100)"
    )

    st.progress(
        min(max(float(emotion["tone_score"]) / 100, 0.0), 1.0)
    )

    st.markdown("#### Emotional Intensity")
    st.write(
        f"**{emotion['intensity_label'].title()}** "
        f"({emotion['intensity_score']}/100)"
    )

    st.progress(
        min(max(float(emotion["intensity_score"]) / 100, 0.0), 1.0)
    )

    st.markdown("#### Strongest Emotional Signals")

    for item in emotion["top_emotions"]:
        st.write(
            f"**{item['label'].title()}**: {item['score']}/100"
        )

    with st.expander("Why this result?"):
        st.write(article["explanation"])


def render():
    st.subheader("Compare Articles")

    st.write(
        "Compare how two articles emotionally frame related subjects."
    )

    input_column_a, input_column_b = st.columns(2)

    with input_column_a:
        url_a = st.text_input(
            "Article A URL",
            placeholder="https://...",
        )

    with input_column_b:
        url_b = st.text_input(
            "Article B URL",
            placeholder="https://...",
        )

    run = st.button(
        "Compare Articles",
        use_container_width=True,
    )

    if not run:
        return

    if not url_a.strip() or not url_b.strip():
        st.error("Paste two article URLs first.")
        return

    try:
        with st.spinner("Comparing articles..."):
            result = compare_urls(
                url_a.strip(),
                url_b.strip(),
            )
    except Exception as error:
        st.error(f"Could not compare these articles: {error}")
        return

    article_column_a, article_column_b = st.columns(2)

    with article_column_a:
        render_article_card(
            "Article A",
            result["article_a"],
        )

    with article_column_b:
        render_article_card(
            "Article B",
            result["article_b"],
        )

    st.divider()

    st.subheader("Comparison Result")

    comparison = result["comparison"]

    st.markdown("#### Topic Similarity")

    st.write(
        f"**{comparison['topic_similarity_score']}/100** "
        f"({comparison['topic_similarity_label']})"
    )

    st.progress(
        min(
            max(float(comparison["topic_similarity_score"]) / 100, 0.0),
            1.0,
        )
    )

    st.markdown("#### Emotional Framing Difference")

    st.write(comparison["explanation"])