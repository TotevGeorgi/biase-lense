import streamlit as st
import plotly.express as px
from pathlib import Path
import json

def render(data_path: str = "data/dashboard_300.json"):
    st.subheader("Dashboard")

    p = Path(data_path)
    if not p.exists():
        st.error(f"Missing file: {p}. Create it from the notebook (dashboard_300.json).")
        return

    with open(p, "r", encoding="utf-8") as f:
        enriched = json.load(f)

    if not enriched:
        st.warning("Empty dashboard file.")
        return

    labels = [e["emotion_label"] for e in enriched]
    counts = {k: labels.count(k) for k in ["happiness", "fear", "motivation"]}

    c1, c2, c3 = st.columns(3)
    c1.metric("Happiness", counts["happiness"])
    c2.metric("Fear", counts["fear"])
    c3.metric("Motivation", counts["motivation"])

    st.subheader("Emotion distribution")
    st.bar_chart(counts)

    years = [e.get("year") for e in enriched]
    animation_frame = "year" if all(y is not None for y in years) else None

    df = {
        "x": [e["x"] for e in enriched],
        "y": [e["y"] for e in enriched],
        "z": [e["z"] for e in enriched],
        "emotion": [e["emotion_label"] for e in enriched],
        "title": [e["title"] for e in enriched],
        "year": years,
        "length": [e["length"] for e in enriched],
        "summary": [e.get("summary", "") for e in enriched],
    }

    st.subheader("Gapminder-style bubble map")
    st.caption("Each bubble is an article. Move the year slider to see how emotions shift over time.")

    fig = px.scatter(
        df,
        x="x",
        y="y",
        size="length",
        color="emotion",
        hover_name="title",
        hover_data={"summary": True, "year": True, "length": True},
        animation_frame=animation_frame,
        size_max=35,
        title="Articles in embedding space over time",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("3D view (static)")
    fig3d = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="emotion",
        hover_name="title",
        hover_data={"summary": True, "year": True, "length": True},
        title="3D PCA space",
    )
    st.plotly_chart(fig3d, use_container_width=True)
