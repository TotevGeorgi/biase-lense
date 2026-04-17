import streamlit as st
import torch

from src.summarizer import summarizer_pipeline
from src.classifier import tokenizer as emo_tokenizer, model as emo_model
from pages.analyzer_page import render as render_analyzer
from pages.dashboard_page import render as render_dashboard

st.set_page_config(page_title="Bias Lense", page_icon="🧠", layout="wide")

@st.cache_resource
def warmup_models():
    _ = summarizer_pipeline("Warmup text.", max_length=30, min_length=10, do_sample=False, truncation=True)
    inputs = emo_tokenizer("warmup", return_tensors="pt", truncation=True, max_length=32)
    with torch.inference_mode():
        _ = emo_model(**inputs).logits
    return True

warmup_models()

st.title("Bias Lense")
tab1, tab2 = st.tabs(["Analyzer", "Dashboard"])

with tab1:
    render_analyzer()

with tab2:
    render_dashboard("notebooks/data/dashboard_300.json")
