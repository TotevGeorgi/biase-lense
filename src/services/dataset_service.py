import json
from pathlib import Path
import streamlit as st

@st.cache_data(show_spinner=False)
def load_docs_json(path: str):
    root = Path(__file__).resolve().parents[2] 
    p = (root / path).resolve()

    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
