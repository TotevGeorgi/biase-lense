from sentence_transformers import SentenceTransformer
import numpy as np

EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
emb_model = SentenceTransformer(EMB_MODEL_NAME)


def build_index(docs):
    """
    docs: list of dicts, each at least {"id": ..., "title": ..., "text": ...}
    Returns an index dict with embeddings.
    """
    texts = [d["text"] for d in docs]
    embeddings = emb_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return {"docs": docs, "embeddings": embeddings}


def semantic_search(index, query: str, k: int = 3):
    """
    Returns top-k documents with similarity scores.
    """
    query_emb = emb_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    sims = index["embeddings"] @ query_emb 
    top_idx = np.argsort(-sims)[:k]

    results = []
    for idx in top_idx:
        doc = index["docs"][idx].copy()
        doc["score"] = float(sims[idx])
        results.append(doc)
    return results
