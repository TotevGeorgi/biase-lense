from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, List
import re

MODEL_NAME = "SamLowe/roberta-base-go_emotions"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
id2label = model.config.id2label

GOEMOTIONS_TO_BIASLENS = {
    "happiness": {"joy","amusement","gratitude","love","admiration","relief","pride","approval","caring","excitement"},
    "fear": {"fear","nervousness","anxiety","worry","terror"},
    "motivation": {"optimism","hope","desire","determination","enthusiasm","inspiration","curiosity"},
}

def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"[.!?]\s+", text) if s.strip()]

def classify_raw(text: str) -> Dict[str, float]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs = torch.sigmoid(logits).tolist()
    return {id2label[i].lower(): float(probs[i]) for i in range(len(probs))}

def map_to_bias_lense_labels(emotions: Dict[str, float]) -> Dict[str, float]:
    def sum_labels(names):
        return float(sum(emotions.get(name, 0.0) for name in names))
    return {
        "happiness": sum_labels(GOEMOTIONS_TO_BIASLENS["happiness"]),
        "fear": sum_labels(GOEMOTIONS_TO_BIASLENS["fear"]),
        "motivation": sum_labels(GOEMOTIONS_TO_BIASLENS["motivation"]),
    }

def analyze_emotion(text: str, mode: str = "chunk_max") -> Dict:
    sentences = split_sentences(text)

    if len(sentences) <= 1:
        raw = classify_raw(text)
        scores = map_to_bias_lense_labels(raw)
        return {"scores": scores, "label": max(scores, key=scores.get)}

    per_chunk_scores = []
    for s in sentences:
        raw = classify_raw(s)
        per_chunk_scores.append(map_to_bias_lense_labels(raw))

    final = {"happiness": 0.0, "fear": 0.0, "motivation": 0.0}
    if mode == "chunk_max":
        for k in final:
            final[k] = max(sc[k] for sc in per_chunk_scores)
    else: 
        for k in final:
            final[k] = sum(sc[k] for sc in per_chunk_scores) / len(per_chunk_scores)

    return {"scores": final, "label": max(final, key=final.get)}
