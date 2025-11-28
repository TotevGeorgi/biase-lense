from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
id2label = model.config.id2label


def classify_raw(text: str) -> dict:
    """
    Run the pre-trained model and return probabilities for each original label.
    Example output: {"joy": 0.72, "anger": 0.03, ...}
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0].tolist()

    result = {}
    for i, p in enumerate(probs):
        label = id2label[i]
        result[label] = float(p)
    return result


def map_to_bias_lense_labels(emotions: dict) -> dict:
    """
    Map model labels to:
    - happiness
    - fear
    - motivation

    Returns:
    {
        "scores": {"happiness": ..., "fear": ..., "motivation": ...},
        "label": "happiness"
    }
    """
    # Lower-case keys for robust matching
    lower = {k.lower(): v for k, v in emotions.items()}

    # You can tweak these groupings once you see real labels
    happiness = 0.0
    for key in ["joy", "happiness", "love"]:
        happiness += lower.get(key, 0.0)

    fear = 0.0
    for key in ["fear", "anxiety", "worry"]:
        fear += lower.get(key, 0.0)

    motivation = 0.0
    for key in ["optimism", "hope", "enthusiasm"]:
        motivation += lower.get(key, 0.0)

    scores = {
        "happiness": happiness,
        "fear": fear,
        "motivation": motivation,
    }

    top_label = max(scores, key=scores.get)
    return {"scores": scores, "label": top_label}


def analyze_emotion(text: str) -> dict:
    """
    Convenience function: text -> final label + scores.
    """
    raw = classify_raw(text)
    return map_to_bias_lense_labels(raw)
