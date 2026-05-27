from typing import Dict, List
import re

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "SamLowe/roberta-base-go_emotions"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

id2label = model.config.id2label


POSITIVE_EMOTIONS = {
    "admiration",
    "amusement",
    "approval",
    "caring",
    "desire",
    "excitement",
    "gratitude",
    "joy",
    "love",
    "optimism",
    "pride",
    "relief",
}

NEGATIVE_EMOTIONS = {
    "anger",
    "annoyance",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "fear",
    "grief",
    "nervousness",
    "remorse",
    "sadness",
}


def split_sentences(text: str) -> List[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"[.!?]\s+", text)
        if sentence.strip()
    ]


def classify_raw(text: str) -> Dict[str, float]:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        logits = model(**inputs).logits[0]

    probabilities = torch.sigmoid(logits).tolist()

    return {
        id2label[i].lower(): float(probabilities[i])
        for i in range(len(probabilities))
    }


def get_tone_label(tone_score: float) -> str:
    if tone_score < 20:
        return "very negative"
    if tone_score < 40:
        return "negative"
    if tone_score < 60:
        return "balanced / neutral"
    if tone_score < 80:
        return "positive"
    return "very positive"


def get_intensity_label(intensity_score: float) -> str:
    if intensity_score < 25:
        return "low"
    if intensity_score < 50:
        return "medium"
    if intensity_score < 75:
        return "high"
    return "very high"


def analyze_emotion(text: str, top_k_sentences: int = 3) -> Dict:
    sentences = split_sentences(text)

    if not sentences:
        return {
            "tone_score": 50.0,
            "tone_label": "balanced / neutral",
            "intensity_score": 0.0,
            "intensity_label": "low",
            "top_emotions": [],
            "evidence_sentences": [],
        }

    total_positive = 0.0
    total_negative = 0.0
    intensity_values = []
    emotion_totals: Dict[str, float] = {}
    analysed_sentences = []

    for sentence in sentences:
        raw = classify_raw(sentence)

        positive = sum(raw.get(label, 0.0) for label in POSITIVE_EMOTIONS)
        negative = sum(raw.get(label, 0.0) for label in NEGATIVE_EMOTIONS)
        emotional_evidence = positive + negative

        if emotional_evidence < 0.10:
            sentence_tone_score = 50.0
        else:
            sentence_tone_score = (positive / emotional_evidence) * 100

        non_neutral_emotions = {
            emotion: score
            for emotion, score in raw.items()
            if emotion != "neutral"
        }

        strongest_emotion = max(
            non_neutral_emotions,
            key=non_neutral_emotions.get,
        )

        strongest_emotion_score = non_neutral_emotions[strongest_emotion]

        total_positive += positive
        total_negative += negative
        intensity_values.append(strongest_emotion_score)

        for emotion, score in non_neutral_emotions.items():
            emotion_totals[emotion] = emotion_totals.get(emotion, 0.0) + score

        analysed_sentences.append({
            "sentence": sentence,
            "positive": positive,
            "negative": negative,
            "tone_score": sentence_tone_score,
            "emotional_evidence": emotional_evidence,
            "strongest_emotion": strongest_emotion,
            "strongest_emotion_score": strongest_emotion_score,
        })

    article_evidence = total_positive + total_negative

    if article_evidence < 0.10:
        article_tone_score = 50.0
    else:
        article_tone_score = (total_positive / article_evidence) * 100

    intensity_score = (
        sum(intensity_values) / len(intensity_values)
    ) * 100

    average_emotion_scores = {
        emotion: score / len(sentences)
        for emotion, score in emotion_totals.items()
    }

    sorted_emotions = sorted(
        average_emotion_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    top_emotions = [
        {
            "label": emotion,
            "score": round(score * 100, 1),
        }
        for emotion, score in sorted_emotions[:3]
    ]

    if article_tone_score < 40:
        analysed_sentences.sort(
            key=lambda item: item["negative"],
            reverse=True,
        )
    elif article_tone_score > 60:
        analysed_sentences.sort(
            key=lambda item: item["positive"],
            reverse=True,
        )
    else:
        analysed_sentences.sort(
            key=lambda item: item["emotional_evidence"],
            reverse=True,
        )

    evidence_sentences = [
        item["sentence"]
        for item in analysed_sentences[:top_k_sentences]
        if len(item["sentence"].strip()) > 10
    ]

    return {
        "tone_score": round(article_tone_score, 1),
        "tone_label": get_tone_label(article_tone_score),
        "intensity_score": round(intensity_score, 1),
        "intensity_label": get_intensity_label(intensity_score),
        "top_emotions": top_emotions,
        "evidence_sentences": evidence_sentences,
    }