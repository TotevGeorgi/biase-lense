from transformers import pipeline, AutoTokenizer
from typing import List
import math

SUM_MODEL_NAME = "facebook/bart-large-cnn"
summarizer_pipeline = pipeline("summarization", model=SUM_MODEL_NAME)
sum_tokenizer = AutoTokenizer.from_pretrained(SUM_MODEL_NAME)

def _chunk_by_tokens(text: str, max_tokens: int = 850) -> List[str]:
    tokens = sum_tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(sum_tokenizer.decode(chunk_tokens, skip_special_tokens=True))
    return [c.strip() for c in chunks if c.strip()]

def _dynamic_lengths_from_tokens(token_count: int) -> tuple[int, int]:
    if token_count < 220:
        return 60, 20
    if token_count < 450:
        return 90, 30
    if token_count < 900:
        return 120, 40
    return 150, 50

def summarize(text: str) -> str:
    text = " ".join(text.split())
    if len(text.split()) < 40:
        return text

    chunks = _chunk_by_tokens(text, max_tokens=850)

    max_chunks = 5
    if len(chunks) > max_chunks:
        idxs = [round(i) for i in [j * (len(chunks) - 1) / (max_chunks - 1) for j in range(max_chunks)]]
        chunks = [chunks[i] for i in idxs]

    chunk_summaries = []
    for ch in chunks:
        tok_len = len(sum_tokenizer.encode(ch, add_special_tokens=False))
        max_len, min_len = _dynamic_lengths_from_tokens(tok_len)

        out = summarizer_pipeline(
            ch,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
            truncation=True,
        )
        chunk_summaries.append(out[0]["summary_text"].strip())

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    combined = " ".join(chunk_summaries)
    tok_len2 = len(sum_tokenizer.encode(combined, add_special_tokens=False))
    max_len2, min_len2 = _dynamic_lengths_from_tokens(tok_len2)

    final = summarizer_pipeline(
        combined,
        max_length=max_len2,
        min_length=min_len2,
        do_sample=False,
        truncation=True,
    )[0]["summary_text"].strip()

    return final
