from transformers import pipeline

SUM_MODEL_NAME = "facebook/bart-large-cnn"
summarizer_pipeline = pipeline("summarization", model=SUM_MODEL_NAME)


def summarize(text: str) -> str:
    if len(text.split()) < 30:
        return text

    result = summarizer_pipeline(
        text,
        max_length=80,   
        min_length=20,  
        do_sample=False,
        truncation=True,
    )
    return result[0]["summary_text"]
