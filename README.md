# Bias Lense – News Emotion Agent

This is my individual AI project for [course name].

It:
- Uses sentence-transformer embeddings to build a semantic search index over news articles
- Summarizes retrieved articles with a BART-based summarization model
- Classifies the emotional tone (happiness, fear, motivation) using a BERT-based emotion classifier
- Includes a Jupyter notebook demo on the CNN News Articles 2011–2022 dataset and a small hand-crafted emotional dataset

Main folders:
- `src/` – core Python modules (agent, embeddings, summarizer, classifier)
- `notebooks/` – Jupyter demo (`cnn_demo.ipynb`)
