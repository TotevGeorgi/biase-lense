# 🧠 Bias Lens

**See beyond the headline.** 

Bias Lense is an AI-powered web application designed to help users analyze how online articles use language, emotional framing, and tone to influence reader perception. By revealing the subtle layers behind the text, Bias Lense fosters critical reading, media literacy, and a deeper understanding of digital information.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Transformers-NLP-FFD21E.svg?logo=huggingface)

---

## 📖 About Bias Lense

In the modern digital landscape, the way a story is told is often as impactful as the facts it contains. Emotional framing and specific language choices can significantly alter how a reader perceives a topic. 

Bias Lense addresses this challenge by providing an intuitive platform to evaluate online news and articles critically. Intended primarily for students, young adults, and anyone interested in media literacy, this application acts as a digital magnifying glass—highlighting emotional undertones, summarizing core facts, and explaining why an article might feel a certain way to the reader.

---

## ✨ Features

Bias Lense is built with a modular architecture to provide a seamless analytical experience:

*   **Article Analyzer (`analyzer_page.py`)**: Users can submit an article URL to receive an AI-assisted breakdown of its emotional tone and structural framing.
*   **Article Comparison (`comparison_page.py`)**: Submit two distinct article URLs to examine them side-by-side, revealing how different sources present the exact same topic.
*   **Automated Article Fetching (`src/article_fetcher.py`)**: Seamlessly retrieves and extracts clean text directly from user-provided URLs.
*   **Intelligent Summarization (`src/summarizer.py`)**: Condenses long-form journalism into digestible, easy-to-read summaries without losing core context.
*   **Emotion & Classification Engine (`src/classifier.py`)**: Scans the text to identify dominant emotional characteristics and framing techniques.
*   **Reasoning & Explanation (`src/explainer.py`)**: Doesn't just give a score—it provides contextual explanations for *why* the AI assigned a specific emotional or bias metric.
*   **Interactive Dashboard (`dashboard_page.py`)**: A centralized hub for tracking and visualizing analytical results.

---

## 🔍 How It Works

Bias Lense uses a streamlined pipeline to process raw web content into structured, actionable insights.

```mermaid
flowchart TD
    A[User Inputs URL] --> B(Article Fetcher)
    B --> C{Text Cleaning & Extraction}
    C --> D[Summarization Model]
    C --> E[Emotion Classifier]
    C --> F[Embeddings Generator]
    
    D --> G(Explainer Engine)
    E --> G
    F --> G
    
    G --> H((Streamlit Interface))
    
    classDef primary fill:#8b5cf6,stroke:#fff,stroke-width:2px,color:#fff;
    classDef secondary fill:#2b2b2b,stroke:#fff,stroke-width:2px,color:#fff;
    class A,H primary;
    class B,C,D,E,F,G secondary;
