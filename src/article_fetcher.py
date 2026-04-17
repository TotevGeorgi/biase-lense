import re
import requests
from bs4 import BeautifulSoup

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text

def fetch_article_text(url: str, timeout: int = 12) -> str:
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    article = soup.find("article")
    if article:
        text = article.get_text(" ", strip=True)
    else:
        ps = soup.find_all("p")
        text = " ".join(p.get_text(" ", strip=True) for p in ps)

    text = _clean_text(text)

    if len(text.split()) < 80:
        raise ValueError("Could not extract enough article text (maybe paywall or JS-rendered page).")

    return text
