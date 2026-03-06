"""
keyword_from_url.py

Extract keywords from any website or article URL using DeepSeek API.

Flow:
  1. Fetch page content from the URL (HTML or text).
  2. Use DeepSeek to extract the main article/body text (removes nav, ads, etc.).
  3. Run the keyword extraction tool on that text.

Requires:
  - DEEPSEEK_API_KEY in the environment (or pass api_key=).
  - pip install openai requests

Usage:
  from keyword_from_url import extract_keywords_from_url

  result = extract_keywords_from_url(
      "https://example.com/article",
      api_key=os.environ.get("DEEPSEEK_API_KEY"),
      top_k=10,
  )
  # result["ok"] and result["data"]["keywords"] or result["error"]
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Mapping, Optional

from tool import KeywordExtractorTool

# Optional deps: fail at call time with a clear message if missing
def _get_openai_client(api_key: Optional[str] = None):
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "DeepSeek integration requires the openai package. "
            "Install with: pip install openai"
        ) from e
    key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError(
            "DeepSeek API key is required. Set DEEPSEEK_API_KEY in the environment "
            "or pass api_key= to extract_keywords_from_url."
        )
    return OpenAI(api_key=key, base_url="https://api.deepseek.com")


def _get_requests():
    try:
        import requests
        return requests
    except ImportError as e:
        raise ImportError(
            "URL fetching requires the requests package. "
            "Install with: pip install requests"
        ) from e


def fetch_page(url: str, timeout: int = 15) -> str:
    """
    Fetch raw body of a URL as text.

    Args:
        url: Full URL (http/https).
        timeout: Request timeout in seconds.

    Returns:
        Response body as string (HTML or plain text).

    Raises:
        ValueError: If URL is invalid or response indicates error.
        requests.RequestException: On network errors.
    """
    if not url or not isinstance(url, str):
        raise ValueError("url must be a non-empty string")
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        raise ValueError("url must start with http:// or https://")

    requests = _get_requests()
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def extract_article_with_deepseek(
    content: str,
    api_key: Optional[str] = None,
    *,
    max_input_chars: int = 100_000,
) -> str:
    """
    Use DeepSeek to extract main article/body text from web page content.

    Sends a prefix of the content (to stay within context limits) and asks
    the model to return only the main textual content, suitable for keyword
    extraction.

    Args:
        content: Raw HTML or page text.
        api_key: DeepSeek API key, or None to use DEEPSEEK_API_KEY env.
        max_input_chars: Truncate content to this many characters before sending.

    Returns:
        Plain-text article body.
    """
    client = _get_openai_client(api_key)
    truncated = content[:max_input_chars] if len(content) > max_input_chars else content
    if len(content) > max_input_chars:
        truncated += "\n\n[Content truncated for API.]"

    system = (
        "You are a helper that extracts the main article or body text from web page content. "
        "Return ONLY the main textual content: paragraphs, headings, and list items that form "
        "the article. Omit navigation, ads, footers, and boilerplate. Output plain text only, "
        "no HTML, no explanations."
    )
    user = f"Extract the main article text from this web page content:\n\n{truncated}"

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        stream=False,
    )
    text = (response.choices[0].message.content or "").strip()
    return text if text else "(No article text extracted)"


def extract_keywords_from_url(
    url: str,
    *,
    api_key: Optional[str] = None,
    top_k: int = 10,
    ngram_range: tuple[int, int] = (1, 2),
    min_token_length: int = 3,
    normalization: str = "lower",
    scoring: str = "tfidf_like",
    timeout: int = 15,
    max_input_chars: int = 100_000,
) -> Dict[str, Any]:
    """
    Extract keywords from a website or article URL using DeepSeek and the keyword tool.

    Fetches the URL, uses DeepSeek to extract main article text, then runs
    keyword extraction. Return shape matches KeywordExtractorTool.run(...)
    (ok, data, error).

    Args:
        url: Full URL of the page or article.
        api_key: DeepSeek API key; defaults to DEEPSEEK_API_KEY env.
        top_k: Maximum number of keywords to return.
        ngram_range: (min_n, max_n) for n-grams.
        min_token_length: Minimum token length.
        normalization: "lower" or "none".
        scoring: "tfidf_like" or "frequency".
        timeout: Seconds to wait when fetching the URL.
        max_input_chars: Max characters of page content to send to DeepSeek.

    Returns:
        Dict with keys:
          - ok: bool
          - data: None or {"keywords": [...], "meta": {...}, "url": str, "article_preview": str}
          - error: None or {"code": str, "message": str, "details": dict}
    """
    try:
        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            return {
                "ok": False,
                "data": None,
                "error": {
                    "code": "MISSING_API_KEY",
                    "message": "DEEPSEEK_API_KEY is not set and api_key was not passed.",
                    "details": {},
                },
            }
    except Exception as e:
        return {
            "ok": False,
            "data": None,
            "error": {
                "code": "CONFIG_ERROR",
                "message": str(e),
                "details": {"exception_type": type(e).__name__},
            },
        }

    try:
        raw = fetch_page(url, timeout=timeout)
    except Exception as e:
        return {
            "ok": False,
            "data": None,
            "error": {
                "code": "FETCH_ERROR",
                "message": str(e),
                "details": {"url": url, "exception_type": type(e).__name__},
            },
        }

    try:
        article_text = extract_article_with_deepseek(
            raw, api_key=api_key, max_input_chars=max_input_chars
        )
    except Exception as e:
        return {
            "ok": False,
            "data": None,
            "error": {
                "code": "DEEPSEEK_ERROR",
                "message": str(e),
                "details": {"url": url, "exception_type": type(e).__name__},
            },
        }

    result = KeywordExtractorTool.run({
        "text": article_text,
        "top_k": top_k,
        "ngram_range": list(ngram_range),
        "min_token_length": min_token_length,
        "normalization": normalization,
        "scoring": scoring,
    })

    if not result.get("ok"):
        return result

    # Attach URL and a short preview for convenience
    preview = (article_text[:500] + "…") if len(article_text) > 500 else article_text
    result["data"] = {
        **result["data"],
        "url": url,
        "article_preview": preview,
    }
    return result
