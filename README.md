# Keyword Extraction Tool + DeepSeek URL Demo

## Overview

This project provides a small, agent-friendly **keyword extraction tool** and a demo workflow.

- **`tool.py`**: local keyword extraction (no external NLP dependencies), with:
  - clear schemas
  - input validation
  - structured `{ok, data, error}` responses
  - a simple `Tool` wrapper (`tool.execute(**kwargs)`)
- **`keyword_from_url.py`**: optional URL pipeline that:
  - fetches a web page
  - uses **DeepSeek** to extract the *main article text*
  - runs the local keyword extractor on the extracted text
- **`demo.py`**: runnable script demonstrating:
  - a workflow calling the tool
  - successful execution
  - error handling (bad inputs, missing API key, invalid URL)

Purpose: make it easy for anyone (or an AI agent) to run keyword extraction on either **raw text** or **any public URL/article** (when DeepSeek is configured).

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Create a `.env` file (DeepSeek configuration)

For URL-based keyword extraction, you need a DeepSeek API key and (optionally) a custom base URL and model.

1. Create a file named **`.env`** in the project root (same folder as `demo.py`).
2. Add the following environment variables:

```bash
# Required
DEEPSEEK_API_KEY="your-api-key-here"

# Optional (defaults shown)
DEEPSEEK_API_BASE="https://api.deepseek.com"
DEEPSEEK_MODEL="deepseek-chat"
```

Notes:
- **`DEEPSEEK_API_KEY`** is required to make requests to DeepSeek.
- **`DEEPSEEK_API_BASE`** lets you point to a compatible endpoint if needed.
- **`DEEPSEEK_MODEL`** controls which DeepSeek model is used (for example `deepseek-chat`).

### Security reminder (important)

- **Never commit your `.env` file** (it contains secrets).
- Add `.env` to `.gitignore` before committing anything.

This repo includes a starter `.gitignore` that ignores `.env`. If you create your own, make sure it contains:

```gitignore
.env
```

## Run the demo

```bash
python3 demo.py
```

If `DEEPSEEK_API_KEY` is not set, the demo will still run and show a structured error response for the URL-based step.

## Examples (inputs/outputs)

## Realistic use case

### Build an SEO/content brief from an article URL

Scenario: you’re writing a blog post or marketing page and want to quickly extract *topic keywords* and *keyphrases* from a competitor or industry article (for SEO research or to build a content outline).

What you do:
- Provide the article URL.
- The workflow fetches the page, uses DeepSeek to extract the main article text, then returns ranked keywords.
- You use the top keywords to:
  - draft headings/subheadings
  - choose internal-link anchor text
  - identify repeated terms to cover in the post

Example:

```python
import os
from keyword_from_url import extract_keywords_from_url

result = extract_keywords_from_url(
    "https://en.wikipedia.org/wiki/Keyword_extraction",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    top_k=12,
    ngram_range=(1, 2),
)

if result["ok"]:
    for item in result["data"]["keywords"]:
        print(item["keyword"], item["score"], item["count"])
else:
    print("Error:", result["error"]["code"], result["error"]["message"])
```

Tip: for SEO-style phrases, try `ngram_range=(1, 2)` or `(2, 3)` to bias toward multi-word keyphrases.

### Example 1: local keyword extraction (text)

Input (tool call):

```python
from tool import keyword_extractor_tool

result = keyword_extractor_tool.execute(
    text="A workflow calls a keyword extraction tool for an AI agent.",
    top_k=5,
    ngram_range=[1, 2],
    scoring="tfidf_like",
)
print(result)
```

Output shape (example):

```json
{
  "ok": true,
  "data": {
    "keywords": [
      {"keyword": "keyword extraction", "score": 2.01, "count": 1},
      {"keyword": "workflow calls", "score": 2.01, "count": 1}
    ],
    "meta": {
      "ngram_range": [1, 2],
      "scoring": "tfidf_like",
      "top_k": 5
    }
  },
  "error": null
}
```

### Example 2: error handling (bad input)

Input (missing required `text`):

```python
from tool import keyword_extractor_tool

print(keyword_extractor_tool.execute(top_k="not-an-int"))
```

Output shape (example):

```json
{
  "ok": false,
  "data": null,
  "error": {
    "code": "INVALID_INPUT",
    "message": "Missing required field: text",
    "details": {"exception_type": "ValueError"}
  }
}
```

### Example 3: URL → DeepSeek → keywords

If you have `DEEPSEEK_API_KEY` set, you can run:

```python
import os
from keyword_from_url import extract_keywords_from_url

result = extract_keywords_from_url(
    "https://en.wikipedia.org/wiki/Keyword_extraction",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    top_k=8,
)
print(result["ok"], result.get("data", {}).get("url"))
```

On success, `data` includes:
- `keywords`: ranked keywords/keyphrases
- `meta`: extractor metadata
- `url`: the input URL
- `article_preview`: first ~500 chars of extracted article text

## Design decisions

- **Dependency-free keyword extraction**: the local extractor uses regex tokenization + n-grams + a simple scoring method. This keeps setup light and behavior predictable.
- **Agent-friendly responses**: the main entrypoint returns a JSON-serializable envelope:
  - `ok: true/false`
  - `data` on success
  - `error` on failure with a stable error code and details
- **Pluggable tool wrapper**: `Tool(name, description, fn)` + `tool.execute(**kwargs)` makes it easy to register tools in workflows.
- **DeepSeek for article text extraction**: HTML parsing and “main content” extraction is delegated to the model so it can work across many websites without site-specific rules.

## Limitations

- **Keyword quality**: the scoring is heuristic (frequency/TF‑IDF-like). It is not a full NLP pipeline (no lemmatization/stemming, no POS tagging, no embeddings).
- **Language/stopwords**: built-in stopwords are **English-focused**; other languages may need custom stopwords.
- **Tokenization**: regex tokenization is conservative and may not handle all scripts/punctuation perfectly.
- **URL extraction costs & constraints**:
  - requires an API key, network access, and incurs token/cost usage
  - page content is truncated (`max_input_chars`) before sending to DeepSeek
  - websites with heavy JS rendering, paywalls, or anti-bot protections may fail to fetch
- **Compliance**: always ensure you have the right to fetch/process a website’s content and respect site terms/robots policies where applicable.

