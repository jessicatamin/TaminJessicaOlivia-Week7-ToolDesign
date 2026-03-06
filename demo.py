"""
demo.py

Demo script showing the keyword extractor tool used in context:
- Tool being called by an agent/workflow
- Successful execution (plain text and URL-based with DeepSeek)
- Error handling (bad input, missing API key, invalid URL)

Run:
    python3 demo.py

For URL-based extraction, set DEEPSEEK_API_KEY in the environment
(or create a .env and load it, e.g. with python-dotenv).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict

from tool import KeywordExtractorTool, Tool, keyword_extractor_tool


def _pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)


@dataclass
class ToolCall:
    """A simple representation of an agent tool call."""

    name: str
    inputs: Dict[str, Any]


class SimpleAgentWorkflow:
    """
    Minimal "agent/workflow" that can call registered tools via the Tool wrapper.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {
            keyword_extractor_tool.name: keyword_extractor_tool,
        }

    def call_tool(self, call: ToolCall) -> Dict[str, Any]:
        if call.name not in self._tools:
            return {
                "ok": False,
                "data": None,
                "error": {
                    "code": "UNKNOWN_TOOL",
                    "message": f"Tool not registered: {call.name}",
                    "details": {"known_tools": sorted(self._tools.keys())},
                },
            }
        return self._tools[call.name].execute(**call.inputs)


def main() -> None:
    workflow = SimpleAgentWorkflow()

    print("=== Tool wrapper (Tool class) ===")
    print("Tool:", keyword_extractor_tool.name)
    print("Description:", keyword_extractor_tool.description)
    print("Execute via: tool.execute(text='...', top_k=10, ...)")
    print()
    print("=== Tool metadata (schema) ===")
    print("Tool:", KeywordExtractorTool.NAME)
    print("Description:", KeywordExtractorTool.DESCRIPTION)
    print("Input schema:\n", _pretty(KeywordExtractorTool.PARAMETER_SCHEMA))
    print("Output schema:\n", _pretty(KeywordExtractorTool.OUTPUT_SCHEMA))
    print()

    print("=== Successful execution (workflow calls tool.execute(**inputs)) ===")
    call_ok = ToolCall(
        name="keyword_extractor",
        inputs={
            "text": (
                "This demo shows a workflow calling a keyword extraction tool. "
                "The tool returns ranked keywords and keyphrases for an AI agent."
            ),
            "top_k": 6,
            "ngram_range": [1, 2],
            "scoring": "tfidf_like",
        },
    )
    result_ok = workflow.call_tool(call_ok)
    print(_pretty(result_ok))
    print()

    print("=== Error handling (bad input) ===")
    call_bad = ToolCall(
        name="keyword_extractor",
        inputs={
            # Missing required field "text"
            "top_k": "not-an-int",
            "ngram_range": "1,2",
        },
    )
    result_bad = workflow.call_tool(call_bad)
    print(_pretty(result_bad))
    print()

    print("=== URL-based keyword extraction (DeepSeek) ===")
    try:
        from keyword_from_url import extract_keywords_from_url
    except ImportError as err:
        print("Skipped (install deps: pip install openai requests):", err)
        return

    # Success path: use a real public URL if API key is set
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if api_key:
        sample_url = "https://en.wikipedia.org/wiki/Keyword_extraction"
        print(f"Fetching and extracting keywords from: {sample_url}")
        result_url = extract_keywords_from_url(
            sample_url,
            api_key=api_key,
            top_k=8,
            timeout=10,
        )
        if result_url.get("ok"):
            print("Success.")
            for r in (result_url.get("data") or {}).get("keywords", [])[:8]:
                print(f"  {r.get('keyword', ''):<30} score={r.get('score', 0):.3f} count={r.get('count', 0)}")
            if result_url.get("data", {}).get("article_preview"):
                preview = (result_url["data"]["article_preview"] or "")[:200]
                print("Article preview:", preview, "…")
        else:
            print(_pretty(result_url))
    else:
        print("DEEPSEEK_API_KEY not set. Showing error response instead:")
        result_no_key = extract_keywords_from_url(
            "https://example.com/some-article",
            top_k=5,
        )
        print(_pretty(result_no_key))
    print()

    print("=== Error handling (invalid URL) ===")
    result_invalid = extract_keywords_from_url(
        "not-a-valid-url",
        api_key=api_key or "dummy-key-for-demo",
        top_k=5,
    )
    print(_pretty(result_invalid))


if __name__ == "__main__":
    main()

