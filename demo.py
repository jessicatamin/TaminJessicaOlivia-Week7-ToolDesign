"""
demo.py

Demo script showing the keyword extractor tool used in context:
- Tool being called by an agent/workflow
- Successful execution
- Error handling (bad input)

Run:
    python3 demo.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping

from tool import KeywordExtractorTool


def _pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)


@dataclass
class ToolCall:
    """A simple representation of an agent tool call."""

    name: str
    inputs: Dict[str, Any]


class SimpleAgentWorkflow:
    """
    Minimal "agent/workflow" that can call registered tools.

    This is a lightweight stand-in for a real agent runtime (e.g., function-calling LLM).
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Callable[[Mapping[str, Any]], Dict[str, Any]]] = {
            KeywordExtractorTool.NAME: KeywordExtractorTool.run,
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
        return self._tools[call.name](call.inputs)


def main() -> None:
    workflow = SimpleAgentWorkflow()

    print("=== Tool metadata (schema) ===")
    print("Tool:", KeywordExtractorTool.NAME)
    print("Description:", KeywordExtractorTool.DESCRIPTION)
    print("Input schema:\n", _pretty(KeywordExtractorTool.PARAMETER_SCHEMA))
    print("Output schema:\n", _pretty(KeywordExtractorTool.OUTPUT_SCHEMA))
    print()

    print("=== Successful execution ===")
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


if __name__ == "__main__":
    main()

