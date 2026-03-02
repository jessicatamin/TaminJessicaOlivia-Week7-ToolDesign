"""
tool.py

Keyword extraction tool (no external dependencies).

This module is designed for AI-agent "tool" style use:
- It provides a clear, machine-readable parameter schema.
- It exposes a single `KeywordExtractorTool.run(...)` entrypoint.
- It returns JSON-serializable output.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union


TokenNormalization = Literal["lower", "none"]
ScoringMethod = Literal["tfidf_like", "frequency"]


DEFAULT_STOPWORDS_EN: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "you",
        "your",
        "we",
        "they",
        "them",
        "this",
        "these",
        "those",
        "or",
        "not",
        "but",
        "if",
        "then",
        "than",
        "can",
        "could",
        "should",
        "would",
        "i",
        "me",
        "my",
        "our",
        "us",
    }
)


@dataclass(frozen=True)
class KeywordResult:
    """
    One extracted keyword (or keyphrase).

    Attributes:
        keyword: The keyword / keyphrase (string).
        score: A relevance score (higher is better).
        count: Number of occurrences in the text (for the keyword itself).
    """

    keyword: str
    score: float
    count: int


@dataclass
class KeywordExtractionInput:
    """
    Input parameters for keyword extraction.

    Notes:
        - This dataclass is optional convenience; the tool also accepts a plain dict.
        - All fields are JSON-serializable.
    """

    text: str
    top_k: int = 10
    ngram_range: Tuple[int, int] = (1, 2)
    min_token_length: int = 3
    normalization: TokenNormalization = "lower"
    scoring: ScoringMethod = "tfidf_like"
    stopwords: Optional[List[str]] = None
    deduplicate: bool = True


@dataclass
class KeywordExtractionOutput:
    """
    Output payload for keyword extraction.

    Attributes:
        keywords: Ranked keywords (best first).
        meta: Small metadata about the extraction run.
    """

    keywords: List[KeywordResult]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolError:
    """
    Structured error returned by tool wrappers.

    Attributes:
        code: Stable, machine-friendly error code.
        message: Human-friendly description of what went wrong.
        details: Optional structured details (JSON-serializable).
    """

    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ToolResponse:
    """
    Standard JSON-serializable response envelope for agent tools.

    Exactly one of `data` or `error` will be present depending on `ok`.
    """

    ok: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[ToolError] = None


def _normalize_token(token: str, normalization: TokenNormalization) -> str:
    if normalization == "lower":
        return token.lower()
    if normalization == "none":
        return token
    raise ValueError(f"Unsupported normalization: {normalization!r}")


def _tokenize(text: str, normalization: TokenNormalization) -> List[str]:
    """
    Tokenize text into word-like tokens.

    Implementation details:
        - Uses a conservative regex to keep letters/numbers and internal apostrophes.
        - Designed to be predictable and dependency-free.
    """

    raw = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text)
    return [_normalize_token(t, normalization) for t in raw]


def _make_ngrams(tokens: Sequence[str], n: int) -> List[str]:
    if n <= 0:
        return []
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1)]


def _validate_input(inp: KeywordExtractionInput) -> None:
    if not isinstance(inp.text, str):
        raise TypeError("text must be a string")
    if inp.top_k <= 0:
        raise ValueError("top_k must be > 0")
    if inp.min_token_length <= 0:
        raise ValueError("min_token_length must be > 0")
    if len(inp.ngram_range) != 2:
        raise ValueError("ngram_range must be a 2-tuple (min_n, max_n)")
    min_n, max_n = inp.ngram_range
    if min_n <= 0 or max_n <= 0:
        raise ValueError("ngram_range values must be > 0")
    if min_n > max_n:
        raise ValueError("ngram_range must satisfy min_n <= max_n")


def _coerce_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        v = value.strip()
        if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
            return int(v)
    raise TypeError(f"{field_name} must be an integer")


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1", "yes", "y", "on"):
            return True
        if v in ("false", "0", "no", "n", "off"):
            return False
    raise TypeError(f"{field_name} must be a boolean")


def _coerce_str(value: Any, *, field_name: str) -> str:
    if isinstance(value, str):
        return value
    raise TypeError(f"{field_name} must be a string")


def _parse_inputs(inputs: Mapping[str, Any]) -> KeywordExtractionInput:
    """
    Parse and validate raw tool inputs into a strongly-typed KeywordExtractionInput.

    Raises:
        TypeError / ValueError: For invalid or missing inputs.
    """

    if not isinstance(inputs, Mapping):
        raise TypeError("inputs must be an object/dict")

    if "text" not in inputs:
        raise ValueError("Missing required field: text")

    text = _coerce_str(inputs.get("text"), field_name="text")
    top_k = _coerce_int(inputs.get("top_k", 10), field_name="top_k")
    min_token_length = _coerce_int(
        inputs.get("min_token_length", 3), field_name="min_token_length"
    )

    normalization_raw = inputs.get("normalization", "lower")
    normalization = _coerce_str(normalization_raw, field_name="normalization")
    if normalization not in ("lower", "none"):
        raise ValueError("normalization must be one of: lower, none")

    scoring_raw = inputs.get("scoring", "tfidf_like")
    scoring = _coerce_str(scoring_raw, field_name="scoring")
    if scoring not in ("tfidf_like", "frequency"):
        raise ValueError("scoring must be one of: tfidf_like, frequency")

    ngram_range_raw = inputs.get("ngram_range", [1, 2])
    if not isinstance(ngram_range_raw, (list, tuple)) or len(ngram_range_raw) != 2:
        raise TypeError("ngram_range must be a 2-item list/tuple: [min_n, max_n]")
    ngram_range = (
        _coerce_int(ngram_range_raw[0], field_name="ngram_range[0]"),
        _coerce_int(ngram_range_raw[1], field_name="ngram_range[1]"),
    )

    stopwords_raw = inputs.get("stopwords", None)
    stopwords: Optional[List[str]]
    if stopwords_raw is None:
        stopwords = None
    else:
        if not isinstance(stopwords_raw, list):
            raise TypeError("stopwords must be a list of strings or null")
        stopwords = []
        for i, s in enumerate(stopwords_raw):
            stopwords.append(_coerce_str(s, field_name=f"stopwords[{i}]"))

    deduplicate = _coerce_bool(inputs.get("deduplicate", True), field_name="deduplicate")

    return KeywordExtractionInput(
        text=text,
        top_k=top_k,
        ngram_range=ngram_range,
        min_token_length=min_token_length,
        normalization=normalization,  # type: ignore[assignment]
        scoring=scoring,  # type: ignore[assignment]
        stopwords=stopwords,
        deduplicate=deduplicate,
    )


def extract_keywords(params: KeywordExtractionInput) -> KeywordExtractionOutput:
    """
    Extract keywords/keyphrases from a text.

    The algorithm is intentionally simple and explainable:
    - Tokenize -> remove stopwords and short tokens
    - Build n-grams for n in `ngram_range`
    - Score candidates using either:
        - "frequency": raw counts
        - "tfidf_like": count * log(1 + N / (1 + count)) where N is token count

    Args:
        params: KeywordExtractionInput with extraction settings.

    Returns:
        KeywordExtractionOutput: ranked keywords and basic metadata.
    """

    _validate_input(params)

    stop = set(DEFAULT_STOPWORDS_EN)
    if params.stopwords is not None:
        stop.update(_normalize_token(s, params.normalization) for s in params.stopwords)

    tokens_all = _tokenize(params.text, params.normalization)
    tokens = [t for t in tokens_all if len(t) >= params.min_token_length and t not in stop]

    # Candidate generation
    min_n, max_n = params.ngram_range
    candidates: List[str] = []
    for n in range(min_n, max_n + 1):
        if n == 1:
            candidates.extend(tokens)
        else:
            candidates.extend(_make_ngrams(tokens, n))

    counts: Dict[str, int] = {}
    for c in candidates:
        counts[c] = counts.get(c, 0) + 1

    # Scoring
    N = max(1, len(tokens))
    scored: List[KeywordResult] = []
    for kw, ct in counts.items():
        if params.scoring == "frequency":
            score = float(ct)
        elif params.scoring == "tfidf_like":
            score = float(ct) * math.log(1.0 + (N / (1.0 + ct)))
        else:
            raise ValueError(f"Unsupported scoring: {params.scoring!r}")
        scored.append(KeywordResult(keyword=kw, score=score, count=ct))

    scored.sort(key=lambda r: (r.score, r.count, len(r.keyword)), reverse=True)

    if params.deduplicate:
        # Keep the best-scoring instance of a keyword (mostly relevant if normalization="none").
        seen: set[str] = set()
        deduped: List[KeywordResult] = []
        for r in scored:
            if r.keyword in seen:
                continue
            seen.add(r.keyword)
            deduped.append(r)
        scored = deduped

    top = scored[: params.top_k]
    return KeywordExtractionOutput(
        keywords=top,
        meta={
            "token_count_raw": len(tokens_all),
            "token_count_filtered": len(tokens),
            "candidate_count": len(candidates),
            "unique_candidate_count": len(counts),
            "scoring": params.scoring,
            "ngram_range": list(params.ngram_range),
            "normalization": params.normalization,
            "min_token_length": params.min_token_length,
            "top_k": params.top_k,
        },
    )


class KeywordExtractorTool:
    """
    AI-agent tool wrapper for keyword extraction.

    This class provides:
    - `PARAMETER_SCHEMA`: a clear input schema (JSON-schema-like)
    - `OUTPUT_SCHEMA`: a clear output schema
    - `run(...)`: a single entrypoint returning a JSON-serializable dict
    - structured error handling via a response envelope
    """

    NAME = "keyword_extractor"
    DESCRIPTION = "Extract ranked keywords/keyphrases from input text."

    PARAMETER_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "required": ["text"],
        "additionalProperties": False,
        "properties": {
            "text": {"type": "string", "description": "Input text to analyze."},
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "default": 10,
                "description": "Maximum number of keywords to return.",
            },
            "ngram_range": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "integer", "minimum": 1},
                "default": [1, 2],
                "description": "Inclusive n-gram range as [min_n, max_n].",
            },
            "min_token_length": {
                "type": "integer",
                "minimum": 1,
                "default": 3,
                "description": "Minimum token length to keep (after normalization).",
            },
            "normalization": {
                "type": "string",
                "enum": ["lower", "none"],
                "default": "lower",
                "description": "Token normalization strategy.",
            },
            "scoring": {
                "type": "string",
                "enum": ["tfidf_like", "frequency"],
                "default": "tfidf_like",
                "description": "How to score candidates.",
            },
            "stopwords": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "default": None,
                "description": "Extra stopwords to exclude (in addition to built-in list).",
            },
            "deduplicate": {
                "type": "boolean",
                "default": True,
                "description": "Remove duplicate keyword strings in output.",
            },
        },
    }

    OUTPUT_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "required": ["ok"],
        "additionalProperties": False,
        "properties": {
            "ok": {"type": "boolean"},
            "data": {
                "type": ["object", "null"],
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["keyword", "score", "count"],
                            "additionalProperties": False,
                            "properties": {
                                "keyword": {"type": "string"},
                                "score": {"type": "number"},
                                "count": {"type": "integer"},
                            },
                        },
                    },
                    "meta": {"type": "object"},
                },
            },
            "error": {
                "type": ["object", "null"],
                "required": ["code", "message"],
                "additionalProperties": False,
                "properties": {
                    "code": {"type": "string"},
                    "message": {"type": "string"},
                    "details": {"type": ["object", "null"]},
                },
            },
        },
    }

    @classmethod
    def run(cls, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Run keyword extraction with input validation and structured error handling.

        Args:
            inputs: Mapping (dict-like) adhering to `PARAMETER_SCHEMA`.

        Returns:
            A JSON-serializable dict adhering to `OUTPUT_SCHEMA`.

            Success shape:
                {"ok": True, "data": {"keywords": [...], "meta": {...}}, "error": None}

            Error shape:
                {"ok": False, "data": None, "error": {"code": "...", "message": "...", "details": {...}}}

        Error handling:
            - This method catches validation/runtime errors and returns them in a structured form.
            - If you prefer exceptions, use `run_or_raise(...)`.
        """

        try:
            data = cls.run_or_raise(inputs)
            resp = ToolResponse(ok=True, data=data, error=None)
            return {
                "ok": resp.ok,
                "data": resp.data,
                "error": None,
            }
        except (TypeError, ValueError) as e:
            err = ToolError(
                code="INVALID_INPUT",
                message=str(e) or "Invalid input",
                details={"exception_type": type(e).__name__},
            )
        except Exception as e:  # defensive: tool wrappers should not crash agents
            err = ToolError(
                code="INTERNAL_ERROR",
                message=str(e) or "Internal error",
                details={"exception_type": type(e).__name__},
            )
        resp = ToolResponse(ok=False, data=None, error=err)
        return {
            "ok": resp.ok,
            "data": None,
            "error": asdict(resp.error) if resp.error else None,
        }

    @classmethod
    def run_or_raise(cls, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Run keyword extraction and raise exceptions on error.

        Args:
            inputs: Mapping (dict-like) adhering to `PARAMETER_SCHEMA`.

        Returns:
            Dict with keys:
                - "keywords": list[{"keyword": str, "score": float, "count": int}]
                - "meta": dict[str, Any]

        Raises:
            TypeError / ValueError: For invalid inputs.
            Exception: For unexpected runtime failures.
        """

        params = _parse_inputs(inputs)
        out = extract_keywords(params)
        return {
            "keywords": [asdict(k) for k in out.keywords],
            "meta": out.meta,
        }


if __name__ == "__main__":
    demo_text = (
        "Designing a custom tool for an AI agent: we need keyword extraction, clear schemas, "
        "and proper documentation. Keyword extraction should return keyphrases too."
    )
    result = KeywordExtractorTool.run({"text": demo_text, "top_k": 8, "ngram_range": [1, 2]})
    if not result["ok"]:
        print("Error:", result["error"])
        raise SystemExit(1)
    for r in (result["data"] or {}).get("keywords", []):
        print(f"{r['keyword']:<25} score={r['score']:.3f} count={r['count']}")
