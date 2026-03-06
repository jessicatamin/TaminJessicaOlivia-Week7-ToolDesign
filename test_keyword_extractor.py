import unittest
from typing import Any, Dict
from unittest.mock import patch

import keyword_from_url


class TestKeywordExtractorTool(unittest.TestCase):
    def test_extract_keywords_from_url_valid_mocks_return_keywords(self) -> None:
        """
        Valid URL should return ok=True and a non-empty keywords list when
        fetch + article extraction succeed.
        """

        url = "https://example.com/article"

        with patch.object(keyword_from_url, "fetch_page", return_value="<html>...</html>"):
            with patch.object(
                keyword_from_url,
                "extract_article_with_deepseek",
                return_value=(
                    "This is the main article text about keyword extraction and AI agents. "
                    "Keyword extraction finds keyphrases in text."
                ),
            ):
                result: Dict[str, Any] = keyword_from_url.extract_keywords_from_url(
                    url,
                    api_key="fake-key",
                    top_k=5,
                    timeout=1,
                )

        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("ok"))
        self.assertIsNone(result.get("error"))
        self.assertIsInstance(result.get("data"), dict)
        self.assertEqual(result["data"].get("url"), url)
        self.assertIn("article_preview", result["data"])
        self.assertIsInstance(result["data"].get("keywords"), list)
        self.assertGreater(len(result["data"]["keywords"]), 0)

    def test_extract_keywords_from_url_missing_api_key(self) -> None:
        """Missing API key should return structured MISSING_API_KEY error."""

        result = keyword_from_url.extract_keywords_from_url(
            "https://example.com/article",
            api_key=None,
            top_k=5,
        )
        self.assertFalse(result.get("ok"))
        self.assertIsNone(result.get("data"))
        self.assertIsInstance(result.get("error"), dict)
        self.assertEqual(result["error"].get("code"), "MISSING_API_KEY")

    def test_extract_keywords_from_url_empty_url(self) -> None:
        """Empty URL should return structured FETCH_ERROR."""

        result = keyword_from_url.extract_keywords_from_url(
            "",
            api_key="fake-key",
        )
        self.assertFalse(result.get("ok"))
        self.assertIsNone(result.get("data"))
        self.assertEqual(result.get("error", {}).get("code"), "FETCH_ERROR")

    def test_extract_keywords_from_url_invalid_url_scheme(self) -> None:
        """Invalid URL (no scheme) should return structured FETCH_ERROR."""

        result = keyword_from_url.extract_keywords_from_url(
            "not-a-valid-url",
            api_key="fake-key",
        )
        self.assertFalse(result.get("ok"))
        self.assertIsNone(result.get("data"))
        self.assertEqual(result.get("error", {}).get("code"), "FETCH_ERROR")
        self.assertIn("http://", result.get("error", {}).get("message", ""))

    def test_extract_keywords_from_url_unreachable_network(self) -> None:
        """Unreachable links (network errors) should return structured FETCH_ERROR."""

        class _DummyRequests:
            class RequestException(Exception):
                pass

            def get(self, *_args: Any, **_kwargs: Any) -> Any:
                raise self.RequestException("network down")

        with patch.object(keyword_from_url, "_get_requests", return_value=_DummyRequests()):
            result = keyword_from_url.extract_keywords_from_url(
                "https://example.com/article",
                api_key="fake-key",
                timeout=1,
            )

        self.assertFalse(result.get("ok"))
        self.assertIsNone(result.get("data"))
        self.assertEqual(result.get("error", {}).get("code"), "FETCH_ERROR")
        self.assertIn("network down", result.get("error", {}).get("message", ""))


if __name__ == "__main__":
    unittest.main()

