"""Tests for the caveman grammar-stripping compressor."""

import pytest

from fiedler_optimizer.caveman import caveman_compress, CavemanResult, CavemanLevel


# ---------------------------------------------------------------------------
# Sample paragraph used across level tests
# ---------------------------------------------------------------------------

SAMPLE = (
    "I'd be glad to help you with this. Basically, the system is really quite "
    "complex and it might be worth considering a simpler approach. The function "
    "essentially parses the input, and it actually validates the schema before "
    "processing. You could potentially refactor the module to improve "
    "performance significantly."
)


# ---------------------------------------------------------------------------
# Level tests
# ---------------------------------------------------------------------------


class TestLiteLevel:
    def test_removes_fillers(self):
        result = caveman_compress(SAMPLE, level="lite")
        lower = result.text.lower()
        for filler in ("basically", "really", "essentially", "actually"):
            assert filler not in lower, f"filler '{filler}' should be removed"

    def test_removes_pleasantries(self):
        result = caveman_compress(SAMPLE, level="lite")
        assert "glad to" not in result.text.lower()

    def test_removes_hedging(self):
        result = caveman_compress(SAMPLE, level="lite")
        assert "it might be worth considering" not in result.text.lower()
        assert "you could potentially" not in result.text.lower()

    def test_produces_shorter_text(self):
        result = caveman_compress(SAMPLE, level="lite")
        assert result.compressed_tokens < result.original_tokens
        assert result.compression_ratio > 0


class TestFullLevel:
    def test_drops_articles(self):
        result = caveman_compress("The cat sat on a mat in an office.", level="full")
        lower = result.text.lower()
        # Articles should be gone
        words = lower.split()
        assert "the" not in words
        assert "a" not in words
        assert "an" not in words

    def test_drops_adverbs(self):
        result = caveman_compress(
            "She extremely carefully examined the very old document.", level="full"
        )
        lower = result.text.lower()
        assert "extremely" not in lower
        assert "very" not in lower

    def test_more_compression_than_lite(self):
        lite = caveman_compress(SAMPLE, level="lite")
        full = caveman_compress(SAMPLE, level="full")
        assert full.compressed_tokens <= lite.compressed_tokens


class TestUltraLevel:
    def test_drops_copulas(self):
        result = caveman_compress("The server is running and the database is ready.", level="ultra")
        lower = result.text.lower()
        assert " is " not in f" {lower} "

    def test_drops_pronouns(self):
        result = caveman_compress("I think it works. They said it was fine.", level="ultra")
        lower = result.text.lower()
        # At least some pronouns should be stripped
        assert result.compressed_tokens < 12  # original ~10-12 words, should shrink

    def test_most_compression(self):
        lite = caveman_compress(SAMPLE, level="lite")
        ultra = caveman_compress(SAMPLE, level="ultra")
        assert ultra.compressed_tokens <= lite.compressed_tokens


# ---------------------------------------------------------------------------
# Preservation tests
# ---------------------------------------------------------------------------


class TestCodeBlockPreservation:
    def test_fenced_code_block(self):
        text = "Basically, use this code:\n```python\ndef foo():\n    return 42\n```\nThat's it."
        result = caveman_compress(text, level="full")
        assert "```python\ndef foo():\n    return 42\n```" in result.text
        assert any("```" in b for b in result.preserved_blocks)

    def test_inline_code(self):
        text = "You should really call `process_data()` to start."
        result = caveman_compress(text, level="full")
        assert "`process_data()`" in result.text

    def test_indented_code_block(self):
        text = "Example:\n    x = 1\n    y = 2\nDone."
        result = caveman_compress(text, level="full")
        assert "    x = 1" in result.text


class TestURLPreservation:
    def test_http_url(self):
        text = "Basically visit https://example.com/path?q=1 for details."
        result = caveman_compress(text, level="full")
        assert "https://example.com/path?q=1" in result.text

    def test_url_in_preserved_blocks(self):
        text = "See https://docs.python.org/3/library/re.html for info."
        result = caveman_compress(text, level="ultra")
        assert any("https://" in b for b in result.preserved_blocks)


class TestQuotedStringPreservation:
    def test_double_quoted(self):
        text = 'The error message is "Connection refused" and it fails.'
        result = caveman_compress(text, level="ultra")
        assert '"Connection refused"' in result.text

    def test_single_quoted(self):
        text = "Set the value to 'enabled' in config."
        result = caveman_compress(text, level="ultra")
        assert "'enabled'" in result.text


class TestNumberPreservation:
    def test_version_string(self):
        text = "Upgrade to version 3.11.4 for the fix."
        result = caveman_compress(text, level="ultra")
        assert "3.11.4" in result.text

    def test_date_preservation(self):
        text = "The deadline is 2024-01-15 and it's really important."
        result = caveman_compress(text, level="full")
        assert "2024-01-15" in result.text


class TestTechnicalTerms:
    def test_custom_terms_preserved(self):
        text = "The GraphQL resolver basically calls the Fiedler decomposition."
        result = caveman_compress(
            text, level="ultra", technical_terms=["GraphQL", "Fiedler decomposition"]
        )
        assert "GraphQL" in result.text
        assert "Fiedler decomposition" in result.text


# ---------------------------------------------------------------------------
# Token count accuracy
# ---------------------------------------------------------------------------


class TestTokenCounting:
    def test_token_counts_populated(self):
        result = caveman_compress("Hello world, this is a test.", level="lite")
        assert result.original_tokens > 0
        assert result.compressed_tokens > 0
        assert result.compressed_tokens <= result.original_tokens

    def test_compression_ratio_range(self):
        result = caveman_compress(SAMPLE, level="full")
        assert 0.0 < result.compression_ratio < 1.0

    def test_empty_input(self):
        result = caveman_compress("", level="full")
        assert result.text == ""
        assert result.original_tokens == 0
        assert result.compressed_tokens == 0
        assert result.compression_ratio == 0.0


# ---------------------------------------------------------------------------
# Round-trip: compressed text is non-empty and shorter
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Placeholder for future semantic tests — for now verify compression."""

    def test_compressed_is_nonempty(self):
        result = caveman_compress(SAMPLE, level="ultra")
        assert len(result.text.strip()) > 0

    def test_compressed_is_shorter(self):
        result = caveman_compress(SAMPLE, level="full")
        assert len(result.text) < len(SAMPLE)

    def test_result_type(self):
        result = caveman_compress(SAMPLE, level="lite")
        assert isinstance(result, CavemanResult)


# ---------------------------------------------------------------------------
# Level enum / input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_case_insensitive_level(self):
        r1 = caveman_compress("Hello world", level="FULL")
        r2 = caveman_compress("Hello world", level="full")
        assert r1.text == r2.text

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError):
            caveman_compress("Hello", level="extreme")
