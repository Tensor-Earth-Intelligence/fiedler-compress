"""
Security hardening tests for the open-core package.

Covers path-traversal rejection, CLI argument bounds, prohibition of unsafe
deserialization (pickle), and prohibition of eval/exec in the source tree.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Path traversal tests
# ---------------------------------------------------------------------------

class TestPathTraversal:
    """Verify that _safe_resolve rejects directory traversal attempts."""

    def test_dotdot_rejected(self):
        from fiedler_optimizer.cli import _safe_resolve
        with pytest.raises(SystemExit):
            _safe_resolve("../etc/passwd", label="test")

    def test_nested_dotdot_rejected(self):
        from fiedler_optimizer.cli import _safe_resolve
        with pytest.raises(SystemExit):
            _safe_resolve("subdir/../../etc/shadow", label="test")

    def test_nonexistent_file_rejected(self):
        from fiedler_optimizer.cli import _safe_resolve
        with pytest.raises(SystemExit):
            _safe_resolve("/nonexistent/path/file.txt", label="test")

    def test_valid_file_accepted(self, tmp_path):
        from fiedler_optimizer.cli import _safe_resolve
        f = tmp_path / "valid.txt"
        f.write_text("hello", encoding="utf-8")
        result = _safe_resolve(str(f), label="test")
        assert result.exists()
        assert result.is_file()

    def test_oversized_file_rejected(self, tmp_path):
        from fiedler_optimizer.cli import _safe_resolve, _MAX_FILE_BYTES
        f = tmp_path / "big.txt"
        # Write just over the limit
        f.write_bytes(b"x" * (_MAX_FILE_BYTES + 1))
        with pytest.raises(SystemExit):
            _safe_resolve(str(f), label="test")

    def test_directory_rejected(self, tmp_path):
        from fiedler_optimizer.cli import _safe_resolve
        with pytest.raises(SystemExit):
            _safe_resolve(str(tmp_path), label="test")

    def test_leading_dotdot_rejected(self):
        """Path starting with .. is rejected."""
        from fiedler_optimizer.cli import _safe_resolve
        with pytest.raises(SystemExit):
            _safe_resolve("../../sensitive_file", label="test")

    def test_middle_dotdot_rejected(self):
        """Path with .. buried in the middle is rejected."""
        from fiedler_optimizer.cli import _safe_resolve
        with pytest.raises(SystemExit):
            _safe_resolve("a/b/../../../etc/passwd", label="test")

    def test_dotdot_at_end_rejected(self):
        """Even trailing .. is rejected (it's a traversal component)."""
        from fiedler_optimizer.cli import _safe_resolve
        with pytest.raises(SystemExit):
            _safe_resolve("somedir/..", label="test")

    def test_empty_path_rejected(self):
        """Empty string path is rejected (not a file)."""
        from fiedler_optimizer.cli import _safe_resolve
        with pytest.raises(SystemExit):
            _safe_resolve("", label="test")

    def test_null_byte_in_filename_rejected(self, tmp_path):
        """Paths with null bytes should raise an error."""
        from fiedler_optimizer.cli import _safe_resolve
        with pytest.raises((SystemExit, ValueError)):
            _safe_resolve(str(tmp_path / "file\x00.txt"), label="test")


# ---------------------------------------------------------------------------
# CLI argument bounds
# ---------------------------------------------------------------------------

class TestCLIBounds:
    def _run_cli(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "fiedler_optimizer.cli", *args],
            capture_output=True, text=True, timeout=10,
        )

    def test_target_ratio_zero_rejected(self):
        result = self._run_cli("optimize", "--target", "0.0", "hello world")
        assert result.returncode != 0
        assert "target" in result.stderr.lower()

    def test_target_ratio_one_rejected(self):
        result = self._run_cli("optimize", "--target", "1.0", "hello world")
        assert result.returncode != 0

    def test_dotdot_file_rejected(self, tmp_path):
        bad_path = str(tmp_path / ".." / "etc" / "passwd")
        result = self._run_cli("optimize", "--file", bad_path)
        assert result.returncode != 0
        assert "traversal" in result.stderr.lower()


# ---------------------------------------------------------------------------
# No-pickle audit
# ---------------------------------------------------------------------------

class TestNoPickle:
    def test_no_pickle_imports_in_source(self):
        """No source file should import pickle, marshal, shelve, or dill."""
        import_pattern = re.compile(
            r"^\s*(import|from)\s+(pickle|_pickle|marshal|shelve|dill)\b",
        )
        src_dir = Path(__file__).resolve().parent.parent / "fiedler_optimizer"
        violations = []
        for py_file in src_dir.rglob("*.py"):
            for i, line in enumerate(py_file.read_text(encoding="utf-8").splitlines(), 1):
                if import_pattern.match(line):
                    violations.append(f"{py_file.name}:{i}: {line.strip()}")
        assert violations == [], f"Pickle/unsafe imports found:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# CLI argument edge cases
# ---------------------------------------------------------------------------

class TestCLIArgumentEdgeCases:
    def _run_cli(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "fiedler_optimizer.cli", *args],
            capture_output=True, text=True, timeout=10,
        )

    def test_target_ratio_negative_rejected(self):
        result = self._run_cli("optimize", "--target", "-0.5", "hello world test text")
        assert result.returncode != 0

    def test_target_ratio_above_one_rejected(self):
        result = self._run_cli("optimize", "--target", "1.5", "hello world test text")
        assert result.returncode != 0

    def test_target_ratio_non_numeric_rejected(self):
        result = self._run_cli("optimize", "--target", "abc", "hello world test text")
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# No eval/exec audit
# ---------------------------------------------------------------------------

class TestNoEvalExec:
    def test_no_eval_or_exec_in_source(self):
        """No source file should use eval() or exec()."""
        dangerous_pattern = re.compile(
            r"(?<!\w)(eval|exec)\s*\(",
        )
        src_dir = Path(__file__).resolve().parent.parent / "fiedler_optimizer"
        violations = []
        for py_file in src_dir.rglob("*.py"):
            for i, line in enumerate(py_file.read_text(encoding="utf-8").splitlines(), 1):
                # Skip comments and strings
                stripped = line.split("#")[0]
                if dangerous_pattern.search(stripped):
                    violations.append(f"{py_file.name}:{i}: {line.strip()}")
        assert violations == [], f"eval/exec usage found:\n" + "\n".join(violations)
