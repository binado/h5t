"""Run mypy and pyright over the acceptance example; assert zero errors.

This locks in the typing spike's result: the frozen annotation forms are
accepted by both checkers and members resolve to the right view types.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).parent / "acceptance_example.py"


@pytest.mark.parametrize("checker", ["mypy", "pyright"])
def test_acceptance_example_is_clean(checker: str, tmp_path: Path):
    # Run from an empty cwd so the project's own (stricter, src-scoped)
    # config does not apply; the promise is default-mode acceptance.
    args = [sys.executable, "-m", checker, str(EXAMPLE)]
    if checker == "mypy":
        args += ["--no-incremental", f"--cache-dir={tmp_path / 'mypy_cache'}"]
    result = subprocess.run(
        args,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"{checker} rejected the frozen annotation forms:\n{result.stdout}\n{result.stderr}"
    )
