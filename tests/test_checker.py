import json
import subprocess
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("case", "errors"),
    [("positive", []), ("negative", ["bad-return", "bad-return", "bad-argument-type"])],
)
def test_shape_contracts(case, errors):
    result = subprocess.run(
        [
            "pyrefly",
            "check",
            "--config",
            "pyrefly.toml",
            "--output-format",
            "json",
            f"tests/cases/{case}.py",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == bool(errors), result.stdout + result.stderr
    diagnostics = json.loads(result.stdout)["errors"]
    assert Counter(error["name"] for error in diagnostics) == Counter(errors), result.stdout
