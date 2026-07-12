import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def check(case: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["pyrefly", "check", "--config", "pyrefly.toml", f"tests/cases/{case}.py"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_positive_contracts() -> None:
    result = check("positive")
    assert result.returncode == 0, result.stdout


def test_negative_contracts() -> None:
    result = check("negative")
    assert result.returncode != 0
    assert "Size mismatch: expected 2, got 3" in result.stdout
    assert "Tensor rank mismatch: expected 3 dimensions, got 2 dimensions" in result.stdout
    assert "Size mismatch: expected 3, got 2" in result.stdout


if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__]))
