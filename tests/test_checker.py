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
    assert "Shape dimension mismatch: expected Int[2], got Int[3]" in result.stdout
    assert "Tensor rank mismatch: expected 3 dimensions, got 2 dimensions" in result.stdout
    assert "Shape dimension mismatch: expected Int[3], got Int[2]" in result.stdout


if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__]))
