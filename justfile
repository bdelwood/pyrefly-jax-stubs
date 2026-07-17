# pyrefly-jax-stubs task runner.

default:
    @just --list

sync:
    uv sync

lint:
    uv run ruff check .
    uv run ruff format --check .

fmt:
    uv run ruff check --fix .
    uv run ruff format .

test:
    uv run pytest

build:
    uv build
