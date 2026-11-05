mod docs

default:
    @just --list

lint:
    uv run ruff check .

test:
    uv run pytest

typecheck:
    uv run ty check

fmt:
    uv run ruff format .
