# Contributing

## Setup

```bash
uv sync
```

installs dependencies into `.venv`, including the `dev` group (pytest, hypothesis, ty, ruff).

Optionally install the `prek` pre-commit hooks (ruff-check, ruff-format, ty), configured in
`prek.toml`:

```bash
uvx prek install
```

## Workflow

```bash
uv run pytest                            # full test suite
uv run pytest tests/test_compile.py::test_inheritance_defaults_and_extras   # single test
uv run --python 3.14 pytest              # PEP 649 paths, skipped on the default interpreter
uv run ty check                          # type check
uv run ruff check .                      # lint
uv run ruff format --check src tests     # format check, as CI runs it
```

`tests/test_pep649.py` only exercises the PEP 649 lazy-annotation paths on Python 3.14+; it is
skipped on the `.python-version` default (3.11), so run it explicitly with
`uv run --python 3.14 pytest` before touching anything in `_compile.py` related to forward
references.

See `AGENTS.md` for the architecture and the invariants each module depends on (compilation
timing, instance construction via `object.__new__`, the frozen `Group`/`Dataset` public surface,
etc.) before making non-trivial changes.

## Commit messages

This project follows [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`,
`fix:`, `chore:`, `refactor:`, and `!` for breaking changes, e.g. `refactor!:`).

## Pull requests

CI (`.github/workflows/ci.yml`) runs the test matrix (Python 3.11–3.14), lint, format check, and
type check on every push to `main` and every pull request. Make sure
`uv run ruff check . && uv run ruff format --check src tests && uv run ty check && uv run pytest`
passes locally first.

## Releasing (maintainers)

1. Update `CHANGELOG.md`: move `[Unreleased]` entries under a new `## [X.Y.Z]` heading and add
   its compare link at the bottom.
2. Bump the version in both `pyproject.toml` (`[project].version`) and `src/h5t/__init__.py`
   (`__version__`) — `tests/test_metadata.py` asserts they match.
3. Commit (`chore: release vX.Y.Z`), tag (`git tag vX.Y.Z`), and push both
   (`git push && git push --tags`).
4. Pushing the tag triggers `.github/workflows/release.yml`, which builds the package and
   publishes it to PyPI via trusted publishing.
