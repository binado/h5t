build:
    uv run --group docs sphinx-build -E -a -b html -W --keep-going docs docs/_build/html

open: build
    uv run --group docs python -c 'import pathlib, webbrowser; webbrowser.open(pathlib.Path("docs/_build/html/index.html").resolve().as_uri())'
