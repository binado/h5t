"""The ``h5t`` command line interface.

v0.1 ships one subcommand::

    h5t check FILE --schema pkg.module:ClassName

Exit codes: 0 when the file validates, 1 when validation finds errors,
2 for usage problems (bad schema reference, unreadable file, incoherent
schema).
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import NoReturn

from h5t._compile import File
from h5t._errors import SchemaError


def _fail(message: str) -> NoReturn:
    """Print a usage-level error to stderr and exit with code 2."""
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def _load_schema(ref: str) -> type[File]:
    """Import a schema class from a ``pkg.module:ClassName`` reference.

    Parameters
    ----------
    ref : str
        Colon-separated module path and class name.

    Returns
    -------
    type of File
        The schema class.

    Raises
    ------
    SystemExit
        With code 2 when the reference is malformed, the module cannot be
        imported, or the attribute is not an ``h5t.File`` schema.
    """
    module_name, sep, class_name = ref.partition(":")
    if not sep or not module_name or not class_name:
        _fail(f"--schema must look like pkg.module:ClassName, got {ref!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        _fail(f"could not import module {module_name!r}: {exc}")
    try:
        schema = getattr(module, class_name)
    except AttributeError:
        _fail(f"module {module_name!r} has no attribute {class_name!r}")
    if not (isinstance(schema, type) and issubclass(schema, File)):
        _fail(f"{ref!r} is not an h5t.File schema class")
    return schema


def _cmd_check(args: argparse.Namespace) -> int:
    """Run ``h5t check``: validate a file against a schema and print the report."""
    schema = _load_schema(args.schema)
    try:
        schema.validate_schema()
    except SchemaError as exc:
        _fail(f"schema {args.schema!r} is incoherent: {exc}")
    try:
        view = schema.open(args.file, validate=False)
    except OSError as exc:
        _fail(f"could not open {args.file!r}: {exc}")
    with view:
        report = view.check()
    print(report.render())
    return 0 if report.ok else 1


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``h5t`` console script.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="h5t",
        description="Schema layer for HDF5: validate files without reading payloads.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="validate an HDF5 file against a schema class")
    check.add_argument("file", help="path to the HDF5 file")
    check.add_argument(
        "--schema",
        required=True,
        help="schema reference, e.g. gwlib.schemas:PEResult",
    )
    check.set_defaults(func=_cmd_check)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
