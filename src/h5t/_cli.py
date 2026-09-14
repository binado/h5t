"""The ``h5t check`` command."""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import NoReturn

from h5t._compile import load
from h5t._errors import SchemaError, ValidationError


def _fail(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def _load_schema(ref: str) -> type:
    module_name, separator, class_name = ref.partition(":")
    if not separator or not module_name or not class_name:
        _fail(f"--schema must look like pkg.module:ClassName, got {ref!r}")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        _fail(f"could not import module {module_name!r}: {exc}")
    try:
        schema = getattr(module, class_name)
    except AttributeError:
        _fail(f"module {module_name!r} has no attribute {class_name!r}")
    if not (isinstance(schema, type) and "__h5t_record__" in schema.__dict__):
        _fail(f"{ref!r} is not a decorated record (@h5t.dataset/@h5t.group)")
    return schema


def _cmd_check(args: argparse.Namespace) -> int:
    schema = _load_schema(args.schema)
    try:
        load(schema, args.file, root=args.root)
    except ValidationError as exc:
        print(f"invalid: {args.file} against {schema.__name__}: {exc}")
        return 1
    except (OSError, ValueError, TypeError, SchemaError) as exc:
        _fail(f"could not check {args.file!r}: {exc}")
    print(f"ok: {args.file} validates against {schema.__name__} at {args.root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the command line interface and return its process exit code."""
    parser = argparse.ArgumentParser(
        prog="h5t",
        description="Validate and materialize an HDF5 group schema.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="check an HDF5 file against a group schema")
    check.add_argument("file", help="path to the HDF5 file")
    check.add_argument("--schema", required=True, help="schema reference, e.g. pkg.schemas:Result")
    check.add_argument("--root", default="/", help="absolute HDF5 group path (default: /)")
    check.set_defaults(func=_cmd_check)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
