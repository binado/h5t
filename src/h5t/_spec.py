"""Public annotation markers and HDF5 path helpers."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from h5t._h5view import _GroupView


@dataclass(frozen=True)
class Glob:
    """Filter which children of a collection field's group are loaded.

    Valid only on a ``dict[str, T]`` / ``Mapping[str, T]`` field.

    Parameters
    ----------
    pattern : str
        A shell-style pattern (``fnmatch.fnmatchcase``: ``*``, ``?``, ``[seq]``),
        matched case-sensitively against each child's name. It may not be empty or
        contain ``/``. Children that do not match are skipped.

    Examples
    --------
    >>> from typing import Annotated
    >>> import h5t
    >>> class Run(h5t.Group):
    ...     chains: Annotated[dict[str, h5t.Dataset], h5t.Glob("chain_?")]
    """

    pattern: str = "*"

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        pattern = self.pattern

        def select(value: Any) -> Any:
            # Only the live HDF5 view is filtered; an in-memory dict is left as given (the
            # writer checks it against the pattern itself).
            if isinstance(value, _GroupView):
                return {name: value[name] for name in value if fnmatch.fnmatchcase(name, pattern)}
            return value

        return core_schema.no_info_before_validator_function(select, handler(source_type))


def child_path(parent: str, name: str) -> str:
    """Join an absolute group path and one immediate child name."""
    return f"/{name}" if parent == "/" else f"{parent.rstrip('/')}/{name}"


def attr_path(parent: str, name: str) -> str:
    """Return a display path for an HDF5 attribute."""
    return f"{parent.rstrip('/') or '/'}@{name}"
