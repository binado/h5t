"""Exception types and the structured validation report.

``SchemaError`` describes an incoherent declaration, ``Invalid`` is the
local signal used inside node hooks, and ``ValidationError`` aggregates a
file's reported problems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class H5TError(Exception):
    """Base class for all h5t exceptions."""


class SchemaError(H5TError):
    """The schema declaration itself is incoherent.

    Raised at class-creation time (or by ``validate_schema()``) for problems
    such as unresolvable dtypes, duplicate HDF5 names within one namespace,
    or conflicting redeclarations across bases.
    """


class Invalid(H5TError):
    """Signal invalid content from a user-defined node validator.

    The validation walk catches this exception, attaches the invoking
    node's path, and records its message in the aggregate report. Other
    exceptions propagate as bugs in validator code.
    """


class ClosedFileError(H5TError):
    """A view was used after its owning file handle was closed."""


class SchemaMismatchError(H5TError):
    """Member access hit a node that does not conform to the schema.

    Raised on attribute access when a file was opened with
    ``validate=False``: the error names the HDF5 path and the expectation
    instead of surfacing a bare ``KeyError`` from h5py internals.

    Parameters
    ----------
    path : str
        Absolute HDF5 path of the offending node or attribute.
    message : str
        Human-readable statement of the expectation that failed.
    """

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message


class Severity(Enum):
    """Severity of a single validation problem."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Problem:
    """One validation finding at one HDF5 path.

    Attributes
    ----------
    path : str
        Absolute HDF5 path of the group or dataset the finding is reported
        against.
    message : str
        Finding text. May span multiple lines; continuation lines are
        indented by the report renderer.
    severity : Severity
        Whether the finding fails validation or is merely notable.
    """

    path: str
    message: str
    severity: Severity


@dataclass
class ValidationReport:
    """Structured result of one validation walk.

    Attributes
    ----------
    filename : str
        Name of the file that was validated.
    schema_name : str
        Name of the schema class the file was validated against.
    problems : list of Problem
        All findings, in traversal order.
    """

    filename: str
    schema_name: str
    problems: list[Problem] = field(default_factory=list)

    @property
    def errors(self) -> list[Problem]:
        """All findings with :attr:`Severity.ERROR`."""
        return [p for p in self.problems if p.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Problem]:
        """All findings with :attr:`Severity.WARNING`."""
        return [p for p in self.problems if p.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        """Whether the walk produced no errors (warnings are allowed)."""
        return not self.errors

    def add_error(self, path: str, message: str) -> None:
        """Record an error finding at ``path``."""
        self.problems.append(Problem(path, message, Severity.ERROR))

    def add_warning(self, path: str, message: str) -> None:
        """Record a warning finding at ``path``."""
        self.problems.append(Problem(path, message, Severity.WARNING))

    def render(self) -> str:
        """Render the report as the per-path tree diff described in PLAN.md.

        Returns
        -------
        str
            Multi-line report; the empty-report rendering states that the
            file validated cleanly.
        """
        n = len(self.problems)
        if n == 0:
            return f"ok: {self.filename} validates against {self.schema_name}"
        noun = "problem" if n == 1 else "problems"
        lines = [f"{n} {noun} in {self.filename} against {self.schema_name}"]
        by_path: dict[str, list[Problem]] = {}
        for problem in self.problems:
            by_path.setdefault(problem.path, []).append(problem)
        for path, problems in by_path.items():
            lines.append("")
            lines.append(path)
            for problem in problems:
                marker = "\u2717" if problem.severity is Severity.ERROR else "!"
                first, *rest = problem.message.splitlines()
                lines.append(f"  {marker} {first}")
                lines.extend(f"      {extra}" for extra in rest)
        return "\n".join(lines)


class ValidationError(H5TError):
    """A file failed validation against a schema.

    All problems found in one walk are batched into a single exception; the
    structured findings are available on :attr:`report`.

    Parameters
    ----------
    report : ValidationReport
        The findings of the failed walk.
    file_closed : bool, optional
        Whether the file handle was closed before raising (the behaviour of
        ``open(..., validate=True)``); if so the rendered message says how
        to reopen for inspection.
    """

    def __init__(self, report: ValidationReport, *, file_closed: bool = False) -> None:
        super().__init__(report.render())
        self.report = report
        self.file_closed = file_closed

    def __str__(self) -> str:
        text = self.report.render()
        if self.file_closed:
            text += (
                "\n\nFile was closed. To inspect, use "
                f"{self.report.schema_name}.open(path, validate=False)."
            )
        return text
