"""The validation engine: a metadata-only walk of a spec tree over a file.

Shapes, dtypes, names, and attrs are read; dataset payloads never are.
All problems found in one walk are batched into a single
:class:`~h5t._errors.ValidationReport`.

Dimension scoping (PLAN.md section 4):

- A class that *declares* a dim (``dims=`` kwarg) binds it locally; every
  use of that class gets an independent binding.
- A dim that appears only in shape strings is *free*: it unifies upward
  through statically declared parents until it reaches a class that
  declares it, or the file root.
- Each child of a dynamic ``Group[T]`` opens a fresh scope; nothing
  unifies across dynamic siblings or escapes to the parent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import h5py
import numpy as np

from h5t._dtypes import dtype_matches
from h5t._errors import ValidationReport
from h5t._shape import AnyDim, FixedDim, NamedDim, format_shape
from h5t._spec import (
    AttrSpec,
    DatasetSpec,
    Extras,
    FromAttr,
    GroupSpec,
    child_path,
)


@dataclass
class _Binding:
    """Collected evidence about one named dimension within one scope."""

    source_value: int | None = None
    source_desc: str | None = None
    sites: list[tuple[str, int, int]] = field(default_factory=list)
    """(dataset path, axis index, observed length) per occurrence."""


@dataclass
class _Frame:
    """One dimension scope on the walk stack."""

    path: str
    catch_all: bool
    declared: frozenset[str] = frozenset()
    bindings: dict[str, _Binding] = field(default_factory=dict)


class _Walker:
    """Stateful spec-vs-file walk accumulating a report."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        self.frames: list[_Frame] = []

    # -- dimension scoping -------------------------------------------------

    def _owning_frame(self, name: str) -> _Frame:
        for frame in reversed(self.frames):
            if frame.catch_all or name in frame.declared:
                return frame
        raise AssertionError("frame stack always ends in a catch-all root frame")

    def _record_site(self, name: str, ds_path: str, axis: int, length: int) -> None:
        frame = self._owning_frame(name)
        frame.bindings.setdefault(name, _Binding()).sites.append((ds_path, axis, length))

    def _push(self, frame: _Frame) -> None:
        self.frames.append(frame)

    def _pop(self) -> None:
        frame = self.frames.pop()
        for name, binding in frame.bindings.items():
            self._finalize_binding(frame, name, binding)

    def _finalize_binding(self, frame: _Frame, name: str, binding: _Binding) -> None:
        def site_line(site: tuple[str, int, int]) -> str:
            path, axis, length = site
            rel = _relpath(path, frame.path)
            suffix = f" (axis {axis})" if axis else ""
            return f"{length} at {rel}{suffix}"

        if binding.source_value is not None:
            bad = [s for s in binding.sites if s[2] != binding.source_value]
            if bad:
                lines = [f"'{name}' does not match its declared source"]
                lines.append(f"{binding.source_value} from {binding.source_desc}")
                lines.extend(site_line(s) for s in bad)
                self.report.add_error(frame.path, "\n".join(lines))
            return
        values = {s[2] for s in binding.sites}
        if len(values) > 1:
            lines = [f"'{name}' has inconsistent values"]
            lines.extend(site_line(s) for s in binding.sites)
            self.report.add_error(frame.path, "\n".join(lines))

    # -- attrs ---------------------------------------------------------------

    def check_attr(
        self,
        spec: AttrSpec,
        node: h5py.Group | h5py.Dataset,
        path: str,
    ) -> None:
        """Validate one declared attribute on ``node``."""
        if spec.h5_name not in node.attrs:
            if spec.optional:
                return
            message = f"required attr '{spec.h5_name}' missing"
            if isinstance(node, h5py.Group) and spec.h5_name in node:
                kind = "dataset" if isinstance(node[spec.h5_name], h5py.Dataset) else "group"
                message += f" \u2014 but a {kind} '{child_path(path, spec.h5_name)}' exists here."
                if kind == "dataset":
                    message += ' Did you mean h5t.Dataset[h5t.f8, "..."]?'
            self.report.add_error(path, message)
            return
        raw = node.attrs[spec.h5_name]
        ok, _, msg = spec.type.check(raw)
        if not ok:
            self.report.add_error(path, f"attr '{spec.h5_name}': {msg}")

    def check_extra_attrs(
        self,
        node: h5py.Group | h5py.Dataset,
        path: str,
        consumed: set[str],
        extras: Extras,
    ) -> None:
        """Apply the extras policy to undeclared attributes on ``node``."""
        if extras is Extras.IGNORE:
            return
        for name in node.attrs:
            if name in consumed:
                continue
            message = f"extra attr '{name}' not in schema"
            if extras is Extras.WARN:
                self.report.add_warning(path, message)
            elif extras is Extras.FORBID:
                self.report.add_error(path, message)

    # -- datasets ------------------------------------------------------------

    def check_dataset(
        self,
        spec: DatasetSpec,
        ds: h5py.Dataset,
        path: str,
        extras: Extras,
    ) -> None:
        """Validate one dataset's dtype, shape terms, and attributes."""
        assert spec.dtype is not None and spec.shape is not None
        if not dtype_matches(spec.dtype, ds.dtype):
            self.report.add_error(
                path,
                f"dtype mismatch: expected {spec.dtype.__name__}, found {ds.dtype}",
            )
        expected_rank = len(spec.shape)
        actual = ds.shape
        if expected_rank == 0:
            if actual != ():
                self.report.add_error(
                    path,
                    f'expected a scalar dataspace (shape ""), found shape {actual}',
                )
        elif len(actual) != expected_rank:
            self.report.add_error(
                path,
                f"rank mismatch: expected {expected_rank} axes"
                f' ("{format_shape(spec.shape)}"), found shape {actual}',
            )
        else:
            for axis, (dim, length) in enumerate(zip(spec.shape, actual, strict=True)):
                match dim:
                    case FixedDim(size=size):
                        if length != size:
                            self.report.add_error(
                                path,
                                f"axis {axis}: expected length {size}, found {length}",
                            )
                    case NamedDim(name=name):
                        self._record_site(name, path, axis, length)
                    case AnyDim():
                        pass
        consumed: set[str] = set()
        for attr_spec in spec.attrs:
            consumed.add(attr_spec.h5_name)
            self.check_attr(attr_spec, ds, path)
        self.check_extra_attrs(ds, path, consumed, extras)

    # -- groups ----------------------------------------------------------------

    def check_group(self, spec: GroupSpec, group: h5py.Group, path: str) -> None:
        """Validate one group: dims, attrs, static children, dynamic children."""
        consumed_attrs: set[str] = set()
        frame: _Frame | None = None
        if spec.dims:
            frame = _Frame(path=path, catch_all=False, declared=frozenset(n for n, _ in spec.dims))
            self._push(frame)
            for name, source in spec.dims:
                if not isinstance(source, FromAttr):
                    continue
                consumed_attrs.add(source.attr)
                binding = frame.bindings.setdefault(name, _Binding())
                if source.attr not in group.attrs:
                    self.report.add_error(
                        path,
                        f"dim '{name}': source attr '{source.attr}' missing",
                    )
                    continue
                raw = group.attrs[source.attr]
                if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, (int, np.integer)):
                    self.report.add_error(
                        path,
                        f"dim '{name}': source attr '{source.attr}' is not an integer"
                        f" (got {type(raw).__name__})",
                    )
                    continue
                binding.source_value = int(raw)
                binding.source_desc = f"attr '{source.attr}' at {path}"

        for attr_spec in spec.attrs:
            consumed_attrs.add(attr_spec.h5_name)
            self.check_attr(attr_spec, group, path)

        consumed_children: set[str] = set()
        for child in spec.children:
            consumed_children.add(child.h5_name)
            cpath = child_path(path, child.h5_name)
            node = group.get(child.h5_name)
            if node is None:
                if not child.optional:
                    self._report_missing_child(child, group, path)
                continue
            if isinstance(child, DatasetSpec):
                if not isinstance(node, h5py.Dataset):
                    self.report.add_error(
                        cpath, "expected a dataset, found a group of the same name"
                    )
                else:
                    self.check_dataset(child, node, cpath, spec.extras)
            else:
                if not isinstance(node, h5py.Group):
                    self.report.add_error(
                        cpath, "expected a group, found a dataset of the same name"
                    )
                else:
                    self.check_group(child, node, cpath)

        if spec.dynamic is not None:
            pattern = spec.dynamic.pattern
            for key in _group_keys(group):
                if key in consumed_children:
                    continue
                if pattern is not None and re.fullmatch(pattern, key) is None:
                    continue
                consumed_children.add(key)
                cpath = child_path(path, key)
                node = group[key]
                if not isinstance(node, (h5py.Group, h5py.Dataset)):
                    self.report.add_error(
                        cpath, f"unsupported HDF5 object of type {type(node).__name__}"
                    )
                    continue
                self._check_dynamic_child(spec, node, cpath)

        self._check_extra_children(spec, group, path, consumed_children)
        self.check_extra_attrs(group, path, consumed_attrs, spec.extras)
        if frame is not None:
            self._pop()

    def _report_missing_child(
        self,
        child: GroupSpec | DatasetSpec,
        group: h5py.Group,
        path: str,
    ) -> None:
        kind = "dataset" if isinstance(child, DatasetSpec) else "group"
        message = f"required {kind} '{child.h5_name}' missing"
        if child.h5_name in group.attrs:
            message += f" \u2014 but an attr '{child.h5_name}' exists here."
        self.report.add_error(path, message)

    def _check_dynamic_child(
        self,
        spec: GroupSpec,
        node: h5py.Group | h5py.Dataset,
        cpath: str,
    ) -> None:
        assert spec.dynamic is not None
        item = spec.dynamic.item
        self._push(_Frame(path=cpath, catch_all=True))
        if isinstance(item, DatasetSpec):
            if not isinstance(node, h5py.Dataset):
                self.report.add_error(cpath, "expected a dataset, found a group")
            else:
                self.check_dataset(item, node, cpath, spec.extras)
        else:
            if not isinstance(node, h5py.Group):
                self.report.add_error(cpath, "expected a group, found a dataset")
            else:
                self.check_group(item, node, cpath)
        self._pop()

    def _check_extra_children(
        self,
        spec: GroupSpec,
        group: h5py.Group,
        path: str,
        consumed: set[str],
    ) -> None:
        if spec.extras is Extras.IGNORE:
            return
        for key in _group_keys(group):
            if key in consumed:
                continue
            kind = "dataset" if isinstance(group[key], h5py.Dataset) else "group"
            message = f"extra {kind} '{key}' not in schema"
            if spec.extras is Extras.WARN:
                self.report.add_warning(path, message)
            elif spec.extras is Extras.FORBID:
                self.report.add_error(path, message)


def _group_keys(group: h5py.Group) -> list[str]:
    """Child names of an h5py group, typed as strings."""
    return [str(key) for key in group.keys()]


def _relpath(path: str, base: str) -> str:
    """Render ``path`` relative to ``base`` for site listings."""
    if base != "/" and path.startswith(base + "/"):
        return "." + path[len(base) :]
    if base == "/" and path.startswith("/"):
        return "." + path
    return path


def run_validation(
    spec: GroupSpec,
    node: h5py.Group,
    *,
    filename: str,
    schema_name: str,
    base_path: str = "/",
) -> ValidationReport:
    """Validate an open HDF5 group (usually the file root) against a spec.

    Parameters
    ----------
    spec : GroupSpec
        The compiled schema spec to check against.
    node : h5py.Group
        The group to walk. Only metadata is read.
    filename : str
        File name used in the rendered report.
    schema_name : str
        Schema class name used in the rendered report.
    base_path : str, optional
        Absolute HDF5 path of ``node``, ``"/"`` by default.

    Returns
    -------
    ValidationReport
        All findings of the walk; empty when the file conforms.
    """
    report = ValidationReport(filename=filename, schema_name=schema_name)
    walker = _Walker(report)
    walker._push(_Frame(path=base_path, catch_all=True))
    walker.check_group(spec, node, base_path)
    walker._pop()
    return report
