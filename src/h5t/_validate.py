"""Validation of built-in schema constraints and user-defined node hooks.

The walker checks structure, dtypes, attrs, and extras before invoking the
``validate()`` hook on each structurally accessible typed view. Hooks run
post-order and may inspect or read anything exposed beneath their node.
"""

from __future__ import annotations

import re
from typing import Any

import h5py

from h5t._dtypes import dtype_matches
from h5t._errors import Invalid, ValidationReport
from h5t._spec import AttrSpec, DatasetSpec, Extras, GroupSpec, NodeSpec, child_path


class _Walker:
    """Stateful spec-vs-file walk accumulating a report."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report

    def _run_hook(self, view: Any, path: str) -> None:
        """Run one user hook, recording only deliberate invalidity."""
        try:
            result = view.validate()
        except Invalid as exc:
            self.report.add_error(path, str(exc))
            return
        if result is not None:
            raise TypeError(
                f"{type(view).__name__}.validate() returned {result!r}; validators must return None"
            )

    # -- attrs -------------------------------------------------------------

    def check_attr(
        self,
        spec: AttrSpec,
        node: h5py.Group | h5py.Dataset,
        path: str,
    ) -> bool:
        """Validate one attr and report whether its descriptor is safe to use."""
        if spec.h5_name not in node.attrs:
            if spec.optional:
                return True
            message = f"required attr '{spec.h5_name}' missing"
            if isinstance(node, h5py.Group) and spec.h5_name in node:
                kind = "dataset" if isinstance(node[spec.h5_name], h5py.Dataset) else "group"
                message += f" — but a {kind} '{child_path(path, spec.h5_name)}' exists here."
                if kind == "dataset":
                    message += " Did you mean h5t.Dataset[h5t.f8]?"
            self.report.add_error(path, message)
            return False
        raw = node.attrs[spec.h5_name]
        ok, _, msg = spec.type.check(raw)
        if not ok:
            self.report.add_error(path, f"attr '{spec.h5_name}': {msg}")
        return ok

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

    # -- datasets ----------------------------------------------------------

    def check_dataset(
        self,
        spec: DatasetSpec,
        dataset: h5py.Dataset,
        path: str,
        extras: Extras,
        view: Any,
    ) -> bool:
        """Validate one dataset and invoke its hook when attrs are accessible."""
        assert spec.dtype is not None
        if not dtype_matches(spec.dtype, dataset.dtype):
            self.report.add_error(
                path,
                f"dtype mismatch: expected {spec.dtype.__name__}, found {dataset.dtype}",
            )

        accessible = True
        consumed: set[str] = set()
        for attr_spec in spec.attrs:
            consumed.add(attr_spec.h5_name)
            accessible = self.check_attr(attr_spec, dataset, path) and accessible
        self.check_extra_attrs(dataset, path, consumed, extras)

        if accessible:
            self._run_hook(view, path)
        return accessible

    # -- groups ------------------------------------------------------------

    def check_group(
        self,
        spec: GroupSpec,
        group: h5py.Group,
        path: str,
        view: Any,
    ) -> bool:
        """Validate one group and invoke its hook after accessible children."""
        accessible = True
        consumed_attrs: set[str] = set()
        for attr_spec in spec.attrs:
            consumed_attrs.add(attr_spec.h5_name)
            accessible = self.check_attr(attr_spec, group, path) and accessible

        consumed_children: set[str] = set()
        for child in spec.children:
            consumed_children.add(child.h5_name)
            cpath = child_path(path, child.h5_name)
            node = group.get(child.h5_name)
            if node is None:
                if not child.optional:
                    self._report_missing_child(child, group, path)
                    accessible = False
                continue

            child_view = view._h5t_bind(child, cpath)
            if isinstance(child, DatasetSpec):
                if not isinstance(node, h5py.Dataset):
                    self.report.add_error(
                        cpath, "expected a dataset, found a group of the same name"
                    )
                    accessible = False
                else:
                    accessible = (
                        self.check_dataset(child, node, cpath, spec.extras, child_view)
                        and accessible
                    )
            elif not isinstance(node, h5py.Group):
                self.report.add_error(cpath, "expected a group, found a dataset of the same name")
                accessible = False
            else:
                accessible = self.check_group(child, node, cpath, child_view) and accessible

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
                    accessible = False
                    continue
                accessible = self._check_dynamic_child(spec, node, cpath, view) and accessible

        self._check_extra_children(spec, group, path, consumed_children)
        self.check_extra_attrs(group, path, consumed_attrs, spec.extras)
        if accessible:
            self._run_hook(view, path)
        return accessible

    def _report_missing_child(
        self,
        child: GroupSpec | DatasetSpec,
        group: h5py.Group,
        path: str,
    ) -> None:
        kind = "dataset" if isinstance(child, DatasetSpec) else "group"
        message = f"required {kind} '{child.h5_name}' missing"
        if child.h5_name in group.attrs:
            message += f" — but an attr '{child.h5_name}' exists here."
        self.report.add_error(path, message)

    def _check_dynamic_child(
        self,
        spec: GroupSpec,
        node: h5py.Group | h5py.Dataset,
        cpath: str,
        parent_view: Any,
    ) -> bool:
        assert spec.dynamic is not None
        item = spec.dynamic.item
        item_view = parent_view._h5t_bind(item, cpath)
        if isinstance(item, DatasetSpec):
            if not isinstance(node, h5py.Dataset):
                self.report.add_error(cpath, "expected a dataset, found a group")
                return False
            return self.check_dataset(item, node, cpath, spec.extras, item_view)
        if not isinstance(node, h5py.Group):
            self.report.add_error(cpath, "expected a group, found a dataset")
            return False
        return self.check_group(item, node, cpath, item_view)

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
    """Return child names of an h5py group as strings."""
    return [str(key) for key in group.keys()]


def run_validation(
    view: Any,
    *,
    filename: str,
    schema_name: str,
) -> ValidationReport:
    """Validate a typed view recursively and return every finding."""
    report = ValidationReport(filename=filename, schema_name=schema_name)
    walker = _Walker(report)
    spec: NodeSpec = view._h5t_spec
    node = view._h5t_node()
    path: str = view._h5t_path
    if isinstance(spec, DatasetSpec):
        if not isinstance(node, h5py.Dataset):
            report.add_error(path, "expected a dataset, found a group")
        else:
            walker.check_dataset(spec, node, path, Extras.IGNORE, view)
    elif not isinstance(node, h5py.Group):
        report.add_error(path, "expected a group, found a dataset")
    else:
        walker.check_group(spec, node, path, view)
    return report
