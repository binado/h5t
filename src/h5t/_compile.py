"""The class-syntax front-end: schema classes compile to plain spec trees.

``__init_subclass__`` on :class:`Group` and :class:`Dataset` compiles class
bodies into :mod:`h5t._spec` nodes stored on ``__h5spec__``. Member kinds
are inferred from the annotation alone (PLAN.md section 4):

============================  =======================================
Annotation                    Kind
============================  =======================================
``h5t.Dataset[...]``          dataset, declared inline
``h5t.Dataset`` subclass      dataset, named type with its own attrs
``h5t.Group`` subclass        subgroup with statically declared contents
``h5t.Group[T]``              subgroup whose dynamic children satisfy T
anything else                 attribute
============================  =======================================
"""

from __future__ import annotations

import inspect
import os
import re
import types
import typing
import warnings
from dataclasses import replace
from typing import Annotated, Any, ClassVar, Generic, Literal, Self, TypeVar, cast

import h5py
import numpy as np

from h5t._dtypes import DType
from h5t._errors import SchemaError, ValidationError
from h5t._spec import (
    AttrSpec,
    AttrType,
    DatasetSpec,
    DynamicSpec,
    Extras,
    GroupSpec,
    Keys,
    MemberSpec,
    Name,
    NodeSpec,
)
from h5t._views import (
    AttrDescriptor,
    DatasetViewOps,
    FileContext,
    GroupViewOps,
    NodeDescriptor,
    make_view,
)

DT = TypeVar("DT", bound=DType)
T = TypeVar("T")

_ATTR_BASES = (str, int, float, bool)


# --------------------------------------------------------------------------
# Annotation classification
# --------------------------------------------------------------------------


def _unwrap_annotation(owner: type, py_name: str, ann: Any) -> tuple[Any, list[Any], bool]:
    """Strip ``Annotated`` and ``| None`` layers off an annotation.

    Returns
    -------
    core : object
        The remaining core annotation.
    metadata : list
        Collected ``Annotated`` metadata, outermost first.
    optional : bool
        Whether ``None`` appeared in a union layer.
    """
    metadata: list[Any] = []
    optional = False
    core = ann
    while True:
        if typing.get_origin(core) is Annotated:
            metadata.extend(core.__metadata__)
            core = typing.get_args(core)[0]
            continue
        origin = typing.get_origin(core)
        if origin is typing.Union or origin is types.UnionType:
            args = list(typing.get_args(core))
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) != len(args):
                optional = True
            if len(non_none) != 1:
                raise SchemaError(
                    f"{owner.__name__}.{py_name}: unions other than '| None'"
                    " are not supported in v0.1"
                )
            core = non_none[0]
            continue
        return core, metadata, optional


def _extract_metadata(
    owner: type, py_name: str, metadata: list[Any]
) -> tuple[str | None, str | None]:
    """Pull ``Name`` and ``Keys`` markers out of Annotated metadata."""
    h5_name: str | None = None
    pattern: str | None = None
    for item in metadata:
        if isinstance(item, Name):
            h5_name = item.name
        elif isinstance(item, Keys):
            try:
                re.compile(item.pattern)
            except re.error as exc:
                raise SchemaError(
                    f"{owner.__name__}.{py_name}: invalid Keys pattern {item.pattern!r}: {exc}"
                ) from exc
            pattern = item.pattern
        elif isinstance(item, str):
            raise SchemaError(
                f"{owner.__name__}.{py_name}: shape metadata is no longer supported;"
                " override validate() to check dataset shapes"
            )
    return h5_name, pattern


def _attr_type_for(owner: type, py_name: str, core: Any) -> AttrType:
    """Build the checked :class:`AttrType` for an attr-by-elimination member."""
    if typing.get_origin(core) is Literal:
        values = typing.get_args(core)
        if not values:
            raise SchemaError(f"{owner.__name__}.{py_name}: empty Literal")
        for value in values:
            if not isinstance(value, _ATTR_BASES):
                raise SchemaError(
                    f"{owner.__name__}.{py_name}: unsupported Literal value {value!r}"
                )
        return AttrType(literals=values)
    if core in _ATTR_BASES or core is np.ndarray:
        return AttrType(base=core)
    raise SchemaError(
        f"{owner.__name__}.{py_name}: unsupported attr type {core!r};"
        " v0.1 supports str, int, float, bool, Literal[...], and np.ndarray"
    )


def _classify_member(
    owner: type,
    py_name: str,
    ann: Any,
    default: Any,
    has_default: bool,
    *,
    allow_children: bool,
    default_extras: Extras,
) -> MemberSpec:
    """Compile one annotated class-body member into its spec."""
    core, metadata, optional = _unwrap_annotation(owner, py_name, ann)
    name_meta, pattern = _extract_metadata(owner, py_name, metadata)
    h5_name = name_meta if name_meta is not None else py_name

    # Dataset member: inline subscript form or named subclass.
    if isinstance(core, type) and issubclass(core, Dataset):
        if not allow_children:
            raise SchemaError(f"{owner.__name__}.{py_name}: a dataset cannot contain child nodes")
        if has_default:
            raise SchemaError(f"{owner.__name__}.{py_name}: defaults apply to attrs only")
        template = core.__h5spec__
        if template.dtype is None:
            raise SchemaError(
                f"{owner.__name__}.{py_name}: dataset member has no dtype;"
                " use h5t.Dataset[h5t.f8] or the dtype= class kwarg"
            )
        return replace(template, py_name=py_name, h5_name=h5_name, optional=optional)

    # Dynamic group member: Group[T] inline.
    origin = typing.get_origin(core)
    if isinstance(origin, type) and issubclass(origin, Group):
        if not allow_children:
            raise SchemaError(f"{owner.__name__}.{py_name}: a dataset cannot contain child nodes")
        item = _dynamic_item_spec(owner, py_name, typing.get_args(core))
        if item is None:
            raise SchemaError(
                f"{owner.__name__}.{py_name}: Group[...] requires a Group or"
                " Dataset schema as its item type"
            )
        return GroupSpec(
            py_name=py_name,
            h5_name=h5_name,
            extras=default_extras,
            dynamic=DynamicSpec(item=item, pattern=pattern),
            optional=optional,
            view_type=origin,
        )

    # Static subgroup member: named Group subclass.
    if isinstance(core, type) and issubclass(core, Group):
        if not allow_children:
            raise SchemaError(f"{owner.__name__}.{py_name}: a dataset cannot contain child nodes")
        if has_default:
            raise SchemaError(f"{owner.__name__}.{py_name}: defaults apply to attrs only")
        spec = core.__h5spec__
        if pattern is not None:
            if spec.dynamic is None:
                raise SchemaError(
                    f"{owner.__name__}.{py_name}: Keys(...) requires a dynamic Group[T] member"
                )
            spec = replace(spec, dynamic=replace(spec.dynamic, pattern=pattern))
        return replace(spec, py_name=py_name, h5_name=h5_name, optional=optional)

    # Everything else is an attribute, by elimination.
    if pattern is not None:
        raise SchemaError(f"{owner.__name__}.{py_name}: Keys(...) applies to dynamic group members")
    attr_type = _attr_type_for(owner, py_name, core)
    return AttrSpec(
        py_name=py_name,
        h5_name=h5_name,
        type=attr_type,
        optional=optional,
        default=default if has_default else None,
    )


def _dynamic_item_spec(owner: type, py_name: str, args: tuple[Any, ...]) -> NodeSpec | None:
    """Resolve the item spec of a ``Group[T]`` subscription, if valid."""
    if len(args) != 1:
        return None
    (arg,) = args
    if not isinstance(arg, type):
        return None
    if issubclass(arg, Dataset):
        item = arg.__h5spec__
        if item.dtype is None:
            raise SchemaError(
                f"{owner.__name__}.{py_name}: dynamic item type {arg.__name__} needs a dtype"
            )
        return item
    if issubclass(arg, Group):
        return arg.__h5spec__
    return None


# --------------------------------------------------------------------------
# Class-body compilation
# --------------------------------------------------------------------------

_MISSING = object()


def _own_members(
    cls: type, *, allow_children: bool, default_extras: Extras
) -> dict[str, MemberSpec]:
    """Compile the annotations declared directly on ``cls``."""
    try:
        annotations = inspect.get_annotations(cls, eval_str=True)
    except NameError as exc:
        raise SchemaError(
            f"{cls.__name__}: could not resolve an annotation ({exc});"
            " forward references to not-yet-defined schemas are not supported in v0.1"
        ) from exc
    members: dict[str, MemberSpec] = {}
    for py_name, ann in annotations.items():
        if py_name.startswith("_"):
            continue
        origin: Any = typing.get_origin(ann)
        if origin is ClassVar:
            continue
        default = cls.__dict__.get(py_name, _MISSING)
        members[py_name] = _classify_member(
            cls,
            py_name,
            ann,
            default if default is not _MISSING else None,
            default is not _MISSING,
            allow_children=allow_children,
            default_extras=default_extras,
        )
    return members


def _merged_members(cls: type) -> dict[str, MemberSpec]:
    """Merge compiled members across the MRO.

    Bases are flattened in MRO order; a more-derived class overrides its
    ancestors, while *sibling* bases redeclaring the same member with
    different specs raise :class:`SchemaError` — that divergence is always
    a bug, and the MRO would otherwise silently pick one.
    """
    merged: dict[str, tuple[type, MemberSpec]] = {}
    for klass in reversed(cls.__mro__):
        own = klass.__dict__.get("__h5members_own__")
        if not own:
            continue
        for name, spec in own.items():
            if name not in merged:
                merged[name] = (klass, spec)
                continue
            owner, existing = merged[name]
            if issubclass(klass, owner):
                merged[name] = (klass, spec)
            elif existing != spec:
                raise SchemaError(
                    f"{cls.__name__}.{name}: conflicting redeclarations in bases"
                    f" {owner.__name__} and {klass.__name__}"
                )
    return {name: spec for name, (_, spec) in merged.items()}


def _check_namespaces(cls: type, members: dict[str, MemberSpec]) -> None:
    """Enforce HDF5 name uniqueness per namespace and API-shadowing rules."""
    child_names: dict[str, str] = {}
    attr_names: dict[str, str] = {}
    for py_name, spec in members.items():
        namespace = attr_names if isinstance(spec, AttrSpec) else child_names
        kind = "attr" if isinstance(spec, AttrSpec) else "child"
        if spec.h5_name in namespace:
            raise SchemaError(
                f"{cls.__name__}: duplicate HDF5 {kind} name {spec.h5_name!r}"
                f" (fields {namespace[spec.h5_name]!r} and {py_name!r})"
            )
        namespace[spec.h5_name] = py_name


def _check_reserved(cls: type, own: dict[str, MemberSpec], reserved: frozenset[str]) -> None:
    """Reject fields that would shadow the public view API."""
    for py_name in own:
        if py_name in reserved:
            raise SchemaError(
                f"{cls.__name__}.{py_name}: field shadows the h5t API;"
                f" use a safe alias such as '{py_name}_:"
                f' Annotated[..., h5t.Name("{py_name}")]\''
            )


def _install_descriptors(cls: type, members: dict[str, MemberSpec]) -> None:
    """Install one lazy, per-instance-cached descriptor per member."""
    for py_name, spec in members.items():
        if isinstance(spec, AttrSpec):
            setattr(cls, py_name, AttrDescriptor(spec))
        else:
            setattr(cls, py_name, NodeDescriptor(spec))


def _resolve_extras(cls: type, extras: str | None) -> Extras:
    """Resolve the extras policy: explicit kwarg, else inherited, else ignore."""
    if extras is not None:
        try:
            return Extras(extras)
        except ValueError:
            raise SchemaError(
                f"{cls.__name__}: extras must be 'ignore', 'warn', or 'forbid', got {extras!r}"
            ) from None
    for base in cls.__mro__[1:]:
        spec = base.__dict__.get("__h5spec__")
        if isinstance(spec, GroupSpec):
            return spec.extras
    return Extras.IGNORE


def _inherited_dynamic(cls: type) -> DynamicSpec | None:
    """Find the dynamic child spec from ``Group[T]`` bases or inherited specs."""
    for base in getattr(cls, "__orig_bases__", ()):
        origin = typing.get_origin(base)
        if isinstance(origin, type) and issubclass(origin, Group):
            args = typing.get_args(base)
            if len(args) == 1 and isinstance(args[0], TypeVar):
                continue
            # Non-schema args such as Any fall through _dynamic_item_spec as None.
            item = _dynamic_item_spec(cls, "<base>", args)
            if item is not None:
                return DynamicSpec(item=item, pattern=None)
    for base in cls.__mro__[1:]:
        spec = base.__dict__.get("__h5spec__")
        if isinstance(spec, GroupSpec) and spec.dynamic is not None:
            return spec.dynamic
    return None


def _reserved_group_names() -> frozenset[str]:
    """Public API names a group schema field may not shadow."""
    names = {n for n in dir(Group) if not n.startswith("_")}
    file_cls = globals().get("File")
    if file_cls is not None:
        names |= {n for n in dir(file_cls) if not n.startswith("_")}
    return frozenset(names)


def _reserved_dataset_names() -> frozenset[str]:
    """Public API names a dataset schema field may not shadow."""
    return frozenset(n for n in dir(Dataset) if not n.startswith("_"))


def _compile_group_class(
    cls: type,
    extras: str | None,
) -> None:
    """Compile a Group (or File) subclass body into its ``__h5spec__``."""
    resolved_extras = _resolve_extras(cls, extras)
    own = _own_members(cls, allow_children=True, default_extras=resolved_extras)
    _check_reserved(cls, own, _reserved_group_names())
    cls.__h5members_own__ = own  # ty: ignore[unresolved-attribute]
    members = _merged_members(cls)
    _check_namespaces(cls, members)
    children = tuple(s for s in members.values() if not isinstance(s, AttrSpec))
    attrs = tuple(s for s in members.values() if isinstance(s, AttrSpec))
    spec = GroupSpec(
        py_name="",
        h5_name="",
        children=children,
        attrs=attrs,
        extras=resolved_extras,
        dynamic=_inherited_dynamic(cls),
        view_type=cls,
    )
    cls.__h5spec__ = spec  # ty: ignore[unresolved-attribute]
    _install_descriptors(cls, members)


def _compile_dataset_class(
    cls: type,
    dtype: type[DType] | None,
) -> None:
    """Compile a Dataset subclass body into its ``__h5spec__``."""
    if dtype is not None and not (isinstance(dtype, type) and issubclass(dtype, DType)):
        raise SchemaError(f"{cls.__name__}: dtype= must be an h5t dtype token, got {dtype!r}")
    inherited = None
    for base in cls.__mro__[1:]:
        candidate = base.__dict__.get("__h5spec__")
        if isinstance(candidate, DatasetSpec):
            inherited = candidate
            break
    if dtype is None and inherited is not None:
        dtype = inherited.dtype
    own = _own_members(cls, allow_children=False, default_extras=Extras.IGNORE)
    _check_reserved(cls, own, _reserved_dataset_names())
    cls.__h5members_own__ = own  # ty: ignore[unresolved-attribute]
    members = _merged_members(cls)
    _check_namespaces(cls, members)
    attrs = tuple(s for s in members.values() if isinstance(s, AttrSpec))
    spec = DatasetSpec(
        py_name="",
        h5_name="",
        dtype=dtype,
        attrs=attrs,
        view_type=cls,
    )
    cls.__h5spec__ = spec  # ty: ignore[unresolved-attribute]
    _install_descriptors(cls, members)


def _check_spec(spec: NodeSpec, context: str) -> None:
    """Recursively check a compiled spec for coherence (``validate_schema``)."""
    if isinstance(spec, DatasetSpec):
        if spec.dtype is None:
            raise SchemaError(f"{context}: dataset has no dtype")
        return
    if spec.dynamic is not None:
        if spec.dynamic.pattern is not None:
            try:
                re.compile(spec.dynamic.pattern)
            except re.error as exc:
                raise SchemaError(
                    f"{context}: invalid Keys pattern {spec.dynamic.pattern!r}: {exc}"
                ) from exc
        _check_spec(spec.dynamic.item, f"{context}[<dynamic>]")
    for child in spec.children:
        _check_spec(child, f"{context}/{child.h5_name}")


# --------------------------------------------------------------------------
# Public schema base classes
# --------------------------------------------------------------------------


class Dataset(DatasetViewOps, Generic[DT]):
    """Base class for dataset schemas; instances are lazy dataset views.

    Configure via class kwargs, not by subclassing the subscript::

        class StrainSeries(h5t.Dataset, dtype=h5t.f8):
            unit: Literal["strain"]

    Ordinary class-body annotations declare *attrs on the dataset* — a
    dataset cannot contain child nodes. The subscript form
    ``h5t.Dataset[h5t.f8]`` builds an inline anonymous subclass at runtime.
    """

    __h5spec__: ClassVar[DatasetSpec]
    __h5members_own__: ClassVar[dict[str, MemberSpec]]

    def __init_subclass__(
        cls,
        *,
        dtype: type[DType] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        _compile_dataset_class(cls, dtype)

    def __class_getitem__(cls, item: Any) -> Any:
        """Build an inline subclass whose subscript sets its dtype.

        TypeVar subscriptions are delegated to ``Generic``.
        """
        parts = item if isinstance(item, tuple) else (item,)
        if any(isinstance(p, TypeVar) for p in parts):
            return super().__class_getitem__(item)  # type: ignore[misc]
        if len(parts) != 1 or not (isinstance(parts[0], type) and issubclass(parts[0], DType)):
            raise SchemaError(
                f"invalid Dataset subscript {item!r}; expected one dtype token;"
                " override validate() to check dataset shapes"
            )
        dtype = parts[0]
        rendered = dtype.__name__
        return type(
            f"{cls.__name__}[{rendered}]",
            (cls,),
            {"__module__": cls.__module__, "__qualname__": f"{cls.__qualname__}[{rendered}]"},
            dtype=dtype,
        )

    @classmethod
    def validate_schema(cls) -> None:
        """Check that this schema itself is coherent.

        Schema errors are a separate channel from file errors: this raises
        :class:`SchemaError` for problems in the *declaration*, never for
        problems in any file.

        Raises
        ------
        SchemaError
            If the compiled spec is incomplete or incoherent.
        """
        _check_spec(cls.__h5spec__, cls.__name__)


class Group(GroupViewOps, Generic[T]):
    """Base class for group schemas; instances are lazy views onto groups.

    Class-body annotations declare child nodes and attrs; ``extras=`` sets
    the policy for undeclared children and attrs (``"ignore"`` by default).

    ``Group[T]`` in an annotation says the group's dynamically named
    children satisfy ``T``; subclassing ``Group[T]`` gives the collection
    group attrs of its own.
    """

    __h5spec__: ClassVar[GroupSpec]
    __h5members_own__: ClassVar[dict[str, MemberSpec]]

    def __init_subclass__(
        cls,
        *,
        extras: Literal["ignore", "warn", "forbid"] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        _compile_group_class(cls, extras)

    def __getitem__(self, key: str) -> T:
        """Address a child by HDF5 name.

        On a dynamic ``Group[T]`` collection, keys selected by the
        ``Keys`` pattern return typed, lazy ``T`` views. On statically
        declared groups this is the untyped mapping escape hatch and
        returns the raw h5py object.
        """
        return cast(T, self._h5t_getitem(key))

    @classmethod
    def validate_schema(cls) -> None:
        """Check that this schema itself is coherent.

        Raises
        ------
        SchemaError
            If the compiled spec is incomplete or incoherent.
        """
        _check_spec(cls.__h5spec__, cls.__name__)


# Base classes carry empty template specs so they are usable as bare
# annotations and as merge anchors; subclass compilation replaces them.
Group.__h5spec__ = GroupSpec(py_name="", h5_name="", view_type=Group)
Dataset.__h5spec__ = DatasetSpec(py_name="", h5_name="", dtype=None, view_type=Dataset)


class File(Group[Any]):
    """Base class for file schemas: a group schema rooted at ``/``.

    ``open()`` validates by default and returns a lazy, typed view; the
    context manager owns the shared handle's lifetime.
    """

    @classmethod
    def open(
        cls,
        path: str | os.PathLike[str],
        *,
        validate: bool = True,
    ) -> Self:
        """Open a file read-only as a validated, lazy view of this schema.

        Parameters
        ----------
        path : str or path-like
            Path of the HDF5 file.
        validate : bool, optional
            Validate on open (the default). Pass ``False`` to inspect
            broken files; member access then raises targeted
            :class:`~h5t._errors.SchemaMismatchError` on nonconforming
            nodes.

        Returns
        -------
        Self
            The root view. Use as a context manager; retained child views
            raise :class:`~h5t._errors.ClosedFileError` after exit.

        Raises
        ------
        ValidationError
            When ``validate`` is true and the file does not conform. All
            problems are batched into one exception and the file handle is
            closed before raising.
        """
        h5file = h5py.File(path, mode="r")
        ctx = FileContext(h5file, str(path))
        spec = cls.__h5spec__
        view = cast(Self, make_view(spec, ctx, "/"))
        if validate:
            try:
                report = view.check()
                for problem in report.warnings:
                    warnings.warn(f"{problem.path}: {problem.message}", stacklevel=2)
                if not report.ok:
                    raise ValidationError(report, file_closed=True)
            except BaseException:
                ctx.close()
                raise
        return view

    def close(self) -> None:
        """Close the shared file handle; all retained views become invalid."""
        self._h5t_ctx.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
