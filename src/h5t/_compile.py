"""``Group[T]``: declaration checks while Pydantic builds the model, and detached loading."""

from __future__ import annotations

import os
import posixpath
import types
import typing
from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Any, Generic, Literal, Self

import h5py
import numpy as np
from jaxtyping import AbstractArray, jaxtyped
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    ModelWrapValidatorHandler,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_validator,
)
from pydantic import ValidationError as PydanticValidationError
from pydantic.fields import FieldInfo
from pydantic_core import PydanticCustomError, core_schema
from typing_extensions import TypeVar

from h5t._array import Dataset, T, _is_mapping_type
from h5t._errors import SchemaError, ValidationError, _located_error
from h5t._h5view import ATTRS_KEY, _ghost, _GroupView


def _schema_error(owner: type, field: str, message: str) -> SchemaError:
    return SchemaError(f"{owner.__name__}.{field}: {message}")


# ---------------------------------------------------------------------------
# Array markers: how an array field reads its raw ``h5py.Dataset``
# ---------------------------------------------------------------------------


def _read_array(value: Any) -> Any:
    """Turn an open ``h5py.Dataset`` into an ndarray; pass any other value through."""
    if isinstance(value, h5py.Dataset):
        return np.asarray(value[()])
    return value


class _NdarraySchema:
    """``Annotated`` marker making a plain ``np.ndarray`` field read its dataset."""

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_before_validator_function(_read_array, handler(source_type))


class _JaxtypingSchema:
    """``Annotated`` marker making a jaxtyping array field check (and read) its dataset."""

    def __init__(self, hint: type[AbstractArray]) -> None:
        self.hint = hint

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        hint = self.hint
        eager = hint.array_type is np.ndarray

        def read(value: Any) -> Any:
            # An ndarray-backed field is read eagerly, but only after jaxtyping has approved
            # the shape and dtype from a metadata-only stand-in, so a mismatch reads no data.
            if eager and isinstance(value, h5py.Dataset):
                reason = hint.__instancecheck_str__(_ghost(value))
                if reason:
                    raise ValidationError(value.name, reason)
                return _read_array(value)
            return value

        def check(value: Any) -> Any:
            reason = hint.__instancecheck_str__(value)
            if reason:
                raise PydanticCustomError("jaxtyping", "{reason}", {"reason": reason})
            return value

        # The backing type's own schema: an isinstance check for ``np.ndarray``, the lazy
        # raw-h5py reader for a ``Dataset``.
        backing = handler.generate_schema(hint.array_type)
        return core_schema.no_info_before_validator_function(
            read,
            core_schema.chain_schema(
                [backing, core_schema.no_info_plain_validator_function(check)]
            ),
        )


_MARKERS = (_NdarraySchema, _JaxtypingSchema)

# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

_NONE = type(None)
_UNIONS = (typing.Union, types.UnionType)
# ``typing_extensions.TypeAliasType`` is a separate class that ``type X = ...`` aliases are
# not instances of, so only the stdlib one (Python >= 3.12) is of use here.
_alias_type = getattr(typing, "TypeAliasType", None)
_TYPE_ALIAS_TYPES: tuple[type, ...] = (_alias_type,) if isinstance(_alias_type, type) else ()
_MAX_ALIAS_DEPTH = 16


def _unwrap(annotation: Any) -> tuple[Any, bool]:
    """Strip ``Annotated`` and ``| None``; return the core type and whether it is nullable."""
    nullable = False
    while True:
        origin = typing.get_origin(annotation)
        args = typing.get_args(annotation)
        if origin is Annotated:
            annotation = args[0]
        elif origin in _UNIONS and _NONE in args:
            nullable = True
            rest = [arg for arg in args if arg is not _NONE]
            if len(rest) != 1:
                return annotation, nullable
            annotation = rest[0]
        else:
            return annotation, nullable


def _is_ndarray_annotation(core: Any) -> bool:
    """Report whether ``core`` denotes ``np.ndarray``, however many aliases deep."""
    for _ in range(_MAX_ALIAS_DEPTH):
        if core is np.ndarray:
            return True
        origin = typing.get_origin(core)
        if origin is np.ndarray:
            return True
        if isinstance(core, _TYPE_ALIAS_TYPES):
            core = core.__value__
        elif isinstance(origin, _TYPE_ALIAS_TYPES):
            core = origin.__value__
        else:
            return False
    return False


def _dataset_class(tp: Any) -> type[Dataset[Any]] | None:
    """Return the `Dataset` class ``tp`` denotes (``Dataset``, ``Dataset[A]``), if any."""
    cls = typing.get_origin(tp) or tp
    return cls if isinstance(cls, type) and issubclass(cls, Dataset) else None


# ---------------------------------------------------------------------------
# Declaration checks, run once while Pydantic builds a group's schema
# ---------------------------------------------------------------------------


def _check_extra(model_cls: type[BaseModel]) -> None:
    raw = model_cls.model_config.get("extra", "ignore")
    if raw not in (None, "ignore", "forbid"):
        raise SchemaError(f"{model_cls.__name__}: extra must be 'ignore' or 'forbid', got {raw!r}")


def _h5_name(owner: type, py_name: str, field: FieldInfo) -> str:
    """Return the on-disk name of a field: its alias, or else its Python name."""
    h5_name = field.alias if field.alias is not None else py_name
    if not isinstance(h5_name, str) or not h5_name:
        raise _schema_error(owner, py_name, "alias requires a non-empty string")
    return h5_name


def _check_unique_names(owner: type, fields: Mapping[str, FieldInfo], what: str) -> None:
    seen: dict[str, str] = {}
    for py_name, field in fields.items():
        h5_name = _h5_name(owner, py_name, field)
        if h5_name in seen:
            raise SchemaError(
                f"{owner.__name__}: duplicate {what} {h5_name!r} for fields "
                f"{seen[h5_name]!r} and {py_name!r}"
            )
        seen[h5_name] = py_name


def _check_attrs(model_type: type[BaseModel]) -> None:
    """Reject what an attributes model cannot be stored as; Pydantic checks the rest."""
    _check_extra(model_type)
    _check_unique_names(model_type, model_type.model_fields, "HDF5 attribute name")
    for py_name, field in model_type.model_fields.items():
        core, _ = _unwrap(field.annotation)
        if isinstance(core, type) and issubclass(core, AbstractArray):
            core = core.array_type
        if isinstance(core, type) and issubclass(core, (Group, Dataset)):
            raise _schema_error(
                model_type, py_name, f"attributes model cannot contain {core.__name__} fields"
            )


def _attrs_model(group_cls: type[Group[Any]]) -> type[BaseModel] | None:
    """Return the attributes model of ``group_cls``, or ``None`` for untyped attributes."""
    attrs_ann: Any = group_cls.model_fields["attrs"].annotation
    if attrs_ann is None or attrs_ann is Any or isinstance(attrs_ann, TypeVar):
        return None
    if _is_mapping_type(attrs_ann):
        return None
    if isinstance(attrs_ann, type) and issubclass(attrs_ann, BaseModel):
        if issubclass(attrs_ann, Group):
            raise _schema_error(
                group_cls, "attrs", "Group[...] type argument cannot be a Group subclass"
            )
        return attrs_ann
    raise _schema_error(
        group_cls,
        "attrs",
        f"attrs must be a BaseModel subclass or Mapping[str, Any], got {attrs_ann!r}",
    )


def _mark(owner: type, name: str, tp: Any, *, item: bool = False) -> Any:
    """Check one child annotation and return it with its array leaves marked.

    The accepted grammar is small: a `Group`, a `Dataset`, an ``np.ndarray`` or a jaxtyping
    array, optionally ``| None`` and optionally ``Annotated``; or a ``dict[str, ...]``
    collection of those (``item`` is set while inside one). Anything else is a
    `SchemaError`. Array leaves come back wrapped in a marker so they can read the raw
    ``h5py.Dataset`` that `_GroupView` hands to Pydantic.
    """
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is Annotated:
        head, *metadata = args
        if any(isinstance(m, _MARKERS) for m in metadata):
            return tp  # already marked by an earlier schema build
        if item and any(isinstance(m, FieldInfo) and m.alias is not None for m in metadata):
            raise _schema_error(owner, name, "alias cannot annotate a collection item")
        return Annotated[(_mark(owner, name, head, item=item), *metadata)]
    if origin in _UNIONS:
        rest = [arg for arg in args if arg is not _NONE]
        if len(rest) != 1:
            raise _schema_error(
                owner, name, "only optional unions of the form 'T | None' are supported"
            )
        if item:
            raise _schema_error(owner, name, "collection items cannot be optional")
        return _mark(owner, name, rest[0]) | None
    if isinstance(tp, type) and issubclass(tp, AbstractArray):
        backing = tp.array_type
        if backing is not np.ndarray and _dataset_class(backing) is None:
            raise _schema_error(
                owner,
                name,
                f"jaxtyping array type {backing!r} is not supported; use np.ndarray or Dataset",
            )
        if backing is not np.ndarray:
            _mark(owner, name, backing)  # checks the Dataset's attributes model
        return Annotated[tp, _JaxtypingSchema(tp)]
    if isinstance(tp, type) and issubclass(tp, Group):
        return tp
    dataset_cls = _dataset_class(tp)
    if dataset_cls is not None:
        if dataset_cls._h5t_attrs_type is not None:
            _check_attrs(dataset_cls._h5t_attrs_type)
        return tp
    if _is_ndarray_annotation(tp):
        return Annotated[tp, _NdarraySchema()]
    if isinstance(tp, _TYPE_ALIAS_TYPES):
        return _mark(owner, name, tp.__value__, item=item)
    container = origin or tp
    if isinstance(container, type) and issubclass(container, Mapping):
        if item:
            raise _schema_error(owner, name, "nested collections are not supported")
        if origin not in (dict, Mapping) or len(args) != 2:
            raise _schema_error(
                owner, name, "a collection must be annotated dict[str, T] or Mapping[str, T]"
            )
        if args[0] is not str:
            raise _schema_error(owner, name, "a collection key type must be str")
        return origin[str, _mark(owner, name, args[1], item=True)]
    if isinstance(tp, (str, typing.ForwardRef)):
        return tp  # unresolved: Pydantic reports it, or defers the build
    raise _schema_error(
        owner,
        name,
        f"unsupported child field annotation {tp!r}; group fields must be "
        "Group, Dataset, np.ndarray, or dict[str, T] (declare HDF5 attributes on Group[Attrs])",
    )


def _check_declarations(group_cls: type[Group[Any]]) -> None:
    """Check ``group_cls``'s own declarations and mark its array fields.

    Only this class is checked: nested groups run the same check when Pydantic builds
    their schema, so nothing here recurses through the model tree.
    """
    _check_extra(group_cls)
    attrs_type = _attrs_model(group_cls)
    if attrs_type is not None:
        _check_attrs(attrs_type)
    children = {n: f for n, f in group_cls.model_fields.items() if n != "attrs"}
    _check_unique_names(group_cls, children, "HDF5 name")
    for name, field in children.items():
        if "/" in _h5_name(group_cls, name, field):
            raise _schema_error(group_cls, name, "a child alias cannot contain '/'")
        if not any(isinstance(m, _MARKERS) for m in field.metadata):
            field.annotation = _mark(group_cls, name, field.annotation)


def _filesystem_path(path: str | os.PathLike[str]) -> str:
    """Return ``path`` as an absolute filesystem string, or raise ``TypeError``."""
    try:
        return os.path.abspath(os.fsdecode(os.fspath(path)))
    except TypeError as exc:
        raise TypeError("path must be a filesystem path") from exc


def _normalize_root(root: str) -> str:
    """Validate and normalize an absolute HDF5 group path."""
    if not isinstance(root, str) or not posixpath.isabs(root):
        raise ValueError("root must be an absolute HDF5 group path")
    normalized = posixpath.normpath(root)
    if normalized.startswith("//"):
        normalized = "/" + normalized.lstrip("/")
    return normalized


# ---------------------------------------------------------------------------
# Group base class
# ---------------------------------------------------------------------------


class Group(BaseModel, Generic[T]):
    """Base class for typed, detached HDF5 group schemas.

    Subclass ``h5t.Group`` (for untyped ``Mapping[str, Any]`` attributes) or
    ``h5t.Group[AttrsModel]`` (where ``AttrsModel`` is a `pydantic.BaseModel` subclass
    describing the group's HDF5 attributes). Every field declared on a ``Group``
    subclass represents an immediate HDF5 child node: a nested `Group`, a `Dataset`, an
    eager ``np.ndarray``, or a ``dict[str, ...]`` collection of children.

    Attributes
    ----------
    attrs : T
        The group's HDF5 attributes (`BaseModel` instance or read-only `Mapping`).

    Notes
    -----
    ``Field(alias="...")`` on a child field or an attribute field sets its on-disk HDF5
    name. ``extra="forbid"`` in ``model_config`` rejects undeclared child nodes (and, on
    ``T``, undeclared attributes); only ``"ignore"`` and ``"forbid"`` are supported.

    Examples
    --------
    >>> import numpy as np
    >>> from pydantic import BaseModel
    >>> import h5t
    >>> class RunAttrs(BaseModel):
    ...     version: int
    >>> class Run(h5t.Group[RunAttrs]):
    ...     values: np.ndarray
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        extra="ignore",
    )

    attrs: T = Field(default_factory=dict, validate_default=True, validation_alias=ATTRS_KEY)  # type: ignore[assignment]

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        # With unresolved forward references the fields are not final yet; Pydantic defers
        # the build and calls this hook again once they resolve.
        if getattr(cls, "__pydantic_fields_complete__", True):
            _check_declarations(cls)
        return handler(source)

    @field_validator("attrs", mode="wrap", check_fields=False)
    @classmethod
    def _h5t_attrs_scope(cls, value: Any, handler: ValidatorFunctionWrapHandler) -> Any:
        # Attribute dims must not bind together with the dims of the group's children.
        with jaxtyped("context"):
            return handler(value)

    @model_validator(mode="wrap")
    @classmethod
    def _h5t_load(cls, values: Any, handler: ModelWrapValidatorHandler[Self]) -> Self:
        loading = isinstance(values, _GroupView)
        try:
            # One jaxtyping scope per group: its children bind dimensions together.
            with jaxtyped("context"):
                model = handler(values)
        except PydanticValidationError as exc:
            if not loading:
                raise
            # This group knows its own HDF5 path, so the error is located here. An
            # ``h5t.ValidationError`` from a nested node already carries its path and is
            # not a ``ValueError``, so it passes through every enclosing group unchanged.
            raise _located_error(exc, values.node.name) from exc
        if loading and isinstance(model.attrs, dict):
            # Untyped attributes of a loaded group are a read-only snapshot.
            object.__setattr__(model, "attrs", MappingProxyType(model.attrs))
        return model

    @classmethod
    def model_validate_hdf5(
        cls,
        path: str | os.PathLike[str],
        root: str = "/",
    ) -> Self:
        """Validate and load an HDF5 group into a detached instance of ``cls``.

        The HDF5 file is closed before ``model_validate_hdf5`` returns, even on error.
        Child `Dataset` fields remain lazy and remember the file path so slices or the full
        array can be read on demand.

        Parameters
        ----------
        path : str or os.PathLike
            The HDF5 file to read.
        root : str
            Absolute path of the group to load (default ``"/"``).

        Returns
        -------
        Self
            A validated instance of ``cls``.

        Raises
        ------
        ValidationError
            The file does not conform to the schema (including a missing ``root``).
        ValueError
            ``root`` is not an absolute HDF5 path.
        OSError
            The file cannot be opened.
        TypeError
            ``path`` is not a string or path-like object.
        """
        if not cls.__pydantic_complete__:
            # Surfaces a still-unresolved annotation as Pydantic's own error, by name.
            cls.model_rebuild()
        normalized_root = _normalize_root(root)
        filename = _filesystem_path(path)
        with h5py.File(filename, mode="r") as h5file:
            node = h5file.get(normalized_root)
            if node is None:
                raise ValidationError(normalized_root, "root group does not exist")
            if not isinstance(node, h5py.Group):
                raise ValidationError(normalized_root, "expected a group, found a dataset")
            try:
                return cls.model_validate(_GroupView(node))
            except PydanticValidationError as exc:
                # A user's own model validator on ``cls`` runs outside `_h5t_load`.
                raise _located_error(exc, normalized_root) from exc

    def model_dump_hdf5(
        self,
        path: str | os.PathLike[str],
        root: str = "/",
        mode: Literal["w-", "x", "w", "a"] = "w-",
        overwrite: bool = False,
    ) -> None:
        """Write ``self`` under ``root`` in ``path``, following its schema.

        Field types, array shapes and attribute values were validated by Pydantic when
        ``self`` was constructed and are not checked again here, so an instance built with
        ``model_construct`` or modified afterwards is written as it is. Only what HDF5
        itself imposes is checked: collection keys and attribute values h5py can store.
        Nodes are written one at a time, each lazy `Dataset` being read just before it is
        stored, so a failure part-way through leaves the nodes written so far in the file.
        ``overwrite=True`` deletes the existing content of ``root`` before writing, as
        ``del group[name]`` does in h5py.

        Parameters
        ----------
        path : str or os.PathLike
            Target HDF5 file.
        root : str
            Absolute group path to write under; missing intermediate groups are created.
        mode : {"w-", "x", "w", "a"}
            ``"w-"`` (default) and ``"x"`` fail if the file exists, ``"w"`` truncates,
            ``"a"`` writes into an existing file or creates one.
        overwrite : bool
            Delete the contents of a non-empty ``root`` group and write over it.

        Raises
        ------
        ValidationError
            A value cannot be stored in HDF5 (an invalid collection key or an unstorable
            or unserializable attribute), a lazy `Dataset` cannot be read, or ``root`` is
            occupied.
        ValueError
            ``mode`` or ``root`` is invalid.
        OSError
            The file cannot be created or opened (for example it exists in ``"w-"``/``"x"``
            mode).
        TypeError
            ``path`` is not a string or path-like object.
        """
        from h5t._write import _dump_group

        _dump_group(self, path, root=root, mode=mode, overwrite=overwrite)
