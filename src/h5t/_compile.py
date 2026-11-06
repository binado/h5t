"""``Group[T]`` declaration checks and detached HDF5 loading."""

from __future__ import annotations

import fnmatch
import os
import posixpath
import re
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

from h5t._array import Dataset, _is_mapping_type
from h5t._errors import SchemaError, ValidationError
from h5t._h5view import ATTRS_KEY, _ghost, _GroupView
from h5t._spec import Glob, attr_path, child_path

T = TypeVar("T", bound=BaseModel | Mapping[str, Any], default=Mapping[str, Any])


def _schema_error(owner: type, field: str, message: str) -> SchemaError:
    return SchemaError(f"{owner.__name__}.{field}: {message}")


# ---------------------------------------------------------------------------
# jaxtyping integration with Pydantic
# ---------------------------------------------------------------------------


def _is_supported_array_type(array_type: Any) -> bool:
    """Report whether jaxtyping may back an array with ``array_type``."""
    return array_type is np.ndarray or (
        isinstance(array_type, type) and issubclass(array_type, Dataset)
    )


def _jaxtyping_pydantic_schema(
    cls: type[AbstractArray], source_type: Any, handler: GetCoreSchemaHandler
) -> core_schema.CoreSchema:
    array_type = cls.array_type
    if not _is_supported_array_type(array_type):
        raise SchemaError(
            f"jaxtyping array type {array_type!r} is not supported; use np.ndarray or Dataset"
        )
    # ``handler(Dataset[T])`` would fall back to a plain isinstance check on this path, so a
    # Dataset-backed array asks the Dataset class for its own schema (the raw-h5py reader).
    if array_type is np.ndarray:
        base_schema = handler(array_type)
    else:
        base_schema = array_type.__get_pydantic_core_schema__(array_type, handler)
    reads_ndarray = array_type is np.ndarray

    def _validate_jaxtyping(value: Any) -> Any:
        reason = cls.__instancecheck_str__(value)
        if reason:
            raise PydanticCustomError("jaxtyping", "{reason}", {"reason": reason})
        return value

    def _read_h5py(value: Any) -> Any:
        # An ndarray-backed field is read eagerly, but only after jaxtyping has approved the
        # shape and dtype from a metadata-only stand-in, so a mismatch never reads data.
        if reads_ndarray and isinstance(value, h5py.Dataset):
            reason = cls.__instancecheck_str__(_ghost(value))
            if reason:
                raise ValidationError(value.name, reason)
            return _read_array(value)
        # Attribute values arrive as Python scalars; a jaxtyping field wants an array.
        if reads_ndarray and isinstance(value, _SCALAR_TYPES):
            return np.asarray(value)
        return value

    return core_schema.no_info_before_validator_function(
        _read_h5py,
        core_schema.chain_schema(
            [
                base_schema,
                core_schema.no_info_plain_validator_function(_validate_jaxtyping),
            ]
        ),
    )


_SCALAR_TYPES = (bool, int, float, complex, str, bytes, np.generic)


setattr(  # noqa: B010
    AbstractArray,
    "__get_pydantic_core_schema__",
    classmethod(_jaxtyping_pydantic_schema),
)


# ---------------------------------------------------------------------------
# Field classification helpers
# ---------------------------------------------------------------------------


def _split_annotation(owner: type, py_name: str, annotation: Any) -> tuple[Any, list[Any], bool]:
    """Extract metadata and optionality from an annotation."""
    metadata: list[Any] = []
    optional = False
    core = annotation
    while True:
        if typing.get_origin(core) is Annotated:
            args = typing.get_args(core)
            core = args[0]
            metadata.extend(args[1:])
            continue
        origin = typing.get_origin(core)
        if origin in (typing.Union, types.UnionType):
            args = typing.get_args(core)
            without_none = tuple(arg for arg in args if arg is not type(None))
            if len(without_none) != len(args):
                optional = True
            if len(without_none) != 1:
                raise _schema_error(
                    owner,
                    py_name,
                    "only optional unions of the form 'T | None' are supported",
                )
            core = without_none[0]
            continue
        return core, metadata, optional


def _marker(metadata: list[Any], marker_type: type, owner: type, py_name: str) -> Any | None:
    found = [item for item in metadata if isinstance(item, marker_type)]
    if len(found) > 1:
        raise _schema_error(owner, py_name, f"{marker_type.__name__} may appear only once")
    return found[0] if found else None


_alias_type = getattr(typing, "TypeAliasType", None)
_TYPE_ALIAS_TYPES: tuple[type, ...] = (_alias_type,) if isinstance(_alias_type, type) else ()
_MAX_ALIAS_DEPTH = 16


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


def _unwrap_jaxtyping(owner: type, py_name: str, core: Any) -> Any:
    """Replace a jaxtyping array type by its backing class; pass any other type through."""
    if not (isinstance(core, type) and issubclass(core, AbstractArray)):
        return core
    array_type = core.array_type
    if not _is_supported_array_type(array_type):
        raise _schema_error(
            owner,
            py_name,
            f"jaxtyping array type {array_type!r} is not supported; use np.ndarray or Dataset",
        )
    return array_type


def _is_mapping_annotation(core: Any) -> bool:
    """Report whether ``core`` denotes a ``Mapping`` (bare, subscripted, or ``dict``)."""
    origin = typing.get_origin(core) or core
    return isinstance(origin, type) and issubclass(origin, Mapping)


def _check_glob(owner: type, py_name: str, marker: Glob) -> re.Pattern[str]:
    pattern = marker.pattern
    if not isinstance(pattern, str) or not pattern:
        raise _schema_error(owner, py_name, "Glob requires a non-empty string pattern")
    if "/" in pattern:
        raise _schema_error(owner, py_name, "a Glob pattern cannot contain '/'")
    return re.compile(fnmatch.translate(pattern))


# ---------------------------------------------------------------------------
# Attribute declaration checks (for T in Group[T] and Dataset[T])
# ---------------------------------------------------------------------------


def _extract_forbid_extras(model_cls: type[BaseModel], owner: type) -> bool:
    raw = model_cls.model_config.get("extra", "ignore")
    if raw is None or raw == "ignore":
        return False
    if raw == "forbid":
        return True
    raise SchemaError(f"{owner.__name__}: extra must be 'ignore' or 'forbid', got {raw!r}")


def _check_attrs(model_type: type[BaseModel]) -> None:
    if not model_type.__pydantic_complete__:
        model_type.model_rebuild()
    _extract_forbid_extras(model_type, model_type)

    # This loop only rejects declarations h5t cannot store or load; Pydantic does the rest.
    seen_h5_names: dict[str, str] = {}
    for py_name, field_info in model_type.model_fields.items():
        raw_ann = field_info.annotation
        core, meta, _ = _split_annotation(model_type, py_name, raw_ann)
        all_meta = [*meta, *field_info.metadata]
        if any(isinstance(m, Glob) for m in all_meta):
            raise _schema_error(
                model_type, py_name, "Glob applies only to dict[str, T] collection fields"
            )

        h5_name = field_info.alias if field_info.alias is not None else py_name
        if not isinstance(h5_name, str) or not h5_name:
            raise _schema_error(model_type, py_name, "alias requires a non-empty string")
        if h5_name in seen_h5_names:
            raise SchemaError(
                f"{model_type.__name__}: duplicate HDF5 attribute name {h5_name!r} for fields "
                f"{seen_h5_names[h5_name]!r} and {py_name!r}"
            )
        seen_h5_names[h5_name] = py_name
        core = _unwrap_jaxtyping(model_type, py_name, core)
        if isinstance(core, type) and issubclass(core, (Group, Dataset)):
            raise _schema_error(
                model_type,
                py_name,
                f"attributes model cannot contain {core.__name__} fields",
            )


def _extract_alias(owner: type, py_name: str, field: FieldInfo) -> str:
    h5_name = field.alias if field.alias is not None else py_name
    if not isinstance(h5_name, str) or not h5_name:
        raise _schema_error(owner, py_name, "alias requires a non-empty string")
    return h5_name


def _collection_item(owner: type, py_name: str, core: Any) -> Any:
    """Check and return the item annotation of a collection."""
    origin = typing.get_origin(core) or core
    args = typing.get_args(core)
    if origin not in (dict, Mapping) or len(args) != 2:
        raise _schema_error(
            owner,
            py_name,
            "a collection must be annotated dict[str, T] or Mapping[str, T]",
        )
    key, item_annotation = args
    if key is not str:
        raise _schema_error(owner, py_name, "a collection key type must be str")
    _, item_metadata, item_optional = _split_annotation(owner, py_name, item_annotation)
    if item_optional:
        raise _schema_error(owner, py_name, "collection items cannot be optional")
    if any(isinstance(m, Glob) for m in item_metadata):
        raise _schema_error(owner, py_name, "Glob cannot annotate a collection item")
    if any(isinstance(m, FieldInfo) and m.alias is not None for m in item_metadata):
        raise _schema_error(owner, py_name, "alias cannot annotate a collection item")
    item_core = _member_annotation(owner, py_name, item_annotation)
    if _is_mapping_annotation(item_core):
        raise _schema_error(owner, py_name, "nested collections are not supported")
    return item_annotation


def _member_annotation(owner: type, name: str, annotation: Any) -> Any:
    core, _, _ = _split_annotation(owner, name, annotation)
    core = _unwrap_jaxtyping(owner, name, core)
    origin = typing.get_origin(core)
    if isinstance(origin, type) and issubclass(origin, Dataset):
        args = typing.get_args(core)
        return origin if not args or isinstance(args[0], TypeVar) else origin[args[0]]
    return core


def _check_member(owner: type, name: str, annotation: Any, seen: set[type]) -> None:
    core = _member_annotation(owner, name, annotation)
    if isinstance(core, type) and issubclass(core, Group):
        _check_group(core, seen)
    elif isinstance(core, type) and issubclass(core, Dataset):
        if core._h5t_attrs_type is not None:
            _check_attrs(core._h5t_attrs_type)
    elif _is_ndarray_annotation(core):
        pass
    elif _is_mapping_annotation(core):
        _check_member(owner, name, _collection_item(owner, name, core), seen)
    else:
        raise _schema_error(
            owner,
            name,
            f"unsupported child field annotation {core!r}; group fields must be "
            "Group, Dataset, np.ndarray, or dict[str, T] (declare HDF5 attributes on Group[Attrs])",
        )


def _field_glob(owner: type, name: str, field: FieldInfo) -> re.Pattern[str] | None:
    core, metadata, _ = _split_annotation(owner, name, field.annotation)
    marker = _marker([*metadata, *field.metadata], Glob, owner, name)
    if marker is None:
        return None
    if not _is_mapping_annotation(core):
        raise _schema_error(owner, name, "Glob applies only to dict[str, T] collection fields")
    return _check_glob(owner, name, marker)


def _resolve_group_attrs_type(group_cls: type[Group[Any]]) -> type[BaseModel] | None:
    """Determine the attributes model for ``group_cls`` from its resolved ``attrs`` field."""
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


def _check_group(group_cls: type[Group[Any]], seen: set[type] | None = None) -> None:
    """Check reachable declarations using only Pydantic's current model metadata."""
    if seen is None:
        seen = set()
    if group_cls in seen:
        return
    seen.add(group_cls)
    if not group_cls.__pydantic_complete__:
        group_cls.model_rebuild()
    _extract_forbid_extras(group_cls, group_cls)
    attrs_type = _resolve_group_attrs_type(group_cls)
    if attrs_type is not None:
        _check_attrs(attrs_type)
    names: dict[str, str] = {}
    for name, field in group_cls.model_fields.items():
        if name == "attrs":
            continue
        h5_name = _extract_alias(group_cls, name, field)
        if "/" in h5_name:
            raise _schema_error(group_cls, name, "a child alias cannot contain '/'")
        if h5_name in names:
            raise SchemaError(
                f"{group_cls.__name__}: duplicate HDF5 name {h5_name!r} for fields "
                f"{names[h5_name]!r} and {name!r}"
            )
        names[h5_name] = name
        _field_glob(group_cls, name, field)
        _check_member(group_cls, name, field.annotation, seen)


def _read_array(value: Any) -> Any:
    """Turn an open ``h5py.Dataset`` into an ndarray; pass any other value through."""
    if isinstance(value, h5py.Dataset):
        return np.asarray(value[()])
    return value


def _with_array_readers(schema: Any, handler: GetCoreSchemaHandler) -> Any:
    """Return ``schema`` with every ``np.ndarray`` instance check preceded by `_read_array`.

    Pydantic knows ``np.ndarray`` only as an ``isinstance`` check, so a raw ``h5py.Dataset``
    would be rejected. Rewriting those nodes where the schema is generated covers plain
    arrays, collections of them and any other nesting, without touching the built model.
    """
    if isinstance(schema, dict):
        target = schema
        seen_refs: set[str] = set()
        while target.get("type") == "definition-ref":
            # PEP 695 array aliases live in Pydantic's definitions rather than in the
            # model schema. Wrap their reference; recursive group refs remain untouched.
            ref = typing.cast(core_schema.DefinitionReferenceSchema, target)["schema_ref"]
            if ref in seen_refs:
                break
            seen_refs.add(ref)
            try:
                target = handler.resolve_ref_schema(target)
            except LookupError:
                break  # a recursive model's definition may still be under construction
        if target.get("type") == "is-instance" and target.get("cls") is np.ndarray:
            return core_schema.no_info_before_validator_function(_read_array, schema)
        return {key: _with_array_readers(value, handler) for key, value in schema.items()}
    if isinstance(schema, list):
        return [_with_array_readers(item, handler) for item in schema]
    return schema


def _first_error_parts(exc: PydanticValidationError) -> tuple[tuple[Any, ...], str]:
    """Return the first error's ``(loc, message)`` pair."""
    errors = exc.errors(include_url=False, include_context=False, include_input=False)
    if not errors:
        return (), str(exc).splitlines()[0]
    first = errors[0]
    return tuple(first.get("loc", ())), str(first.get("msg", "invalid value"))


def _h5_error_path(model: type[Group[Any]], root: str, loc: tuple[Any, ...]) -> tuple[str, str]:
    """Translate error locations by traversing model fields and their annotations."""
    path = root
    current: type[Group[Any]] | None = model
    i = 0
    while i < len(loc) and current is not None:
        part = loc[i]
        if part == ATTRS_KEY:
            if i + 1 < len(loc):
                return attr_path(path, str(loc[i + 1])), _dotted(loc[i + 2 :])
            return path, ""
        owner = current
        entry = next(
            (
                (name, field)
                for name, field in owner.model_fields.items()
                if name != "attrs" and (field.alias or name) == part
            ),
            None,
        )
        path = child_path(path, str(part))
        i += 1
        current = None
        if entry is None:
            break
        name, field = entry
        core = _member_annotation(owner, name, field.annotation)
        if _is_mapping_annotation(core) and i < len(loc):
            path = child_path(path, str(loc[i]))
            i += 1
            core = _member_annotation(owner, name, _collection_item(owner, name, core))
        if isinstance(core, type) and issubclass(core, Group):
            current = core
        elif (
            isinstance(core, type)
            and issubclass(core, Dataset)
            and core._h5t_attrs_type is not None
            and i < len(loc)
        ):
            return attr_path(path, str(loc[i])), _dotted(loc[i + 1 :])
    return path, _dotted(loc[i:])


def _dotted(loc: tuple[Any, ...]) -> str:
    return ".".join(str(part) for part in loc)


def _first_error(
    exc: PydanticValidationError, model: type[Group[Any]], root: str
) -> ValidationError:
    """Convert the first Pydantic error into an `h5t.ValidationError` at its HDF5 path."""
    loc, message = _first_error_parts(exc)
    path, rest = _h5_error_path(model, root, loc)
    return ValidationError(path, f"{rest}: {message}" if rest else message)


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
    Annotate a ``dict[str, ...]`` field with `Glob` to restrict which child names it
    collects.

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
        return _with_array_readers(handler(source), handler)

    @field_validator("attrs", mode="wrap", check_fields=False)
    @classmethod
    def _h5t_attrs_scope(cls, value: Any, handler: ValidatorFunctionWrapHandler) -> Any:
        # Attribute dims must not bind together with the dims of the group's children.
        with jaxtyped("context"):
            return handler(value)

    @model_validator(mode="wrap")
    @classmethod
    def _h5t_jaxtyping_scope(cls, values: Any, handler: ModelWrapValidatorHandler[Self]) -> Self:
        is_view = isinstance(values, _GroupView)
        untyped = is_view and _resolve_group_attrs_type(cls) is None
        if untyped and _extract_forbid_extras(cls, cls):
            name = next(iter(values.node.attrs), None)
            if name is not None:
                path = attr_path(values.node.name, str(name))
                raise ValidationError(path, "unexpected attribute")
        with jaxtyped("context"):
            model = handler(values)
        if untyped:
            # Untyped attributes of a loaded group are a read-only snapshot.
            object.__setattr__(
                model, "attrs", MappingProxyType(dict(typing.cast(Mapping[str, Any], model.attrs)))
            )
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
        SchemaError
            An HDF5 declaration is unsupported. Pydantic model-building errors propagate unchanged.
        ValidationError
            The file does not conform to the schema (including a missing ``root``).
        ValueError
            ``root`` is not an absolute HDF5 path.
        OSError
            The file cannot be opened.
        TypeError
            ``path`` is not a string or path-like object.
        """
        _check_group(cls)
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
                raise _first_error(exc, cls, normalized_root) from exc

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
        itself imposes is checked: collection keys, `Glob` patterns, and attribute values
        h5py can store. The group tree is fully planned before the file is opened, so a
        failed check never touches the target file. If writing fails midway, any
        created groups or files are rolled back. ``overwrite=True`` deletes the existing
        content of ``root`` before writing, as ``del group[name]`` does in h5py; the old
        content is not restored if the write then fails. A problem during rollback itself
        is attached to the raised exception as a note.

        Parameters
        ----------
        path : str or os.PathLike
            Target HDF5 file.
        root : str
            Absolute group path to write under; missing intermediate groups are created.
        mode : {"w-", "x", "w", "a"}
            ``"w-"`` (default) and ``"x"`` fail if the file exists, ``"w"`` truncates (the previous
            file is discarded even if the write then fails), ``"a"`` writes into an
            existing file or creates one.
        overwrite : bool
            Delete the contents of a non-empty ``root`` group and write over it.

        Raises
        ------
        ValidationError
            A value cannot be stored in HDF5 (an invalid collection key, a `Glob` mismatch,
            an unstorable or unserializable attribute), or ``root`` is occupied.
        SchemaError
            The group class has an invalid HDF5 declaration.
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
