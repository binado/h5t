"""``Group[T]`` schema compilation and detached HDF5 loading."""

from __future__ import annotations

import contextlib
import dataclasses
import fnmatch
import os
import posixpath
import re
import sys
import types
import typing
import weakref
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
    PlainSerializer,
    PydanticSchemaGenerationError,
    PydanticUndefinedAnnotation,
    PydanticUserError,
    TypeAdapter,
    ValidatorFunctionWrapHandler,
    WrapSerializer,
    field_validator,
    model_validator,
)
from pydantic import ValidationError as PydanticValidationError
from pydantic._internal._model_construction import ModelMetaclass
from pydantic.fields import FieldInfo
from pydantic_core import PydanticCustomError, core_schema
from typing_extensions import TypeVar

from h5t._array import Dataset, _is_mapping_type
from h5t._errors import SchemaError, ValidationError
from h5t._h5view import ATTRS_KEY, _ghost, _GroupView
from h5t._spec import (
    ArraySpec,
    AttrFieldSpec,
    AttrsSpec,
    FieldSpec,
    Glob,
    GroupSpec,
    MemberKind,
    attr_path,
    child_path,
)

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


def _unwrap_jaxtyping(owner: type, py_name: str, core: Any) -> tuple[Any, ArraySpec | None]:
    """Split a jaxtyping array type into its backing class and an ``ArraySpec``."""
    if not (isinstance(core, type) and issubclass(core, AbstractArray)):
        return core, None
    array_type = core.array_type
    if not _is_supported_array_type(array_type):
        raise _schema_error(
            owner,
            py_name,
            f"jaxtyping array type {array_type!r} is not supported; use np.ndarray or Dataset",
        )
    return array_type, ArraySpec(core, core.dtype, core.dim_str)


def _check_array(spec: ArraySpec, candidate: Any, path: str) -> None:
    """Ask jaxtyping whether ``candidate`` satisfies ``spec``; call inside a jaxtyping scope."""
    reason = spec.hint.__instancecheck_str__(candidate)  # ty: ignore[unresolved-attribute]
    if reason:
        raise ValidationError(path, reason)


def _is_mapping_annotation(core: Any) -> bool:
    """Report whether ``core`` denotes a ``Mapping`` (bare, subscripted, or ``dict``)."""
    origin = typing.get_origin(core) or core
    return isinstance(origin, type) and issubclass(origin, Mapping)


def _compile_glob(owner: type, py_name: str, marker: Glob) -> re.Pattern[str]:
    pattern = marker.pattern
    if not isinstance(pattern, str) or not pattern:
        raise _schema_error(owner, py_name, "Glob requires a non-empty string pattern")
    if "/" in pattern:
        raise _schema_error(owner, py_name, "a Glob pattern cannot contain '/'")
    return re.compile(fnmatch.translate(pattern))


def _build_adapter(owner: type, py_name: str, annotation: Any) -> TypeAdapter[Any]:
    try:
        return TypeAdapter(annotation, config=ConfigDict(arbitrary_types_allowed=True))
    except Exception as exc:
        raise _schema_error(owner, py_name, f"cannot construct a validator: {exc}") from exc


# ---------------------------------------------------------------------------
# AttrsSpec compilation (for T in Group[T] and Dataset[T])
# ---------------------------------------------------------------------------

_ATTRS_CACHE: weakref.WeakKeyDictionary[type[BaseModel], AttrsSpec] = weakref.WeakKeyDictionary()


def _extract_forbid_extras(model_cls: type[BaseModel], owner: type) -> bool:
    raw = model_cls.model_config.get("extra", "ignore")
    if raw is None or raw == "ignore":
        return False
    if raw == "forbid":
        return True
    raise SchemaError(f"{owner.__name__}: extra must be 'ignore' or 'forbid', got {raw!r}")


def _compile_attrs_spec(model_type: type[BaseModel], owner: type) -> AttrsSpec:
    cached = _ATTRS_CACHE.get(model_type)
    if cached is not None:
        return cached

    forbid_extras = _extract_forbid_extras(model_type, model_type)
    if not getattr(model_type, "__pydantic_complete__", True):
        try:
            module = sys.modules.get(model_type.__module__)
            globalns = vars(module) if module is not None else {}
            model_type.model_rebuild(_types_namespace=globalns)
        except Exception as exc:
            raise SchemaError(
                f"{model_type.__name__}: could not resolve attribute annotations: {exc}"
            ) from exc

    attr_fields: list[AttrFieldSpec] = []
    seen_h5_names: dict[str, str] = {}
    for py_name, field_info in model_type.model_fields.items():
        raw_ann = field_info.annotation
        core, meta, optional = _split_annotation(model_type, py_name, raw_ann)
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

        core, array_spec = _unwrap_jaxtyping(model_type, py_name, core)
        if isinstance(core, type) and issubclass(core, (Group, Dataset)):
            raise _schema_error(
                model_type,
                py_name,
                f"attributes model cannot contain {core.__name__} fields",
            )
        if _is_ndarray_annotation(core):
            core = np.ndarray

        # Dump-time adapter: type check plus serializers only. Validators already ran when
        # the attrs model was built and must not run again on every save.
        save_meta = [m for m in all_meta if isinstance(m, (PlainSerializer, WrapSerializer))]
        save_ann: Any = Annotated[core, *save_meta] if save_meta else core
        if optional:
            save_ann = save_ann | None
        save_adapter = _build_adapter(model_type, py_name, save_ann)

        attr_fields.append(
            AttrFieldSpec(
                py_name=py_name,
                h5_name=h5_name,
                save_adapter=save_adapter,
                optional=optional,
                array=array_spec,
            )
        )

    spec = AttrsSpec(
        model_type_ref=weakref.ref(model_type),
        fields=tuple(attr_fields),
        forbid_extras=forbid_extras,
    )
    _ATTRS_CACHE[model_type] = spec
    return spec


# ---------------------------------------------------------------------------
# Child FieldSpec & GroupSpec compilation
# ---------------------------------------------------------------------------


def _extract_alias(
    owner: type,
    py_name: str,
    metadata: list[Any],
    raw_default: Any,
) -> str:
    alias: str | None = None
    field_infos = [m for m in metadata if isinstance(m, FieldInfo)]
    if isinstance(raw_default, FieldInfo):
        field_infos.append(raw_default)
    for fi in field_infos:
        if fi.alias is not None:
            alias = fi.alias
    h5_name = alias if alias is not None else py_name
    if not isinstance(h5_name, str) or not h5_name:
        raise _schema_error(owner, py_name, "alias requires a non-empty string")
    return h5_name


def _collection_item(owner: type, py_name: str, core: Any) -> FieldSpec:
    """Compile the item type of a ``dict[str, T]`` / ``Mapping[str, T]`` collection."""
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
    item = _field_spec(owner, py_name, item_annotation, None)
    if item.kind is MemberKind.COLLECTION:
        raise _schema_error(owner, py_name, "nested collections are not supported")
    return item


def _field_spec(
    owner: type,
    py_name: str,
    annotation: Any,
    raw_default: Any,
) -> FieldSpec:
    core, metadata, optional = _split_annotation(owner, py_name, annotation)
    if isinstance(raw_default, FieldInfo):
        metadata = [*metadata, *raw_default.metadata]
    glob_marker = _marker(metadata, Glob, owner, py_name)
    h5_name = _extract_alias(owner, py_name, metadata, raw_default)
    if "/" in h5_name:
        raise _schema_error(owner, py_name, "a child alias cannot contain '/'")

    core, array_spec = _unwrap_jaxtyping(owner, py_name, core)

    # Check if core is Dataset[T] or a Dataset subclass
    origin = typing.get_origin(core)
    dataset_attrs: AttrsSpec | None = None
    is_dataset = False
    dataset_cls: type[Dataset[Any]] | None = None
    if isinstance(core, type) and issubclass(core, Dataset):
        is_dataset = True
        dataset_cls = core
        if core._h5t_attrs_type is not None:
            dataset_attrs = _compile_attrs_spec(core._h5t_attrs_type, owner)
    elif isinstance(origin, type) and issubclass(origin, Dataset):
        is_dataset = True
        args = typing.get_args(core)
        # A still-open TypeVar (``Dataset[T]`` in a generic group) says nothing about
        # the attrs model, so it behaves like a bare ``Dataset``.
        if args and not isinstance(args[0], TypeVar):
            dataset_cls = origin[args[0]]
            if dataset_cls._h5t_attrs_type is not None:
                dataset_attrs = _compile_attrs_spec(dataset_cls._h5t_attrs_type, owner)
        else:
            dataset_cls = origin

    is_group = isinstance(core, type) and issubclass(core, Group) and core is not Group

    item: FieldSpec | None = None
    glob: re.Pattern[str] | None = None
    member_type: type | None = None

    if is_group:
        kind = MemberKind.GROUP
        member_type = core
    elif is_dataset:
        kind = MemberKind.DATASET
        member_type = dataset_cls
    elif _is_ndarray_annotation(core):
        kind = MemberKind.ARRAY
        core = np.ndarray
    elif _is_mapping_annotation(core):
        kind = MemberKind.COLLECTION
        item = _collection_item(owner, py_name, core)
        glob = _compile_glob(owner, py_name, glob_marker) if glob_marker is not None else None
    else:
        raise _schema_error(
            owner,
            py_name,
            f"unsupported child field annotation {core!r}; group fields must be "
            "Group, Dataset, np.ndarray, or dict[str, T] (declare HDF5 attributes on Group[Attrs])",
        )

    if glob_marker is not None and kind is not MemberKind.COLLECTION:
        raise _schema_error(owner, py_name, "Glob applies only to dict[str, T] collection fields")

    return FieldSpec(
        py_name=py_name,
        h5_name=h5_name,
        kind=kind,
        optional=optional,
        member_type=member_type,
        dataset_attrs=dataset_attrs,
        array=array_spec,
        item=item,
        glob=glob,
    )


def _resolve_group_attrs_type(group_cls: type[Group[Any]]) -> AttrsSpec | None:
    """Determine the ``AttrsSpec`` for ``group_cls`` from its resolved ``attrs`` field."""
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
        return _compile_attrs_spec(attrs_ann, group_cls)
    raise _schema_error(
        group_cls,
        "attrs",
        f"attrs must be a BaseModel subclass or Mapping[str, Any], got {attrs_ann!r}",
    )


def _compile_group_fields(group_cls: type[Group[Any]]) -> GroupSpec:
    """Compile a *complete* ``group_cls`` from the fields Pydantic has resolved."""
    forbid_extras = _extract_forbid_extras(group_cls, group_cls)
    attrs_spec = _resolve_group_attrs_type(group_cls)

    fields: dict[str, FieldSpec] = {}
    seen_h5_names: dict[str, str] = {}
    for field_name, field_info in group_cls.model_fields.items():
        if field_name == "attrs":
            continue
        fspec = _field_spec(group_cls, field_name, field_info.annotation, field_info)
        if fspec.h5_name in seen_h5_names:
            raise SchemaError(
                f"{group_cls.__name__}: duplicate HDF5 name {fspec.h5_name!r} for fields "
                f"{seen_h5_names[fspec.h5_name]!r} and {field_name!r}"
            )
        seen_h5_names[fspec.h5_name] = field_name
        fields[field_name] = fspec

    return GroupSpec(
        group_type=group_cls,
        attrs=attrs_spec,
        fields=tuple(fields.values()),
        forbid_extras=forbid_extras,
    )


@dataclasses.dataclass(frozen=True)
class _PendingGroup:
    """A `Group` subclass whose annotations could not resolve yet."""

    localns: Mapping[str, Any] | None


def _rebuild_model(
    cls: type[BaseModel], namespace: dict[str, Any]
) -> PydanticUndefinedAnnotation | None:
    """Rebuild ``cls``; return the still-undefined forward reference, if there is one."""
    try:
        cls.model_rebuild(_types_namespace=namespace)
    except PydanticUndefinedAnnotation as exc:
        return exc
    except (PydanticUserError, PydanticSchemaGenerationError) as exc:
        raise SchemaError(f"{cls.__name__}: could not build the Pydantic schema: {exc}") from exc
    return None


def _read_array(value: Any) -> Any:
    """Turn an open ``h5py.Dataset`` into an ndarray; pass any other value through."""
    if isinstance(value, h5py.Dataset):
        return np.asarray(value[()])
    return value


def _with_array_readers(schema: Any) -> Any:
    """Return ``schema`` with every ``np.ndarray`` instance check preceded by `_read_array`.

    Pydantic knows ``np.ndarray`` only as an ``isinstance`` check, so a raw ``h5py.Dataset``
    would be rejected. Rewriting those nodes where the schema is generated covers plain
    arrays, collections of them and any other nesting, without touching the built model.
    """
    if isinstance(schema, dict):
        if schema.get("type") == "is-instance" and schema.get("cls") is np.ndarray:
            return core_schema.no_info_before_validator_function(_read_array, schema)
        return {key: _with_array_readers(value) for key, value in schema.items()}
    if isinstance(schema, list):
        return [_with_array_readers(item) for item in schema]
    return schema


def _group_namespace(cls: type, localns: Mapping[str, Any] | None) -> dict[str, Any]:
    module = sys.modules.get(cls.__module__)
    globalns = vars(module) if module is not None else {}
    ns: dict[str, Any] = {**globalns, cls.__name__: cls}
    if localns:
        ns.update(localns)
    return ns


def _compiled_group(cls: type[Group[Any]]) -> GroupSpec:
    """Return ``cls``'s spec, finishing a deferred compilation first."""
    marker = cls.__dict__.get("__h5t_group__")
    if isinstance(marker, GroupSpec):
        return marker
    if not getattr(cls, "__pydantic_complete__", True):
        localns = marker.localns if isinstance(marker, _PendingGroup) else None
        undefined = _rebuild_model(cls, _group_namespace(cls, localns))
        cls.__pydantic_parent_namespace__ = None
        if undefined is not None:
            raise SchemaError(
                f"{cls.__name__}: could not resolve annotation name {undefined.name!r}. If it "
                f"is defined in a function scope, declare the group with "
                f"`localns={{...}}` as a class keyword argument."
            ) from undefined
    spec = _compile_group_fields(cls)
    setattr(cls, "__h5t_group__", spec)  # noqa: B010
    return spec


# ---------------------------------------------------------------------------
# Validation and loading helpers
# ---------------------------------------------------------------------------


def _first_error_parts(exc: PydanticValidationError) -> tuple[tuple[Any, ...], str]:
    """Return the first error's ``(loc, message)`` pair."""
    errors = exc.errors(include_url=False, include_context=False, include_input=False)
    if not errors:
        return (), str(exc).splitlines()[0]
    first = errors[0]
    return tuple(first.get("loc", ())), str(first.get("msg", "invalid value"))


def _short_pydantic_error(exc: PydanticValidationError) -> str:
    loc, message = _first_error_parts(exc)
    if not loc:
        return message
    return f"{_dotted(loc)}: {message}"


def _validate_adapter(adapter: TypeAdapter[Any], value: Any, path: str) -> Any:
    try:
        return adapter.validate_python(value)
    except PydanticValidationError as exc:
        raise ValidationError(path, _short_pydantic_error(exc)) from exc


def _h5_error_path(spec: GroupSpec, root: str, loc: tuple[Any, ...]) -> tuple[str, str]:
    """Translate a Pydantic error ``loc`` into ``(hdf5_path, unplaced_location)``.

    Walks ``loc`` down the compiled specs: a field alias becomes a child path, `ATTRS_KEY`
    followed by a name becomes an attribute path, and a collection key becomes the item's
    path. For a typed `Dataset[T]`, a field name on ``T`` becomes ``/dataset@attr``.
    Whatever cannot be placed (an index into an array, a nested dict key inside an
    attribute, ...) is returned as a dotted remainder for the message. A name no field
    claims is an undeclared child, so it is reported at that child.
    """
    path = root
    current: GroupSpec | None = spec
    i = 0
    while i < len(loc) and current is not None:
        part = loc[i]
        if part == ATTRS_KEY:
            if i + 1 < len(loc):
                return attr_path(path, str(loc[i + 1])), _dotted(loc[i + 2 :])
            return path, ""
        field = next((f for f in current.fields if f.h5_name == part), None)
        path = child_path(path, str(part))
        i += 1
        current = None
        if field is None:
            break
        member = field
        if field.kind is MemberKind.COLLECTION and field.item is not None and i < len(loc):
            path = child_path(path, str(loc[i]))
            i += 1
            member = field.item
        if member.kind is MemberKind.GROUP and member.member_type is not None:
            current = _compiled_group(typing.cast("type[Group[Any]]", member.member_type))
        elif (
            member.kind is MemberKind.DATASET and member.dataset_attrs is not None and i < len(loc)
        ):
            return attr_path(path, str(loc[i])), _dotted(loc[i + 1 :])
    return path, _dotted(loc[i:])


def _dotted(loc: tuple[Any, ...]) -> str:
    return ".".join(str(part) for part in loc)


def _compile_closure(spec: GroupSpec) -> None:
    """Compile every group reachable from ``spec``, so an unresolved name is a `SchemaError`."""
    seen: set[type] = {spec.group_type}
    stack = [spec]
    while stack:
        for fspec in stack.pop().fields:
            member = fspec.item if fspec.item is not None else fspec
            if member.kind is not MemberKind.GROUP or member.member_type is None:
                continue
            if member.member_type in seen:
                continue
            seen.add(member.member_type)
            stack.append(_compiled_group(typing.cast("type[Group[Any]]", member.member_type)))


def _first_error(exc: PydanticValidationError, spec: GroupSpec, root: str) -> ValidationError:
    """Convert the first Pydantic error into an `h5t.ValidationError` at its HDF5 path."""
    loc, message = _first_error_parts(exc)
    path, rest = _h5_error_path(spec, root, loc)
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
# Group base class & metaclass
# ---------------------------------------------------------------------------


class _GroupMeta(ModelMetaclass):
    """Metaclass for `Group` that compiles child and attribute specs on subclass creation."""

    def __new__(
        mcs,
        cls_name: str,
        bases: tuple[type[Any], ...],
        namespace: dict[str, Any],
        *,
        localns: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        group_base = globals().get("Group")
        is_generic_sub = "[" in cls_name
        is_user_group = (
            group_base is not None
            and not is_generic_sub
            and any(isinstance(b, type) and issubclass(b, group_base) for b in bases)
        )

        try:
            raw_cls = super().__new__(mcs, cls_name, bases, namespace, **kwargs)
        except SchemaError:
            raise
        except Exception as exc:
            raise SchemaError(f"{cls_name}: could not resolve annotations: {exc}") from exc

        cls = typing.cast("type[Group[Any]]", raw_cls)
        cls.__pydantic_parent_namespace__ = None

        if is_generic_sub and group_base is not None:
            _resolve_group_attrs_type(cls)
            return cls

        if not is_user_group:
            return cls

        caller_locals = dict(sys._getframe(1).f_locals)
        if localns:
            caller_locals.update(localns)
        caller_locals[cls_name] = cls

        setattr(cls, "__h5t_group__", _PendingGroup(localns=localns))  # noqa: B010
        try:
            if not getattr(cls, "__pydantic_complete__", True):
                undefined = _rebuild_model(cls, caller_locals)
                cls.__pydantic_parent_namespace__ = None
                if undefined is not None:
                    return cls  # a forward reference: retried when the group is first used
            spec = _compile_group_fields(cls)
        except BaseException:
            # Never leave the pending marker behind on a class whose definition failed.
            with contextlib.suppress(AttributeError):
                delattr(cls, "__h5t_group__")
            raise
        setattr(cls, "__h5t_group__", spec)  # noqa: B010
        return cls


class Group(BaseModel, Generic[T], metaclass=_GroupMeta):
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
        return _with_array_readers(handler(source))

    @field_validator("attrs", mode="wrap", check_fields=False)
    @classmethod
    def _h5t_attrs_scope(cls, value: Any, handler: ValidatorFunctionWrapHandler) -> Any:
        # Attribute dims must not bind together with the dims of the group's children.
        with jaxtyped("context"):
            return handler(value)

    @model_validator(mode="wrap")
    @classmethod
    def _h5t_jaxtyping_scope(cls, values: Any, handler: ModelWrapValidatorHandler[Self]) -> Self:
        spec = _compiled_group(cls) if isinstance(values, _GroupView) else None
        if spec is not None and spec.attrs is None and spec.forbid_extras:
            name = next(iter(values.node.attrs), None)
            if name is not None:
                path = attr_path(values.node.name, str(name))
                raise ValidationError(path, "unexpected attribute")
        with jaxtyped("context"):
            model = handler(values)
        if spec is not None and spec.attrs is None:
            # Untyped attributes of a loaded group are a read-only snapshot.
            model.attrs = typing.cast(
                T, MappingProxyType(dict(typing.cast(Mapping[str, Any], model.attrs)))
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
            One of ``cls``'s annotations cannot be resolved.
        ValidationError
            The file does not conform to the schema (including a missing ``root``).
        ValueError
            ``root`` is not an absolute HDF5 path.
        OSError
            The file cannot be opened.
        TypeError
            ``path`` is not a string or path-like object.
        """
        spec = _compiled_group(cls)
        _compile_closure(spec)
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
                raise _first_error(exc, spec, normalized_root) from exc

    def model_dump_hdf5(
        self,
        path: str | os.PathLike[str],
        root: str = "/",
        mode: Literal["w-", "x", "w", "a"] = "w-",
        overwrite: bool = False,
    ) -> None:
        """Validate ``self`` against its schema and write it under ``root`` in ``path``.

        The group tree is fully planned and validated before the file is opened, so an
        invalid model never touches the target file. If writing fails midway, any
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
            The model does not conform to its schema, or ``root`` is occupied.
        SchemaError
            The group class has an invalid declaration or an unresolved annotation.
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
