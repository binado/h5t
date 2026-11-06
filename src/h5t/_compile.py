"""``Group[T]`` schema compilation and detached HDF5 loading."""

from __future__ import annotations

import dataclasses
import fnmatch
import inspect
import os
import posixpath
import re
import sys
import types
import typing
import weakref
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from types import MappingProxyType
from typing import Annotated, Any, ClassVar, Generic, Literal, Self

import h5py
import numpy as np
from jaxtyping import AbstractArray, jaxtyped
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    ModelWrapValidatorHandler,
    TypeAdapter,
    model_validator,
)
from pydantic import ValidationError as PydanticValidationError
from pydantic._internal._model_construction import ModelMetaclass
from pydantic.fields import FieldInfo
from pydantic_core import PydanticCustomError, PydanticUndefined, core_schema
from typing_extensions import TypeVar

from h5t._array import Dataset, _is_mapping_type
from h5t._errors import SchemaError, ValidationError
from h5t._spec import (
    _NO_DEFAULT,
    ArraySpec,
    AttrFieldSpec,
    AttrsSpec,
    Extras,
    FieldSpec,
    Glob,
    GroupSpec,
    MemberKind,
    attr_path,
    child_path,
)

T = TypeVar("T", bound=BaseModel | Mapping[str, Any], default=Mapping[str, Any])

_IN_H5T_LOAD: ContextVar[bool] = ContextVar("_IN_H5T_LOAD", default=False)

_ANNOTATION_FILE = "<h5t annotation>"


def _schema_error(owner: type, field: str, message: str) -> SchemaError:
    return SchemaError(f"{owner.__name__}.{field}: {message}")


# ---------------------------------------------------------------------------
# jaxtyping integration with Pydantic
# ---------------------------------------------------------------------------


def _jaxtyping_pydantic_schema(
    cls: type[AbstractArray], source_type: Any, handler: GetCoreSchemaHandler
) -> core_schema.CoreSchema:
    array_type = cls.array_type
    if not (
        array_type is np.ndarray
        or (isinstance(array_type, type) and issubclass(array_type, Dataset))
    ):
        raise SchemaError(
            f"jaxtyping array type {array_type!r} is not supported; use np.ndarray or Dataset"
        )
    base_schema = handler(array_type)

    def _validate_jaxtyping(value: Any) -> Any:
        if _IN_H5T_LOAD.get():
            return value
        reason = cls.__instancecheck_str__(value)
        if reason:
            raise PydanticCustomError("jaxtyping", "{reason}", {"reason": reason})
        return value

    return core_schema.chain_schema(
        [
            base_schema,
            core_schema.no_info_plain_validator_function(_validate_jaxtyping),
        ]
    )


setattr(  # noqa: B010
    AbstractArray,
    "__get_pydantic_core_schema__",
    classmethod(_jaxtyping_pydantic_schema),
)


# ---------------------------------------------------------------------------
# Annotation resolution & deferral
# ---------------------------------------------------------------------------


class _UnresolvedAnnotation(Exception):
    """An annotation names something not defined yet, so compilation must defer."""

    def __init__(self, owner: type, name: str | None, message: str) -> None:
        super().__init__(message)
        self.owner = owner
        self.name = name


def _raised_by_annotation(exc: NameError, cls: type) -> bool:
    """Report whether ``exc`` came from the annotation expression itself, not a callee."""
    tb = exc.__traceback__
    while tb is not None and tb.tb_next is not None:
        tb = tb.tb_next
    if tb is None:
        return False
    code = tb.tb_frame.f_code
    if code.co_filename in (_ANNOTATION_FILE, "<string>"):
        return True
    return code is getattr(getattr(cls, "__annotate__", None), "__code__", None)


def _annotation_failure(cls: type, exc: Exception) -> Exception:
    """Classify a failure to resolve ``cls``'s annotations."""
    if isinstance(exc, SchemaError):
        return exc
    if isinstance(exc, NameError) and _raised_by_annotation(exc, cls):
        return _UnresolvedAnnotation(cls, exc.name, str(exc))
    return SchemaError(f"{cls.__name__}: could not resolve annotations: {exc}")


def _resolved_annotations(cls: type, localns: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return ``cls``'s own annotations with strings and lazy values evaluated."""
    namespace: dict[str, Any] = {cls.__name__: cls}
    if localns:
        namespace.update(localns)
    try:
        if sys.version_info >= (3, 14):
            import annotationlib

            raw = annotationlib.get_annotations(cls, format=annotationlib.Format.VALUE)
            module = sys.modules.get(cls.__module__)
            globalns = vars(module) if module is not None else {}
            return {
                name: eval(compile(value, _ANNOTATION_FILE, "eval"), globalns, namespace)
                if isinstance(value, str)
                else value
                for name, value in raw.items()
            }
        own = inspect.get_annotations(cls)
        module = sys.modules.get(cls.__module__)
        globalns = vars(module) if module is not None else {}
        resolved: dict[str, Any] = {}
        for name, value in own.items():
            if isinstance(value, str):
                resolved[name] = eval(compile(value, _ANNOTATION_FILE, "eval"), globalns, namespace)
            else:
                resolved[name] = value
        return resolved
    except Exception as exc:
        raise _annotation_failure(cls, exc) from exc


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
    if not (
        array_type is np.ndarray
        or (isinstance(array_type, type) and issubclass(array_type, Dataset))
    ):
        raise _schema_error(
            owner,
            py_name,
            f"jaxtyping array type {array_type!r} is not supported; use np.ndarray or Dataset",
        )
    return array_type, ArraySpec(core, core.dtype, core.dim_str)


def _check_array(spec: ArraySpec, candidate: Any, path: str) -> None:
    """Ask jaxtyping whether ``candidate`` satisfies ``spec``; call inside a record scope."""
    reason = spec.hint.__instancecheck_str__(candidate)  # ty: ignore[unresolved-attribute]
    if reason:
        raise ValidationError(path, reason)


def _ghost(node: h5py.Dataset) -> np.ndarray:
    """Return an ndarray with ``node``'s shape and dtype that allocates one scalar only."""
    return np.broadcast_to(np.empty((), node.dtype), node.shape)


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


class _IsInstance:
    __slots__ = ("optional", "target_type")

    def __init__(self, target_type: type, optional: bool) -> None:
        self.target_type = target_type
        self.optional = optional

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        schema = core_schema.is_instance_schema(self.target_type)
        if self.optional:
            return core_schema.nullable_schema(schema)
        return schema


def _build_instance_adapter(
    owner: type, py_name: str, target_type: type, optional: bool
) -> TypeAdapter[Any]:
    try:
        return TypeAdapter(_IsInstance(target_type, optional))
    except Exception as exc:
        raise _schema_error(owner, py_name, f"cannot construct a validator: {exc}") from exc


# ---------------------------------------------------------------------------
# AttrsSpec compilation (for T in Group[T] and Dataset[T])
# ---------------------------------------------------------------------------

_ATTRS_CACHE: weakref.WeakKeyDictionary[type[BaseModel], AttrsSpec] = weakref.WeakKeyDictionary()


def _extract_extras(model_cls: type[BaseModel], owner: type) -> Extras:
    raw = model_cls.model_config.get("extra", "ignore")
    if raw is None or raw == "ignore":
        return Extras.IGNORE
    if raw == "forbid":
        return Extras.FORBID
    raise SchemaError(f"{owner.__name__}: extra must be 'ignore' or 'forbid', got {raw!r}")


def _compile_attrs_spec(model_type: type[BaseModel], owner: type) -> AttrsSpec:
    cached = _ATTRS_CACHE.get(model_type)
    if cached is not None:
        return cached

    extras = _extract_extras(model_type, model_type)
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

        adapter_ann: Any = Annotated[core, *all_meta] if all_meta else core
        if optional:
            adapter_ann = adapter_ann | None
        adapter = _build_adapter(model_type, py_name, adapter_ann)

        save_meta = [m for m in all_meta if not isinstance(m, BeforeValidator)]
        save_ann: Any = Annotated[core, *save_meta] if save_meta else core
        if optional:
            save_ann = save_ann | None
        save_adapter = _build_adapter(model_type, py_name, save_ann)

        default: Any = _NO_DEFAULT
        default_factory: Callable[[], Any] | None = None
        if field_info.default is not PydanticUndefined:
            default = field_info.default
        elif field_info.default_factory is not None:
            default_factory = field_info.default_factory  # type: ignore[assignment]

        attr_fields.append(
            AttrFieldSpec(
                py_name=py_name,
                h5_name=h5_name,
                adapter=adapter,
                save_adapter=save_adapter,
                optional=optional,
                default=default,
                default_factory=default_factory,
                array=array_spec,
            )
        )

    model_type.__pydantic_parent_namespace__ = None
    spec = AttrsSpec(
        model_type_ref=weakref.ref(model_type),
        fields=tuple(attr_fields),
        extras=extras,
    )
    _ATTRS_CACHE[model_type] = spec
    return spec


# ---------------------------------------------------------------------------
# Child FieldSpec & GroupSpec compilation
# ---------------------------------------------------------------------------


def _extract_alias_and_default(
    owner: type,
    py_name: str,
    metadata: list[Any],
    raw_default: Any,
) -> tuple[str, Any, Callable[[], Any] | None]:
    alias: str | None = None
    default: Any = _NO_DEFAULT
    default_factory: Callable[[], Any] | None = None

    field_infos = [m for m in metadata if isinstance(m, FieldInfo)]
    if isinstance(raw_default, FieldInfo):
        field_infos.append(raw_default)
        if raw_default.default is not PydanticUndefined:
            default = raw_default.default
        elif raw_default.default_factory is not None:
            default_factory = raw_default.default_factory  # type: ignore[assignment]
    elif raw_default is not _NO_DEFAULT and raw_default is not PydanticUndefined:
        default = raw_default

    for fi in field_infos:
        if fi.alias is not None:
            alias = fi.alias

    h5_name = alias if alias is not None else py_name
    if not isinstance(h5_name, str) or not h5_name:
        raise _schema_error(owner, py_name, "alias requires a non-empty string")
    return h5_name, default, default_factory


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
    item = _field_spec(owner, py_name, item_annotation, _NO_DEFAULT)
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
    glob_marker = _marker(metadata, Glob, owner, py_name)
    h5_name, default, default_factory = _extract_alias_and_default(
        owner, py_name, metadata, raw_default
    )
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
        if args:
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

    if kind is MemberKind.COLLECTION:
        adapter_ann: Any = dict
        adapter = _build_instance_adapter(owner, py_name, dict, optional)
    elif kind in (MemberKind.GROUP, MemberKind.DATASET):
        assert member_type is not None
        adapter_ann = member_type
        adapter = _build_instance_adapter(owner, py_name, member_type, optional)
    else:
        adapter_ann = np.ndarray | None if optional else np.ndarray
        adapter = _build_adapter(owner, py_name, adapter_ann)

    return FieldSpec(
        py_name=py_name,
        h5_name=h5_name,
        kind=kind,
        annotation=adapter_ann,
        adapter=adapter,
        optional=optional,
        default=default,
        default_factory=default_factory,
        member_type=member_type,
        dataset_attrs=dataset_attrs,
        array=array_spec,
        item=item,
        glob=glob,
    )


def _resolve_group_attrs_type(
    group_cls: type[Group[Any]],
    own_annotations: Mapping[str, Any],
) -> AttrsSpec | None:
    """Determine the ``AttrsSpec`` for ``group_cls`` from ``Group[T]`` or ``attrs: T``."""
    attrs_ann: Any = own_annotations.get("attrs")
    if attrs_ann is None:
        for base in group_cls.__mro__:
            meta = base.__dict__.get("__pydantic_generic_metadata__")
            if meta and meta.get("origin") is Group:
                args = meta.get("args", ())
                if args:
                    attrs_ann = args[0]
                    break
            if base is not Group and issubclass(base, Group):
                base_spec = base.__dict__.get("__h5t_group__")
                if isinstance(base_spec, GroupSpec) and base_spec.attrs is not None:
                    return base_spec.attrs

    if (
        attrs_ann is None
        or attrs_ann is Any
        or isinstance(attrs_ann, TypeVar)
        or _is_mapping_type(attrs_ann)
    ):
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


def _compile_group_fields(
    group_cls: type[Group[Any]],
    localns: Mapping[str, Any] | None = None,
) -> GroupSpec:
    extras = _extract_extras(group_cls, group_cls)

    annotations: dict[str, Any] = {}
    raw_defaults: dict[str, Any] = {}
    for base in reversed(group_cls.__mro__):
        if base in (object, BaseModel, Generic, Group):
            continue
        meta = base.__dict__.get("__pydantic_generic_metadata__")
        if meta and meta.get("origin") is Group and not base.__dict__.get("__annotations__"):
            continue
        base_annotations = _resolved_annotations(base, localns)
        for field_name, field_annotation in base_annotations.items():
            if field_name.startswith("_") or typing.get_origin(field_annotation) is ClassVar:
                continue
            annotations[field_name] = field_annotation
            if field_name in base.__dict__:
                raw_defaults[field_name] = base.__dict__[field_name]
            else:
                base_fields = getattr(base, "model_fields", None)
                if base_fields is not None and field_name in base_fields:
                    raw_defaults[field_name] = base_fields[field_name]

    attrs_spec = _resolve_group_attrs_type(group_cls, annotations)
    annotations.pop("attrs", None)

    fields: dict[str, FieldSpec] = {}
    seen_h5_names: dict[str, str] = {}
    for field_name, field_annotation in annotations.items():
        raw_default = raw_defaults.get(field_name, _NO_DEFAULT)
        fspec = _field_spec(group_cls, field_name, field_annotation, raw_default)
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
        extras=extras,
    )


@dataclasses.dataclass(frozen=True)
class _PendingGroup:
    """A `Group` subclass whose annotations could not resolve yet."""

    localns: Mapping[str, Any] | None


def _ensure_group_compiled(cls: type[Group[Any]]) -> GroupSpec:
    marker = cls.__dict__.get("__h5t_group__")
    if isinstance(marker, GroupSpec):
        return marker
    localns = marker.localns if isinstance(marker, _PendingGroup) else None
    spec = _compile_group_fields(cls, localns=localns)
    if not getattr(cls, "__pydantic_complete__", True):
        module = sys.modules.get(cls.__module__)
        globalns = vars(module) if module is not None else {}
        ns: dict[str, Any] = {**globalns, cls.__name__: cls}
        if localns:
            ns.update(localns)
        with contextlib_suppress_pydantic():
            cls.model_rebuild(_types_namespace=ns)
        cls.__pydantic_parent_namespace__ = None
    setattr(cls, "__h5t_group__", spec)  # noqa: B010
    return spec


def _compiled_group(cls: type[Group[Any]]) -> GroupSpec:
    try:
        return _ensure_group_compiled(cls)
    except _UnresolvedAnnotation as exc:
        raise SchemaError(f"{exc.owner.__name__}: could not resolve annotations: {exc}") from exc


class contextlib_suppress_pydantic:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return exc_type is not None and not issubclass(exc_type, SchemaError)


# ---------------------------------------------------------------------------
# Validation and loading helpers
# ---------------------------------------------------------------------------


def _short_pydantic_error(exc: PydanticValidationError) -> str:
    errors = exc.errors(include_url=False, include_context=False, include_input=False)
    if not errors:
        return str(exc).splitlines()[0]
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", ()))
    message = str(first.get("msg", "invalid value"))
    return f"{location}: {message}" if location else message


def _validate_adapter(adapter: TypeAdapter[Any], value: Any, path: str) -> Any:
    try:
        return adapter.validate_python(value)
    except PydanticValidationError as exc:
        raise ValidationError(path, _short_pydantic_error(exc)) from exc


def _missing_attr_value(field: AttrFieldSpec, path: str) -> Any:
    if field.has_default:
        return _validate_adapter(field.adapter, field.default, path)
    if field.default_factory is not None:
        return _validate_adapter(field.adapter, field.default_factory(), path)
    if field.optional:
        return None
    raise ValidationError(path, f"required attribute {field.h5_name!r} is missing")


def _missing_child_value(field: FieldSpec, path: str) -> Any:
    if field.has_default:
        return _validate_adapter(field.adapter, field.default, path)
    if field.default_factory is not None:
        return _validate_adapter(field.adapter, field.default_factory(), path)
    if field.optional:
        return None
    kind = field.kind.value
    raise ValidationError(path, f"required {kind} {field.h5_name!r} is missing")


def _raw_attrs(node: h5py.Group | h5py.Dataset) -> dict[str, Any]:
    return {str(name): node.attrs[name] for name in node.attrs}


def _normalize_raw_attr(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value)
    if isinstance(value, np.str_):
        return str(value)
    if isinstance(value, np.generic) and not isinstance(value, np.ndarray):
        return value.item()
    return value


def _construct_validated_model(
    model_cls: type[BaseModel],
    values: dict[str, Any],
    path: str,
) -> Any:
    token = _IN_H5T_LOAD.set(True)
    try:
        instance = model_cls.model_construct(**values)
        decorators = getattr(model_cls, "__pydantic_decorators__", None)
        if decorators is not None:
            for dec in decorators.field_validators.values():
                if dec.info.mode == "after":
                    for fname in dec.info.fields:
                        if fname in values:
                            setattr(instance, fname, dec.func(getattr(instance, fname)))
            for dec in decorators.model_validators.values():
                if dec.info.mode == "after":
                    res = dec.func(instance)
                    if isinstance(res, model_cls):
                        instance = res
        return instance
    except (ValidationError, SchemaError):
        raise
    except PydanticValidationError as exc:
        raise ValidationError(path, _short_pydantic_error(exc)) from exc
    except Exception as exc:
        raise ValidationError(path, f"{model_cls.__name__} validation failed: {exc}") from exc
    finally:
        _IN_H5T_LOAD.reset(token)


def _load_attrs(
    attrs_spec: AttrsSpec | None,
    node: h5py.Group | h5py.Dataset,
    path: str,
    *,
    forbid_untyped_extras: bool = False,
) -> Any:
    raw = _raw_attrs(node)
    if attrs_spec is None:
        if forbid_untyped_extras and raw:
            first_extra = next(iter(raw))
            raise ValidationError(attr_path(path, first_extra), "unexpected attribute")
        return MappingProxyType(raw)

    if attrs_spec.extras is Extras.FORBID:
        declared = {f.h5_name for f in attrs_spec.fields}
        for name in raw:
            if name not in declared:
                raise ValidationError(attr_path(path, name), "unexpected attribute")

    with jaxtyped("context"):
        values: dict[str, Any] = {}
        for field in attrs_spec.fields:
            apath = attr_path(path, field.h5_name)
            if field.h5_name not in raw:
                values[field.py_name] = _missing_attr_value(field, apath)
                continue
            raw_val = raw[field.h5_name]
            if field.array is not None:
                arr_val = np.asarray(raw_val)
                _check_array(field.array, arr_val, apath)
                values[field.py_name] = _validate_adapter(field.adapter, arr_val, apath)
            else:
                norm_val = (
                    raw_val if isinstance(raw_val, np.ndarray) else _normalize_raw_attr(raw_val)
                )
                values[field.py_name] = _validate_adapter(field.adapter, norm_val, apath)

        return _construct_validated_model(attrs_spec.model_type, values, path)


def _load_dataset_node(
    dataset_cls: type[Dataset[Any]],
    attrs_spec: AttrsSpec | None,
    array_spec: ArraySpec | None,
    node: h5py.Dataset,
    filename: str,
    path: str,
) -> Dataset[Any]:
    shape, dtype = tuple(node.shape), np.dtype(node.dtype)
    attrs_val = _load_attrs(attrs_spec, node, path, forbid_untyped_extras=False)
    ds = dataset_cls._from_hdf5(filename, path, shape, dtype, attrs_val)
    if array_spec is not None:
        _check_array(array_spec, ds, path)
    return ds


def _load_member(field: FieldSpec, node: Any, filename: str, path: str) -> Any:
    """Load one child member from ``node`` and validate it."""
    if field.kind is MemberKind.GROUP:
        if not isinstance(node, h5py.Group):
            raise ValidationError(path, "expected a group, found a dataset")
        assert field.member_type is not None
        group_cls = typing.cast("type[Group[Any]]", field.member_type)
        value = _load_group(_compiled_group(group_cls), node, filename, path)
    elif field.kind is MemberKind.DATASET:
        if not isinstance(node, h5py.Dataset):
            raise ValidationError(path, "expected a dataset, found a group")
        assert field.member_type is not None
        dataset_cls = typing.cast("type[Dataset[Any]]", field.member_type)
        value = _load_dataset_node(
            dataset_cls, field.dataset_attrs, field.array, node, filename, path
        )
    else:
        if not isinstance(node, h5py.Dataset):
            raise ValidationError(path, "expected a dataset, found a group")
        if field.array is not None:
            _check_array(field.array, _ghost(node), path)
        value = np.asarray(node[()])
    return _validate_adapter(field.adapter, value, path)


def _load_collection(field: FieldSpec, node: Any, filename: str, path: str) -> dict[str, Any]:
    """Load each (glob-matching) child of the collection's group as one item."""
    if not isinstance(node, h5py.Group):
        raise ValidationError(path, "expected a group, found a dataset")
    assert field.item is not None
    items: dict[str, Any] = {}
    for name in node.keys():
        key = str(name)
        if field.glob is not None and field.glob.match(key) is None:
            continue
        items[key] = _load_member(field.item, node[name], filename, child_path(path, key))
    return items


def _check_group_extras(spec: GroupSpec, group: h5py.Group, path: str) -> None:
    if spec.extras is Extras.IGNORE:
        return
    children = {field.h5_name for field in spec.fields}
    for name in group.keys():
        rendered = str(name)
        if rendered not in children:
            raise ValidationError(child_path(path, rendered), "unexpected child node")


def _load_group(spec: GroupSpec, group: h5py.Group, filename: str, path: str) -> Any:
    with jaxtyped("context"):
        _check_group_extras(spec, group, path)
        attrs_val = _load_attrs(
            spec.attrs,
            group,
            path,
            forbid_untyped_extras=(spec.extras is Extras.FORBID and spec.attrs is None),
        )
        values: dict[str, Any] = {"attrs": attrs_val}
        for field in spec.fields:
            member_path = child_path(path, field.h5_name)
            if field.h5_name not in group:
                values[field.py_name] = _missing_child_value(field, member_path)
                continue
            node = group[field.h5_name]
            if field.kind is MemberKind.COLLECTION:
                values[field.py_name] = _load_collection(field, node, filename, member_path)
            else:
                values[field.py_name] = _load_member(field, node, filename, member_path)

        return _construct_validated_model(spec.group_type, values, path)


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
            _resolve_group_attrs_type(cls, {})
            return cls

        if not is_user_group:
            return cls

        caller_locals = dict(sys._getframe(1).f_locals)
        if localns:
            caller_locals.update(localns)
        caller_locals[cls_name] = cls

        pending = _PendingGroup(localns=localns)
        setattr(cls, "__h5t_group__", pending)  # noqa: B010
        try:
            spec = _compile_group_fields(cls, localns=caller_locals)
        except _UnresolvedAnnotation:
            return cls
        except BaseException:
            delattr(cls, "__h5t_group__")
            raise
        if not getattr(cls, "__pydantic_complete__", True):
            with contextlib_suppress_pydantic():
                cls.model_rebuild(_types_namespace=caller_locals)
            cls.__pydantic_parent_namespace__ = None
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

    attrs: T = Field(default_factory=dict, validate_default=True)  # type: ignore[assignment]

    @model_validator(mode="wrap")
    @classmethod
    def _h5t_jaxtyping_scope(cls, values: Any, handler: ModelWrapValidatorHandler[Self]) -> Self:
        if _IN_H5T_LOAD.get():
            return handler(values)
        with jaxtyped("context"):
            return handler(values)

    @classmethod
    def model_validate_hdf5(
        cls,
        path: str | os.PathLike[str],
        root: str = "/",
    ) -> Self:
        """Validate and load an HDF5 group into a detached instance of ``cls``.

        The HDF5 file is closed before ``model_validate_hdf5`` returns, even on error.
        Child `Dataset` fields remain lazy and remember the file path so slices or full
        payloads can be read on demand.

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
        """
        spec = _compiled_group(cls)
        normalized_root = _normalize_root(root)
        filename = _filesystem_path(path)
        with h5py.File(filename, mode="r") as h5file:
            node = h5file.get(normalized_root)
            if node is None:
                raise ValidationError(normalized_root, "root group does not exist")
            if not isinstance(node, h5py.Group):
                raise ValidationError(normalized_root, "expected a group, found a dataset")
            return typing.cast(Self, _load_group(spec, node, filename, normalized_root))

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
        created groups or files are rolled back.

        Parameters
        ----------
        path : str or os.PathLike
            Target HDF5 file.
        root : str
            Absolute group path to write under; missing intermediate groups are created.
        mode : {"w-", "x", "w", "a"}
            ``"w-"``/``"x"`` fail if the file exists, ``"w"`` truncates, ``"a"`` writes
            into an existing file or creates one.
        overwrite : bool
            Allow replacing the contents of a non-empty ``root`` group.

        Raises
        ------
        ValidationError
            The model does not conform to its schema, or ``root`` is occupied.
        ValueError
            ``mode`` or ``root`` is invalid.
        """
        from h5t._write import _dump_group

        _dump_group(self, path, root=root, mode=mode, overwrite=overwrite)
