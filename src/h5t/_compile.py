"""Schema compilation and detached HDF5 loading."""

from __future__ import annotations

import dataclasses
import inspect
import os
import posixpath
import reprlib
import sys
import types
import typing
import weakref
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import contextmanager
from enum import Enum
from types import MappingProxyType
from typing import Annotated, Any, ClassVar, Literal, Self

import h5py
import numpy as np
from pydantic import ConfigDict, TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from h5t._array import LazyArray
from h5t._errors import ConversionError, SchemaError, ValidationError
from h5t._spec import (
    _NO_DEFAULT,
    Attr,
    ClassSpec,
    Eager,
    Extras,
    FieldSpec,
    ForeignSpec,
    MemberKind,
    Name,
    Payload,
    attr_path,
    child_path,
)

_SCALAR_TYPES = (str, int, float, bool, bytes, complex)


def _schema_error(owner: type, field: str, message: str) -> SchemaError:
    return SchemaError(f"{owner.__name__}.{field}: {message}")


_ANNOTATION_FILE = "<h5t annotation>"


class _UnresolvedAnnotation(Exception):
    """An annotation names something not defined yet, so compilation must defer.

    Private to this module: raised while resolving annotations and turned into the
    user-visible ``SchemaError`` by ``_ensure_compiled`` if the name is still missing
    when the spec is finally read.
    """

    def __init__(self, owner: type, name: str | None, message: str) -> None:
        super().__init__(message)
        self.owner = owner
        self.name = name


def _raised_by_annotation(exc: NameError, cls: type) -> bool:
    """Report whether ``exc`` came from the annotation expression itself, not a callee.

    Only an unbound name in the expression is a forward reference worth deferring for.
    A ``NameError`` escaping a function the annotation calls is a bug in that function,
    so it must take the ``SchemaError`` path instead. The frame that raised identifies
    the two: a PEP 563 string annotation is compiled under ``_ANNOTATION_FILE``, and a
    PEP 649 lazy annotation is evaluated by ``cls``'s own ``__annotate__`` function.
    """
    tb = exc.__traceback__
    while tb is not None and tb.tb_next is not None:
        tb = tb.tb_next
    if tb is None:
        return False
    code = tb.tb_frame.f_code
    if code.co_filename == _ANNOTATION_FILE:
        return True
    # A class with no annotations of its own has ``__annotate__ is None``.
    return code is getattr(getattr(cls, "__annotate__", None), "__code__", None)


def _code_names(code: types.CodeType) -> set[str]:
    """Collect every name ``code`` and the code objects nested in it could load.

    ``co_names`` over-approximates -- it also holds attribute names -- which is the
    safe direction, since a name the snapshot drops is one resolution cannot find.
    Nested code objects (a lambda inside an annotation) load their names the same way.

    ``co_freevars`` is deliberately not collected. A class ``__annotate__`` has free
    variables only when an enclosing function scope exists, which is exactly when its
    closure keeps those cells reachable on its own -- so for that set the snapshot is
    redundant, and adding it would only widen what ``_weak_scope`` holds strongly.
    """
    names: set[str] = set()
    pending = [code]
    while pending:
        current = pending.pop()
        names.update(current.co_names)
        pending.extend(c for c in current.co_consts if isinstance(c, types.CodeType))
    return names


def _names_in_source(source: str) -> set[str]:
    """Collect every name the annotation expression ``source`` could load."""
    try:
        code = compile(source, _ANNOTATION_FILE, "eval")
    except SyntaxError:
        # Unparseable: reading the spec raises SchemaError, so nothing is needed.
        return set()
    return _code_names(code)


def _referenced_names(annotations: Mapping[str, Any]) -> set[str]:
    """Collect every name the string annotations in ``annotations`` could load."""
    names: set[str] = set()
    for annotation in annotations.values():
        if isinstance(annotation, str):
            names.update(_names_in_source(annotation))
    return names


def _quoted_names(code: types.CodeType) -> set[str]:
    """Collect every name the string constants in ``code`` could load.

    A quoted annotation is a plain string constant in ``__annotate__``: the names it
    mentions never reach ``co_names``, and the compiler makes no closure cell for them
    either, so unlike a bare annotation it has nothing else to resolve through.
    Reading them back out of the constants is what keeps a quoted forward reference
    resolvable in a class that some *other* annotation forced onto the deferred path.
    """
    names: set[str] = set()
    pending = [code]
    while pending:
        current = pending.pop()
        for const in current.co_consts:
            if isinstance(const, types.CodeType):
                pending.append(const)
            elif isinstance(const, str):
                names.update(_names_in_source(const))
    return names


def _annotation_names(cls: type) -> set[str]:
    """Collect every name ``cls``'s own annotations could load, however they failed.

    Reading the annotations yields PEP 563 strings, which name what they mention. On
    3.14 the read is itself the evaluation (PEP 649), so an annotation that could not
    resolve has no value left to inspect: the names come straight off the
    ``__annotate__`` code object instead, without re-running an expression that
    already raised. ``annotationlib``'s ``STRING`` and ``FORWARDREF`` formats look
    like the obvious source here and are not: both re-execute the annotation with the
    real globals first and swallow whatever it raises, which would run a user's
    annotation helper -- and hide its bug -- from inside an exception handler.
    """
    try:
        annotations = inspect.get_annotations(cls, eval_str=False)
    except Exception:
        annotate = getattr(cls, "__annotate__", None)
        code = getattr(annotate, "__code__", None)
        if code is None:
            return set()
        # _weak_scope keeps only names the scope holds, so the compiler's own
        # __classdict__ freevar drops out.
        return _code_names(code) | _quoted_names(code)
    return _referenced_names(annotations)


def _weak_scope(scope: Mapping[str, Any], names: Collection[str]) -> dict[str, Any]:
    """Snapshot the ``names`` entries of ``scope``, weakly wherever the value allows.

    Scopes are only retained by the deferred path, where the snapshot may outlive the
    frame it came from, so it keeps just the names the unresolved annotations mention.
    Values that reject ``weakref.ref`` (``int``, ``str``, tuples, dicts) are stored
    directly, and an unfiltered snapshot would let one of those containers pin an
    unrelated local object graph for as long as the schema class lives.
    """
    snapshot: dict[str, Any] = {}
    for name in names:
        if name not in scope:
            continue
        value = scope[name]
        try:
            snapshot[name] = weakref.ref(value)
        except TypeError:
            snapshot[name] = value
    return snapshot


def _unpack_scope(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Reverse ``_weak_scope``, dropping entries whose referent was collected."""
    scope: dict[str, Any] = {}
    for name, value in snapshot.items():
        if isinstance(value, weakref.ref):
            referent = value()
            if referent is not None:
                scope[name] = referent
        else:
            scope[name] = value
    return scope


def _split_annotation(owner: type, py_name: str, annotation: Any) -> tuple[Any, list[Any], bool]:
    """Extract h5t/constraint metadata and optionality from an annotation."""
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


def _adapter_annotation(core: Any, metadata: list[Any], optional: bool) -> Any:
    foreign = tuple(item for item in metadata if not isinstance(item, (Name, Attr, Eager)))
    annotation = Annotated[core, *foreign] if foreign else core
    return annotation | None if optional else annotation


def _build_adapter(owner: type, py_name: str, annotation: Any) -> TypeAdapter[Any]:
    try:
        return TypeAdapter(annotation, config=ConfigDict(arbitrary_types_allowed=True))
    except Exception as exc:
        raise _schema_error(owner, py_name, f"cannot construct a validator: {exc}") from exc


def _marker(metadata: list[Any], marker_type: type, owner: type, py_name: str) -> Any | None:
    found = [item for item in metadata if isinstance(item, marker_type)]
    if len(found) > 1:
        raise _schema_error(owner, py_name, f"{marker_type.__name__} may appear only once")
    return found[0] if found else None


# PEP 695 aliases are 3.12+; on 3.11 the empty tuple makes every isinstance False.
_alias_type = getattr(typing, "TypeAliasType", None)
_TYPE_ALIAS_TYPES: tuple[type, ...] = (_alias_type,) if isinstance(_alias_type, type) else ()
_MAX_ALIAS_DEPTH = 16


def _is_ndarray_annotation(core: Any) -> bool:
    """Report whether ``core`` denotes ``np.ndarray``, however many aliases deep.

    numpy >= 2.5 defines ``npt.NDArray`` as a PEP 695 ``TypeAliasType``, which defers
    its right-hand side -- deferring is what lets an alias recurse -- so
    ``typing.get_origin`` stops at the alias itself. numpy <= 2.4 built the same name
    out of eager substitution, leaving ``np.ndarray`` directly visible as the origin.
    Expanding ``__value__`` sees through either shape, and through a user's own alias
    of one. The walk is depth-bounded because a PEP 695 alias may be self-referential.
    """
    for _ in range(_MAX_ALIAS_DEPTH):
        if core is np.ndarray:
            return True
        origin = typing.get_origin(core)
        if origin is np.ndarray:
            return True
        if isinstance(core, _TYPE_ALIAS_TYPES):
            core = core.__value__
        elif isinstance(origin, _TYPE_ALIAS_TYPES):
            # A subscripted alias: __value__ carries the parameter, which is dropped
            # here along with the dtype the classification already ignores.
            core = origin.__value__
        else:
            return False
    return False


def _is_scalar_annotation(core: Any) -> bool:
    if core in _SCALAR_TYPES or typing.get_origin(core) is Literal:
        return True
    return isinstance(core, type) and (issubclass(core, Enum) or issubclass(core, np.generic))


def _field_spec(
    owner: type,
    py_name: str,
    annotation: Any,
    default: Any,
    *,
    dataset_owner: bool,
    default_factory: Callable[[], Any] | None = None,
) -> FieldSpec:
    core, metadata, optional = _split_annotation(owner, py_name, annotation)
    name_marker = _marker(metadata, Name, owner, py_name)
    attr_marker = _marker(metadata, Attr, owner, py_name)
    eager_marker = _marker(metadata, Eager, owner, py_name)
    payload_marker = _marker(metadata, Payload, owner, py_name)

    h5_name = name_marker.name if name_marker is not None else py_name
    if not isinstance(h5_name, str) or not h5_name:
        raise _schema_error(owner, py_name, "Name requires a non-empty string")
    if attr_marker is not None and attr_marker.converter is not None:
        if not callable(attr_marker.converter):
            raise _schema_error(owner, py_name, "Attr.converter must be callable")
    if attr_marker is not None and eager_marker is not None:
        raise _schema_error(owner, py_name, "Attr and Eager cannot be combined")
    if payload_marker is not None and attr_marker is not None:
        raise _schema_error(owner, py_name, "Attr and Payload cannot be combined")
    if payload_marker is not None and eager_marker is not None:
        raise _schema_error(owner, py_name, "Eager and Payload cannot be combined")

    is_dataset = isinstance(core, type) and issubclass(core, Dataset)
    is_group = isinstance(core, type) and issubclass(core, Group)

    if payload_marker is not None and (is_dataset or is_group):
        raise _schema_error(owner, py_name, "Payload cannot annotate a Group or Dataset field")

    foreign: ForeignSpec | None = None
    if attr_marker is not None:
        if is_dataset or is_group:
            raise _schema_error(owner, py_name, "Attr cannot annotate a Group or Dataset field")
        kind = MemberKind.ATTRIBUTE
        member_type = None
    elif payload_marker is not None:
        if not isinstance(core, type):
            raise _schema_error(owner, py_name, "Payload requires a class annotation")
        kind = MemberKind.DATASET
        member_type = core
        foreign = _foreign_spec(core, payload_marker, owner, py_name)
    elif is_dataset:
        kind = MemberKind.DATASET
        member_type = core
    elif is_group:
        kind = MemberKind.GROUP
        member_type = core
    elif _is_ndarray_annotation(core):
        # Accept npt.NDArray[...] aliases. The dtype parameter is not validated
        # (the adapter degrades to an isinstance check), so normalize to the
        # plain type to keep declaration equivalence dtype-agnostic.
        kind = MemberKind.ARRAY
        member_type = None
        core = np.ndarray
    elif _is_scalar_annotation(core):
        kind = MemberKind.ATTRIBUTE
        member_type = None
    else:
        raise _schema_error(
            owner,
            py_name,
            f"unsupported annotation {core!r}; containers require Attr(), and dynamic "
            "collections are not supported",
        )

    if eager_marker is not None and kind is not MemberKind.DATASET:
        raise _schema_error(owner, py_name, "Eager applies only to Dataset fields")
    if dataset_owner and kind is not MemberKind.ATTRIBUTE:
        raise _schema_error(owner, py_name, "a dataset record may declare only attributes")
    if kind is not MemberKind.ATTRIBUTE and "/" in h5_name:
        raise _schema_error(owner, py_name, "a child Name cannot contain '/'")

    if foreign is not None:
        # A member-level TypeAdapter cannot validate a foreign record: pydantic
        # rejects `config=` alongside a dataclass/BaseModel/TypedDict type, and
        # revalidating an already-built instance without one is a no-op anyway
        # (see AGENTS.md). Any accepts the constructed instance unchanged; keep
        # the record type itself as `annotation` for repr/equivalence purposes.
        adapter_ann = core
        adapter = _build_adapter(owner, py_name, Any)
    else:
        adapter_ann = _adapter_annotation(core, metadata, optional)
        adapter = _build_adapter(owner, py_name, adapter_ann)

    return FieldSpec(
        py_name=py_name,
        h5_name=h5_name,
        kind=kind,
        annotation=adapter_ann,
        adapter=adapter,
        optional=optional,
        default=default,
        converter=attr_marker.converter if attr_marker is not None else None,
        eager=eager_marker is not None,
        member_type=member_type,
        foreign=foreign,
        default_factory=default_factory,
    )


_FOREIGN_CACHE: weakref.WeakKeyDictionary[type, dict[tuple[str, Extras], ForeignSpec]] = (
    weakref.WeakKeyDictionary()
)


def _foreign_spec(record_type: type, marker: Payload, owner: type, py_name: str) -> ForeignSpec:
    """Compile a plain ``record_type`` into a dataset record's fields and payload.

    Cached per ``(record_type, data_attr, extras)``, since the same foreign class may
    be reused as a ``Payload`` target from more than one field or schema. Foreign
    annotations resolve against ``record_type``'s module globals and class dict only
    (see ``_resolved_annotations``'s ``scope=None`` default): there is no
    ``__init_subclass__`` hook here to capture a defining frame, so an unresolved
    annotation is a ``SchemaError`` rather than something the deferred-compilation
    machinery can retry later.
    """
    try:
        extras = Extras(marker.extras)
    except ValueError:
        raise _schema_error(
            owner, py_name, f"extras must be 'ignore' or 'forbid', got {marker.extras!r}"
        ) from None

    cache = _FOREIGN_CACHE.setdefault(record_type, {})
    cache_key = (marker.data_attr, extras)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    is_dataclass_type = dataclasses.is_dataclass(record_type)
    dc_fields = {f.name: f for f in dataclasses.fields(record_type)} if is_dataclass_type else {}

    annotations: dict[str, Any] = {}
    for base in reversed(record_type.__mro__):
        if base is object:
            continue
        try:
            base_annotations = _resolved_annotations(base)
        except _UnresolvedAnnotation as exc:
            raise _schema_error(
                owner, py_name, f"{record_type.__name__} has unresolved annotations: {exc}"
            ) from exc
        for field_name, field_annotation in base_annotations.items():
            if field_name.startswith("_") or typing.get_origin(field_annotation) is ClassVar:
                continue
            if is_dataclass_type:
                dc_field = dc_fields.get(field_name)
                if dc_field is not None and not dc_field.init:
                    continue  # derived; cannot be passed to the constructor
            annotations[field_name] = field_annotation

    if marker.data_attr not in annotations:
        raise _schema_error(
            owner,
            py_name,
            f"Payload data_attr {marker.data_attr!r} is not a field of {record_type.__name__}",
        )
    payload_annotation = annotations.pop(marker.data_attr)
    payload_core, _, _ = _split_annotation(record_type, marker.data_attr, payload_annotation)
    if _is_ndarray_annotation(payload_core):
        lazy = False
    elif isinstance(payload_core, type) and issubclass(payload_core, LazyArray):
        lazy = True
    else:
        raise _schema_error(
            owner,
            py_name,
            f"Payload field {marker.data_attr!r} must be annotated np.ndarray or LazyArray",
        )

    fields: dict[str, FieldSpec] = {}
    for field_name, field_annotation in annotations.items():
        default: Any = _NO_DEFAULT
        default_factory: Callable[[], Any] | None = None
        if is_dataclass_type:
            dc_field = dc_fields[field_name]
            if dc_field.default is not dataclasses.MISSING:
                default = dc_field.default
            elif dc_field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                default_factory = dc_field.default_factory
        else:
            default = record_type.__dict__.get(field_name, _NO_DEFAULT)
        fields[field_name] = _field_spec(
            record_type,
            field_name,
            field_annotation,
            default,
            dataset_owner=True,
            default_factory=default_factory,
        )

    _check_duplicate_names(record_type.__name__, fields)
    foreign = ForeignSpec(
        record_type=record_type,
        spec=ClassSpec(tuple(fields.values()), extras),
        data_attr=marker.data_attr,
        lazy=lazy,
    )
    cache[cache_key] = foreign
    return foreign


def _annotation_failure(cls: type, exc: Exception) -> Exception:
    """Classify a failure to resolve ``cls``'s annotations."""
    if isinstance(exc, NameError) and _raised_by_annotation(exc, cls):
        return _UnresolvedAnnotation(cls, exc.name, str(exc))
    return SchemaError(f"{cls.__name__}: could not resolve annotations: {exc}")


def _resolved_annotations(cls: type, scope: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return ``cls``'s own annotations, evaluating PEP 563 strings.

    Names resolve against the defining module, then ``scope`` -- the live locals of
    the defining frame while the class is being created, or the weak snapshot kept
    by the deferred path -- then the class body.
    """
    try:
        annotations = inspect.get_annotations(cls, eval_str=False)
    except Exception as exc:
        # PEP 649 (3.14+): this call evaluates lazy annotations, so it fails here
        # where older versions failed at the ``class`` statement itself.
        raise _annotation_failure(cls, exc) from exc
    if not any(isinstance(annotation, str) for annotation in annotations.values()):
        return annotations
    if scope is None:
        scope = _unpack_scope(cls.__dict__.get("_h5t_scope", {}))
    module = sys.modules.get(cls.__module__)
    namespace: dict[str, Any] = dict(vars(module)) if module is not None else {}
    namespace.update(scope)
    namespace.update(vars(cls))
    try:
        return {
            name: eval(compile(annotation, _ANNOTATION_FILE, "eval"), namespace)
            if isinstance(annotation, str)
            else annotation
            for name, annotation in annotations.items()
        }
    except Exception as exc:
        raise _annotation_failure(cls, exc) from exc


def _own_fields(
    cls: type, *, dataset_owner: bool, scope: Mapping[str, Any] | None = None
) -> dict[str, FieldSpec]:
    annotations = _resolved_annotations(cls, scope)
    fields: dict[str, FieldSpec] = {}
    for py_name, annotation in annotations.items():
        if py_name.startswith("_") or typing.get_origin(annotation) is ClassVar:
            continue
        default = cls.__dict__.get(py_name, _NO_DEFAULT)
        fields[py_name] = _field_spec(
            cls,
            py_name,
            annotation,
            default,
            dataset_owner=dataset_owner,
        )
    return fields


def _merged_fields(cls: type) -> dict[str, FieldSpec]:
    merged: dict[str, tuple[type, FieldSpec]] = {}
    for candidate in reversed(cls.__mro__):
        own = candidate.__dict__.get("_h5t_own")
        if not own:
            continue
        for name, field in own.items():
            previous = merged.get(name)
            if previous is None:
                merged[name] = (candidate, field)
                continue
            previous_owner, previous_field = previous
            if issubclass(candidate, previous_owner):
                merged[name] = (candidate, field)
            elif not _fields_equivalent(field, previous_field):
                raise SchemaError(
                    f"{cls.__name__}.{name}: conflicting declarations in bases "
                    f"{previous_owner.__name__} and {candidate.__name__}"
                )
    return {name: field for name, (_, field) in merged.items()}


def _foreign_equivalent(left: ForeignSpec | None, right: ForeignSpec | None) -> bool:
    if left is None or right is None:
        return left is right
    return (
        left.record_type is right.record_type
        and left.data_attr == right.data_attr
        and left.lazy == right.lazy
        and left.spec.extras is right.spec.extras
    )


def _fields_equivalent(left: FieldSpec, right: FieldSpec) -> bool:
    """Compare declarations without adapter identity or array-valued equality."""
    return (
        left.py_name == right.py_name
        and left.h5_name == right.h5_name
        and left.kind is right.kind
        and repr(left.annotation) == repr(right.annotation)
        and left.optional == right.optional
        and repr(left.default) == repr(right.default)
        and left.converter is right.converter
        and left.eager == right.eager
        and left.member_type is right.member_type
        and _foreign_equivalent(left.foreign, right.foreign)
    )


def _resolve_extras(cls: type) -> Extras:
    requested = cls.__dict__.get("_h5t_extras")
    if requested is not None:
        try:
            return Extras(requested)
        except ValueError:
            raise SchemaError(
                f"{cls.__name__}: extras must be 'ignore' or 'forbid', got {requested!r}"
            ) from None
    for base in cls.__mro__[1:]:
        spec = base.__dict__.get("_h5t_spec")
        if spec is not None:
            return spec.extras
    return Extras.IGNORE


def _reserved_names(base: type) -> frozenset[str]:
    return frozenset(name for name in dir(base) if not name.startswith("_"))


def _check_duplicate_names(label: str, fields: Mapping[str, FieldSpec]) -> None:
    attr_names: dict[str, str] = {}
    child_names: dict[str, str] = {}
    for py_name, field in fields.items():
        namespace = attr_names if field.kind is MemberKind.ATTRIBUTE else child_names
        if field.h5_name in namespace:
            raise SchemaError(
                f"{label}: duplicate HDF5 name {field.h5_name!r} for fields "
                f"{namespace[field.h5_name]!r} and {py_name!r}"
            )
        namespace[field.h5_name] = py_name


def _check_fields(cls: type, own: Mapping[str, FieldSpec], fields: Mapping[str, FieldSpec]) -> None:
    base = Dataset if issubclass(cls, Dataset) else Group
    reserved = _reserved_names(base)
    for name in own:
        if name in reserved:
            raise _schema_error(
                cls,
                name,
                f"field shadows the h5t API; rename it and use Name({name!r})",
            )
    _check_duplicate_names(cls.__name__, fields)


def _compile_class(cls: SchemaMeta, scope: Mapping[str, Any] | None = None) -> None:
    """Compile ``cls`` into its own fields and its flattened schema."""
    for base in cls.__mro__[1:]:
        if isinstance(base, SchemaMeta) and "_h5t_spec" not in base.__dict__:
            # _merged_fields and _resolve_extras read compiled state off the bases.
            # A base that is still waiting on a forward reference defers ``cls`` too.
            _compile_class(base)
    own = _own_fields(cls, dataset_owner=issubclass(cls, Dataset), scope=scope)
    cls._h5t_own = own
    fields = _merged_fields(cls)
    _check_fields(cls, own, fields)
    cls._h5t_spec = ClassSpec(tuple(fields.values()), _resolve_extras(cls))


def _ensure_compiled(cls: SchemaMeta) -> None:
    if "_h5t_spec" in cls.__dict__:
        return
    try:
        _compile_class(cls)
    except _UnresolvedAnnotation as exc:
        raise SchemaError(f"{exc.owner.__name__}: could not resolve annotations: {exc}") from exc


class SchemaMeta(type):
    """Metaclass exposing a schema class's compiled spec.

    Schemas compile at the ``class`` statement, so a bad declaration raises there.
    This property is the fallback route for the classes that could not: one whose
    annotations name something defined later compiles on the first read instead.
    """

    _h5t_own: dict[str, FieldSpec]
    _h5t_spec: ClassSpec

    @property
    def __h5spec__(cls) -> ClassSpec:
        """The flattened schema of this class, compiled on first access."""
        _ensure_compiled(cls)
        return cls._h5t_spec


def _short_pydantic_error(exc: PydanticValidationError) -> str:
    errors = exc.errors(include_url=False, include_context=False, include_input=False)
    if not errors:
        return str(exc).splitlines()[0]
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", ()))
    message = str(first.get("msg", "invalid value"))
    return f"{location}: {message}" if location else message


def _validate_value(field: FieldSpec, value: Any, path: str) -> Any:
    try:
        return field.adapter.validate_python(value)
    except PydanticValidationError as exc:
        raise ValidationError(path, _short_pydantic_error(exc)) from exc


def _missing_value(field: FieldSpec, path: str) -> Any:
    if field.has_default:
        return _validate_value(field, field.default, path)
    if field.default_factory is not None:
        return _validate_value(field, field.default_factory(), path)
    if field.optional:
        return None
    kind = field.kind.value
    raise ValidationError(path, f"required {kind} {field.h5_name!r} is missing")


def _raw_attrs(node: h5py.Group | h5py.Dataset) -> dict[str, Any]:
    return {str(name): node.attrs[name] for name in node.attrs}


def _check_group_extras(spec: ClassSpec, group: h5py.Group, path: str) -> None:
    if spec.extras is Extras.IGNORE:
        return
    attrs = {field.h5_name for field in spec.fields if field.kind is MemberKind.ATTRIBUTE}
    children = {field.h5_name for field in spec.fields if field.kind is not MemberKind.ATTRIBUTE}
    for name in group.attrs:
        rendered = str(name)
        if rendered not in attrs:
            raise ValidationError(attr_path(path, rendered), "unexpected attribute")
    for name in group.keys():
        rendered = str(name)
        if rendered not in children:
            raise ValidationError(child_path(path, rendered), "unexpected child node")


def _check_dataset_extras(spec: ClassSpec, dataset: h5py.Dataset, path: str) -> None:
    if spec.extras is Extras.IGNORE:
        return
    attrs = {field.h5_name for field in spec.fields}
    for name in dataset.attrs:
        rendered = str(name)
        if rendered not in attrs:
            raise ValidationError(attr_path(path, rendered), "unexpected attribute")


def _load_attribute(
    field: FieldSpec,
    raw: Mapping[str, Any],
    attrs: dict[str, Any],
    parent_path: str,
) -> Any:
    path = attr_path(parent_path, field.h5_name)
    if field.h5_name not in raw:
        value = _missing_value(field, path)
    else:
        value = raw[field.h5_name]
        if field.converter is not None:
            try:
                value = field.converter(value)
            except Exception as exc:
                raise ConversionError(path, f"attribute converter failed: {exc}") from exc
        value = _validate_value(field, value, path)
    attrs[field.h5_name] = value
    return value


def _load_dataset(
    dataset_type: type[Dataset],
    dataset: h5py.Dataset,
    filename: str,
    path: str,
    *,
    eager: bool = False,
) -> Dataset:
    spec = dataset_type.__h5spec__
    _check_dataset_extras(spec, dataset, path)
    raw = _raw_attrs(dataset)
    attrs = dict(raw)
    values: dict[str, Any] = {}
    for field in spec.fields:
        values[field.py_name] = _load_attribute(field, raw, attrs, path)

    array = LazyArray(
        filename,
        path,
        tuple(dataset.shape),
        np.dtype(dataset.dtype),
        data=np.asarray(dataset[()]) if eager else None,
    )
    instance = object.__new__(dataset_type)
    object.__setattr__(instance, "_h5t_array", array)
    object.__setattr__(instance, "_h5t_attrs", MappingProxyType(attrs))
    for name, value in values.items():
        object.__setattr__(instance, name, value)
    return instance


def _load_foreign_dataset(
    foreign: ForeignSpec, node: h5py.Dataset, filename: str, path: str
) -> Any:
    """Load ``node`` into an instance of ``foreign.record_type`` via its constructor."""
    spec = foreign.spec
    _check_dataset_extras(spec, node, path)
    raw = _raw_attrs(node)
    attrs = dict(raw)
    values: dict[str, Any] = {}
    for field in spec.fields:
        values[field.py_name] = _load_attribute(field, raw, attrs, path)

    if foreign.lazy:
        values[foreign.data_attr] = LazyArray(
            filename, path, tuple(node.shape), np.dtype(node.dtype)
        )
    else:
        values[foreign.data_attr] = np.asarray(node[()])

    try:
        return foreign.record_type(**values)
    except TypeError as exc:
        raise SchemaError(
            f"{foreign.record_type.__name__}: cannot construct from loaded fields: {exc}"
        ) from exc
    except (ValidationError, SchemaError):
        raise
    except Exception as exc:
        raise ValidationError(
            path, f"{foreign.record_type.__name__} constructor failed: {exc}"
        ) from exc


def _load_group(group_type: type[Group], group: h5py.Group, filename: str, path: str) -> Group:
    spec = group_type.__h5spec__
    _check_group_extras(spec, group, path)
    raw = _raw_attrs(group)
    attrs = dict(raw)
    values: dict[str, Any] = {}

    for field in spec.fields:
        if field.kind is MemberKind.ATTRIBUTE:
            values[field.py_name] = _load_attribute(field, raw, attrs, path)
            continue

        member_path = child_path(path, field.h5_name)
        if field.h5_name not in group:
            values[field.py_name] = _missing_value(field, member_path)
            continue
        node = group[field.h5_name]
        if field.kind is MemberKind.GROUP:
            if not isinstance(node, h5py.Group):
                raise ValidationError(member_path, "expected a group, found a dataset")
            assert field.member_type is not None
            nested_group_type = typing.cast(type[Group], field.member_type)
            value = _load_group(nested_group_type, node, filename, member_path)
        elif field.kind is MemberKind.DATASET:
            if not isinstance(node, h5py.Dataset):
                raise ValidationError(member_path, "expected a dataset, found a group")
            if field.foreign is not None:
                value = _load_foreign_dataset(field.foreign, node, filename, member_path)
            else:
                assert field.member_type is not None
                dataset_type = typing.cast(type[Dataset], field.member_type)
                value = _load_dataset(dataset_type, node, filename, member_path, eager=field.eager)
        else:
            if not isinstance(node, h5py.Dataset):
                raise ValidationError(member_path, "expected a dataset, found a group")
            value = np.asarray(node[()])
        values[field.py_name] = _validate_value(field, value, member_path)

    instance = object.__new__(group_type)
    object.__setattr__(instance, "_h5t_attrs", MappingProxyType(attrs))
    for name, value in values.items():
        object.__setattr__(instance, name, value)
    return instance


class _Record(metaclass=SchemaMeta):
    """Shared declaration handling and repr for detached schema records."""

    _h5t_extras: ClassVar[str | None] = None
    _h5t_scope: ClassVar[Mapping[str, Any]] = {}

    def __init_subclass__(
        cls,
        *,
        extras: Literal["ignore", "forbid"] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        cls._h5t_extras = extras
        if cls.__module__ == __name__:
            # Group and Dataset themselves: the module globals _compile_class reads
            # (issubclass(cls, Dataset)) are not bound yet. They compile on first use.
            return
        # The defining frame is still live, so its locals resolve annotations naming
        # function-local classes without retaining anything. Only a genuine forward
        # reference defers, and only that path keeps a (weak) snapshot of the scope.
        frame = inspect.currentframe()
        scope: Mapping[str, Any] = (
            frame.f_back.f_locals if frame is not None and frame.f_back else {}
        )
        try:
            _compile_class(cls, scope)
        except _UnresolvedAnnotation:
            cls._h5t_scope = _weak_scope(scope, _annotation_names(cls))

    def _h5t_repr_fields(self) -> list[tuple[str, Any]]:
        spec = type(self).__h5spec__
        return [(field.py_name, getattr(self, field.py_name)) for field in spec.fields]

    def __repr__(self) -> str:
        fields = ", ".join(
            f"{name}={reprlib.repr(value)}" for name, value in self._h5t_repr_fields()
        )
        return f"{type(self).__name__}({fields})"


class Dataset(_Record):
    """A detached HDF5 dataset with snapshotted metadata and lazy payload data."""

    _h5t_array: LazyArray
    _h5t_attrs: Mapping[str, Any]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError(f"{type(self).__name__} objects are created by Group.from_file()")

    @property
    def attrs(self) -> Mapping[str, Any]:
        """Immutable snapshot of all dataset attributes."""
        return self._h5t_attrs

    @property
    def path(self) -> str:
        """Absolute HDF5 path of this dataset."""
        return self._h5t_array.path

    @property
    def shape(self) -> tuple[int, ...]:
        """Dataset shape captured while the model was loaded."""
        return self._h5t_array.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        """Dataset dtype captured while the model was loaded."""
        return self._h5t_array.dtype

    @property
    def ndim(self) -> int:
        """Number of dimensions in the captured shape."""
        return self._h5t_array.ndim

    @property
    def data(self) -> np.ndarray:
        """Read and cache the complete current payload as a NumPy array."""
        return self._h5t_array.data

    def read(self) -> np.ndarray:
        """Return the same cached complete payload as ``data``."""
        return self._h5t_array.read()

    @contextmanager
    def open(self) -> Iterator[h5py.Dataset]:
        """Open the current source file and yield this dataset for live access."""
        with self._h5t_array.open() as node:
            yield node

    def _h5t_repr_fields(self) -> list[tuple[str, Any]]:
        fields: list[tuple[str, Any]] = [
            ("path", self.path),
            ("shape", self.shape),
            ("dtype", self.dtype),
        ]
        fields.extend(super()._h5t_repr_fields())
        return fields


class Group(_Record):
    """Base class for detached, typed HDF5 group records."""

    _h5t_attrs: Mapping[str, Any]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError(
            f"{type(self).__name__} objects are created by {type(self).__name__}.from_file()"
        )

    @property
    def attrs(self) -> Mapping[str, Any]:
        """Immutable snapshot of all group attributes."""
        return self._h5t_attrs

    @classmethod
    def from_file(cls, path: str | os.PathLike[str], root: str = "/") -> Self:
        """Load this group schema from ``root`` and close the HDF5 file."""
        _ensure_compiled(cls)  # compile a deferred schema before touching the filesystem
        try:
            filesystem_path = os.fsdecode(os.fspath(path))
        except TypeError as exc:
            raise TypeError("path must be a filesystem path") from exc
        if not isinstance(root, str) or not posixpath.isabs(root):
            raise ValueError("root must be an absolute HDF5 group path")
        normalized_root = posixpath.normpath(root)
        if normalized_root.startswith("//"):
            normalized_root = "/" + normalized_root.lstrip("/")
        filename = os.path.abspath(filesystem_path)
        with h5py.File(filename, mode="r") as h5file:
            node = h5file.get(normalized_root)
            if node is None:
                raise ValidationError(normalized_root, "root group does not exist")
            if not isinstance(node, h5py.Group):
                raise ValidationError(normalized_root, "expected a group, found a dataset")
            return typing.cast(Self, _load_group(cls, node, filename, normalized_root))
