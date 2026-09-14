"""Lazy annotations under PEP 649/749.

Deliberately no ``from __future__ import annotations``: this module exists to exercise
the annotation forms every other test module opts out of. Record types are declared
inside the test functions because a module-level bare forward reference would raise at
collection time on the versions ``pytestmark`` skips.
"""

import dataclasses
import gc
import inspect
import sys
import weakref
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import pytest

import h5t
from h5t._compile import _PendingRecord
from h5t._spec import ForeignSpec, MemberKind

from .conftest import write_result

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 14), reason="PEP 649 lazy annotations require Python 3.14"
)


def marker(record: type) -> Any:
    return record.__dict__["__h5t_record__"]


def test_this_module_keeps_lazy_annotations() -> None:
    # Guards the guard: adding `from __future__ import annotations` to this module
    # would stringize every annotation below and silently void the whole file.
    class Probe:
        value: int

    assert inspect.get_annotations(Probe)["value"] is int


def test_a_lazy_annotation_may_name_a_class_defined_later() -> None:
    # Under PEP 649 the annotation is not evaluated at the class statement, so the
    # bare name below is legal. Compilation still has to defer: inspect.get_annotations
    # is the evaluation point, and it cannot resolve _DefinedLater yet.
    @h5t.group()
    @dataclasses.dataclass
    class Deferred:
        later: _DefinedLater  # noqa: F821 -- bound below; lazy under PEP 649

    assert isinstance(marker(Deferred), _PendingRecord)

    @h5t.group()
    @dataclasses.dataclass
    class _DefinedLater:
        value: int

    foreign = h5t._compile._ensure_record_compiled(Deferred)
    fields = {field.py_name: field for field in foreign.spec.fields}
    assert fields["later"].kind is MemberKind.GROUP
    assert fields["later"].member_type is _DefinedLater


def test_a_lazy_forward_reference_still_unresolved_reports_a_schema_error() -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Deferred:
        undefined: _NeverDefined  # noqa: F821 -- never bound anywhere

    # h5t.load retries the record before touching the filesystem, so the still-
    # unresolved name surfaces as the user-visible SchemaError there.
    with pytest.raises(h5t.SchemaError, match="_NeverDefined"):
        h5t.load(Deferred, "never-opened.h5")


def test_a_broken_lazy_annotation_helper_is_not_a_forward_reference() -> None:
    # A NameError escaping a function the annotation calls is a bug in that function.
    # The lazy form moves where it surfaces -- inside inspect.get_annotations rather
    # than the class statement's own evaluation -- but it must still raise there.
    def alias():
        return _typo_inside_the_helper  # noqa: F821

    with pytest.raises(h5t.SchemaError, match="_typo_inside_the_helper"):

        @h5t.group()
        @dataclasses.dataclass
        class Broken:
            value: alias()


def test_a_lazy_annotation_helper_that_raises_becomes_a_schema_error() -> None:
    # Not a NameError at all: it propagates out of inspect.get_annotations and must
    # be classified the same way the string path already classifies it.
    def alias():
        raise ValueError("kaboom")

    with pytest.raises(h5t.SchemaError, match="kaboom"):

        @h5t.group()
        @dataclasses.dataclass
        class Broken:
            value: alias()


def test_a_mixed_record_snapshots_the_scope_for_its_quoted_annotations() -> None:
    # A quoted annotation still resolves through the namespace h5t builds, so the
    # deferred path must snapshot its dependency. The sources it reads them from are
    # version-dependent: strings on Python versions through 3.13, and __annotate__
    # code-object names via _code_names(code) and _quoted_names(code) on Python 3.14+.
    @h5t.group()
    @dataclasses.dataclass
    class Dependency:
        value: int

    label = "on-disk"

    @h5t.group()
    @dataclasses.dataclass
    class Deferred:
        dependency: "Dependency"
        renamed: "Annotated[int, h5t.Name(label)]"
        undefined: _NeverDefined  # noqa: F821 -- forces the deferred path

    snapshot = marker(Deferred).scope
    assert isinstance(snapshot["Dependency"], weakref.ref)
    assert snapshot["Dependency"]() is Dependency
    # A str rejects weakref.ref, so it is the one kind of entry held strongly.
    assert snapshot["label"] == label


def test_a_record_whose_base_carries_the_unresolved_annotation_defers_too() -> None:
    # A record's bases never compile or snapshot a scope of their own, so
    # _record_annotation_names unions across the MRO -- mirroring the MRO walk
    # _compile_foreign_fields does when it actually resolves the annotations.
    @dataclasses.dataclass
    class Base:
        later: _DefinedLater  # noqa: F821 -- bound below; lazy under PEP 649

    @h5t.group()
    @dataclasses.dataclass
    class Sub(Base):
        also: int

    assert isinstance(marker(Sub), _PendingRecord)

    @h5t.group()
    @dataclasses.dataclass
    class _DefinedLater:
        value: int

    foreign = h5t._compile._ensure_record_compiled(Sub)
    assert [field.py_name for field in foreign.spec.fields] == ["later", "also"]


def test_the_dataset_decorator_does_not_defer_even_under_pep_649() -> None:
    # @h5t.dataset is eager by design: an attributes-only record can never forward-
    # reference another schema type, so nothing is gained by deferring, and the
    # decorator forces __annotate__ to evaluate right here, before Later is bound.
    # Contrast with test_the_group_decorator_defers_under_pep_649 below.
    def declare() -> None:
        with pytest.raises(h5t.SchemaError, match="Later"):

            @h5t.dataset(data="data")
            @dataclasses.dataclass
            class Recording:
                unit: Later  # noqa: F821 -- bound below; never resolves for this decorator
                data: np.ndarray

        class Later:
            pass

    declare()


def test_the_group_decorator_defers_under_pep_649() -> None:
    # The PEP 649 win. A GROUP-kind record's fields are not restricted to attributes,
    # so it may forward-reference another schema type and must defer. Its __annotate__
    # closes over the defining function's cells, so a bare annotation naming a local
    # bound *after* the decorator runs resolves without the scope snapshot -- and so
    # escapes the accepted limitation the quoted form carries, where the referent may
    # be collected first.
    def declare() -> type:
        @h5t.group()
        @dataclasses.dataclass
        class Local:
            later: Later  # noqa: F821 -- bound below; reached via __annotate__'s closure

        @h5t.group()
        @dataclasses.dataclass
        class Later:
            value: int

        return Local

    record = declare()
    gc.collect()
    assert marker(record).scope == {}  # nothing snapshotted; the closure carries it
    foreign = h5t._compile._ensure_record_compiled(record)
    fields = {field.py_name: field for field in foreign.spec.fields}
    assert fields["later"].member_type.__name__ == "Later"


def test_a_lazy_schema_loads_from_a_file(tmp_path: Path) -> None:
    path = tmp_path / "result.h5"
    write_result(path)

    @h5t.group(extras="ignore")
    @dataclasses.dataclass
    class Lazy:
        version: int
        values: np.ndarray
        measurement: LazyRecording  # noqa: F821 -- bound below; lazy under PEP 649
        nested: LazyNested  # noqa: F821 -- same
        optional_note: str | None

    @h5t.dataset(data="data", extras="ignore")
    @dataclasses.dataclass
    class LazyRecording:
        unit: Literal["m"]
        data: h5t.LazyArray
        scale: float = 1.0

    @h5t.group(extras="forbid")
    @dataclasses.dataclass
    class LazyNested:
        answer: int

    assert isinstance(marker(Lazy), _PendingRecord)
    record = h5t.load(Lazy, path)
    assert isinstance(marker(Lazy), ForeignSpec)  # resolved by the load
    assert record.version == 2
    assert np.array_equal(record.values, np.arange(4))
    assert record.measurement.unit == "m"
    assert record.measurement.scale == 1.0
    assert np.array_equal(record.measurement.data.read(), np.arange(5))
    assert record.nested.answer == 42
    assert record.optional_note is None
