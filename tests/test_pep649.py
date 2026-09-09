"""Lazy annotations under PEP 649/749.

Deliberately no ``from __future__ import annotations``: this module exists to exercise
the annotation forms every other test module opts out of. Schema classes are declared
inside the test functions because a module-level bare forward reference would raise at
collection time on the versions ``pytestmark`` skips.
"""

import dataclasses
import gc
import inspect
import sys
import weakref
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pytest

import h5t
from h5t._spec import MemberKind

from .conftest import write_result

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 14), reason="PEP 649 lazy annotations require Python 3.14"
)


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
    class Deferred(h5t.Group):
        later: _DefinedLater  # noqa: F821 -- bound below; lazy under PEP 649

    assert "_h5t_spec" not in Deferred.__dict__

    class _DefinedLater(h5t.Group):
        value: int

    fields = {field.py_name: field for field in Deferred.__h5spec__.fields}
    assert fields["later"].kind is MemberKind.GROUP
    assert fields["later"].member_type is _DefinedLater


def test_a_lazy_forward_reference_still_unresolved_reports_a_schema_error() -> None:
    class Deferred(h5t.Group):
        undefined: _NeverDefined  # noqa: F821 -- never bound anywhere

    with pytest.raises(h5t.SchemaError, match="_NeverDefined"):
        Deferred.__h5spec__


def test_a_broken_lazy_annotation_helper_is_not_a_forward_reference() -> None:
    # A NameError escaping a function the annotation calls is a bug in that function.
    # The lazy form moves where it surfaces -- inside inspect.get_annotations rather
    # than the class statement's own evaluation -- but it must still raise there.
    def alias():
        return _typo_inside_the_helper  # noqa: F821

    with pytest.raises(h5t.SchemaError, match="_typo_inside_the_helper"):

        class Broken(h5t.Group):
            value: alias()


def test_a_lazy_annotation_helper_that_raises_becomes_a_schema_error() -> None:
    # Not a NameError at all: it propagates out of inspect.get_annotations and must
    # be classified the same way the string path already classifies it.
    def alias():
        raise ValueError("kaboom")

    with pytest.raises(h5t.SchemaError, match="kaboom"):

        class Broken(h5t.Group):
            value: alias()


def test_a_lazy_forward_reference_in_a_function_scope_resolves_via_the_closure() -> None:
    # The PEP 649 win. A class body's __annotate__ closes over the defining function's
    # cells, so a bare annotation naming a local bound *after* the class statement
    # resolves without the scope snapshot -- and so escapes the accepted limitation
    # that the quoted form carries, where the referent may be collected first.
    def declare() -> type[h5t.Group]:
        class Local(h5t.Group):
            later: Later  # noqa: F821 -- bound below; reached via __annotate__'s closure

        class Later(h5t.Group):
            value: int

        return Local

    schema = declare()
    gc.collect()
    assert schema.__dict__["_h5t_scope"] == {}
    fields = {field.py_name: field for field in schema.__h5spec__.fields}
    assert fields["later"].kind is MemberKind.GROUP
    assert fields["later"].member_type.__name__ == "Later"


def test_a_mixed_schema_snapshots_the_scope_for_its_quoted_annotations() -> None:
    # A quoted annotation still resolves through the namespace h5t builds, so the
    # deferred path must snapshot its dependency. The sources it reads them from are
    # version-dependent: strings on Python versions through 3.13, and __annotate__
    # code-object names via _code_names(code) and _quoted_names(code) on Python 3.14+.
    class Dependency(h5t.Group):
        value: int

    label = "on-disk"

    class Deferred(h5t.Group):
        dependency: "Dependency"
        renamed: "Annotated[int, h5t.Name(label)]"
        undefined: _NeverDefined  # noqa: F821 -- forces the deferred path

    snapshot = Deferred.__dict__["_h5t_scope"]
    assert isinstance(snapshot["Dependency"], weakref.ref)
    assert snapshot["Dependency"]() is Dependency
    # A str rejects weakref.ref, so it is the one kind of entry held strongly.
    assert snapshot["label"] == label


def test_a_lazy_subclass_of_a_deferred_base_defers_too() -> None:
    # The second unguarded call site: the base's deferral is raised from inside
    # _compile_class's base loop, and __init_subclass__ then has to snapshot the
    # scope for a subclass whose own annotations are equally unevaluable.
    class Base(h5t.Group):
        later: _DefinedLater  # noqa: F821 -- bound below; lazy under PEP 649

    class Sub(Base):
        also: _DefinedLater  # noqa: F821 -- same

    assert "_h5t_spec" not in Base.__dict__
    assert "_h5t_spec" not in Sub.__dict__

    class _DefinedLater(h5t.Group):
        value: int

    assert [field.py_name for field in Sub.__h5spec__.fields] == ["later", "also"]


def test_the_dataset_decorator_does_not_defer_even_under_pep_649() -> None:
    # A Group/Dataset field defers via __h5spec__'s lazy metaclass property, so a bare
    # annotation naming a not-yet-bound local resolves once the enclosing function
    # continues past its binding (see test_a_lazy_forward_reference_in_a_function_
    # scope_resolves_via_the_closure above). @h5t.dataset has no such hook and stays
    # eager by design -- an attributes-only record can never forward-reference another
    # schema type, so nothing is gained by deferring, and the decorator forces
    # __annotate__ to evaluate right here, before Later is ever bound.
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
    # Contrast with test_the_dataset_decorator_does_not_defer_even_under_pep_649 above:
    # a GROUP-kind record's fields are not restricted to attributes, so it can
    # forward-reference another schema type (including itself), and the
    # attributes-only premise that keeps @h5t.dataset eager no longer holds. @h5t.group
    # therefore gets the same tolerance _Record.__init_subclass__ gives Group/Dataset --
    # here demonstrated via the PEP 649 closure win, exactly like the Group case above.
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

    schema = declare()
    gc.collect()
    foreign = h5t._compile._ensure_record_compiled(schema)
    fields = {field.py_name: field for field in foreign.spec.fields}
    assert fields["later"].member_type.__name__ == "Later"


def test_a_lazy_schema_loads_from_a_file(tmp_path: Path) -> None:
    path = tmp_path / "result.h5"
    write_result(path)

    class Lazy(h5t.Group, extras="ignore"):
        version: int
        values: np.ndarray
        measurement: LazyMeasurement  # noqa: F821 -- bound below; lazy under PEP 649
        nested: LazyNested  # noqa: F821 -- same
        optional_note: str | None

    class LazyMeasurement(h5t.Dataset, extras="ignore"):
        unit: Literal["m"]
        scale: float = 1.0

    class LazyNested(h5t.Group, extras="forbid"):
        answer: int

    assert "_h5t_spec" not in Lazy.__dict__
    record = Lazy.from_file(path)
    assert record.version == 2
    assert np.array_equal(record.values, np.arange(4))
    assert record.measurement.unit == "m"
    assert record.measurement.scale == 1.0
    assert record.nested.answer == 42
    assert record.optional_note is None
