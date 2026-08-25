"""The class compiler: annotations -> __h5spec__, and class-creation errors.

Note: no ``from __future__ import annotations`` here. PEP 563 stringifies
annotations, and schema classes defined inside test functions reference
local names that string evaluation cannot resolve (a documented v0.1
limitation of the compiler).
"""

from typing import Annotated, Literal

import numpy as np
import pytest

import h5t
from h5t._spec import DatasetSpec, Extras, GroupSpec

from .conftest import PEResult, Posterior


class TestMemberKinds:
    def test_inline_dataset_subscript(self):
        class S(h5t.Group):
            x: h5t.Dataset[h5t.f8]

        (child,) = S.__h5spec__.children
        assert isinstance(child, DatasetSpec)
        assert child.dtype is h5t.f8

    def test_named_dataset_subclass_with_attrs(self):
        class StrainSeries(h5t.Dataset, dtype=h5t.f8):
            unit: Literal["strain"]
            t0: float
            dt: float

        class Detector(h5t.Group):
            strain: StrainSeries

        (child,) = Detector.__h5spec__.children
        assert isinstance(child, DatasetSpec)
        assert child.dtype is h5t.f8
        assert {a.py_name for a in child.attrs} == {"unit", "t0", "dt"}

    def test_named_dataset_validator_is_preserved_at_use_site(self):
        class Mass(h5t.Dataset, dtype=h5t.f8):
            unit: Literal["Msun"]

            def validate(self) -> None:
                raise h5t.Invalid("mass failure")

        class S(h5t.Group):
            mass: Mass

        (mass,) = S.__h5spec__.children
        assert mass.view_type is Mass
        assert {attr.py_name for attr in mass.attrs} == {"unit"}

    def test_group_subclass_member(self):
        class Inner(h5t.Group):
            x: h5t.Dataset[h5t.f8]

        class Outer(h5t.Group):
            inner: Inner

        (child,) = Outer.__h5spec__.children
        assert isinstance(child, GroupSpec)
        assert child.py_name == "inner"
        assert child.view_type is Inner

    def test_dynamic_group_subscript(self):
        assert PEResult.__h5spec__.children[0].dynamic is not None
        dynamic = PEResult.__h5spec__.children[0].dynamic
        assert dynamic.pattern == r"C\d+:.*"
        assert dynamic.item.view_type is Posterior

    def test_named_dynamic_group_with_attrs(self):
        class Runs(h5t.Group[Posterior]):
            created_by: str

        class F(h5t.File):
            runs: Runs

        (child,) = F.__h5spec__.children
        assert child.dynamic is not None
        assert child.dynamic.item.view_type is Posterior
        assert {a.py_name for a in child.attrs} == {"created_by"}

    def test_attrs_by_elimination(self):
        class S(h5t.Group):
            a: str
            b: float = 20.0
            c: Literal["x", "y"]
            d: np.ndarray
            e: int | None

        spec = S.__h5spec__
        assert not spec.children
        by_name = {a.py_name: a for a in spec.attrs}
        assert by_name["a"].type.base is str
        assert by_name["b"].default == 20.0
        assert by_name["c"].type.literals == ("x", "y")
        assert by_name["d"].type.base is np.ndarray
        assert by_name["e"].optional


class TestNames:
    def test_name_marker_maps_h5_name(self):
        spec = PEResult.__h5spec__
        (attr,) = spec.attrs
        assert attr.py_name == "format_version"
        assert attr.h5_name == "version"

    def test_collision_across_namespaces_is_allowed(self):
        class Observation(h5t.Group):
            duration: h5t.Dataset[h5t.f8]
            duration_attr: Annotated[float, h5t.Name("duration")]

        spec = Observation.__h5spec__
        assert spec.children[0].h5_name == "duration"
        assert spec.attrs[0].h5_name == "duration"

    def test_duplicate_names_within_namespace_raise(self):
        with pytest.raises(h5t.SchemaError, match="duplicate HDF5"):

            class S(h5t.Group):
                a: Annotated[str, h5t.Name("x")]
                b: Annotated[str, h5t.Name("x")]

    def test_api_shadowing_raises(self):
        with pytest.raises(h5t.SchemaError, match="shadows the h5t API"):

            class S(h5t.Group):
                keys: h5t.Dataset[h5t.i8]

    def test_safe_alias_for_reserved_names(self):
        class S(h5t.Group):
            keys_: Annotated[h5t.Dataset[h5t.i8], h5t.Name("keys")]

        assert S.__h5spec__.children[0].h5_name == "keys"


class TestErrors:
    def test_removed_shape_api_is_not_exported(self):
        assert not hasattr(h5t, "Shape")
        assert not hasattr(h5t, "FromAttr")

    def test_dataset_body_cannot_declare_children(self):
        with pytest.raises(h5t.SchemaError, match="cannot contain child nodes"):

            class D(h5t.Dataset, dtype=h5t.f8):
                child: h5t.Dataset[h5t.f8]

    def test_dataset_member_needs_dtype(self):
        with pytest.raises(h5t.SchemaError, match="no dtype"):

            class S(h5t.Group):
                x: h5t.Dataset

    def test_default_on_dataset_member_raises(self):
        with pytest.raises(h5t.SchemaError, match="attrs only"):

            class S(h5t.Group):
                x: h5t.Dataset[h5t.f8] = 3  # type: ignore[assignment]

    def test_non_optional_union_raises(self):
        with pytest.raises(h5t.SchemaError, match="unions"):

            class S(h5t.Group):
                x: int | str

    def test_unsupported_attr_type_raises(self):
        with pytest.raises(h5t.SchemaError, match="unsupported attr type"):

            class S(h5t.Group):
                x: list[str]

    def test_invalid_keys_pattern_raises(self):
        with pytest.raises(h5t.SchemaError, match="invalid Keys pattern"):

            class S(h5t.Group):
                runs: Annotated[h5t.Group[Posterior], h5t.Keys(pattern="(")]

    def test_keys_on_attr_raises(self):
        with pytest.raises(h5t.SchemaError, match="Keys"):

            class S(h5t.Group):
                x: Annotated[str, h5t.Keys(pattern=".*")]

    def test_shape_metadata_is_rejected(self):
        with pytest.raises(h5t.SchemaError, match="no longer supported"):

            class S(h5t.Group):
                x: Annotated[h5t.Dataset[h5t.f8], "n"]

    def test_invalid_subscript_raises(self):
        with pytest.raises(h5t.SchemaError, match="invalid Dataset subscript"):
            h5t.Dataset[h5t.f8, 3]

    def test_shape_subscript_is_rejected(self):
        with pytest.raises(h5t.SchemaError, match="override validate"):
            h5t.Dataset[h5t.f8, "n"]

    def test_removed_shape_and_dims_kwargs_are_rejected(self):
        with pytest.raises(TypeError):

            class Shaped(h5t.Dataset, dtype=h5t.f8, shape="n"):
                pass

        with pytest.raises(TypeError):

            class Dimmed(h5t.Group, dims={"n": None}):
                pass


class TestInheritance:
    def test_partial_schemas_flatten(self):
        class HasFoo(h5t.File):
            foo: h5t.Dataset[h5t.f8]
            calibration: str

        class NeedsBar(HasFoo):
            bar: h5t.Dataset[h5t.f8]

        class NeedsBaz(HasFoo):
            baz: h5t.Dataset[h5t.f8]

        class MyFile(NeedsBar, NeedsBaz):
            pass

        spec = MyFile.__h5spec__
        # Flatten order follows reversed MRO, exactly like get_type_hints:
        # for MyFile(NeedsBar, NeedsBaz) that is HasFoo, NeedsBaz, NeedsBar.
        assert sorted(c.py_name for c in spec.children) == ["bar", "baz", "foo"]
        assert [c.py_name for c in spec.children][0] == "foo"
        assert [a.py_name for a in spec.attrs] == ["calibration"]
        assert issubclass(MyFile, NeedsBar) and issubclass(MyFile, NeedsBaz)

    def test_conflicting_sibling_redeclarations_raise(self):
        class A(h5t.File):
            foo: h5t.Dataset[h5t.f8]

        class B(h5t.File):
            foo: h5t.Dataset[h5t.f4]

        with pytest.raises(h5t.SchemaError, match="conflicting redeclarations"):

            class C(A, B):
                pass

    def test_identical_sibling_redeclarations_are_fine(self):
        class A(h5t.File):
            foo: h5t.Dataset[h5t.f8]

        class B(h5t.File):
            foo: h5t.Dataset[h5t.f8]

        class C(A, B):
            pass

        assert [c.py_name for c in C.__h5spec__.children] == ["foo"]

    def test_subclass_override_wins(self):
        class A(h5t.Group):
            x: h5t.Dataset[h5t.f8]

        class B(A):
            x: h5t.Dataset[h5t.f4]

        assert B.__h5spec__.children[0].dtype is h5t.f4

    def test_extras_is_inherited_unless_overridden(self):
        class A(h5t.Group, extras="forbid"):
            pass

        class B(A):
            pass

        class C(A, extras="warn"):
            pass

        assert B.__h5spec__.extras is Extras.FORBID
        assert C.__h5spec__.extras is Extras.WARN

    def test_dataset_inherits_dtype(self):
        class Base(h5t.Dataset, dtype=h5t.f8):
            unit: str

        class Derived(Base):
            frame: str

        spec = Derived.__h5spec__
        assert spec.dtype is h5t.f8
        assert {a.py_name for a in spec.attrs} == {"unit", "frame"}


class TestValidateSchema:
    def test_good_schema_passes(self):
        PEResult.validate_schema()
        Posterior.validate_schema()

    def test_incomplete_dataset_template_fails(self):
        with pytest.raises(h5t.SchemaError, match="no dtype"):
            h5t.Dataset.validate_schema()

    def test_schema_errors_are_not_validation_errors(self):
        assert not issubclass(h5t.SchemaError, h5t.ValidationError)
        assert not issubclass(h5t.ValidationError, h5t.SchemaError)


class TestSpec:
    def test_optional_members(self):
        spec = Posterior.__h5spec__
        by_name = {c.py_name: c for c in spec.children}
        assert by_name["spins"].optional
        assert not by_name["mass_1"].optional
