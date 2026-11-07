"""jaxtyping-annotated arrays and datasets: dtype and shape validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from jaxtyping import AbstractArray, AbstractDtype, Float, Float64, Num, Shaped
from pydantic import BaseModel, ConfigDict, PydanticSchemaGenerationError
from pydantic import ValidationError as PydanticValidationError

import h5t


def write(path: Path, **datasets: Any) -> Path:
    with h5py.File(path, "w") as f:
        for name, value in datasets.items():
            f[name] = value
    return path


def test_guard_jaxtyping_exposes_what_h5t_relies_on() -> None:
    cls = Float[np.ndarray, "n"]
    assert issubclass(cls, AbstractArray)
    assert (cls.dim_str, cls.array_type, cls.dtype) == ("n", np.ndarray, Float)
    assert callable(cls.__instancecheck_str__)
    assert cls.__instancecheck_str__(np.zeros(3)) == ""


def test_jaxtyping_field_loads_an_array(tmp_path: Path) -> None:
    class R(h5t.Group):
        x: Float64[np.ndarray, "*batch n 3 _"]

    value = np.zeros((2, 3, 4))
    loaded = R.model_validate_hdf5(write(tmp_path / "array.h5", x=value))
    np.testing.assert_array_equal(loaded.x, value)


@pytest.mark.parametrize("value", [np.zeros(3), np.zeros((2, 2), dtype="int8")])
def test_plain_ndarray_field_accepts_any_shape_and_dtype(tmp_path: Path, value: Any) -> None:
    class R(h5t.Group):
        x: np.ndarray

    loaded = R.model_validate_hdf5(write(tmp_path / "plain.h5", x=value))
    assert loaded.x.shape == value.shape
    assert loaded.x.dtype == value.dtype


def test_float_accepts_any_width_float64_is_exact(tmp_path: Path) -> None:
    class Wide(h5t.Group):
        x: Float[np.ndarray, "n"]

    class Exact(h5t.Group):
        x: Float64[np.ndarray, "n"]

    for dtype in ("float32", "float64"):
        path = write(tmp_path / f"{dtype}.h5", x=np.zeros(3, dtype))
        assert Wide.model_validate_hdf5(path).x.dtype == dtype
    path = write(tmp_path / "f32.h5", x=np.zeros(3, "float32"))
    with pytest.raises(h5t.ValidationError, match=r"/x: this array has dtype float32, not float64"):
        Exact.model_validate_hdf5(path)


def test_dtype_mismatch(tmp_path: Path) -> None:
    class R(h5t.Group):
        x: Float[np.ndarray, "n"]

    path = write(tmp_path / "i.h5", x=np.arange(3))
    with pytest.raises(h5t.ValidationError, match=r"/x: this array has dtype int64, not any of"):
        R.model_validate_hdf5(path)


def test_big_endian_file_dtype_is_accepted(tmp_path: Path) -> None:
    class R(h5t.Group):
        x: Float64[np.ndarray, "n"]

    path = write(tmp_path / "be.h5", x=np.arange(3, dtype=">f8"))
    assert R.model_validate_hdf5(path).x.shape == (3,)


def test_shaped_accepts_any_dtype(tmp_path: Path) -> None:
    class R(h5t.Group):
        x: Shaped[np.ndarray, "n"]

    assert R.model_validate_hdf5(write(tmp_path / "a.h5", x=np.arange(3))).x.shape == (3,)


def test_rank_mismatch(tmp_path: Path) -> None:
    class R(h5t.Group):
        x: Num[np.ndarray, "n n"]

    path = write(tmp_path / "a.h5", x=np.zeros(3))
    with pytest.raises(
        h5t.ValidationError, match=r"/x: this array has 1 dimensions, not the 2 expected"
    ):
        R.model_validate_hdf5(path)


def test_fixed_size_mismatch(tmp_path: Path) -> None:
    class R(h5t.Group):
        x: Num[np.ndarray, "n 3"]

    path = write(tmp_path / "a.h5", x=np.zeros((2, 4)))
    with pytest.raises(h5t.ValidationError, match=r"/x: the dimension size 4 does not equal 3"):
        R.model_validate_hdf5(path)


class _Pair(h5t.Group):
    a: Float[np.ndarray, "n n"]
    b: Float[np.ndarray, "n"]


def test_named_dim_unifies_across_fields(tmp_path: Path) -> None:
    class Pair(_Pair):
        pass

    good = write(tmp_path / "g.h5", a=np.zeros((3, 3)), b=np.zeros(3))
    assert Pair.model_validate_hdf5(good).b.shape == (3,)
    bad = write(tmp_path / "b.h5", a=np.zeros((3, 3)), b=np.zeros(4))
    with pytest.raises(
        h5t.ValidationError,
        match=r"/b: the size of dimension n is 4 which does not equal the existing value of 3",
    ):
        Pair.model_validate_hdf5(bad)

    # In-memory construction also enforces shared symbolic dimensions!
    assert Pair(a=np.zeros((3, 3)), b=np.zeros(3)).b.shape == (3,)
    with pytest.raises(PydanticValidationError, match="dimension n"):
        Pair(a=np.zeros((3, 3)), b=np.zeros(4))


def test_named_dim_must_agree_within_one_array(tmp_path: Path) -> None:
    class R(h5t.Group):
        x: Float[np.ndarray, "n n"]

    path = write(tmp_path / "a.h5", x=np.zeros((2, 3)))
    with pytest.raises(
        h5t.ValidationError,
        match=r"the size of dimension n is 3 which does not equal the existing value of 2",
    ):
        R.model_validate_hdf5(path)


def test_variadic_forms(tmp_path: Path) -> None:
    class R(h5t.Group):
        a: Float[np.ndarray, "*batch n"]
        b: Float[np.ndarray, "*batch"]
        c: Float[np.ndarray, "... 2"]
        d: Float[np.ndarray, "*_ _"]

    path = write(
        tmp_path / "v.h5",
        a=np.zeros((5, 6, 3)),
        b=np.zeros((5, 6)),
        c=np.zeros(2),
        d=np.zeros(4),
    )
    assert R.model_validate_hdf5(path).a.shape == (5, 6, 3)

    bad = write(
        tmp_path / "bad.h5",
        a=np.zeros((5, 6, 3)),
        b=np.zeros((5, 7)),
        c=np.zeros(2),
        d=np.zeros(4),
    )
    with pytest.raises(
        h5t.ValidationError,
        match=r"/b: the shape of its variadic dimensions '\*batch' is \(5, 7\)",
    ):
        R.model_validate_hdf5(bad)


def test_variadic_rank_too_small(tmp_path: Path) -> None:
    class R(h5t.Group):
        x: Float[np.ndarray, "*b n 3"]

    with pytest.raises(
        h5t.ValidationError, match=r"this array has 1 dimensions, which is fewer than 2"
    ):
        R.model_validate_hdf5(write(tmp_path / "a.h5", x=np.zeros(3)))


def test_scalar(tmp_path: Path) -> None:
    class R(h5t.Group):
        x: Float[np.ndarray, ""]

    assert R.model_validate_hdf5(write(tmp_path / "s.h5", x=np.float64(2.0))).x.shape == ()
    with pytest.raises(
        h5t.ValidationError, match=r"this array has 1 dimensions, not the 0 expected"
    ):
        R.model_validate_hdf5(write(tmp_path / "v.h5", x=np.zeros(1)))


def test_nested_record_has_independent_bindings(tmp_path: Path) -> None:
    class Inner(h5t.Group):
        y: Float[np.ndarray, "n"]

    class Outer(h5t.Group):
        x: Float[np.ndarray, "n"]
        inner: Inner

    path = tmp_path / "n.h5"
    with h5py.File(path, "w") as f:
        f["x"] = np.zeros(3)
        f.create_group("inner")["y"] = np.zeros(7)
    assert Outer.model_validate_hdf5(path).inner.y.shape == (7,)


def test_dataset_generic_with_jaxtyping(tmp_path: Path) -> None:
    class SeriesAttrs(BaseModel):
        unit: str = "s"

    class R(h5t.Group):
        series: Float[h5t.Dataset[SeriesAttrs], "n 2"]
        other: Float[np.ndarray, "n"]

    path = write(tmp_path / "d.h5", series=np.zeros((4, 2)), other=np.zeros(4))
    loaded = R.model_validate_hdf5(path)
    assert loaded.series.shape == (4, 2)
    assert loaded.series.attrs.unit == "s"

    with pytest.raises(
        h5t.ValidationError, match=r"/series: the dimension size 3 does not equal 2"
    ):
        R.model_validate_hdf5(
            write(tmp_path / "bad.h5", series=np.zeros((4, 3)), other=np.zeros(4))
        )


def test_optional_field_missing(tmp_path: Path) -> None:
    class R(h5t.Group):
        x: Float[np.ndarray, "n"] | None = None
        y: Float[h5t.Dataset, "n"] | None = None

    result = R.model_validate_hdf5(write(tmp_path / "e.h5", other=np.zeros(1)))
    assert result.x is None and result.y is None


def test_dataset_is_checked_without_reading_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class R(h5t.Group):
        samples: Float[h5t.Dataset, "n 2"]

    good = write(tmp_path / "g.h5", samples=np.zeros((5, 2)))
    bad = write(tmp_path / "b.h5", samples=np.zeros((5, 3)))

    def forbidden(self: h5py.Dataset, *args: Any) -> Any:
        raise AssertionError("dataset data was read")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", forbidden)
    loaded = R.model_validate_hdf5(good)
    assert loaded.samples.shape == (5, 2)
    assert type(loaded.samples) is h5t.Dataset
    with pytest.raises(
        h5t.ValidationError, match=r"/samples: the dimension size 3 does not equal 2"
    ):
        R.model_validate_hdf5(bad)


def test_jaxtyping_grammar_beyond_the_basics_comes_for_free(tmp_path: Path) -> None:
    class R(h5t.Group):
        a: Float[np.ndarray, "#n"]
        b: Float[np.ndarray, "n m"]
        c: Float[np.ndarray, "m+1"]

    ok = write(tmp_path / "ok.h5", a=np.zeros(1), b=np.zeros((4, 2)), c=np.zeros(3))
    assert R.model_validate_hdf5(ok).c.shape == (3,)
    bad = write(tmp_path / "bad.h5", a=np.zeros(1), b=np.zeros((4, 2)), c=np.zeros(4))
    with pytest.raises(h5t.ValidationError, match=r"/c: "):
        R.model_validate_hdf5(bad)


def test_custom_jaxtyping_dtype_family(tmp_path: Path) -> None:
    class Single(AbstractDtype):
        dtypes = ["float32", "float64"]

    class R(h5t.Group):
        x: Single[np.ndarray, "n"]
        y: Single[h5t.Dataset, "n"]

    path = write(tmp_path / "ok.h5", x=np.zeros(3, "float32"), y=np.zeros(3))
    assert R.model_validate_hdf5(path).x.dtype == "float32"
    with pytest.raises(h5t.ValidationError, match=r"/x: this array has dtype int8"):
        R.model_validate_hdf5(write(tmp_path / "bad.h5", x=np.zeros(3, "int8"), y=np.zeros(3)))


def test_invalid_shape_string_error_propagates() -> None:
    with pytest.raises(ValueError, match="comma"):

        class R(h5t.Group):
            x: Float[np.ndarray, "n, m"]


def test_unsupported_array_type_is_a_schema_error(tmp_path: Path) -> None:
    # The class itself builds (the annotation only concerns h5t); the declaration check
    # fires at operation time, from _unwrap_jaxtyping via _check_group.
    class R(h5t.Group):
        x: Float[list, "n"]  # ty: ignore

    with pytest.raises(h5t.SchemaError, match=r"not supported"):
        R.model_validate_hdf5(write(tmp_path / "a.h5", x=np.zeros(3)))
    with pytest.raises(h5t.SchemaError, match=r"not supported"):
        R.model_construct(x=[1, 2]).model_dump_hdf5(tmp_path / "b.h5")


def test_foreign_jaxtyping_models_do_not_break_when_h5t_is_imported() -> None:
    # h5t does not mutate jaxtyping's base class, so foreign models keep their native
    # Pydantic schema behavior.
    assert not hasattr(AbstractArray, "__get_pydantic_core_schema__")

    class DuckArray:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape
            self.dtype = np.dtype("float32")

    class Foreign(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        x: Float[DuckArray, "n"]

    assert Foreign(x=DuckArray((3,))).x.shape == (3,)
    with pytest.raises(PydanticValidationError):
        Foreign(x=object())
    with pytest.raises(PydanticSchemaGenerationError):
        # Without arbitrary_types_allowed the error is Pydantic's own, as without h5t.

        class Strict(BaseModel):
            x: Float[DuckArray, "n"]


def test_ndarray_jaxtyping_rejects_mismatch_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class R(h5t.Group):
        x: Float[np.ndarray, "n n"]

    path = write(tmp_path / "a.h5", x=np.zeros((2, 3)))
    reads: list[object] = []
    real_getitem = h5py.Dataset.__getitem__

    def spy(self: h5py.Dataset, *args: Any, **kwargs: Any) -> Any:
        reads.append(args)
        return real_getitem(self, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", spy)
    with pytest.raises(h5t.ValidationError) as info:
        R.model_validate_hdf5(path)
    assert info.value.path == "/x"
    assert reads == []
