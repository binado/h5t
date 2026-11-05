"""jaxtyping-annotated arrays: dtype and shape validation against HDF5 metadata."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Annotated, Any

import h5py
import numpy as np
import pytest
from jaxtyping import AbstractArray, AbstractDtype, Float, Float64, Num, Shaped

import h5t
from h5t._spec import ArraySpec, MemberKind


def write(path: Path, **datasets: Any) -> Path:
    with h5py.File(path, "w") as f:
        for name, value in datasets.items():
            f[name] = value
    return path


def spec_of(record: type, name: str = "x") -> Any:
    return next(f for f in record.__dict__["__h5t_record__"].spec.fields if f.py_name == name)


def test_guard_jaxtyping_exposes_what_h5t_relies_on() -> None:
    cls = Float[np.ndarray, "n"]
    assert issubclass(cls, AbstractArray)
    assert (cls.dim_str, cls.array_type, cls.dtype) == ("n", np.ndarray, Float)
    assert callable(cls.__instancecheck_str__)
    assert cls.__instancecheck_str__(np.zeros(3)) == ""  # "" means the value conforms


def test_compiles_to_array_spec() -> None:
    @h5t.group()
    @dataclasses.dataclass
    class R:
        x: Float64[np.ndarray, "*batch n 3 _"]
        y: Shaped[np.ndarray, ""]

    x = spec_of(R)
    assert x.kind is MemberKind.ARRAY
    assert x.array == ArraySpec(Float64[np.ndarray, "*batch n 3 _"], Float64, "*batch n 3 _")
    assert x.array != ArraySpec(Float64[np.ndarray, "n"], Float64, "n")
    assert spec_of(R, "y").array.dim_str == ""


def test_plain_ndarray_stays_unchecked() -> None:
    @h5t.group()
    @dataclasses.dataclass
    class R:
        x: np.ndarray

    assert spec_of(R).array is None


def test_float_accepts_any_width_float64_is_exact(tmp_path: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Wide:
        x: Float[np.ndarray, "n"]

    @h5t.group()
    @dataclasses.dataclass
    class Exact:
        x: Float64[np.ndarray, "n"]

    for dtype in ("float32", "float64"):
        path = write(tmp_path / f"{dtype}.h5", x=np.zeros(3, dtype))
        assert h5t.load(Wide, path).x.dtype == dtype
    path = write(tmp_path / "f32.h5", x=np.zeros(3, "float32"))
    with pytest.raises(h5t.ValidationError, match=r"/x: this array has dtype float32, not float64"):
        h5t.load(Exact, path)


def test_dtype_mismatch(tmp_path: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class R:
        x: Float[np.ndarray, "n"]

    path = write(tmp_path / "i.h5", x=np.arange(3))
    with pytest.raises(h5t.ValidationError, match=r"/x: this array has dtype int64, not any of"):
        h5t.load(R, path)


def test_big_endian_file_dtype_is_accepted(tmp_path: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class R:
        x: Float64[np.ndarray, "n"]

    path = write(tmp_path / "be.h5", x=np.arange(3, dtype=">f8"))
    assert h5t.load(R, path).x.shape == (3,)


def test_shaped_accepts_any_dtype(tmp_path: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class R:
        x: Shaped[np.ndarray, "n"]

    assert h5t.load(R, write(tmp_path / "a.h5", x=np.arange(3))).x.shape == (3,)


def test_rank_mismatch(tmp_path: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class R:
        x: Num[np.ndarray, "n n"]

    path = write(tmp_path / "a.h5", x=np.zeros(3))
    with pytest.raises(
        h5t.ValidationError, match=r"/x: this array has 1 dimensions, not the 2 expected"
    ):
        h5t.load(R, path)


def test_fixed_size_mismatch(tmp_path: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class R:
        x: Num[np.ndarray, "n 3"]

    path = write(tmp_path / "a.h5", x=np.zeros((2, 4)))
    with pytest.raises(h5t.ValidationError, match=r"/x: the dimension size 4 does not equal 3"):
        h5t.load(R, path)


@dataclasses.dataclass
class _Pair:
    a: Float[np.ndarray, "n n"]
    b: Float[np.ndarray, "n"]


def test_named_dim_unifies_across_fields(tmp_path: Path) -> None:
    Pair = h5t.group()(dataclasses.make_dataclass("Pair", [], bases=(_Pair,)))
    good = write(tmp_path / "g.h5", a=np.zeros((3, 3)), b=np.zeros(3))
    assert h5t.load(Pair, good).b.shape == (3,)
    bad = write(tmp_path / "b.h5", a=np.zeros((3, 3)), b=np.zeros(4))
    with pytest.raises(
        h5t.ValidationError,
        match=r"/b: the size of dimension n is 4 which does not equal the existing value of 3",
    ):
        h5t.load(Pair, bad)


def test_named_dim_must_agree_within_one_array(tmp_path: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class R:
        x: Float[np.ndarray, "n n"]

    path = write(tmp_path / "a.h5", x=np.zeros((2, 3)))
    with pytest.raises(
        h5t.ValidationError,
        match=r"the size of dimension n is 3 which does not equal the existing value of 2",
    ):
        h5t.load(R, path)


def test_variadic_forms(tmp_path: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class R:
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
    assert h5t.load(R, path).a.shape == (5, 6, 3)

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
        h5t.load(R, bad)


def test_variadic_rank_too_small(tmp_path: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class R:
        x: Float[np.ndarray, "*b n 3"]

    with pytest.raises(
        h5t.ValidationError, match=r"this array has 1 dimensions, which is fewer than 2"
    ):
        h5t.load(R, write(tmp_path / "a.h5", x=np.zeros(3)))


def test_scalar(tmp_path: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class R:
        x: Float[np.ndarray, ""]

    assert h5t.load(R, write(tmp_path / "s.h5", x=np.float64(2.0))).x.shape == ()
    with pytest.raises(
        h5t.ValidationError, match=r"this array has 1 dimensions, not the 0 expected"
    ):
        h5t.load(R, write(tmp_path / "v.h5", x=np.zeros(1)))


def test_nested_record_has_independent_bindings(tmp_path: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class Inner:
        y: Float[np.ndarray, "n"]

    @h5t.group()
    @dataclasses.dataclass
    class Outer:
        x: Float[np.ndarray, "n"]
        inner: Inner

    path = tmp_path / "n.h5"
    with h5py.File(path, "w") as f:
        f["x"] = np.zeros(3)
        f.create_group("inner")["y"] = np.zeros(7)
    assert h5t.load(Outer, path).inner.y.shape == (7,)


def test_dataset_record_payload(tmp_path: Path) -> None:
    @h5t.dataset(data="values")
    @dataclasses.dataclass
    class Series:
        values: Float[np.ndarray, "n 2"]

    @h5t.group()
    @dataclasses.dataclass
    class R:
        series: Series
        other: Float[np.ndarray, "n"]

    path = write(tmp_path / "d.h5", series=np.zeros((4, 2)), other=np.zeros(9))
    assert h5t.load(R, path).series.values.shape == (4, 2)  # `n` is the payload's own scope

    with pytest.raises(
        h5t.ValidationError, match=r"/series: the dimension size 3 does not equal 2"
    ):
        h5t.load(R, write(tmp_path / "bad.h5", series=np.zeros((4, 3)), other=np.zeros(9)))


def test_dataset_record_lazy_payload(tmp_path: Path) -> None:
    @h5t.dataset(data="values")
    @dataclasses.dataclass
    class Series:
        values: Float[h5t.LazyArray, "n"]

    @h5t.group()
    @dataclasses.dataclass
    class R:
        series: Series

    path = write(tmp_path / "d.h5", series=np.zeros(4))
    assert h5t.load(R, path).series.values.shape == (4,)
    with pytest.raises(h5t.ValidationError, match=r"/series: this array has dtype int64"):
        h5t.load(R, write(tmp_path / "bad.h5", series=np.arange(4)))


def test_optional_field_missing(tmp_path: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class R:
        x: Float[np.ndarray, "n"] | None
        y: Float[h5t.LazyArray, "n"] | None = None

    result = h5t.load(R, write(tmp_path / "e.h5", other=np.zeros(1)))
    assert result.x is None and result.y is None


def test_lazy_array_is_checked_without_reading_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class R:
        samples: Float[h5t.LazyArray, "n 2"]

    good = write(tmp_path / "g.h5", samples=np.zeros((5, 2)))
    bad = write(tmp_path / "b.h5", samples=np.zeros((5, 3)))

    def forbidden(self: h5py.Dataset, *args: Any) -> Any:
        raise AssertionError("dataset data was read")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", forbidden)
    loaded = h5t.load(R, good)
    assert loaded.samples._h5t_data is not None
    assert loaded.samples.shape == (5, 2)
    assert type(loaded.samples) is h5t.LazyArray
    with pytest.raises(
        h5t.ValidationError, match=r"/samples: the dimension size 3 does not equal 2"
    ):
        h5t.load(R, bad)


def test_lazy_array_eager_marker_is_allowed(tmp_path: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class R:
        samples: Annotated[Float[h5t.LazyArray, "n"], h5t.Eager()]

    loaded = h5t.load(R, write(tmp_path / "g.h5", samples=np.arange(3.0)))
    assert loaded.samples.data.tolist() == [0.0, 1.0, 2.0]


def test_jaxtyping_grammar_beyond_the_basics_comes_for_free(tmp_path: Path) -> None:
    @h5t.group()
    @dataclasses.dataclass
    class R:
        a: Float[np.ndarray, "#n"]  # broadcastable: a size-1 axis matches anything
        b: Float[np.ndarray, "n m"]
        c: Float[np.ndarray, "m+1"]  # arithmetic over a bound name

    ok = write(tmp_path / "ok.h5", a=np.zeros(1), b=np.zeros((4, 2)), c=np.zeros(3))
    assert h5t.load(R, ok).c.shape == (3,)
    bad = write(tmp_path / "bad.h5", a=np.zeros(1), b=np.zeros((4, 2)), c=np.zeros(4))
    with pytest.raises(h5t.ValidationError, match=r"/c: "):
        h5t.load(R, bad)


def test_custom_jaxtyping_dtype_family(tmp_path: Path) -> None:
    class Single(AbstractDtype):
        dtypes = ["float32", "float64"]

    @h5t.group()
    @dataclasses.dataclass
    class R:
        x: Single[np.ndarray, "n"]
        y: Single[h5t.LazyArray, "n"]

    path = write(tmp_path / "ok.h5", x=np.zeros(3, "float32"), y=np.zeros(3))
    assert h5t.load(R, path).x.dtype == "float32"
    with pytest.raises(h5t.ValidationError, match=r"/x: this array has dtype int8"):
        h5t.load(R, write(tmp_path / "bad.h5", x=np.zeros(3, "int8"), y=np.zeros(3)))


def test_invalid_shape_string_is_a_schema_error() -> None:
    # jaxtyping rejects it when the annotation is evaluated; h5t reports it as a schema error.
    with pytest.raises(h5t.SchemaError, match=r"R: could not resolve annotations"):

        @h5t.group()
        @dataclasses.dataclass
        class R:
            x: Float[np.ndarray, "n, m"]


def test_unsupported_array_type_is_a_schema_error() -> None:
    with pytest.raises(h5t.SchemaError, match=r"R\.x: .*not supported"):

        @h5t.group()
        @dataclasses.dataclass
        class R:
            x: Float[list, "n"]  # ty: ignore


def test_attr_cannot_annotate_a_jaxtyping_array() -> None:
    with pytest.raises(h5t.SchemaError, match=r"Attr cannot annotate a jaxtyping"):

        @h5t.group()
        @dataclasses.dataclass
        class R:
            x: Annotated[Float[np.ndarray, "n"], h5t.Attr()]


def test_dataset_record_still_declares_only_attributes() -> None:
    with pytest.raises(h5t.SchemaError, match=r"a dataset record may declare only attributes"):

        @h5t.dataset(data="values")
        @dataclasses.dataclass
        class R:
            values: Float[np.ndarray, "n"]
            extra: Float[np.ndarray, "n"]
