"""Property-based coverage of the validation matrix, driven by generated specs.

`mock` is deferred past v0.1, so the strategies build files directly with
h5py from the same structure description the schema class is generated
from: a conforming file must validate clean, and truncating one dataset on
a shared dimension must produce the specific dim-conflict error.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

import h5t
from h5t._dtypes import DType
from h5t._validate import run_validation

DTYPE_TOKENS: list[type[DType]] = [h5t.f4, h5t.f8, h5t.i2, h5t.i4, h5t.i8, h5t.u1, h5t.c16]
DIM_NAMES = ["na", "nb", "nc"]
ANON_FILL = 2


@dataclass(frozen=True)
class DatasetDef:
    name: str
    dtype: type[DType]
    dims: tuple[str, ...]


@dataclass(frozen=True)
class AttrDef:
    name: str
    type: type
    value: object


@dataclass(frozen=True)
class Structure:
    datasets: tuple[DatasetDef, ...]
    attrs: tuple[AttrDef, ...]
    dim_values: dict[str, int]


@st.composite
def structures(draw: st.DrawFn, *, shared_dim: bool = False) -> Structure:
    n_ds = draw(st.integers(2, 5))
    datasets = []
    for i in range(n_ds):
        dtype = draw(st.sampled_from(DTYPE_TOKENS))
        ndim = draw(st.integers(0, 3))
        dims = tuple(draw(st.sampled_from(DIM_NAMES + ["_", "4"])) for _ in range(ndim))
        if shared_dim and i < 2:
            dims = ("na", *dims)
        datasets.append(DatasetDef(f"d{i}", dtype, dims))
    attrs = []
    for i in range(draw(st.integers(0, 3))):
        kind = draw(st.sampled_from(["str", "int", "float", "bool"]))
        value: object
        if kind == "str":
            value = draw(st.text(alphabet=string.ascii_letters, max_size=8))
        elif kind == "int":
            value = draw(st.integers(-1000, 1000))
        elif kind == "float":
            value = draw(st.floats(allow_nan=False, allow_infinity=False, width=32))
        else:
            value = draw(st.booleans())
        attrs.append(AttrDef(f"a{i}", type(value), value))
    dim_values = {name: draw(st.integers(1, 6)) for name in DIM_NAMES}
    return Structure(tuple(datasets), tuple(attrs), dim_values)


def build_schema(structure: Structure) -> type[h5t.File]:
    annotations: dict[str, object] = {}
    for ds in structure.datasets:
        annotations[ds.name] = h5t.Dataset[ds.dtype, " ".join(ds.dims)]
    for attr in structure.attrs:
        annotations[attr.name] = attr.type
    return type(
        "GeneratedSchema",
        (h5t.File,),
        {"__annotations__": annotations, "__module__": __name__},
    )


def _axis_length(dim: str, dim_values: dict[str, int]) -> int:
    if dim == "_":
        return ANON_FILL
    if dim.isdigit():
        return int(dim)
    return dim_values[dim]


def build_file(path: Path, structure: Structure) -> None:
    with h5py.File(path, "w") as f:
        for ds in structure.datasets:
            shape = tuple(_axis_length(d, structure.dim_values) for d in ds.dims)
            np_dtype = np.dtype(f"{ds.dtype.kind}{ds.dtype.itemsize}")
            f.create_dataset(ds.name, shape=shape, dtype=np_dtype)
        for attr in structure.attrs:
            f.attrs[attr.name] = attr.value


def check(schema: type[h5t.File], path: Path) -> h5t.ValidationReport:
    with h5py.File(path, "r") as f:
        return run_validation(schema.__h5spec__, f, filename=str(path), schema_name=schema.__name__)


@settings(max_examples=30, deadline=None)
@given(structure=structures())
def test_conforming_files_validate_clean(structure: Structure, tmp_path_factory):
    path = tmp_path_factory.mktemp("hyp") / "t.h5"
    schema = build_schema(structure)
    schema.validate_schema()
    build_file(path, structure)
    report = check(schema, path)
    assert report.ok, report.render()


@settings(max_examples=30, deadline=None)
@given(structure=structures(shared_dim=True))
def test_truncating_a_shared_dim_is_detected(structure: Structure, tmp_path_factory):
    path = tmp_path_factory.mktemp("hyp") / "t.h5"
    schema = build_schema(structure)
    build_file(path, structure)
    victim = structure.datasets[0]
    with h5py.File(path, "a") as f:
        old = f[victim.name]
        assert isinstance(old, h5py.Dataset)
        shape = list(old.shape)
        shape[0] += 1  # 'na' axis now disagrees with its sibling
        dtype = old.dtype
        del f[victim.name]
        f.create_dataset(victim.name, shape=tuple(shape), dtype=dtype)
    report = check(schema, path)
    assert not report.ok
    assert any("'na' has inconsistent values" in p.message for p in report.errors), report.render()
    # Both conflicting sites are named: order never implies which is correct.
    (problem,) = [p for p in report.errors if "'na'" in p.message]
    assert f"./{structure.datasets[0].name}" in problem.message
    assert f"./{structure.datasets[1].name}" in problem.message
