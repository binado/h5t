"""Static acceptance surface checked by ty."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from jaxtyping import Float
from pydantic import BaseModel

import h5t


class RecordingAttrs(BaseModel):
    unit: str


class NestedAttrs(BaseModel):
    label: str


class ResultAttrs(BaseModel):
    version: int
    optional: str | None = None
    defaulted: int = 3


class Nested(h5t.Group[NestedAttrs]):
    pass


class UntypedGroup(h5t.Group):
    values: np.ndarray


class Result(h5t.Group[ResultAttrs]):
    values: np.ndarray
    samples: h5t.Dataset
    fisher: Float[np.ndarray, "n n"]
    lazy_fisher: Float[h5t.Dataset, "n _"]
    recording: Float[h5t.Dataset[RecordingAttrs], "n"]
    nested: Nested
    untyped: UntypedGroup
    snaps: dict[str, Nested]
    chains: Annotated[dict[str, h5t.Dataset], h5t.Glob("chain_?")]


result: Result = Result.model_validate_hdf5(Path("result.h5"))
fisher: np.ndarray = result.fisher
lazy_fisher: h5t.Dataset[Mapping[str, Any]] = result.lazy_fisher
version: int = result.attrs.version
optional: str | None = result.attrs.optional
defaulted: int = result.attrs.defaulted
snaps: dict[str, Nested] = result.snaps
values: np.ndarray = result.values
samples: h5t.Dataset[Mapping[str, Any]] = result.samples
samples_attrs: Mapping[str, Any] = result.samples.attrs
data: np.ndarray = result.samples.data
sliced: np.ndarray = result.samples[:10]
shape: tuple[int, ...] = result.samples.shape
nested: Nested = result.nested
label: str = result.nested.attrs.label
untyped_attrs: Mapping[str, Any] = result.untyped.attrs
recording: h5t.Dataset[RecordingAttrs] = result.recording
unit: str = result.recording.attrs.unit
full: np.ndarray = result.recording[()]

saved: None = result.model_dump_hdf5(Path("out.h5"), root="/run", mode="a", overwrite=True)


# Negative cases: each line must stay a type error, so the ignores below fail
# ``ty check`` (unused-ignore) if the public typing surface loosens.
bad_unit: int = result.recording.attrs.unit  # ty: ignore[invalid-assignment]
bad_version: str = result.attrs.version  # ty: ignore[invalid-assignment]
bad_untyped: h5t.Dataset[RecordingAttrs] = result.samples  # ty: ignore[invalid-assignment]
bad_mode: None = result.model_dump_hdf5(Path("out.h5"), mode="r")  # ty: ignore[invalid-argument-type]
