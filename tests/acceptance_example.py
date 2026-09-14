"""Static acceptance surface checked by ty."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import numpy as np

import h5t


@dataclasses.dataclass
class Recording:
    unit: str
    payload: np.ndarray
    attrs: Mapping[str, Any]


@h5t.dataset(data="payload", attrs="attrs")
@dataclasses.dataclass
class DecoratedRecording:
    unit: str
    payload: np.ndarray
    attrs: Mapping[str, Any]


@h5t.group()
@dataclasses.dataclass
class Nested:
    label: str


@h5t.group()
@dataclasses.dataclass
class Result:
    version: int
    values: np.ndarray
    samples: h5t.LazyArray
    eager: Annotated[h5t.LazyArray, h5t.Eager()]
    nested: Nested
    recording: DecoratedRecording
    marked: Annotated[Recording, h5t.Payload("payload", attrs="attrs")]
    optional: str | None
    defaulted: int = 3


result: Result = h5t.load(Result, Path("result.h5"))
version: int = result.version
values: np.ndarray = result.values
samples: h5t.LazyArray = result.samples
data: np.ndarray = result.samples.data
shape: tuple[int, ...] = result.eager.shape
nested: Nested = result.nested
label: str = result.nested.label
optional: str | None = result.optional
defaulted: int = result.defaulted
recording: DecoratedRecording = result.recording
unit: str = result.recording.unit
payload: np.ndarray = result.recording.payload
attrs: Mapping[str, Any] = result.recording.attrs
marked_payload: np.ndarray = result.marked.payload
