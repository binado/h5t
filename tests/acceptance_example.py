"""Static acceptance surface checked by ty."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import numpy as np

import h5t


class Samples(h5t.Dataset):
    unit: str


class Nested(h5t.Group):
    label: str


@dataclasses.dataclass
class Recording:
    unit: str
    payload: np.ndarray
    attrs: Mapping[str, Any]


class Result(h5t.Group):
    version: int
    values: np.ndarray
    samples: Samples
    eager: Annotated[Samples, h5t.Eager()]
    nested: Nested
    optional: str | None
    defaulted: int = 3
    recording: Annotated[Recording, h5t.Payload("payload", attrs="attrs")]


result: Result = Result.from_file(Path("result.h5"))
version: int = result.version
values: np.ndarray = result.values
data: np.ndarray = result.samples.data
unit: str = result.samples.unit
nested: Nested = result.nested
optional: str | None = result.optional
defaulted: int = result.defaulted
recording: Recording = result.recording
recording_payload: np.ndarray = result.recording.payload
attrs: Mapping[str, Any] = result.recording.attrs
