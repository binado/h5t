"""The frozen public annotation forms, checked by mypy and pyright.

This module is never executed; test_typing_acceptance.py runs both type
checkers over it and asserts zero errors. It exercises every form the
typing spike froze: dtype-only datasets, Group[T], optional members,
custom validators, class kwargs, and inherited partial schemas.
"""

from __future__ import annotations

from typing import Annotated, Literal, assert_type

import numpy as np
import numpy.typing as npt

import h5t


class Mass(h5t.Dataset[h5t.f8], dtype=h5t.f8):
    unit: Literal["Msun"]
    frame: Literal["source", "detector"]


class Posterior(h5t.Group):
    mass_1: h5t.Dataset[h5t.f8]
    mass_2: Mass
    log_likelihood: h5t.Dataset[h5t.f8]
    spins: h5t.Dataset[h5t.f8] | None
    psd: h5t.Dataset[h5t.f8]

    n_samples: int
    approximant: str
    f_ref: float = 20.0

    def validate(self) -> None:
        if self.mass_1.shape != (self.n_samples,):
            raise h5t.Invalid("mass_1 must match n_samples")
        if self.mass_2.shape != (self.n_samples,):
            raise h5t.Invalid("mass_2 must match n_samples")

    @property
    def chirp_mass(self) -> npt.NDArray[np.float64]:
        m1, m2 = self.mass_1[:], self.mass_2[:]
        result: npt.NDArray[np.float64] = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
        return result


class PEResult(h5t.File, extras="ignore"):
    runs: Annotated[
        h5t.Group[Posterior],
        h5t.Keys(pattern=r"C\d+:.*"),
    ]
    format_version: Annotated[
        Literal["1.0"],
        h5t.Name("version"),
    ]


def access(path: str) -> None:
    with PEResult.open(path) as f:
        assert_type(f, PEResult)
        assert_type(f.runs, h5t.Group[Posterior])
        run = f.runs["C01:IMRPhenomXPHM"]
        assert_type(run, Posterior)
        assert_type(run.mass_1, h5t.Dataset[h5t.f8])
        assert_type(run.mass_2, Mass)
        assert_type(run.spins, h5t.Dataset[h5t.f8] | None)
        assert_type(run.approximant, str)
        assert_type(run.f_ref, float)
        assert_type(f.format_version, Literal["1.0"])
        assert_type(run.chirp_mass, npt.NDArray[np.float64])


# Inherited partial schemas: the subtyping relation comes free.


class HasFoo(h5t.File):
    foo: h5t.Dataset[h5t.f8]
    calibration: str


class NeedsBar(HasFoo):
    bar: h5t.Dataset[h5t.f8]


class NeedsBaz(HasFoo):
    baz: h5t.Dataset[h5t.f8]


class MyFile(NeedsBar, NeedsBaz):
    """The canonical format."""


def step_one(f: NeedsBar) -> None:
    assert_type(f.foo, h5t.Dataset[h5t.f8])
    assert_type(f.bar, h5t.Dataset[h5t.f8])


def compose(path: str) -> None:
    f = MyFile.open(path)
    assert_type(f, MyFile)  # open() returns Self
    step_one(f)  # MyFile IS-A NeedsBar
    f.close()


# The mapping API is deliberately untyped: bracket access on a statically
# declared group is the escape hatch for generic code.


def escape_hatch(f: PEResult) -> None:
    _child = f["runs"]
    _attr = f.attrs["version"]
    for _key in f.keys():
        pass
