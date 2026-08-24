"""The frozen public annotation forms, checked by mypy and pyright.

This module is never executed; test_typing_acceptance.py runs both type
checkers over it and asserts zero errors. It exercises every form the
typing spike froze (see typing_spike/FINDINGS.md): the Annotated dataset
spelling, Group[T], optional members, use-site rebinding of a named
dataset type, class kwargs, and inherited partial schemas.
"""

from __future__ import annotations

from typing import Annotated, Literal, assert_type

import numpy as np
import numpy.typing as npt

import h5t


class Mass(h5t.Dataset[h5t.f8], dtype=h5t.f8, shape="n"):
    unit: Literal["Msun"]
    frame: Literal["source", "detector"]


class Posterior(h5t.Group, dims={"n_samples": h5t.FromAttr("n_samples")}):
    mass_1: Annotated[h5t.Dataset[h5t.f8], h5t.Shape("n_samples")]
    mass_2: Annotated[Mass, h5t.Shape("n_samples")]
    log_likelihood: Annotated[h5t.Dataset[h5t.f8], "n_samples"]
    spins: Annotated[h5t.Dataset[h5t.f8], h5t.Shape("n_samples 3")] | None
    psd: Annotated[h5t.Dataset[h5t.f8], h5t.Shape("n_freq 2")]

    approximant: str
    f_ref: float = 20.0

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
    foo: Annotated[h5t.Dataset[h5t.f8], h5t.Shape("n_samples")]
    calibration: str


class NeedsBar(HasFoo):
    bar: Annotated[h5t.Dataset[h5t.f8], h5t.Shape("n_samples")]


class NeedsBaz(HasFoo):
    baz: Annotated[h5t.Dataset[h5t.f8], h5t.Shape("n_samples")]


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
