"""Shared example schemas and file builders for the h5t test suite."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal

import h5py
import numpy as np
import pytest

import h5t


class Posterior(h5t.Group, dims={"n_samples": h5t.FromAttr("n_samples")}):
    """Posterior samples from one PE run (the PLAN.md example)."""

    mass_1: h5t.Dataset[h5t.f8, "n_samples"]
    mass_2: h5t.Dataset[h5t.f8, "n_samples"]
    log_likelihood: h5t.Dataset[h5t.f8, "n_samples"]
    spins: h5t.Dataset[h5t.f8, "n_samples 3"] | None
    psd: h5t.Dataset[h5t.f8, "n_freq 2"]

    approximant: str
    f_ref: float = 20.0

    @property
    def chirp_mass(self) -> np.ndarray:
        m1, m2 = self.mass_1[:], self.mass_2[:]
        return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


class PEResult(h5t.File, extras="ignore"):
    """LVK-style parameter estimation result (the PLAN.md example)."""

    runs: Annotated[
        h5t.Group[Posterior],
        h5t.Keys(pattern=r"C\d+:.*"),
    ]
    format_version: Annotated[
        Literal["1.0"],
        h5t.Name("version"),
    ]


def write_run(group: h5py.Group, n_samples: int = 100, n_freq: int = 32) -> None:
    """Populate one conforming Posterior group."""
    group.attrs["n_samples"] = n_samples
    group.attrs["approximant"] = "IMRPhenomXPHM"
    group.attrs["f_ref"] = 20.0
    rng = np.random.default_rng(0)
    group.create_dataset("mass_1", data=rng.random(n_samples))
    group.create_dataset("mass_2", data=rng.random(n_samples))
    group.create_dataset("log_likelihood", data=rng.random(n_samples))
    group.create_dataset("psd", data=rng.random((n_freq, 2)))


def write_pe_result(path: Path | str, run_keys: tuple[str, ...] = ("C01:XPHM",)) -> None:
    """Write a conforming PEResult file."""
    with h5py.File(path, "w") as f:
        f.attrs["version"] = "1.0"
        for key in run_keys:
            write_run(f.create_group(f"runs/{key}"))


@pytest.fixture
def pe_file(tmp_path: Path) -> Path:
    """A conforming PEResult file on disk."""
    path = tmp_path / "pe.h5"
    write_pe_result(path)
    return path


def open_fd_count() -> int:
    """Number of open file descriptors of this process (POSIX)."""
    return len(os.listdir("/dev/fd"))
