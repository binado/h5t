"""Dtype tokens and the section-6 compatibility policy."""

from __future__ import annotations

import numpy as np
import pytest

import h5t
from h5t._dtypes import dtype_matches, normalize_str


def test_matches_on_kind_and_itemsize():
    assert dtype_matches(h5t.f8, np.dtype("f8"))
    assert dtype_matches(h5t.i4, np.dtype("i4"))
    assert dtype_matches(h5t.u2, np.dtype("u2"))
    assert dtype_matches(h5t.c16, np.dtype("c16"))


def test_byte_order_is_ignored():
    assert dtype_matches(h5t.f8, np.dtype(">f8"))
    assert dtype_matches(h5t.f8, np.dtype("<f8"))
    assert dtype_matches(h5t.i8, np.dtype(">i8"))


def test_no_width_tolerance():
    assert not dtype_matches(h5t.f8, np.dtype("f4"))
    assert not dtype_matches(h5t.f4, np.dtype("f8"))
    assert not dtype_matches(h5t.i8, np.dtype("i4"))


def test_kind_mismatch():
    assert not dtype_matches(h5t.f8, np.dtype("i8"))
    assert not dtype_matches(h5t.i8, np.dtype("u8"))


def test_normalize_str_flavours():
    assert normalize_str("plain") == "plain"
    assert normalize_str(b"bytes") == "bytes"
    assert normalize_str(np.bytes_(b"IMRPhenomXPHM")) == "IMRPhenomXPHM"
    assert normalize_str(np.str_("vlen")) == "vlen"
    assert type(normalize_str(np.str_("vlen"))) is str


def test_undecodable_bytes_raise():
    with pytest.raises(ValueError, match="UTF-8"):
        normalize_str(b"\xff\xfe\xfa")


def test_non_string_raises():
    with pytest.raises(ValueError):
        normalize_str(3.5)


def test_dtype_token_requires_kind_and_itemsize():
    with pytest.raises(TypeError):

        class Broken(h5t.DType):
            pass
