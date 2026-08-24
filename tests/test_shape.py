"""The shape grammar: dimension := NAME | INTEGER | _ ."""

from __future__ import annotations

import pytest

import h5t
from h5t._shape import AnyDim, FixedDim, NamedDim, format_shape, parse_shape


def test_scalar_is_empty_tuple():
    assert parse_shape("") == ()


def test_named_dimension():
    assert parse_shape("n_samples") == (NamedDim("n_samples"),)


def test_name_and_literal():
    assert parse_shape("n_samples 3") == (NamedDim("n_samples"), FixedDim(3))


def test_anonymous_axis():
    assert parse_shape("n_samples _") == (NamedDim("n_samples"), AnyDim())


@pytest.mark.parametrize(
    "bad",
    ["n+1", "n  m", " n", "n ", "3n", "n-1", "a,b", "*"],
)
def test_malformed_shapes_raise(bad: str):
    with pytest.raises(h5t.SchemaError):
        parse_shape(bad)


def test_non_string_raises():
    with pytest.raises(h5t.SchemaError):
        parse_shape(3)  # type: ignore[arg-type]


@pytest.mark.parametrize("text", ["", "n", "n 3", "n _ 2"])
def test_format_round_trips(text: str):
    assert format_shape(parse_shape(text)) == text
