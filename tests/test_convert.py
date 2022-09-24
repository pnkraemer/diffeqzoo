"""Tests for ODE conversion."""

from odezoo import convert, ivps, vector_fields


def test_second_to_first_order_autonomous():
    ivp = ivps.three_body()

    assert len(ivp.initial_values) == 2
    assert ivp.order == 2

    ivp_converted = convert.second_to_first_order_autonomous(ivp=ivp)
    assert len(ivp_converted.initial_values) == 1
    assert ivp_converted.order == 1

    assert ivp_converted.dimension / 2 == ivp.dimension
