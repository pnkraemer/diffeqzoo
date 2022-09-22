"""Tests for initial value problems."""


from odezoo import ivp


def test_lotka_volterra():
    f, f_args, u0, _ = ivp.lotka_volterra()
    assert f(u0, *f_args).shape == u0.shape
