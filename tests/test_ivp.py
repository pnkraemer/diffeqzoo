"""Tests for initial value problems."""


import pytest_cases

from odezoo import ivp


@pytest_cases.case
def case_lotka_volterra():
    return ivp.lotka_volterra()


@pytest_cases.case
def case_van_der_pol():
    return ivp.van_der_pol()


@pytest_cases.case
def case_three_body():
    return ivp.three_body()


@pytest_cases.parametrize_with_cases(argnames=("ode_model",), cases=".")
def test_evaluate_ode(ode_model):

    f, u0, (t0, _), f_args, *_ = ode_model

    if ode_model.is_autonomous:
        assert f(*u0, *f_args).shape == u0[0].shape
    else:
        assert f(*u0, t0, *f_args).shape == u0[0].shape
