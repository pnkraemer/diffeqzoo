"""Tests for initial value problems."""


import pytest_cases

from odezoo import ivps


@pytest_cases.case
def case_lotka_volterra():
    return ivps.lotka_volterra()


@pytest_cases.case
def case_van_der_pol():
    return ivps.van_der_pol()


@pytest_cases.case
def case_three_body():
    return ivps.three_body()


@pytest_cases.case
def case_pleiades():
    return ivps.pleiades()


@pytest_cases.parametrize_with_cases(argnames=("ode_model",), cases=".")
def test_evaluate_ode(ode_model):

    f, u0, (t0, _), f_args, *_ = ode_model

    if ode_model.is_autonomous:
        assert f(*u0, *f_args).shape == u0[0].shape
    else:
        assert f(*u0, t0, *f_args).shape == u0[0].shape
