"""Tests for initial value problems."""


import pytest_cases

from diffeqzoo import ivps


@pytest_cases.case
def case_lotka_volterra():
    f, u0, time_span, f_args = ivps.lotka_volterra()
    return lambda y, _, *args: f(y, *args), (u0,), time_span, f_args


@pytest_cases.case
def case_van_der_pol():
    f, u0s, time_span, f_args = ivps.van_der_pol()
    return lambda y, dy, _, *args: f(y, dy, *args), u0s, time_span, f_args


@pytest_cases.case
def case_van_der_pol_first_order():
    f, u0, time_span, f_args = ivps.van_der_pol_first_order()
    return lambda y, _, *args: f(y, *args), (u0,), time_span, f_args


@pytest_cases.case
def case_three_body_restricted():
    f, u0s, time_span, f_args = ivps.three_body_restricted()
    return lambda y, dy, _, *args: f(y, dy, *args), u0s, time_span, f_args


@pytest_cases.case
def case_three_body_restricted_first_order():
    f, u0, time_span, f_args = ivps.three_body_restricted_first_order()
    return lambda y, _, *args: f(y, *args), (u0,), time_span, f_args


@pytest_cases.case
def case_pleiades():
    f, u0s, time_span, f_args = ivps.pleiades()
    return lambda y, dy, _, *args: f(y, *args), u0s, time_span, f_args


@pytest_cases.case
def case_pleiades_first_order():
    f, u0, time_span, f_args = ivps.pleiades_first_order()
    return lambda y, _, *args: f(y, *args), (u0,), time_span, f_args


@pytest_cases.case
def case_lorenz96():
    f, u0, time_span, f_args = ivps.lorenz96()
    return lambda y, _, *args: f(y, *args), (u0,), time_span, f_args


@pytest_cases.case
def case_lorenz63():
    f, u0, time_span, f_args = ivps.lorenz63()
    return lambda y, _, *args: f(y, *args), (u0,), time_span, f_args


@pytest_cases.case
def case_rigid_body():
    f, u0, time_span, f_args = ivps.rigid_body()
    return lambda y, _, *args: f(y, *args), (u0,), time_span, f_args


@pytest_cases.case
def case_logistic():
    f, u0, time_span, f_args = ivps.logistic()
    return lambda y, _, *args: f(y, *args), (u0,), time_span, f_args


@pytest_cases.case
def case_fitzhhugh_nagumo():
    f, u0, time_span, f_args = ivps.fitzhugh_nagumo()
    return lambda y, _, *args: f(y, *args), (u0,), time_span, f_args


@pytest_cases.case
def case_sir():
    f, u0, time_span, f_args = ivps.sir()
    return lambda y, _, *args: f(y, *args), (u0,), time_span, f_args


@pytest_cases.case
def case_seir():
    f, u0, time_span, f_args = ivps.seir()
    return lambda y, _, *args: f(y, *args), (u0,), time_span, f_args


@pytest_cases.case
def case_sird():
    f, u0, time_span, f_args = ivps.sird()
    return lambda y, _, *args: f(y, *args), (u0,), time_span, f_args


@pytest_cases.case
def case_hires():
    f, u0, time_span, f_args = ivps.hires()
    return lambda y, _, *args: f(y, *args), (u0,), time_span, f_args


@pytest_cases.case
def case_rober():
    f, u0, time_span, f_args = ivps.rober()
    return lambda y, _, *args: f(y, *args), (u0,), time_span, f_args


@pytest_cases.parametrize_with_cases(argnames=("ode_model",), cases=".")
def test_evaluate_ode(ode_model):
    """All IVPs are forced into the interface.

    f(y, du, ..., t, p)
    (t0, t1)
    (u0, du0)

    so that all tests become checking for

    f(*initvals, t, *p).shape == initvals[0].shape

    """
    f, inits, (t0, _), f_args, *_ = ode_model

    assert f(*inits, t0, *f_args).shape == inits[0].shape
