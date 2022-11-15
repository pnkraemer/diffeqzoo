"""Tests for initial value problems."""


from typing import Any, Callable, NamedTuple

import pytest_cases

from diffeqzoo import ivps


class _WrappedIVP(NamedTuple):
    """Test-case for IVP evaluation."""

    wrapped_vector_field: Callable
    """Vector field of the IVP.

    Wrapped into a function with signature ``vf(*initial_values, t, *f_args)``.
    """

    initial_values: Any
    """Tuple of initial values."""

    t: float
    """Dummy time variable."""

    f_args: Any
    """Additional arguments to the vector field."""


@pytest_cases.case
def case_lotka_volterra():
    f, u0, time_span, f_args = ivps.lotka_volterra()
    return _WrappedIVP(lambda y, _, *args: f(y, *args), (u0,), time_span[0], f_args)


@pytest_cases.case
def case_van_der_pol():
    f, u0s, time_span, f_args = ivps.van_der_pol()
    return _WrappedIVP(
        lambda y, dy, _, *args: f(y, dy, *args), u0s, time_span[0], f_args
    )


@pytest_cases.case
def case_van_der_pol_first_order():
    f, u0, time_span, f_args = ivps.van_der_pol_first_order()
    return _WrappedIVP(lambda y, _, *args: f(y, *args), (u0,), time_span[0], f_args)


@pytest_cases.case
def case_three_body_restricted():
    f, u0s, time_span, f_args = ivps.three_body_restricted()
    return _WrappedIVP(
        lambda y, dy, _, *args: f(y, dy, *args), u0s, time_span[0], f_args
    )


@pytest_cases.case
def case_three_body_restricted_first_order():
    f, u0, time_span, f_args = ivps.three_body_restricted_first_order()
    return _WrappedIVP(lambda y, _, *args: f(y, *args), (u0,), time_span[0], f_args)


@pytest_cases.case
def case_pleiades():
    f, u0s, time_span, f_args = ivps.pleiades()
    return _WrappedIVP(lambda y, dy, _, *args: f(y, *args), u0s, time_span[0], f_args)


@pytest_cases.case
def case_pleiades_with_unused_derivative_argument():
    f, u0s, time_span, f_args = ivps.pleiades_with_unused_derivative_argument()
    return _WrappedIVP(
        lambda y, dy, _, *args: f(y, dy, *args), u0s, time_span[0], f_args
    )


@pytest_cases.case
def case_pleiades_first_order():
    f, u0, time_span, f_args = ivps.pleiades_first_order()
    return _WrappedIVP(lambda y, _, *args: f(y, *args), (u0,), time_span[0], f_args)


@pytest_cases.case
def case_henon_heiles():
    f, u0s, time_span, f_args = ivps.henon_heiles()
    return _WrappedIVP(lambda y, dy, _, *args: f(y, *args), u0s, time_span[0], f_args)


@pytest_cases.case
def case_henon_heiles_with_unused_derivative_argument():
    f, u0s, time_span, f_args = ivps.henon_heiles_with_unused_derivative_argument()
    return _WrappedIVP(
        lambda y, dy, _, *args: f(y, dy, *args), u0s, time_span[0], f_args
    )


@pytest_cases.case
def case_henon_heiles_first_order():
    f, u0, time_span, f_args = ivps.henon_heiles_first_order()
    return _WrappedIVP(lambda y, _, *args: f(y, *args), (u0,), time_span[0], f_args)


@pytest_cases.case
def case_lorenz96():
    f, u0, time_span, f_args = ivps.lorenz96()
    return _WrappedIVP(lambda y, _, *args: f(y, *args), (u0,), time_span[0], f_args)


@pytest_cases.case
def case_lorenz63():
    f, u0, time_span, f_args = ivps.lorenz63()
    return _WrappedIVP(lambda y, _, *args: f(y, *args), (u0,), time_span[0], f_args)


@pytest_cases.case
def case_roessler():
    f, u0, time_span, f_args = ivps.roessler()
    return _WrappedIVP(lambda y, _, *args: f(y, *args), (u0,), time_span[0], f_args)


@pytest_cases.case
def case_rigid_body():
    f, u0, time_span, f_args = ivps.rigid_body()
    return _WrappedIVP(lambda y, _, *args: f(y, *args), (u0,), time_span[0], f_args)


@pytest_cases.case
def case_logistic():
    f, u0, time_span, f_args = ivps.logistic()
    return _WrappedIVP(lambda y, _, *args: f(y, *args), (u0,), time_span[0], f_args)


@pytest_cases.case
def case_fitzhugh_nagumo():
    f, u0, time_span, f_args = ivps.fitzhugh_nagumo()
    return _WrappedIVP(lambda y, _, *args: f(y, *args), (u0,), time_span[0], f_args)


@pytest_cases.case
def case_sir():
    f, u0, time_span, f_args = ivps.sir()
    return _WrappedIVP(lambda y, _, *args: f(y, *args), (u0,), time_span[0], f_args)


@pytest_cases.case
def case_seir():
    f, u0, time_span, f_args = ivps.seir()
    return _WrappedIVP(lambda y, _, *args: f(y, *args), (u0,), time_span[0], f_args)


@pytest_cases.case
def case_sird():
    f, u0, time_span, f_args = ivps.sird()
    return _WrappedIVP(lambda y, _, *args: f(y, *args), (u0,), time_span[0], f_args)


@pytest_cases.case
def case_hires():
    f, u0, time_span, f_args = ivps.hires()
    return _WrappedIVP(lambda y, _, *args: f(y, *args), (u0,), time_span[0], f_args)


@pytest_cases.case
def case_rober():
    f, u0, time_span, f_args = ivps.rober()
    return _WrappedIVP(lambda y, _, *args: f(y, *args), (u0,), time_span[0], f_args)


@pytest_cases.case
def case_affine_independent():
    f, u0, time_span, f_args = ivps.affine_independent()
    return _WrappedIVP(lambda y, _, *args: f(y, *args), (u0,), time_span[0], f_args)


@pytest_cases.case
def case_oregonator():
    f, u0, time_span, f_args = ivps.oregonator()
    return _WrappedIVP(lambda y, _, *args: f(y, *args), (u0,), time_span[0], f_args)


@pytest_cases.case
def case_goodwin():
    f, u0, time_span, f_args = ivps.goodwin()
    return _WrappedIVP(lambda y, _, *args: f(y, *args), (u0,), time_span[0], f_args)


@pytest_cases.case
def case_nonlinear_chemical_reaction():
    f, u0, time_span, f_args = ivps.nonlinear_chemical_reaction()
    return _WrappedIVP(lambda y, _, *args: f(y, *args), (u0,), time_span[0], f_args)


@pytest_cases.case
def case_neural_ode_mlp():
    f, u0, time_span, f_args = ivps.neural_ode_mlp()
    return _WrappedIVP(f, (u0,), time_span[0], f_args)


@pytest_cases.parametrize_with_cases(argnames=("ode_model",), cases=".")
def test_evaluate_ode(ode_model: _WrappedIVP):
    """All IVPs are forced into the interface.

    f(y, du, ..., t, p)
    (t0, t1)
    (u0, du0)

    so that all tests become checking for

    f(*initvals, t, *p).shape == initvals[0].shape

    """
    vf, inits, t0, f_args = ode_model

    assert vf(*inits, t0, *f_args).shape == inits[0].shape
