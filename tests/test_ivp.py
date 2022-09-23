"""Tests for initial value problems."""


import pytest_cases

from odezoo import ivp


@pytest_cases.case
def case_lotka_volterra():
    return ivp.lotka_volterra()


@pytest_cases.parametrize_with_cases(argnames=("ode_model",), cases=".")
def test_evaluate_ode(ode_model):

    f = ode_model.vector_field
    f_args = ode_model.parameters.vector_field_args
    u0 = ode_model.parameters.initial_values

    if ode_model.meta_information.is_autonomous:
        assert f(u0, *f_args).shape == u0.shape
    else:
        t0 = ode_model.parameters.time_span[0]
        assert f(u0, t0, *f_args).shape == u0.shape


@pytest_cases.case(tags=("second_order",))
def case_second_order_vanderpol():
    return ivp.vanderpol_second_order()


@pytest_cases.case(tags=("second_order",))
def case_second_order_threebody():
    return ivp.threebody_second_order()


@pytest_cases.parametrize_with_cases(
    argnames=("ode_model",), cases=".", has_tag=("second_order",)
)
def test_evaluate_second_order_ode(ode_model):

    f = ode_model.vector_field
    f_args = ode_model.parameters.vector_field_args
    u0, du0 = ode_model.parameters.initial_values

    if ode_model.meta_information.is_autonomous:
        assert f(u0, du0, *f_args).shape == u0.shape
    else:
        t0 = ode_model.parameters.time_span[0]
        assert f(u0, du0, t0, *f_args).shape == u0.shape
