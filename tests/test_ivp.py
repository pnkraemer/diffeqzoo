"""Tests for initial value problems."""


from odezoo import ivp


def test_lotka_volterra():
    ode_model = ivp.lotka_volterra()

    f = ode_model.vector_field
    f_params = ode_model.vector_field_parameters_proposal
    u0 = ode_model.initial_values_proposal

    assert f(u0, *f_params).shape == u0.shape


def test_vanderpol_second_order():
    ode_model = ivp.vanderpol_second_order()

    f = ode_model.vector_field
    f_params = ode_model.vector_field_parameters_proposal
    (u0, du0) = ode_model.initial_values_proposal

    assert f(u0, du0, *f_params).shape == u0.shape

def test_threebody_second_order():
    ode_model = ivp.threebody_second_order()

    f = ode_model.vector_field
    f_params = ode_model.vector_field_parameters_proposal
    (u0, du0) = ode_model.initial_values_proposal

    assert f(u0, du0, *f_params).shape == u0.shape

