"""Tests for BVPs."""

import inspect

import pytest_cases

from diffeqzoo import backend, bvps


@pytest_cases.case(tags="two_point")
def case_bratu():
    f_, (g0_, g1_), tspan, f_args = bvps.bratu()

    def f(u, _, *args):
        return f_(u, *args)

    def g0(u, _, *args):
        return g0_(u, *args)

    def g1(u, _, *args):
        return g1_(u, *args)

    u_dummy = (backend.numpy.ones(()),) * 2
    return f, (g0, g1), tspan, f_args, u_dummy


@pytest_cases.case(tags="two_point")
def case_pendulum():
    f_, (g0_, g1_), tspan, f_args = bvps.pendulum()

    def f(u, _, *args):
        return f_(u, *args)

    def g0(u, _, *args):
        return g0_(u, *args)

    def g1(u, _, *args):
        return g1_(u, *args)

    u_dummy = (backend.numpy.ones(()),) * 2
    return f, (g0, g1), tspan, f_args, u_dummy


@pytest_cases.case(tags="two_point")
def case_bratu_with_unused_derivative_argument():
    f, bconds, tspan, f_args = bvps.bratu_with_unused_derivative_argument()
    u_dummy = (backend.numpy.ones(()),) * 2
    return f, bconds, tspan, f_args, u_dummy


@pytest_cases.case(tags="two_point")
def case_pendulum_with_unused_derivative_argument():
    f, bconds, tspan, f_args = bvps.pendulum_with_unused_derivative_argument()
    u_dummy = (backend.numpy.ones(()),) * 2
    return f, bconds, tspan, f_args, u_dummy


@pytest_cases.parametrize_with_cases("ode_model", cases=".", has_tag=("two_point",))
def test_two_point_bvp(ode_model):
    f, (g0, g1), _, f_args, u_dummy = ode_model

    assert f(*u_dummy, *f_args).shape == u_dummy[0].shape
    assert g0(*u_dummy).shape == u_dummy[0].shape
    assert g1(*u_dummy).shape == u_dummy[0].shape


@pytest_cases.case(tags="not_two_point")
def case_measles():
    f, bconds, tspan, f_args = bvps.measles()
    u_dummy = (backend.numpy.ones((3,)),)
    return f, bconds, tspan, f_args, u_dummy


@pytest_cases.parametrize_with_cases("ode_model", cases=".", has_tag=("not_two_point",))
def test_bvp(ode_model):
    f, bcond, (t0, _), f_args, u_dummy = ode_model

    assert f(t0, *u_dummy, *f_args).shape == u_dummy[0].shape
    assert bcond(*u_dummy, *u_dummy).shape == u_dummy[0].shape
