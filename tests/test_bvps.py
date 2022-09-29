"""Tests for BVPs."""

import inspect

import pytest_cases

from odezoo import backend, bvps


@pytest_cases.case
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


@pytest_cases.case
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


@pytest_cases.case
def case_bratu_autonomous_api():
    f, bconds, tspan, f_args = bvps.bratu_autonomous_api()
    u_dummy = (backend.numpy.ones(()),) * 2
    return f, bconds, tspan, f_args, u_dummy


@pytest_cases.case
def case_pendulum_autonomous_api():
    f, bconds, tspan, f_args = bvps.pendulum_autonomous_api()
    u_dummy = (backend.numpy.ones(()),) * 2
    return f, bconds, tspan, f_args, u_dummy


@pytest_cases.parametrize_with_cases("ode_model", cases=".")
def test_separable_bvp(ode_model):
    f, (g0, g1), _, f_args, u_dummy = ode_model

    assert f(*u_dummy, *f_args).shape == u_dummy[0].shape
    assert g0(*u_dummy).shape == u_dummy[0].shape
    assert g1(*u_dummy).shape == u_dummy[0].shape
