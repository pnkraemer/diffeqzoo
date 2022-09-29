"""Tests for BVPs."""

import inspect

import pytest_cases

from odezoo import backend, bvps


@pytest_cases.case
def case_bratu():
    f_, ((L, l), (R, r)), tspan, f_args = bvps.bratu()

    def g0(*state):
        full_state = backend.numpy.concatenate(state, axis=None)
        return backend.numpy.dot(L, full_state) - l

    def g1(*state):
        full_state = backend.numpy.concatenate(state, axis=None)
        return backend.numpy.dot(R, full_state) - r

    def f(u, _, *args):
        return f_(u, *args)

    u_dummy = (backend.numpy.ones(()),) * 2
    return f, (g0, g1), tspan, f_args, u_dummy


@pytest_cases.case
def case_bratu_autonomous_api():
    f, ((L, l), (R, r)), tspan, f_args = bvps.bratu_autonomous_api()

    def g0(*state):
        full_state = backend.numpy.concatenate(state, axis=None)
        return backend.numpy.dot(L, full_state) - l

    def g1(*state):
        full_state = backend.numpy.concatenate(state, axis=None)
        return backend.numpy.dot(R, full_state) - r

    u_dummy = (backend.numpy.ones(()),) * 2
    return f, (g0, g1), tspan, f_args, u_dummy


@pytest_cases.parametrize_with_cases("ode_model", cases=".")
def test_separable_bvp(ode_model):
    f, (g0, g1), _, f_args, u_dummy = ode_model

    assert f(*u_dummy, *f_args).shape == u_dummy[0].shape
    assert g0(*u_dummy).shape == u_dummy[0].shape
    assert g1(*u_dummy).shape == u_dummy[0].shape
