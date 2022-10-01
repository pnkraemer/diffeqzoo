"""Boundary value problems."""

import math  # for PI
from typing import Any, Callable, Iterable, NamedTuple, Union

from diffeqzoo import _vector_fields, backend, transform


class _BoundaryValueProblem(NamedTuple):
    vector_field: Callable
    boundary_conditions: Iterable  # (g0, g1) or ((L, l), (R, r))
    time_span: Iterable  # (t0, t1)
    vector_field_args: Iterable = ()


def bratu(*, time_span=(0.0, 1.0), parameters=(1.0,)):
    """Bratu's problem."""

    def g0(u):
        return u

    def g1(u):
        return u

    return _BoundaryValueProblem(
        vector_field=_vector_fields.bratu,
        boundary_conditions=(g0, g1),
        vector_field_args=parameters,
        time_span=time_span,
    )


def bratu_autonomous_api(*, time_span=(0.0, 1.0), parameters=(1.0,)):
    """Bratu's problem with a signature (u, u')."""

    def g0(u, _):
        return u

    def g1(u, _):
        return u

    return _BoundaryValueProblem(
        vector_field=_vector_fields.bratu_autonomous_api,
        boundary_conditions=(g0, g1),
        vector_field_args=parameters,
        time_span=time_span,
    )
    return f_bratu, (t0, tmax), (u0, umax)


def pendulum_autonomous_api(*, time_span=(0.0, math.pi / 2.0), parameters=(9.81,)):
    """Pendulum problem with a signature (u, u')."""

    def g0(u, _):
        return u

    def g1(u, _):
        return u

    return _BoundaryValueProblem(
        vector_field=_vector_fields.pendulum_autonomous_api,
        boundary_conditions=(g0, g1),
        vector_field_args=parameters,
        time_span=time_span,
    )


def pendulum(*, time_span=(0.0, math.pi / 2.0), parameters=(9.81,)):
    """Pendulum problem."""

    def g0(u):
        return u

    def g1(u):
        return u

    return _BoundaryValueProblem(
        vector_field=_vector_fields.pendulum,
        boundary_conditions=(g0, g1),
        vector_field_args=parameters,
        time_span=time_span,
    )


def measles(*, time_span=(0.0, 1.0), mu=0.02, lmbda=0.0279, eta=0.01, beta0=1575):
    """Measles problem.

    Example 1.10 in Ascher et al., which contains a reference to the original paper.
    """

    parameters = (mu, lmbda, eta, beta0)

    def bcond(u_left, u_right):
        return u_left - u_right

    return _BoundaryValueProblem(
        vector_field=_vector_fields.measles,
        boundary_conditions=bcond,
        vector_field_args=parameters,
        time_span=time_span,
    )
