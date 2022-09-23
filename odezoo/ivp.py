"""Initial value problem examples."""

from collections import namedtuple
from typing import Any, Callable, Iterable, NamedTuple, Optional

from odezoo import numpy_like


class MetaInformation(NamedTuple):
    """Store meta-information about the ODE (periodicity, etc.)."""

    is_autonomous: Optional[bool] = None
    has_periodic_solution: Optional[bool] = None
    order: Optional[int] = None


class Parameters(NamedTuple):
    """Store the ODE parameters separately from the ODE vector field."""

    initial_values: Iterable
    vector_field_args: Iterable = ()
    time_span: Iterable = ()


class InitialValueProblem(NamedTuple):
    vector_field: Callable
    parameters: Parameters
    jacobian: Optional[Callable] = None
    solution: Optional[Callable] = None
    meta_information: Optional[MetaInformation] = None


def lotka_volterra():
    """Lotka--Volterra / predator-prey model."""

    meta_information = MetaInformation(
        is_autonomous=True, has_periodic_solution=True, order=1
    )

    def f(y, /, *params):
        a, b, c, d = params
        return numpy_like.asarray(
            [a * y[0] - b * y[0] * y[1], -c * y[1] + d * y[0] * y[1]]
        )

    p = (0.5, 0.05, 0.5, 0.05)
    u0 = numpy_like.asarray([20.0, 20.0])
    time_span = (0.0, 20.0)
    parameters_proposed = Parameters(
        vector_field_args=p, initial_values=(u0,), time_span=time_span
    )

    return InitialValueProblem(
        vector_field=f,
        parameters=parameters_proposed,
        meta_information=meta_information,
    )


def vanderpol_second_order():
    """Van-der-Pol system as a second order differential equation."""

    def f(u, du, /, stiffness_constant):
        return stiffness_constant * ((1.0 - u**2) * du - u)

    meta_information = MetaInformation(
        is_autonomous=True, has_periodic_solution=True, order=2
    )

    p = (1.0,)
    u0 = numpy_like.asarray([2.0])
    du0 = numpy_like.asarray([0.0])
    time_span = (0.0, 6.3)
    parameters_proposed = Parameters(
        vector_field_args=p, initial_values=(u0, du0), time_span=time_span
    )
    return InitialValueProblem(
        vector_field=f,
        parameters=parameters_proposed,
        meta_information=meta_information,
    )


def threebody_second_order():
    """Restricted three-body problem as a second order differential equation."""

    def f(Y, dY, /, standardised_moon_mass):
        mu = standardised_moon_mass
        mp = 1.0 - mu
        D1 = numpy_like.linalg.norm(numpy_like.asarray([Y[0] + mu, Y[1]])) ** 3.0
        D2 = numpy_like.linalg.norm(numpy_like.asarray([Y[0] - mp, Y[1]])) ** 3.0
        du0p = Y[0] + 2 * dY[1] - mp * (Y[0] + mu) / D1 - mu * (Y[0] - mp) / D2
        du1p = Y[1] - 2 * dY[0] - mp * Y[1] / D1 - mu * Y[1] / D2
        return numpy_like.asarray([du0p, du1p])

    meta_information = MetaInformation(
        is_autonomous=True, has_periodic_solution=True, order=2
    )

    # Some parameter suggestions for nice simulation
    p = (0.012277471,)
    u0 = numpy_like.asarray([0.994, 0])
    du0 = numpy_like.asarray([0, -2.00158510637908252240537862224])
    time_span = (0.0, 17.0652165601579625588917206249)
    parameters_proposed = Parameters(
        vector_field_args=p, initial_values=(u0, du0), time_span=time_span
    )
    return InitialValueProblem(
        vector_field=f,
        parameters=parameters_proposed,
        meta_information=meta_information,
    )
