"""Initial value problem examples."""

from typing import Callable, Iterable, NamedTuple, Optional

from odezoo import numpy_like, vector_fields


class InitialValueProblem(NamedTuple):
    """Initial value problems."""

    vector_field: Callable
    initial_values: Iterable
    time_span: Iterable

    vector_field_args: Iterable = ()

    jacobian: Optional[Callable] = None
    solution: Optional[Callable] = None

    is_autonomous: Optional[bool] = None
    has_periodic_solution: Optional[bool] = None
    order: Optional[int] = None


def lotka_volterra():
    """Lotka--Volterra / predator-prey model."""
    p = (0.5, 0.05, 0.5, 0.05)
    u0 = numpy_like.asarray([20.0, 20.0])
    time_span = (0.0, 20.0)

    return InitialValueProblem(
        vector_field=vector_fields.lotka_volterra,
        vector_field_args=p,
        initial_values=(u0,),
        time_span=time_span,
        is_autonomous=True,
        has_periodic_solution=True,
        order=1,
    )


def lorenz96(*, num_variables=10, forcing=8.0):
    """Lorenz96 model."""

    p = (forcing,)
    u0 = _lorenz96_chaotic_u0(forcing=forcing, num_variables=num_variables)
    time_span = (0.0, 30.0)

    return InitialValueProblem(
        vector_field=vector_fields.lorenz96,
        vector_field_args=p,
        initial_values=(u0,),
        time_span=time_span,
        is_autonomous=True,
        has_periodic_solution=True,
        order=1,
    )


def _lorenz96_chaotic_u0(*, forcing, num_variables, perturb=0.01):
    u0_equilibrium = numpy_like.ones(num_variables) * forcing
    return numpy_like.concatenate(([u0_equilibrium[0] + perturb], u0_equilibrium[1:]))


def pleiades():
    """Pleiades problem."""

    x0 = [3.0, 3.0, -1.0, -3.0, 2.0, -2.0, 2.0]
    y0 = [3.0, -3.0, 2.0, 0.0, 0.0, -4.0, 4.0]
    dx0 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.75, -1.5]
    dy0 = [0.0, 0.0, 0.0, -1.25, 1.0, 0.0, 0.0]
    u0 = numpy_like.asarray(x0 + y0)
    du0 = numpy_like.asarray(dx0 + dy0)

    time_span = (0.0, 3.0)

    return InitialValueProblem(
        vector_field=vector_fields.pleiades,
        initial_values=(u0, du0),
        time_span=time_span,
        is_autonomous=True,
        has_periodic_solution=False,
        order=2,
    )


def van_der_pol():
    """Van-der-Pol system as a second order differential equation."""
    p = (1.0,)
    u0 = numpy_like.asarray([2.0])
    du0 = numpy_like.asarray([0.0])
    time_span = (0.0, 6.3)

    return InitialValueProblem(
        vector_field=vector_fields.van_der_pol,
        vector_field_args=p,
        initial_values=(u0, du0),
        time_span=time_span,
        is_autonomous=True,
        has_periodic_solution=True,
        order=2,
    )


def three_body():
    """Restricted three-body problem as a second order differential equation."""
    # Some parameter suggestions for nice simulation
    p = (0.012277471,)
    u0 = numpy_like.asarray([0.994, 0])
    du0 = numpy_like.asarray([0, -2.00158510637908252240537862224])
    time_span = (0.0, 17.0652165601579625588917206249)

    return InitialValueProblem(
        vector_field=vector_fields.three_body,
        is_autonomous=True,
        has_periodic_solution=True,
        order=2,
        vector_field_args=p,
        initial_values=(u0, du0),
        time_span=time_span,
    )
