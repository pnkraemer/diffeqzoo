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
