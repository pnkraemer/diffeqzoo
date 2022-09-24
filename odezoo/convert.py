"""Conversion of ODE problems."""
from odezoo import backend, ivps


def second_to_first_order_autonomous(*, ivp):
    """Convert an autonomous second-order problem to a first-order problem."""

    def f(y, /, *args):
        u, du = y[: ivp.dimension], y[ivp.dimension :]
        return ivp.vector_field(u, du, *args)

    def df(y, /, *args):
        u, du = y[: ivp.dimension], y[ivp.dimension :]
        return ivp.jacobian(u, du, *args)

    inits = (backend.numpy.concatenate(ivp.initial_values, axis=0),)

    if ivp.order is not None:
        order = 1
    else:
        order = None

    dimension = ivp.dimension * 2 or None

    return ivps.InitialValueProblem(
        vector_field=f,  # new
        jacobian=df,  # new
        vector_field_args=ivp.vector_field_args,
        initial_values=inits,  # new
        time_span=ivp.time_span,
        is_autonomous=True,
        has_periodic_solution=ivp.has_periodic_solution,
        order=order,  # new
        dimension=dimension,  # new
    )
