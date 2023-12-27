"""Discretised partial differential equations."""
from diffeqzoo import backend, vector_fields
from diffeqzoo.ivps import _ivp

# todo: Should the "info" be part of the IVP data structure, i.e., always returned?


def heat_1d_dirichlet(
    *,
    initial_values=None,
    time_span=(0.0, 1.0),
    bounding_box=(0.0, 1.0),
    num_gridpoints=10,
    coefficient=1.0
):
    r"""Discretised heat equation in 1d with Dirichlet boundary.

    The discretisation uses second-order central differences.
    The vector field is evaluated with a convolution with zero-padding.
    """
    # Make the grid
    x0, x1 = bounding_box
    grid = backend.numpy.linspace(x0, x1, num=num_gridpoints, endpoint=True)

    # Make FD coefficients
    dx = backend.numpy.diff(grid)[0]
    stencil_weights = backend.numpy.array([1.0, -2.0, 1.0]) / dx**2

    # Make initial condition
    if initial_values is None:
        midpoint = 0.5 * (x1 - x0)
        initial_values = backend.numpy.exp(-50.0 * (grid - midpoint) ** 2)
    else:
        initial_values = backend.numpy.asarray(initial_values)

    # The initial condition should conform with the Dirichlet-boundary
    tol = 0.1 * dx**2  # central difference error * 0.1
    u0_0, u0_1 = initial_values[0], initial_values[-1]
    assert backend.numpy.allclose(u0_0, 0.0, atol=tol), (u0_0, tol)
    assert backend.numpy.allclose(u0_1, 0.0, atol=tol), (u0_1, tol)

    # Load the vector field and return the IVP
    ivp = _ivp.InitialValueProblem(
        vector_field=vector_fields.heat_1d_dirichlet,
        vector_field_args=(stencil_weights, coefficient),
        initial_values=initial_values,
        time_span=time_span,
    )
    return ivp, {"grid": grid}
