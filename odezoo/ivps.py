"""Initial value problem examples.

The following information should be available for each equation:

- The equation (in a copy/pastable latex-math format)
- The original reference
- The meaning of each parameter (if possible)
- "Notable" info (i.e. why this problem may be interesting)

Providing this information makes it easy to use the problems
as benchmark problems (e.g., in papers).
"""
from typing import Callable, Iterable, NamedTuple, Optional

from odezoo import backend, vector_fields


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
    dimension: Optional[int] = None


def lotka_volterra(
    *, initial_values=None, time_span=(0.0, 20.0), parameters=(0.5, 0.05, 0.5, 0.05)
):
    """Lotka--Volterra / predator-prey model."""
    if initial_values is None:
        initial_values = (backend.numpy.asarray([20.0, 20.0]),)

    return InitialValueProblem(
        vector_field=vector_fields.lotka_volterra,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
        is_autonomous=True,
        has_periodic_solution=True,
        order=1,
        dimension=2,
    )


def fitzhugh_nagumo(
    *, initial_values=None, time_span=(0.0, 20.0), parameters=(0.2, 0.2, 3.0, 1.0)
):
    r"""FitzHugh-Nagumo model."""
    if initial_values is None:
        initial_values = (backend.numpy.asarray([1.0, -1.0]),)

    return InitialValueProblem(
        vector_field=vector_fields.fitzhugh_nagumo,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
        is_autonomous=True,
        has_periodic_solution=True,
        order=1,
        dimension=2,
    )


def logistic(*, initial_values=None, time_span=(0.0, 2.5), parameters=(1.0, 1.0)):
    """Logistic ODE model."""
    if initial_values is None:
        initial_values = (backend.numpy.asarray([0.1]),)

    return InitialValueProblem(
        vector_field=vector_fields.logistic,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
        is_autonomous=True,
        has_periodic_solution=False,
        order=1,
        dimension=1,
    )


def sir(*, initial_values=None, time_span=(0.0, 200.0), beta=0.3, gamma=0.1):
    """SIR model."""
    if initial_values is None:
        u0 = backend.numpy.asarray([998.0, 1.0, 1.0])
        initial_values = (u0,)

    parameters = (beta, gamma, backend.numpy.sum(initial_values[0]))

    return InitialValueProblem(
        vector_field=vector_fields.sir,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
        is_autonomous=True,
        has_periodic_solution=False,
        order=1,
        dimension=3,
    )


def seir(
    *, initial_values=None, time_span=(0.0, 200.0), alpha=0.3, beta=0.3, gamma=0.1
):
    """SEIR model."""
    if initial_values is None:
        u0 = backend.numpy.asarray([998.0, 1.0, 1.0, 1.0])
        initial_values = (u0,)

    parameters = (alpha, beta, gamma, backend.numpy.sum(initial_values[0]))

    return InitialValueProblem(
        vector_field=vector_fields.seir,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
        is_autonomous=True,
        has_periodic_solution=False,
        order=1,
        dimension=4,
    )


def sird(
    *, initial_values=None, time_span=(0.0, 200.0), beta=0.3, gamma=0.1, eta=0.005
):
    """SIRD model."""
    if initial_values is None:
        u0 = backend.numpy.asarray([998.0, 1.0, 1.0, 0.0])
        initial_values = (u0,)

    parameters = (beta, gamma, eta, backend.numpy.sum(initial_values[0]))

    return InitialValueProblem(
        vector_field=vector_fields.sird,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
        is_autonomous=True,
        has_periodic_solution=False,
        order=1,
        dimension=4,
    )


def lorenz96(
    *,
    initial_values=None,
    time_span=(0.0, 30.0),
    num_variables=10,
    forcing=8.0,
    perturb=0.01,
):
    """Lorenz96 model."""
    if initial_values is None:
        u0 = _lorenz96_chaotic_u0(
            forcing=forcing, num_variables=num_variables, perturb=perturb
        )
        initial_values = (u0,)

    return InitialValueProblem(
        vector_field=vector_fields.lorenz96,
        vector_field_args=(forcing,),
        initial_values=initial_values,
        time_span=time_span,
        is_autonomous=True,
        has_periodic_solution=False,
        order=1,
        dimension=num_variables,
    )


def lorenz63(
    *,
    initial_values=None,
    time_span=(0.0, 20.0),
    parameters=(10.0, 28.0, 8.0 / 3.0),
):
    """Lorenz63 model."""
    if initial_values is None:
        u0 = backend.numpy.asarray([0.0, 1.0, 1.05])
        initial_values = (u0,)

    return InitialValueProblem(
        vector_field=vector_fields.lorenz63,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
        is_autonomous=True,
        has_periodic_solution=False,
        order=1,
        dimension=3,
    )


def rigid_body(
    *, time_span=(0.0, 20.0), initial_values=None, parameters=(-2.0, 1.25, -0.5)
):
    r"""Rigid body dynamics without external forces."""
    if initial_values is None:
        u0 = backend.numpy.array([1.0, 0.0, 0.9])
        initial_values = (u0,)

    return InitialValueProblem(
        vector_field=vector_fields.rigid_body,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
        is_autonomous=True,
        order=1,
        dimension=3,
    )


def _lorenz96_chaotic_u0(*, forcing, num_variables, perturb):
    u0_equilibrium = backend.numpy.ones(num_variables) * forcing
    return backend.numpy.concatenate(
        [backend.numpy.asarray([u0_equilibrium[0] + perturb]), u0_equilibrium[1:]]
    )


def pleiades(*, initial_values=None, time_span=(0.0, 3.0)):
    """Pleiades problem."""
    if initial_values is None:
        x0 = [3.0, 3.0, -1.0, -3.0, 2.0, -2.0, 2.0]
        y0 = [3.0, -3.0, 2.0, 0.0, 0.0, -4.0, 4.0]
        dx0 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.75, -1.5]
        dy0 = [0.0, 0.0, 0.0, -1.25, 1.0, 0.0, 0.0]
        u0 = backend.numpy.asarray(x0 + y0)
        du0 = backend.numpy.asarray(dx0 + dy0)
        initial_values = (u0, du0)

    return InitialValueProblem(
        vector_field=vector_fields.pleiades,
        initial_values=initial_values,
        time_span=time_span,
        is_autonomous=True,
        has_periodic_solution=False,
        order=2,
        dimension=14,
    )


def van_der_pol(*, stiffness_constant=1.0, initial_values=None, time_span=(0.0, 6.3)):
    """Van-der-Pol system as a second order differential equation."""
    if initial_values is None:
        u0 = backend.numpy.asarray([2.0])
        du0 = backend.numpy.asarray([0.0])
        initial_values = (u0, du0)

    return InitialValueProblem(
        vector_field=vector_fields.van_der_pol,
        vector_field_args=(stiffness_constant,),
        initial_values=initial_values,
        time_span=time_span,
        is_autonomous=True,
        has_periodic_solution=True,
        order=2,
        dimension=1,
    )


def three_body(
    *,
    initial_values=None,
    standardised_moon_mass=0.012277471,
    time_span=(0.0, 17.0652165601579625588917206249),
):
    """Restricted three-body problem as a second order differential equation."""
    if initial_values is None:
        u0 = backend.numpy.asarray([0.994, 0])
        du0 = backend.numpy.asarray([0, -2.00158510637908252240537862224])
        initial_values = (u0, du0)

    return InitialValueProblem(
        vector_field=vector_fields.three_body,
        is_autonomous=True,
        has_periodic_solution=True,
        order=2,
        vector_field_args=(standardised_moon_mass,),
        initial_values=initial_values,
        time_span=time_span,
        dimension=2,
    )


def hires(*, initial_values=None, time_span=(0.0, 321.8122)):
    """High Irradiance Response (HIRES).

    A chemical reaction involving eight reactants.
    """
    if initial_values is None:
        u0 = backend.numpy.asarray([1.0, 0.0, 0.0, 0, 0, 0, 0, 0.0057])
        initial_values = (u0,)

    return InitialValueProblem(
        vector_field=vector_fields.hires,
        is_autonomous=True,
        order=1,
        vector_field_args=(),  # todo: move vf-params here
        initial_values=initial_values,
        time_span=time_span,
        dimension=8,
    )


def rober(*, initial_values=None, time_span=(0.0, 1e5), k1=0.04, k2=3e7, k3=1e4):
    """Rober ODE problem due to Robertson (1966)."""
    if initial_values is None:
        u0 = backend.numpy.asarray([1.0, 0.0, 0.0])
        initial_values = (u0,)

    return InitialValueProblem(
        vector_field=vector_fields.rober,
        is_autonomous=True,
        order=1,
        vector_field_args=(k1, k2, k3),
        initial_values=initial_values,
        time_span=time_span,
        dimension=3,
    )
