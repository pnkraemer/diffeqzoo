"""Initial value problem examples.

The following information should be available for each equation:

- The equation (in a copy/pastable latex-math format)
- The original reference
- The meaning of each parameter (if possible)
- "Notable" info (i.e. why this problem may be interesting)

Providing this information makes it easy to use the problems
as benchmark problems (e.g., in papers).
"""
from typing import Any, Callable, Iterable, NamedTuple, Union

from odezoo import _descriptions, _docstring_utils, backend, transform, vector_fields


class _InitialValueProblem(NamedTuple):
    vector_field: Callable
    initial_values: Union[Iterable, Any]  # u0 or (u0, du0, ddu0, ...)
    time_span: Iterable  # (t0, t1)
    vector_field_args: Iterable = ()


def lotka_volterra(
    *, initial_values=None, time_span=(0.0, 20.0), parameters=(0.5, 0.05, 0.5, 0.05)
):
    """Lotka--Volterra / predator-prey model."""
    if initial_values is None:
        initial_values = backend.numpy.asarray([20.0, 20.0])

    return _InitialValueProblem(
        vector_field=vector_fields.lotka_volterra,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
    )


def fitzhugh_nagumo(
    *, initial_values=None, time_span=(0.0, 20.0), parameters=(0.2, 0.2, 3.0, 1.0)
):
    r"""FitzHugh-Nagumo model."""
    if initial_values is None:
        initial_values = backend.numpy.asarray([1.0, -1.0])

    return _InitialValueProblem(
        vector_field=vector_fields.fitzhugh_nagumo,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
    )


def logistic(*, initial_values=None, time_span=(0.0, 2.5), parameters=(1.0, 1.0)):
    """Logistic ODE model."""
    if initial_values is None:
        initial_values = backend.numpy.asarray([0.1])

    return _InitialValueProblem(
        vector_field=vector_fields.logistic,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
    )


def sir(*, initial_values=None, time_span=(0.0, 200.0), beta=0.3, gamma=0.1):
    """SIR model."""
    if initial_values is None:
        initial_values = backend.numpy.asarray([998.0, 1.0, 1.0])

    parameters = (beta, gamma, backend.numpy.sum(initial_values[0]))

    return _InitialValueProblem(
        vector_field=vector_fields.sir,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
    )


def seir(
    *, initial_values=None, time_span=(0.0, 200.0), alpha=0.3, beta=0.3, gamma=0.1
):
    """SEIR model."""
    if initial_values is None:
        initial_values = backend.numpy.asarray([998.0, 1.0, 1.0, 1.0])

    parameters = (alpha, beta, gamma, backend.numpy.sum(initial_values[0]))

    return _InitialValueProblem(
        vector_field=vector_fields.seir,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
    )


def sird(
    *, initial_values=None, time_span=(0.0, 200.0), beta=0.3, gamma=0.1, eta=0.005
):
    """SIRD model."""
    if initial_values is None:
        initial_values = backend.numpy.asarray([998.0, 1.0, 1.0, 0.0])

    parameters = (beta, gamma, eta, backend.numpy.sum(initial_values[0]))

    return _InitialValueProblem(
        vector_field=vector_fields.sird,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
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
        initial_values = _lorenz96_chaotic_u0(
            forcing=forcing, num_variables=num_variables, perturb=perturb
        )

    return _InitialValueProblem(
        vector_field=vector_fields.lorenz96,
        vector_field_args=(forcing,),
        initial_values=initial_values,
        time_span=time_span,
    )


def lorenz63(
    *,
    initial_values=None,
    time_span=(0.0, 20.0),
    parameters=(10.0, 28.0, 8.0 / 3.0),
):
    """Lorenz63 model."""
    if initial_values is None:
        initial_values = backend.numpy.asarray([0.0, 1.0, 1.05])

    return _InitialValueProblem(
        vector_field=vector_fields.lorenz63,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
    )


def rigid_body(
    *, time_span=(0.0, 20.0), initial_values=None, parameters=(-2.0, 1.25, -0.5)
):
    r"""Rigid body dynamics without external forces."""
    if initial_values is None:
        initial_values = backend.numpy.array([1.0, 0.0, 0.9])

    return _InitialValueProblem(
        vector_field=vector_fields.rigid_body,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
    )


def _lorenz96_chaotic_u0(*, forcing, num_variables, perturb):
    u0_equilibrium = backend.numpy.ones(num_variables) * forcing
    return backend.numpy.concatenate(
        [backend.numpy.asarray([u0_equilibrium[0] + perturb]), u0_equilibrium[1:]]
    )


def pleiades(*, initial_values=None, time_span=(0.0, 3.0)):
    """Construct the Pleiades problem in its original, second-order form."""
    if initial_values is None:
        x0 = [3.0, 3.0, -1.0, -3.0, 2.0, -2.0, 2.0]
        y0 = [3.0, -3.0, 2.0, 0.0, 0.0, -4.0, 4.0]
        dx0 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.75, -1.5]
        dy0 = [0.0, 0.0, 0.0, -1.25, 1.0, 0.0, 0.0]
        u0 = backend.numpy.asarray(x0 + y0)
        du0 = backend.numpy.asarray(dx0 + dy0)
        initial_values = (u0, du0)

    return _InitialValueProblem(
        vector_field=vector_fields.pleiades,
        initial_values=initial_values,
        time_span=time_span,
    )


pleiades.__doc__ = _docstring_utils.add_long_description(
    pleiades.__doc__, long_description=_descriptions.PLEIADES
)


def pleiades_autonomous_api(**kwargs):
    """Construct the Pleiades problem as \
    :math:`\\ddot u(t) = f(u(t), \\dot u(t))` \
    (with an unused second argument)."""  # noqa: D301
    _, initial_values, time_span, args = pleiades(**kwargs)

    return _InitialValueProblem(
        vector_field=vector_fields.pleiades_autonomous_api,
        initial_values=initial_values,
        time_span=time_span,
    )


pleiades_autonomous_api.__doc__ = _docstring_utils.add_long_description(
    pleiades_autonomous_api.__doc__, long_description=_descriptions.PLEIADES
)


pleiades_first_order = transform.second_to_first_order_auto(
    pleiades_autonomous_api,
    short_summary="""The Pleiades problem as a first-order differential equation.""",
)


def van_der_pol(*, stiffness_constant=1.0, initial_values=None, time_span=(0.0, 6.3)):
    """Construct the Van-der-Pol system as a second-order differential equation."""
    if initial_values is None:
        u0 = backend.numpy.asarray([2.0])
        du0 = backend.numpy.asarray([0.0])
        initial_values = (u0, du0)

    return _InitialValueProblem(
        vector_field=vector_fields.van_der_pol,
        vector_field_args=(stiffness_constant,),
        initial_values=initial_values,
        time_span=time_span,
    )


van_der_pol_first_order = transform.second_to_first_order_auto(
    van_der_pol,
    short_summary="""The Van-der-Pol system as a first-order differential equation.""",
)


def three_body(
    *,
    initial_values=None,
    standardised_moon_mass=0.012277471,
    time_span=(0.0, 17.0652165601579625588917206249),
):
    """Construct the restricted three-body problem as \
    a second-order differential equation."""
    if initial_values is None:
        u0 = backend.numpy.asarray([0.994, 0])
        du0 = backend.numpy.asarray([0, -2.00158510637908252240537862224])
        initial_values = (u0, du0)

    return _InitialValueProblem(
        vector_field=vector_fields.three_body,
        vector_field_args=(standardised_moon_mass,),
        initial_values=initial_values,
        time_span=time_span,
    )


_3bdocs = "The restricted three-body problem as a first-order differential equation."
three_body_first_order = transform.second_to_first_order_auto(
    three_body,
    short_summary=_3bdocs,
)


def hires(*, initial_values=None, time_span=(0.0, 321.8122)):
    """High Irradiance Response (HIRES).

    A chemical reaction involving eight reactants.
    """
    if initial_values is None:
        initial_values = backend.numpy.asarray([1.0, 0.0, 0.0, 0, 0, 0, 0, 0.0057])

    return _InitialValueProblem(
        vector_field=vector_fields.hires,
        vector_field_args=(),  # todo: move vf-params here
        initial_values=initial_values,
        time_span=time_span,
    )


def rober(*, initial_values=None, time_span=(0.0, 1e5), k1=0.04, k2=3e7, k3=1e4):
    """Rober ODE problem due to Robertson (1966)."""
    if initial_values is None:
        initial_values = backend.numpy.asarray([1.0, 0.0, 0.0])

    return _InitialValueProblem(
        vector_field=vector_fields.rober,
        vector_field_args=(k1, k2, k3),
        initial_values=initial_values,
        time_span=time_span,
    )
