"""Initial value problem examples."""
from typing import Any, Callable, Iterable, NamedTuple, Union

from odezoo import _vector_fields, backend, transform


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
        vector_field=_vector_fields.lotka_volterra,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
    )


def fitzhugh_nagumo(
    *, initial_values=None, time_span=(0.0, 20.0), parameters=(0.2, 0.2, 3.0, 1.0)
):
    r"""Construct the FitzHugh-Nagumo model.

    The FitzHugh-Nagumo model is a simple example of an excitable system
    (for example: a neuron).
    This simplified, 2d-version of the Hodgkin-Huxley model
    (which describes the spike generation in squid giant axons)
    was suggested by FitzHugh (1961) and Nagumo et al. (1962)

    It is a non-stiff, first-order problem,

    .. math::
        \dot u(t) = f(u(t), \theta)

    and generally easy to solve by most ODE solvers.

    The following bibtex(s) point to the original papers about
    the FitzHugh-Nagumo models. (Source: Google Scholar).

    .. collapse:: BibTex for FitzHugh (1961)

        .. code-block:: tex

            @article{fitzhugh1961impulses,
                title={Impulses and physiological states in
                theoretical models of nerve membrane},
                author={FitzHugh, Richard},
                journal={Biophysical Journal},
                volume={1},
                number={6},
                pages={445--466},
                year={1961},
                publisher={Elsevier}
            }

    .. collapse:: BibTex for Nagumo et al. (1962)

        .. code-block:: tex

            @article{nagumo1962active,
                title={An active pulse transmission line simulating nerve axon},
                author={Nagumo, Jinichi and Arimoto, Suguru and Yoshizawa, Shuji},
                journal={Proceedings of the IRE},
                volume={50},
                number={10},
                pages={2061--2070},
                year={1962},
                publisher={IEEE}
            }

    """
    if initial_values is None:
        initial_values = backend.numpy.asarray([1.0, -1.0])

    return _InitialValueProblem(
        vector_field=_vector_fields.fitzhugh_nagumo,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
    )


def logistic(*, initial_values=None, time_span=(0.0, 2.5), parameters=(1.0, 1.0)):
    """Logistic ODE model."""
    if initial_values is None:
        initial_values = backend.numpy.asarray(0.1)

    return _InitialValueProblem(
        vector_field=_vector_fields.logistic,
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
        vector_field=_vector_fields.sir,
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
        vector_field=_vector_fields.seir,
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
        vector_field=_vector_fields.sird,
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
        vector_field=_vector_fields.lorenz96,
        vector_field_args=(forcing,),
        initial_values=initial_values,
        time_span=time_span,
    )


def _lorenz96_chaotic_u0(*, forcing, num_variables, perturb):
    u0_equilibrium = backend.numpy.ones(num_variables) * forcing
    return backend.numpy.concatenate(
        [backend.numpy.asarray([u0_equilibrium[0] + perturb]), u0_equilibrium[1:]]
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
        vector_field=_vector_fields.lorenz63,
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
        vector_field=_vector_fields.rigid_body,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
    )


def pleiades(*, initial_values=None, time_span=(0.0, 3.0)):
    r"""Construct the Pleiades problem in its original, second-order form.

    The Pleiades problem from celestial mechanics describes the
    gravitational interaction(s) of seven stars
    (the "Pleiades", or "Seven Sisters") in a plane.
    It is a 14-dimensional, second-order differential equation
    and commonly solved as a 28-dimensional, first-order equation.
    In in its original, second-order form, it is

    .. math::
        \ddot u(t) = f(u(t)),

    with nonlinear dynamics :math:`f: \mathbb{R}^{14} \rightarrow  \mathbb{R}^{14}`.

    The Pleiades problem is not stiff.
    It is a popular benchmark problem because
    it is not very difficult to solve numerically, but
    (a) it requires high accuracy in each ODE solver step, and
    (b) its 14 (or 28) dimensions start to expose those numerical solvers
    that do not scale well to high dimensions.

    A common citation for the Pleiades problem is p. 245 in the book
    by Hairer et al. (1993):

    .. collapse:: BibTex for Hairer et al. (1993)

        .. code-block:: tex

            @book{hairer1993solving,
                title={Solving Ordinary Differential equations I, Nonstiff Problems},
                author={Hairer, Ernst and N{\o}rsett, Syvert P and Wanner, Gerhard},
                year={1993},
                publisher={Springer}
                edition={2}
            }

    Note
    ----
    If you know a better source, please make some noise!

    See Also
    --------
    odezoo.ivps.pleiades
    odezoo.ivps.pleiades_autonomous_api
    odezoo.ivps.pleiades_first_order

    """
    if initial_values is None:
        x0 = [3.0, 3.0, -1.0, -3.0, 2.0, -2.0, 2.0]
        y0 = [3.0, -3.0, 2.0, 0.0, 0.0, -4.0, 4.0]
        dx0 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.75, -1.5]
        dy0 = [0.0, 0.0, 0.0, -1.25, 1.0, 0.0, 0.0]
        u0 = backend.numpy.asarray(x0 + y0)
        du0 = backend.numpy.asarray(dx0 + dy0)
        initial_values = (u0, du0)

    return _InitialValueProblem(
        vector_field=_vector_fields.pleiades,
        initial_values=initial_values,
        time_span=time_span,
    )


def pleiades_autonomous_api(**kwargs):
    """Construct the Pleiades problem as \
    :math:`\\ddot u(t) = f(u(t), \\dot u(t))` \
    (with an unused second argument).

    See :func:`pleiades` for a more detailed problem description.

    See Also
    --------
    odezoo.ivps.pleiades
    odezoo.ivps.pleiades_autonomous_api
    odezoo.ivps.pleiades_first_order

    """  # noqa: D301
    _, initial_values, time_span, args = pleiades(**kwargs)

    return _InitialValueProblem(
        vector_field=_vector_fields.pleiades_autonomous_api,
        initial_values=initial_values,
        time_span=time_span,
    )


pleiades_first_order = transform.second_to_first_order_auto(
    pleiades_autonomous_api,
    short_summary="""The Pleiades problem as a first-order differential equation.""",
)


def van_der_pol(*, stiffness_constant=1.0, initial_values=None, time_span=(0.0, 6.3)):
    """Construct the Van-der-Pol system as a second-order differential equation."""
    if initial_values is None:
        u0 = backend.numpy.asarray(2.0)
        du0 = backend.numpy.asarray(0.0)
        initial_values = (u0, du0)

    return _InitialValueProblem(
        vector_field=_vector_fields.van_der_pol,
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
        vector_field=_vector_fields.three_body,
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
    r"""Construct the High Irradiance Response (HIRES) problem.

    The "High Irradiance Response" ODE (HIRES) from plant physiology describes how light
    is involved in morphogenesis.
    It was proposed by Schäfer (1975) and named "HIRES" by Hairer and Wanner (1996).

    It is a system of 8 nonlinear differential equations,

    .. math::
        \dot u(t) = f(u(t))

    and a common testproblem for ODE solvers that can handle stiff problems.


    The following bibtex(s) point to the original paper about
    the HIRES model and the book by Hairer and Wanner. (Source: Google Scholar).

    .. collapse:: BibTex for Schäfer (1975)

        .. code-block:: tex

            @article{schafer1975new,
                title={A new approach to explain the "high irradiance responses"
                of photomorphogenesis on the basis of phytochrome},
                author={Sch{\"a}fer, E},
                journal={Journal of Mathematical Biology},
                volume={2},
                number={1},
                pages={41--56},
                year={1975},
                publisher={Springer}
            }

    .. collapse:: BibTex for Hairer and Wanner (1996)

        .. code-block:: tex

            @book{hairer1996solving,
                title={Solving Ordinary Differential Equations II,
                Stiff and Differential-Algebraic Problems},
                author={Hairer, Ernst and Wanner, Gerhard},
                year={1996},
                publisher={Springer}
            }

    """
    if initial_values is None:
        initial_values = backend.numpy.asarray([1.0, 0.0, 0.0, 0, 0, 0, 0, 0.0057])

    return _InitialValueProblem(
        vector_field=_vector_fields.hires,
        vector_field_args=(),  # todo: move vf-params here
        initial_values=initial_values,
        time_span=time_span,
    )


def rober(*, initial_values=None, time_span=(0.0, 1e5), k1=0.04, k2=3e7, k3=1e4):
    r"""Construct the ROBER problem due to Robertson (1966).

    The ROBER problem describes the kinetics of an autocatalytic reaction,
    and was proposed by Robertson (1966).
    It was named "ROBER" by Hairer and Wanner (1996).

    It is a three-dimensional, stiff initial value problem,

    .. math::
        \dot u(t) = f(u(t))

    and a common test problem for numerical solvers for stiff differential equations.

    The following bibtex(s) point to the original paper about
    the ROBER model and the book by Hairer and Wanner. (Source: Google Scholar).

    .. collapse:: BibTex for Robertson (1966)

        .. code-block:: tex

            @article{robertson1966solution,
                title={The solution of a set of reaction rate equations},
                author={Robertson, HH},
                journal={Numerical Analysis: An Introduction},
                publisher={Academic Press},
                year={1966},
                pages={178-182},
            }

    .. collapse:: BibTex for Hairer and Wanner (1996)

        .. code-block:: tex

            @book{wanner1996solving,
                title={Solving Ordinary Differential Equations II,
                Stiff and Differential-Algebraic Problems},
                author={Hairer, Ernst and Wanner, Gerhard},
                year={1996},
                publisher={Springer}
            }

    """
    if initial_values is None:
        initial_values = backend.numpy.asarray([1.0, 0.0, 0.0])

    return _InitialValueProblem(
        vector_field=_vector_fields.rober,
        vector_field_args=(k1, k2, k3),
        initial_values=initial_values,
        time_span=time_span,
    )
