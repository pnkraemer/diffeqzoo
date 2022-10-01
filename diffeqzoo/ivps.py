r"""Initial value problem examples.

This module provides a number of example implementations
of initial value problems (IVPs).
IVPs are a combination of an ordinary differential equation

.. math:: \dot u(t) = f(u(t), t, \theta)

and an initial condition :math:`u(0) = u_0`.
The initial values :math:`u_0` and the vector field :math:`f`
are known, the parameters :math:`\theta` might be known,
and :math:`u` is unknown.

The functions in this module construct implementations of this
kind of problem. They (loosely) follow the input/output rule

.. code:: python

    f, u0, (t0, tmax), param = constructor()

where the constructor is, e.g., :code:`lotka_volterra()`
or :code:`sir()`.
This API specification is only loose, because every problem is different.
For example, second-order problems implement a second-order differential
equation

.. math:: \ddot u(t) = f(u(t), \dot u(t), t, \theta)

subject to the initial conditions
:math:`u(0) = u_0` and :math:`\dot u(0) = u_1`.
For these problems (e.g., :code:`three_body_restricted()` or :code:`van_der_pol()`),
there are two initial values:

.. code:: python

    f, (u0, u1), (t0, tmax), param = constructor()

We try to stick as closely as possible to the above signature,
but if problem-specific issues arise, we allow ourselves to deviate from
this specification.
When in doubt, consult the documentation of the respect constructor function.
"""
from typing import Any, Callable, Iterable, NamedTuple, Union

from diffeqzoo import _vector_fields, backend, transform


class _InitialValueProblem(NamedTuple):
    vector_field: Callable
    initial_values: Union[Iterable, Any]  # u0 or (u0, du0, ddu0, ...)
    time_span: Iterable  # (t0, t1)
    vector_field_args: Iterable = ()


def lotka_volterra(
    *, initial_values=None, time_span=(0.0, 20.0), parameters=(0.5, 0.05, 0.5, 0.05)
):
    """Construct the Lotka--Volterra / predator-prey model.

    The Lotka--Volterra equations describe the dynamics of biological systems
    in which two species, predators and prey, interact.

    The original version is due to Lotka (1910).
    Its application to predator-Prey dynamics is due to Lotka (1925).
    The same model was discovered by Volterra (1926).

    .. collapse:: BibTex for Lotka (1910)

        .. code-block:: tex

            @article{lotka1910contribution,
                title={Contribution to the theory of periodic reactions},
                author={Lotka, Alfred J},
                journal={The Journal of Physical Chemistry},
                volume={14},
                number={3},
                pages={271--274},
                year={1910},
                publisher={ACS Publications}
            }

    .. collapse:: BibTex for Lotka (1925)

        .. code-block:: tex

            @book{lotka1925elements,
                title={Elements of physical biology},
                author={Lotka, Alfred James},
                year={1925},
                publisher={Williams \& Wilkins}
            }

    .. collapse:: BibTex for Volterra (1926)

        .. code-block:: tex

            @book{volterra1926variazioni,
                title={Variazioni e fluttuazioni del numero d'individui in specie animali conviventi},
                author={Volterra, Vito},
                year={1926},
                publisher={Societ{\`a} anonima tipografica" Leonardo da Vinci"}
            }

    """
    if initial_values is None:
        initial_values = backend.numpy.asarray([20.0, 20.0])

    return _InitialValueProblem(
        vector_field=_vector_fields.lotka_volterra,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
    )


def fitzhugh_nagumo(
    *, initial_values=None, time_span=(0.0, 20.0), parameters=(0.2, 0.2, 3.0)
):
    r"""Construct the FitzHugh-Nagumo model.

    The FitzHugh-Nagumo model is a simple example of an excitable system
    (for example: a neuron).
    This simplified, 2d-version of the Hodgkin-Huxley model
    (which describes the spike generation in squid giant axons)
    was suggested by FitzHugh (1961) and Nagumo et al. (1962)

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
        initial_values = backend.numpy.asarray([-1.0, 1.0])

    return _InitialValueProblem(
        vector_field=_vector_fields.fitzhugh_nagumo,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
    )


def logistic(*, initial_values=None, time_span=(0.0, 2.5), parameters=(1.0, 1.0)):
    """Construct the logistic ODE model.

    The logistic ODE is a differential equation model whose solution
    exhibits exponential growth early in the time interval,
    and approaches a constant value over time.

    It is a differential equation version of the sigmoid and the logistic function.
    The logistic ODE has a closed-form solution.

    .. note::
        **Help wanted!**

        If you know which paper/book to cite when the logistic ODE is used
        in a paper, please consider making a contribution.

    """
    if initial_values is None:
        initial_values = backend.numpy.asarray(0.1)

    return _InitialValueProblem(
        vector_field=_vector_fields.logistic,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
    )


def sir(*, initial_values=None, time_span=(0.0, 200.0), beta=0.3, gamma=0.1):
    """Construct the SIR model without vital dynamics.

    The SIR model describes the spread of a virus in a population.
    More specifically, it describes how populations move from being
    susceptible, to being infected, to being removed from the population.

    It was first proposed by Kermack and McKendrick (1927).

    .. collapse:: BibTex for Kermack and McKendrick (1927).

        .. code-block:: tex

            @article{kermack1927contribution,
                title={A contribution to the mathematical theory of epidemics},
                author={Kermack, William Ogilvy and McKendrick, Anderson G},
                journal={Proceedings of the Royal Society of London. Series A},
                volume={115},
                number={772},
                pages={700--721},
                year={1927},
                publisher={The Royal Society London}
            }


    See Also
    --------
    ivps.seir
    ivps.sird

    """
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
    """Construct the SEIR model.

    The SEIR model is a variant of the SIR model,
    but additionally includes a compartment of the population that
    has been exposed to the virus (but is not infected yet).
    See Hethcode (2000).

    .. collapse:: BibTex for Hethcote (2000).

        .. code-block:: tex

            @article{hethcote2000mathematics,
                title={The Mathematics of Infectious Diseases},
                author={Hethcote, Herbert W},
                journal={SIAM Review},
                volume={42},
                number={4},
                pages={599--653},
                year={2000},
                publisher={SIAM}
            }

    Note
    ----
    If you know a more suitable original reference, please make some noise!

    See Also
    --------
    ivps.sir
    ivps.sird

    """
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
    """Construct the SIRD model.

    The SIRD model is a variant of the SIR model that
    distinguishes the recovered compartment from the deceased compartment
    in the population.
    See Hethcode (2000).

    .. collapse:: BibTex for Hethcote (2000).

        .. code-block:: tex

            @article{hethcote2000mathematics,
                title={The Mathematics of Infectious Diseases},
                author={Hethcote, Herbert W},
                journal={SIAM Review},
                volume={42},
                number={4},
                pages={599--653},
                year={2000},
                publisher={SIAM}
            }

    Note
    ----
    If you know a more suitable original reference, please make some noise!

    See Also
    --------
    ivps.sir
    ivps.seir
    """
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
    """Construct the Lorenz96 model.

    The Lorenz96 is a chaotic initial value problem, due to Lorenz (1996),
    and commonly used as a testproblem in data assimilation.

    .. collapse:: BibTex for Lorenz (1996)

        .. code-block:: tex

            @inproceedings{lorenz1996predictability,
                title={Predictability: A problem partly solved},
                author={Lorenz, Edward N},
                booktitle={Proceedings of the Seminar on Predictability},
                volume={1},
                number={1},
                year={1996}
            }

    """
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
    """Construct the Lorenz63 model.

    The Lorenz63 model, initially used for atmospheric convection,
    is a common example of an initial value problem that
    has a chaotic solution.

    It was proposed by Lorenz (1963).

    .. collapse:: BibTex for Lorenz (1963)

        .. code-block:: tex

            @article{lorenz1963deterministic,
                title={Deterministic nonperiodic flow},
                author={Lorenz, Edward N},
                journal={Journal of atmospheric sciences},
                volume={20},
                number={2},
                pages={130--141},
                year={1963}
            }
    """
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
    r"""Construct the rigid body dynamics without external forces.

    The rigid body dynamics from classical mechanics,
    or "Euler's rotation equations",
    describe the rotation of a rigid body in three-dimensional, principal,
    orthogonal coordinates.


    A common citation for the Pleiades problem is p. 244 in the book
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
    If you know a more suitable original reference, please make some noise!

    """
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
    If you know a more suitable original reference, please make some noise!

    See Also
    --------
    diffeqzoo.ivps.pleiades
    diffeqzoo.ivps.pleiades_autonomous_api
    diffeqzoo.ivps.pleiades_first_order

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
    diffeqzoo.ivps.pleiades
    diffeqzoo.ivps.pleiades_autonomous_api
    diffeqzoo.ivps.pleiades_first_order

    """  # noqa: D301
    _, initial_values, time_span, args = pleiades(**kwargs)

    return _InitialValueProblem(
        vector_field=_vector_fields.pleiades_autonomous_api,
        initial_values=initial_values,
        time_span=time_span,
    )


pleiades_first_order = transform.second_to_first_order_auto(
    pleiades_autonomous_api,
    short_summary="Construct the Pleiades problem as a first-order differential equation.",
)


def van_der_pol(*, stiffness_constant=1.0, initial_values=None, time_span=(0.0, 6.3)):
    r"""Construct the Van-der-Pol system as a second-order differential equation.

    The Van-der-Pol system is a non-conservative oscillator subject to non-linear damping.
    It is a popular benchmark problem, because it involves a parameter :math:`\mu`
    (the "stiffness constant") which governs the stiffness of the problem.
    For :math:`\mu ~ 1`, the problen is not stiff.
    For large values (e.g. :math:`\mu ~ 10^6`) the problem is stiff.
    It was first published by Van der Pol (1920).

    .. collapse:: BibTex for Van der Pol (1920).

        .. code-block:: tex

            @article{van1920theory,
                title={Theory of the amplitude of free and forced triode vibrations},
                author={Van der Pol, Balthasar},
                journal={Radio Review},
                volume={1},
                pages={701--710},
                year={1920}
            }

    """
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
    short_summary="Construct the Van-der-Pol system as a first-order differential equation.",
)


def three_body_restricted(
    *,
    initial_values=None,
    standardised_moon_mass=0.012277471,
    time_span=(0.0, 17.0652165601579625588917206249),
):
    """Construct the restricted three-body problem as \
    a second-order differential equation.

    The restricted three-body problem describes how
    a body of negligible mass moves under the influence of two massive bodies.
    It can be described in terms of two-body motion.

    It is commonly pointed to p. 129 of Hairer et al. (1993) as the first reference.

    .. collapse:: BibTex for Hairer et al. (1993)

        .. code-block:: tex

            @book{hairer1993solving,
                title={Solving Ordinary Differential equations I, Nonstiff Problems},
                author={Hairer, Ernst and N{\o}rsett, Syvert P and Wanner, Gerhard},
                year={1993},
                publisher={Springer}
                edition={2}
            }

    """
    if initial_values is None:
        u0 = backend.numpy.asarray([0.994, 0])
        du0 = backend.numpy.asarray([0, -2.00158510637908252240537862224])
        initial_values = (u0, du0)

    return _InitialValueProblem(
        vector_field=_vector_fields.three_body_restricted,
        vector_field_args=(standardised_moon_mass,),
        initial_values=initial_values,
        time_span=time_span,
    )


_3bdocs = "Construct the restricted three-body problem as a first-order differential equation."
three_body_restricted_first_order = transform.second_to_first_order_auto(
    three_body_restricted,
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
