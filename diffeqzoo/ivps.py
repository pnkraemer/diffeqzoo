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
    *,
    initial_values=(20.0, 20.0),
    time_span=(0.0, 20.0),
    parameters=(0.5, 0.05, 0.5, 0.05),
):
    r"""Construct the Lotka--Volterra / predator-prey model.

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
    initial_values = backend.numpy.asarray(initial_values)

    return _InitialValueProblem(
        vector_field=_vector_fields.lotka_volterra,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
    )


def fitzhugh_nagumo(
    *, initial_values=(-1.0, 1.0), time_span=(0.0, 20.0), parameters=(0.2, 0.2, 3.0)
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
    initial_values = backend.numpy.asarray(initial_values)

    return _InitialValueProblem(
        vector_field=_vector_fields.fitzhugh_nagumo,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
    )


def logistic(*, initial_value=0.1, time_span=(0.0, 2.5), parameters=(1.0, 1.0)):
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
    initial_value = backend.numpy.asarray(initial_value)

    return _InitialValueProblem(
        vector_field=_vector_fields.logistic,
        vector_field_args=parameters,
        initial_values=initial_value,
        time_span=time_span,
    )


def sir(
    *, initial_values=(998.0, 1.0, 1.0), time_span=(0.0, 200.0), beta=0.3, gamma=0.1
):
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

    initial_values = backend.numpy.asarray(initial_values)
    parameters = (beta, gamma, backend.numpy.sum(initial_values[0]))

    return _InitialValueProblem(
        vector_field=_vector_fields.sir,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
    )


def seir(
    *,
    initial_values=(998.0, 1.0, 1.0, 1.0),
    time_span=(0.0, 200.0),
    alpha=0.3,
    beta=0.3,
    gamma=0.1,
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

    initial_values = backend.numpy.asarray(initial_values)
    parameters = (alpha, beta, gamma, backend.numpy.sum(initial_values[0]))

    return _InitialValueProblem(
        vector_field=_vector_fields.seir,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
    )


def sird(
    *,
    initial_values=(998.0, 1.0, 1.0, 0.0),
    time_span=(0.0, 200.0),
    beta=0.3,
    gamma=0.1,
    eta=0.005,
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

    initial_values = backend.numpy.asarray(initial_values)
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
    initial_values=(0.0, 1.0, 1.05),
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
    initial_values = backend.numpy.asarray(initial_values)

    return _InitialValueProblem(
        vector_field=_vector_fields.lorenz63,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
    )


def roessler(
    *,
    initial_values=(1.0, 0.0, 0.0),
    time_span=(0.0, 100.0),
    parameters=(0.1, 0.1, 14.0),
):
    """Construct the Roessler model.

    The Roessler model is a three-dimensional chaotic system,
    that was proposed by Roessler (1990).

    .. collapse:: BibTex for Roessler (1990)

        .. code-block:: tex

            @article{rossler1976equation,
                title={An equation for continuous chaos},
                author={R{\"o}ssler, Otto E},
                journal={Physics Letters A},
                volume={57},
                number={5},
                pages={397--398},
                year={1976},
                publisher={Elsevier}
            }
    """
    initial_values = backend.numpy.asarray(initial_values)

    return _InitialValueProblem(
        vector_field=_vector_fields.roessler,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
    )


def rigid_body(
    *,
    time_span=(0.0, 20.0),
    initial_values=(1.0, 0.0, 0.9),
    parameters=(-2.0, 1.25, -0.5),
):
    r"""Construct the rigid body dynamics without external forces.

    The rigid body dynamics from classical mechanics,
    or "Euler's rotation equations",
    describe the rotation of a rigid body in three-dimensional, principal,
    orthogonal coordinates.


    A common citation for the rigid-body problem is p. 244 in the book
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
    initial_values = backend.numpy.asarray(initial_values)

    return _InitialValueProblem(
        vector_field=_vector_fields.rigid_body,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
    )


# Pleiades initial values
_X0 = (3.0, 3.0, -1.0, -3.0, 2.0, -2.0, 2.0)
_Y0 = (3.0, -3.0, 2.0, 0.0, 0.0, -4.0, 4.0)
_DX0 = (0.0, 0.0, 0.0, 0.0, 0.0, 1.75, -1.5)
_DY0 = (0.0, 0.0, 0.0, -1.25, 1.0, 0.0, 0.0)
_U0 = _X0 + _Y0
_DU0 = _DX0 + _DY0


def pleiades(*, initial_values=(_U0, _DU0), time_span=(0.0, 3.0)):
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
    diffeqzoo.ivps.pleiades_with_unused_derivative_argument
    diffeqzoo.ivps.pleiades_first_order

    """
    u0, du0 = initial_values
    u0 = backend.numpy.asarray(u0)
    du0 = backend.numpy.asarray(du0)
    initial_values = (u0, du0)

    return _InitialValueProblem(
        vector_field=_vector_fields.pleiades,
        initial_values=initial_values,
        time_span=time_span,
    )


def pleiades_with_unused_derivative_argument(**kwargs):
    """Construct the Pleiades problem as \
    :math:`\\ddot u(t) = f(u(t), \\dot u(t))` \
    (with an unused second argument).

    See :func:`pleiades` for a more detailed problem description.

    See Also
    --------
    diffeqzoo.ivps.pleiades
    diffeqzoo.ivps.pleiades_with_unused_derivative_argument
    diffeqzoo.ivps.pleiades_first_order

    """  # noqa: D301
    _, initial_values, time_span, args = pleiades(**kwargs)

    return _InitialValueProblem(
        vector_field=_vector_fields.pleiades_with_unused_derivative_argument,
        initial_values=initial_values,
        time_span=time_span,
    )


pleiades_first_order = transform.second_to_first_order_auto(
    pleiades_with_unused_derivative_argument,
    short_summary="Construct the Pleiades problem as a first-order differential equation.",
)

_HENON_HEILES_INITS = ((0.5, 0.0), (0.0, 0.1))


def henon_heiles(*, initial_values=_HENON_HEILES_INITS, time_span=(0.0, 100.0), p=1.0):
    r"""Construct the Henon-Heiles problem.

    The Henon-Heiles problem relates to the non-linear motion
    of a star around a galactic center with the motion restricted to a plane.
    It is a 2-dimensional, second-order differential equation
    and commonly solved as a 4-dimensional, first-order equation.
    In in its original, second-order form, it is

    .. math::
        \ddot u(t) = f(u(t)),

    with nonlinear dynamics :math:`f: \mathbb{R}^{2} \rightarrow  \mathbb{R}^{2}`.

    The Henon-Heiles problem is not stiff.
    It is a popular benchmark problem because of its well-known Hamiltonian,
    which makes it a good test for symplectic integrators.

    The Henon-Heiles problem is due to Henon and Heiles (1964).


    .. collapse:: BibTex for Henon and Heiles (1964)

        .. code-block:: tex

            @article{henon1964applicability,
                title={The applicability of the third integral of motion: some numerical experiments},
                author={H{\'e}non, Michel and Heiles, Carl},
                journal={The astronomical journal},
                volume={69},
                pages={73},
                year={1964}
            }

    See Also
    --------
    diffeqzoo.ivps.henon_heiles
    diffeqzoo.ivps.henon_heiles_with_unused_derivative_argument
    diffeqzoo.ivps.henon_heiles_first_order

    """
    u0, du0 = initial_values
    initial_values = (backend.numpy.asarray(u0), backend.numpy.asarray(du0))

    return _InitialValueProblem(
        vector_field=_vector_fields.henon_heiles,
        initial_values=initial_values,
        time_span=time_span,
        vector_field_args=(p,),
    )


def henon_heiles_with_unused_derivative_argument(**kwargs):
    """Construct the Henon-Heiles problem as \
    :math:`\\ddot u(t) = f(u(t), \\dot u(t))` \
    (with an unused second argument).

    See :func:`henon_heiles` for a more detailed problem description.

    See Also
    --------
    diffeqzoo.ivps.henon_heiles
    diffeqzoo.ivps.henon_heiles_with_unused_derivative_argument
    diffeqzoo.ivps.henon_heiles_first_order

    """  # noqa: D301
    _, initial_values, time_span, args = henon_heiles(**kwargs)

    return _InitialValueProblem(
        vector_field=_vector_fields.henon_heiles_with_unused_derivative_argument,
        initial_values=initial_values,
        time_span=time_span,
        vector_field_args=args,
    )


henon_heiles_first_order = transform.second_to_first_order_auto(
    henon_heiles_with_unused_derivative_argument,
    short_summary="Construct the Henon-Heiles problem as a first-order differential equation.",
)


def van_der_pol(
    *, stiffness_constant=1.0, initial_values=(2.0, 0.0), time_span=(0.0, 6.3)
):
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

    u0, du0 = initial_values
    u0 = backend.numpy.asarray(u0)
    du0 = backend.numpy.asarray(du0)
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


_Y0_3 = (0.994, 0)
_DY0_3 = (0, -2.00158510637908252240537862224)


def three_body_restricted(
    *,
    initial_values=(_Y0_3, _DY0_3),
    standardised_moon_mass=0.012277471,
    time_span=(0.0, 17.0652165601579625588917206249),
):
    r"""Construct the restricted three-body problem as \
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

    u0, du0 = initial_values
    u0 = backend.numpy.asarray(u0)
    du0 = backend.numpy.asarray(du0)
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


def hires(
    *, initial_values=(1.0, 0.0, 0.0, 0, 0, 0, 0, 0.0057), time_span=(0.0, 321.8122)
):
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
    initial_values = backend.numpy.asarray(initial_values)

    return _InitialValueProblem(
        vector_field=_vector_fields.hires,
        vector_field_args=(),  # todo: move vf-params here
        initial_values=initial_values,
        time_span=time_span,
    )


def rober(
    *, initial_values=(1.0, 0.0, 0.0), time_span=(0.0, 1e5), k1=0.04, k2=3e7, k3=1e4
):
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
    initial_values = backend.numpy.asarray(initial_values)

    return _InitialValueProblem(
        vector_field=_vector_fields.rober,
        vector_field_args=(k1, k2, k3),
        initial_values=initial_values,
        time_span=time_span,
    )


def affine_independent(*, initial_values=1.0, time_span=(0.0, 1.0), a=1.0, b=0.0):
    r"""Construct an IVP with an affine vector field, \
    where each dimension is treated independently.

    In Python code, this means :code:`f(y, a, b)=a * y + b`.

    By default, this is a scalar problem.
    Change the initial value to make this a multidimensional problem.
    """
    initial_values = backend.numpy.asarray(initial_values)

    return _InitialValueProblem(
        vector_field=_vector_fields.affine_independent,
        vector_field_args=(a, b),
        initial_values=initial_values,
        time_span=time_span,
    )


def affine_dependent(
    *, initial_values=(1.0, 1.0), time_span=(0.0, 1.0), A=((1, 0), (0, 1)), b=(0, 0)
):
    r"""Construct an IVP with an affine vector field.

    In Python code, this means :code:`f(y, A, b) = A @ y + b`.

    By default, this is a 2d-problem.
    Change the initial value to make this a multidimensional problem.
    """

    initial_values = backend.numpy.asarray(initial_values)
    A = backend.numpy.asarray(A)
    b = backend.numpy.asarray(b)

    return _InitialValueProblem(
        vector_field=_vector_fields.affine_dependent,
        vector_field_args=(A, b),
        initial_values=initial_values,
        time_span=time_span,
    )


def oregonator(
    *,
    initial_values=(1.0, 2.0, 3.0),
    time_span=(0.0, 1e5),
    s=77.27,
    q=8.375e-6,
    w=0.161,
):
    r"""Construct the scaled Oregonator Mass-Action dynamics \
    in a well-stirred, homogeneous system.

    What is often referred to as the "Oregonator" problem is a simplified
    model of the chemical dynamics of the oscillatory Belousov-Zhabotinsky
    reaction and due to Fields and Noyes (1974).
    It is a three-dimensional, stiff initial value problem,

    .. math::
        \dot u(t) = f(u(t))

    and a common test problem for numerical solvers for stiff differential equations.

    .. collapse:: BibTex for Fields and Noyes (1974)

        .. code-block:: tex

            @article{field1974oscillations,
                title={Oscillations in chemical systems. IV. Limit cycle behavior in a model of a real chemical reaction},
                author={Field, Richard J and Noyes, Richard M},
                journal={The Journal of Chemical Physics},
                volume={60},
                number={5},
                pages={1877--1884},
                year={1974},
                publisher={American Institute of Physics}
            }

    """

    initial_values = backend.numpy.asarray(initial_values)

    return _InitialValueProblem(
        vector_field=_vector_fields.rober,
        vector_field_args=(s, w, q),
        initial_values=initial_values,
        time_span=time_span,
    )


def goodwin(
    *,
    initial_values=(0.0, 0.0),
    time_span=(0.0, 25.0),
    r=10,
    a1=1.0,
    a2=3.0,
    alpha=0.5,
    k=(1.0,),
):
    r"""Construct the Goodwin Oscillator dynamics.

    Describes a mechanism for periodic protein expression described by Goodwin (1965).
    The first dimension describes the mRNA concentration,
    the last dimension the protein inhibiting mRNA production,
    and the remaining dimensions correspond to intermediate protein species.
    r > 8 leads to oscillatory behavior.
    It is a n-dimensional ODE initial value problem.
    The length of the `k` needs to be `len(initial_values)-1`.

    Common problem for parameter inference, where the posterior has a multimodal distribution.


    .. collapse:: BibTex for Goodwin (1965)

        .. code-block:: tex

            @article{goodwin1965oscillatory,
                title = {Oscillatory behavior in enzymatic control processes},
                journal = {Advances in Enzyme Regulation},
                volume = {3},
                pages = {425-437},
                year = {1965},
                author = {Brian C. Goodwin},
            }
    """

    initial_values = backend.numpy.asarray(initial_values)

    k = backend.numpy.asarray(k)
    return _InitialValueProblem(
        vector_field=_vector_fields.goodwin,
        vector_field_args=(r, a1, a2, alpha, k),
        initial_values=initial_values,
        time_span=time_span,
    )


def nonlinear_chemical_reaction(
    *,
    initial_values=(1.0, 0.0, 0.0),
    time_span=(0.0, 1.0),
    k1=1,
    k2=1,
):
    r"""Construct the Nonlinear Chemical Reaction dynamics.

    We use the version described by Liu et al. (2012).

    .. collapse:: BibTex for Liu et al. (2012)

        .. code-block:: tex

            @article{liu2012analytic,
                title={Analytic solution for a nonlinear chemistry system of ordinary differential equations},
                author={Liu, Li-Cai and Tian, Bo and Xue, Yu-Shan and Wang, Ming and Liu, Wen-Jun},
                journal={Nonlinear Dynamics},
                volume={68},
                number={1},
                pages={17--21},
                year={2012},
                publisher={Springer}
            }


    Note
    ----
    If you know a better source of this (class of) problem(s), please consider making a pull request.
    """

    initial_values = backend.numpy.asarray(initial_values)

    return _InitialValueProblem(
        vector_field=_vector_fields.nonlinear_chemical_reaction,
        vector_field_args=(k1, k2),
        initial_values=initial_values,
        time_span=time_span,
    )


def neural_ode_mlp(
    *,
    initial_values=((0.0,)),
    time_span=(0.0, 1.0),
    scale=1.0,
    layer_sizes=(2, 20, 1),
    seed=1234,
):
    r"""Construct an IVP with a neural ODE vector field.

    The vector field is given by a neural network.

    The neural network is a multi-layer perceptron with `tanh` activation functions.

    We implement the dynamics used in the "implicit-layers" tutorial:
    ``http://implicit-layers-tutorial.org/neural_odes/``.

    Note
    ----
    The neural network is not trained in this function. This function only constructs the IVP.
    """
    initial_values = backend.numpy.asarray(initial_values)
    params = _init_random_params(scale, layer_sizes, seed)
    return _InitialValueProblem(
        vector_field=_vector_fields.neural_ode_mlp,
        vector_field_args=(params,),
        initial_values=initial_values,
        time_span=time_span,
    )


def _init_random_params(scale, layer_sizes, seed):
    if backend.name == "jax":
        params = _init_random_params_jax(
            scale=scale, layer_sizes=layer_sizes, seed=seed
        )
    elif backend.name == "numpy":
        params = _init_random_params_numpy(
            scale=scale, layer_sizes=layer_sizes, seed=seed
        )
    else:
        msg1 = f"Neural ODE is not compatible with the current backend {backend.name}. "
        msg2 = "Please use `jax` or `numpy` and/or consider raising an Issue."
        raise ValueError(msg1 + msg2)
    return params


def _init_random_params_numpy(*, scale, layer_sizes, seed):
    rng = backend.random.default_rng(seed=seed)
    return [
        (
            scale * rng.standard_normal(size=(m, n)),
            scale * rng.standard_normal(size=(n,)),
        )
        for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])
    ]


def _init_random_params_jax(*, scale, layer_sizes, seed):
    key = backend.random.PRNGKey(seed=seed)
    return [
        (
            scale * backend.random.normal(key, shape=(m, n)),
            scale * backend.random.normal(key, shape=(n,)),
        )
        for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])
    ]
