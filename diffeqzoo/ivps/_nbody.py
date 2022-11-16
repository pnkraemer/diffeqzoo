"""N-Body problems and celestial mechanics."""

from diffeqzoo import _vector_fields, backend, transform
from diffeqzoo.ivps import _ivp


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

    return _ivp.InitialValueProblem(
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

    return _ivp.InitialValueProblem(
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

    return _ivp.InitialValueProblem(
        vector_field=_vector_fields.pleiades_with_unused_derivative_argument,
        initial_values=initial_values,
        time_span=time_span,
        vector_field_args=args,
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

    return _ivp.InitialValueProblem(
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

    return _ivp.InitialValueProblem(
        vector_field=_vector_fields.henon_heiles_with_unused_derivative_argument,
        initial_values=initial_values,
        time_span=time_span,
        vector_field_args=args,
    )


henon_heiles_first_order = transform.second_to_first_order_auto(
    henon_heiles_with_unused_derivative_argument,
    short_summary="Construct the Henon-Heiles problem as a first-order differential equation.",
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

    return _ivp.InitialValueProblem(
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
