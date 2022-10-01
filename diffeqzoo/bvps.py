r"""Boundary value problem examples.

This module provides a number of example implementations
of boundary value problems (BVPs).
BVPs are a combination of a (commonly second-order)
ordinary differential equation

.. math:: \ddot u(t) = f(u(t), \dot u(t), t, \theta)

and a set of boundary conditions: often, it is either a two-point
boundary condition, :math:`g_0(u(0)) = g_1(u(1)) = 0`, or a
general boundary condition, :math:`g(u(0), u(1)) = 0`.
The boundary conditions :math:`g_0`, :math:`g_1`, or :math:`g`,
and the vector field :math:`f`
are known, the parameters :math:`\theta` might be known,
and :math:`u` is unknown.


The functions in this module construct implementations of this
kind of problem. They (loosely) follow the input/output rule

.. code:: python

    f, g, (t0, tmax), param = constructor()
    f, (g0, g1), (t0, tmax), param = constructor_two_point()

where the constructor is, e.g., :code:`pendulum()`
or :code:`measles()`.
This API specification is only loose, because every problem is different.
For example, first-order problems implement a differential
equation

.. math:: \dot u(t) = f(u(t), t, \theta).

We try to stick as closely as possible to the above signature,
but if problem-specific issues arise, we allow ourselves to deviate from
this specification.
When in doubt, consult the documentation of the respect constructor function.
Each function's documentation also explains whether the problem is a two-point
boundary value problem, and whether
the differential equation is first-, second-, or higher order.

"""

import math  # for PI
from typing import Any, Callable, Iterable, NamedTuple, Union

from diffeqzoo import _vector_fields, backend, transform


class _BoundaryValueProblem(NamedTuple):
    vector_field: Callable
    boundary_conditions: Iterable  # (g0, g1) or ((L, l), (R, r))
    time_span: Iterable  # (t0, t1)
    vector_field_args: Iterable = ()


def bratu(*, time_span=(0.0, 1.0), parameters=(1.0,)):
    r"""Construct Bratu's problem.

    Bratu's problem consists of a second-order differential equation
    and two-point boundary conditions.
    It is a common example problem to showcase BVP solvers and due to Bratu (1913).

    .. collapse:: BibTex for Bratu (1913)

        .. code-block:: tex

            @article{bratu1913equations,
                title={Sur les {\'e}quations int{\'e}grales non lin{\'e}aires},
                author={Bratu, G},
                journal={Bulletin de la Soci{\'e}t{\'e} Math{\'e}matique de France},
                volume={41},
                pages={346--350},
                year={1913}
            }

    """

    def g0(u):
        return u

    def g1(u):
        return u

    return _BoundaryValueProblem(
        vector_field=_vector_fields.bratu,
        boundary_conditions=(g0, g1),
        vector_field_args=parameters,
        time_span=time_span,
    )


def bratu_autonomous_api(*, time_span=(0.0, 1.0), parameters=(1.0,)):
    r"""Construct Bratu's problem with a signature :math:`(u, \dot u)` /
    and an unused second argument.

    See :func:`bratu` for a more detailed problem description.
    """

    def g0(u, _):
        return u

    def g1(u, _):
        return u

    return _BoundaryValueProblem(
        vector_field=_vector_fields.bratu_autonomous_api,
        boundary_conditions=(g0, g1),
        vector_field_args=parameters,
        time_span=time_span,
    )
    return f_bratu, (t0, tmax), (u0, umax)


def pendulum_autonomous_api(*, time_span=(0.0, math.pi / 2.0), parameters=(9.81,)):
    r"""Construct the pendulum problem with a signature :math:`(u, \dot u)` /
    and an unused second argument.

    See :func:`pendulum` for a more detailed problem description.
    """

    def g0(u, _):
        return u

    def g1(u, _):
        return u

    return _BoundaryValueProblem(
        vector_field=_vector_fields.pendulum_autonomous_api,
        boundary_conditions=(g0, g1),
        vector_field_args=parameters,
        time_span=time_span,
    )


def pendulum(*, time_span=(0.0, math.pi / 2.0), parameters=(9.81,)):
    r"""Construct the pendulum problem.

    The pendulum problem consists of a second-order differential equation
    and two-point boundary conditions. It is a common example BVP
    to showcase a boundary value problem solver.

    .. note::
        **Help wanted!**

        If you know which paper/book to cite when the pendulum problem
        is used in a paper, please consider making a contribution.

    """

    def g0(u):
        return u

    def g1(u):
        return u

    return _BoundaryValueProblem(
        vector_field=_vector_fields.pendulum,
        boundary_conditions=(g0, g1),
        vector_field_args=parameters,
        time_span=time_span,
    )


def measles(*, time_span=(0.0, 1.0), mu=0.02, lmbda=0.0279, eta=0.01, beta0=1575):
    r"""Construct the Measles problem.

    The Measles problem is a first-order differential equation with
    general boundary conditions (specifically: periodic boundary conditions).
    It describes the dynamics of a seasonal disease, and is a standard BVP testproblem.

    Ascher et al. (1995) point to Schwartz (1983) as the original source.

    .. collapse:: BibTex for Schwartz (1983)

        .. code-block:: tex

            @article{schwartz1983estimating,
                title={Estimating regions of existence of unstable periodic orbits using computer-based techniques},
                author={Schwartz, Ira Bruce},
                journal={SIAM Journal on Numerical Analysis},
                volume={20},
                number={1},
                pages={106--120},
                year={1983},
                publisher={SIAM}
            }


    .. collapse:: BibTex for Ascher et al. (1995)

        .. code-block:: tex

            @book{ascher1995numerical,
                title={Numerical Solution of Boundary Value Problems for Ordinary Differential Equations},
                author={Ascher, Uri M and Mattheij, Robert MM and Russell, Robert D},
                year={1995},
                publisher={SIAM}
            }

    """

    parameters = (mu, lmbda, eta, beta0)

    def bcond(u_left, u_right):
        return u_left - u_right

    return _BoundaryValueProblem(
        vector_field=_vector_fields.measles,
        boundary_conditions=bcond,
        vector_field_args=parameters,
        time_span=time_span,
    )
