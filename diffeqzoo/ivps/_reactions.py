"""Chemical reactions."""

from diffeqzoo import backend, vector_fields
from diffeqzoo.ivps import _ivp


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

    return _ivp.InitialValueProblem(
        vector_field=vector_fields.hires,
        vector_field_args=(),  # todo: move vf-params here
        initial_values=initial_values,
        time_span=time_span,
    )


def rober(
    *, initial_values=(1.0, 0.0, 0.0), time_span=(0.0, 1e5), k1=0.04, k2=3e7, k3=1e4
):
    r"""Construct the ROBER problem due to Robertson (1966).

    The ROBER problem describes the kinetics of an auto-catalytic reaction,
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

    return _ivp.InitialValueProblem(
        vector_field=vector_fields.rober,
        vector_field_args=(k1, k2, k3),
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

    return _ivp.InitialValueProblem(
        vector_field=vector_fields.rober,
        vector_field_args=(s, w, q),
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

    return _ivp.InitialValueProblem(
        vector_field=vector_fields.nonlinear_chemical_reaction,
        vector_field_args=(k1, k2),
        initial_values=initial_values,
        time_span=time_span,
    )
