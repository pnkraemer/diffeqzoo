"""Initial value problem examples."""

from odezoo import numpy_like


from collections import namedtuple

InitialValueProblem = namedtuple(
    "InitialValueProblem",
    (
        "vector_field",
        "vector_field_jacobian",
        "vector_field_parameters_proposal",
        "initial_values_proposal",
        "time_span_proposal",
    ),
    defaults=(None,) * 5,
)


def lotka_volterra():
    """Lotka--Volterra / predator-prey model."""

    def f_lv(y, /, *params):
        a, b, c, d = params
        return numpy_like.asarray(
            [a * y[0] - b * y[0] * y[1], -c * y[1] + d * y[0] * y[1]]
        )

    p_proposal = (0.5, 0.05, 0.5, 0.05)
    u0_proposal = numpy_like.asarray([20.0, 20.0])
    t_span_proposal = (0.0, 20.0)
    return InitialValueProblem(
        vector_field=f_lv,
        vector_field_parameters_proposal=p_proposal,
        initial_values_proposal=u0_proposal,
        time_span_proposal=t_span_proposal,
    )


def vanderpol_second_order():
    """Van-der-Pol system as a second order differential equation."""

    def f_vdp(u, du, /, stiffness_constant):
        return stiffness_constant * ((1.0 - u**2) * du - u)

    p_proposal = (1.0,)
    u0_proposal = numpy_like.asarray([2.])
    du0_proposal = numpy_like.asarray([0.])
    t_span_proposal = (0.0, 6.3)

    return InitialValueProblem(
        vector_field=f_vdp,
        vector_field_parameters_proposal=p_proposal,
        initial_values_proposal=(u0_proposal, du0_proposal),
        time_span_proposal=t_span_proposal,
    )

def threebody_second_order():
    """Restricted three-body problem as a second order differential equation."""

    def f_3b(Y, dY, /, standardised_moon_mass):
        mu = standardised_moon_mass
        mp = 1.0 - mu
        D1 = numpy_like.linalg.norm(numpy_like.asarray([Y[0] + mu, Y[1]])) ** 3.0
        D2 = numpy_like.linalg.norm(numpy_like.asarray([Y[0] - mp, Y[1]])) ** 3.0
        du0p = Y[0] + 2 * dY[1] - mp * (Y[0] + mu) / D1 - mu * (Y[0] - mp) / D2
        du1p = Y[1] - 2 * dY[0] - mp * Y[1] / D1 - mu * Y[1] / D2
        return numpy_like.asarray([du0p, du1p])

    # Some parameter suggestions for nice simulation
    p_proposal = (0.012277471,)
    u0_proposal = numpy_like.asarray([0.994, 0])
    du0_proposal = numpy_like.asarray([0, -2.00158510637908252240537862224])
    t_span_proposal = (0.0, 17.0652165601579625588917206249)

    return InitialValueProblem(
        vector_field=f_3b,
        vector_field_parameters_proposal=p_proposal,
        initial_values_proposal=(u0_proposal, du0_proposal),
        time_span_proposal=t_span_proposal,
    )

