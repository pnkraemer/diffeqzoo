"""Initial value problem examples."""

from odezoo import numpy_like


def lotka_volterra(*, t0=0.0, tmax=20.0, u0=None):
    """Lotka--Volterra / predator-prey model."""

    def f_lv(y, /, *params):
        a, b, c, d = params
        return numpy_like.asarray(
            [a * y[0] - b * y[0] * y[1], -c * y[1] + d * y[0] * y[1]]
        )

    params_proposal = (0.5, 0.05, 0.5, 0.05)
    u0_proposal = numpy_like.asarray([20.0, 20.0])
    t_span_proposal = (0., 20.)
    return f_lv, params_proposal, u0_proposal, t_span_proposal
