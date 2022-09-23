"""ODE vector fields."""

from odezoo import numpy_like


def lotka_volterra(y, /, a, b, c, d):
    """Lotka--Volterra dynamics."""

    return numpy_like.asarray([a * y[0] - b * y[0] * y[1], -c * y[1] + d * y[0] * y[1]])


def van_der_pol(u, du, /, stiffness_constant):
    """Van-der-Pol dynamics."""

    return stiffness_constant * ((1.0 - u**2) * du - u)


def three_body(Y, dY, /, standardised_moon_mass):
    """Restricted three body dynamics."""

    mu, mp = standardised_moon_mass, 1.0 - standardised_moon_mass
    D1 = numpy_like.linalg.norm(numpy_like.asarray([Y[0] + mu, Y[1]])) ** 3.0
    D2 = numpy_like.linalg.norm(numpy_like.asarray([Y[0] - mp, Y[1]])) ** 3.0
    du0p = Y[0] + 2 * dY[1] - mp * (Y[0] + mu) / D1 - mu * (Y[0] - mp) / D2
    du1p = Y[1] - 2 * dY[0] - mp * Y[1] / D1 - mu * Y[1] / D2
    return numpy_like.asarray([du0p, du1p])
