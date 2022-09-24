"""ODE vector fields."""

from odezoo import _descriptions, backend


def lotka_volterra(y, /, a, b, c, d):
    """Lotka--Volterra dynamics."""
    return backend.numpy.asarray(
        [a * y[0] - b * y[0] * y[1], -c * y[1] + d * y[0] * y[1]]
    )


def van_der_pol(u, du, /, stiffness_constant):
    """Van-der-Pol dynamics."""
    return stiffness_constant * ((1.0 - u**2) * du - u)


def three_body(Y, dY, /, standardised_moon_mass):
    """Restricted three body dynamics."""
    mu, mp = standardised_moon_mass, 1.0 - standardised_moon_mass
    D1 = backend.numpy.linalg.norm(backend.numpy.asarray([Y[0] + mu, Y[1]])) ** 3.0
    D2 = backend.numpy.linalg.norm(backend.numpy.asarray([Y[0] - mp, Y[1]])) ** 3.0
    du0p = Y[0] + 2 * dY[1] - mp * (Y[0] + mu) / D1 - mu * (Y[0] - mp) / D2
    du1p = Y[1] - 2 * dY[0] - mp * Y[1] / D1 - mu * Y[1] / D2
    return backend.numpy.asarray([du0p, du1p])


def pleiades(u, /):
    """Apply the second-order dynamics :math:`f` as follows.

    Parameters
    ----------
    u
        Current value of the dynamical system. Positional only.
        ``u.shape = (14,)``.

    Returns
    -------
    ddu: array
        Second time-derivative of the current state. ``ddu.shape = (14,)``.

    See Also
    --------
    odezoo.ivps.pleiades : Full specification of the Pleiades problem.

    """
    x, y = u[:7], u[7:]
    x_diff = x[:, None] - x[None, :]
    y_diff = y[:, None] - y[None, :]
    r = (x_diff**2 + y_diff**2) ** 1.5

    # We divide by r (elementwise) further below, but r contains zeros.
    # Those zeros in r imply zeros in the nominator, so the following
    # manipulation does not alter the result (but avoids warnings)
    r = backend.numpy.where(r == 0, r + 1e-12, r)

    mj = backend.numpy.arange(1, 8)[None, :]
    ddx = backend.numpy.sum(mj * x_diff / r, axis=1)
    ddy = backend.numpy.sum(mj * y_diff / r, axis=1)
    return backend.numpy.concatenate((ddx, ddy))


pleiades.__doc__ = _descriptions.PLEIADES + pleiades.__doc__


def lorenz96(y, /, forcing):
    """Lorenz96 dynamics."""
    A = backend.numpy.roll(y, shift=-1)
    B = backend.numpy.roll(y, shift=2)
    C = backend.numpy.roll(y, shift=1)
    D = y
    return (A - B) * C - D + forcing


def lorenz63(u, /, a, b, c):
    """Lorenz63 dynamics."""
    return backend.numpy.asarray(
        [a * (u[1] - u[0]), u[0] * (b - u[2]) - u[1], u[0] * u[1] - c * u[2]]
    )


def rigid_body(y, /, p1, p2, p3):
    r"""Rigid body dynamics without external forces."""
    return backend.numpy.asarray([p1 * y[1] * y[2], p2 * y[0] * y[2], p3 * y[0] * y[1]])


def logistic(u, p0, p1, /):
    """Logistic ODE dynamics."""
    return p0 * u * (1.0 - p1 * u)


def fitzhugh_nagumo(u, /, a, b, c, d):
    """FitzHugh--Nagumo model."""
    return backend.numpy.asarray(
        [u[0] - u[0] ** 3.0 / 3.0 - u[1] + a, (u[0] + b - c * u[1]) / d]
    )


def sir(u, /, beta, gamma, population_count):
    """SIR model."""
    du0_next = -beta * u[0] * u[1] / population_count
    du1_next = beta * u[0] * u[1] / population_count - gamma * u[1]
    du2_next = gamma * u[1]
    return backend.numpy.asarray([du0_next, du1_next, du2_next])


def seir(u, /, alpha, beta, gamma, population_count):
    """SEIR model."""
    du0_next = -beta * u[0] * u[2] / population_count
    du1_next = beta * u[0] * u[2] / population_count - alpha * u[1]
    du2_next = alpha * u[1] - gamma * u[2]
    du3_next = gamma * u[2]
    return backend.numpy.asarray([du0_next, du1_next, du2_next, du3_next])


def sird(u, /, beta, gamma, eta, population_count):
    """SIRD model."""
    du0_next = -beta * u[0] * u[1] / population_count
    du1_next = beta * u[0] * u[1] / population_count - gamma * u[1] - eta * u[1]
    du2_next = gamma * u[1]
    du3_next = eta * u[1]
    return backend.numpy.asarray([du0_next, du1_next, du2_next, du3_next])


def hires(u, /):  # todo: move parameters here
    """High irradiance response."""
    du1 = -1.71 * u[0] + 0.43 * u[1] + 8.32 * u[2] + 0.0007
    du2 = 1.71 * u[0] - 8.75 * u[1]
    du3 = -10.03 * u[2] + 0.43 * u[3] + 0.035 * u[4]
    du4 = 8.32 * u[1] + 1.71 * u[2] - 1.12 * u[3]
    du5 = -1.745 * u[4] + 0.43 * u[5] + 0.43 * u[6]
    du6 = -280.0 * u[5] * u[7] + 0.69 * u[3] + 1.71 * u[4] - 0.43 * u[5] + 0.69 * u[6]
    du7 = 280.0 * u[5] * u[7] - 1.81 * u[6]
    du8 = -280.0 * u[5] * u[7] + 1.81 * u[6]
    return backend.numpy.asarray([du1, du2, du3, du4, du5, du6, du7, du8])


def rober(u, /, k1, k2, k3):
    """Rober ODE problem."""
    du0 = -k1 * u[0] + k3 * u[1] * u[2]
    du1 = k1 * u[0] - k2 * u[1] ** 2 - k3 * u[1] * u[2]
    du2 = k2 * u[1] ** 2
    return backend.numpy.asarray([du0, du1, du2])
