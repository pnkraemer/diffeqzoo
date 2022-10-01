"""ODE vector fields."""

from diffeqzoo import backend, transform


def lotka_volterra(y, /, a, b, c, d):
    """Lotka--Volterra dynamics."""
    return backend.numpy.asarray(
        [a * y[0] - b * y[0] * y[1], -c * y[1] + d * y[0] * y[1]]
    )


def pleiades(u, /):
    """Evaluate the Pleiades vector field in its original, second-order form."""
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


def pleiades_autonomous_api(u, _, /):
    """Evaluate the Pleiades vector field as \
    :math:`\\ddot u(t) = f(u(t), \\dot u(t))` \
    (with an unused second argument)."""  # noqa: D301
    return pleiades(u)


# Transform the autonomous-API-version into a first-order problem.
pleiades_first_order = transform.second_to_first_order_vf_auto(
    pleiades_autonomous_api,
    short_summary="The Pleiades problem as a first-order differential equation.",
)


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


# todo: make external forces a parameter
def rigid_body(y, /, p1, p2, p3):
    r"""Rigid body dynamics without external forces."""
    return backend.numpy.asarray([p1 * y[1] * y[2], p2 * y[0] * y[2], p3 * y[0] * y[1]])


def logistic(u, p0, p1, /):
    """Logistic ODE dynamics."""
    return p0 * u * (1.0 - p1 * u)


def fitzhugh_nagumo(u, /, a, b, c):
    """FitzHugh--Nagumo model."""
    return backend.numpy.asarray(
        [c * (u[0] - u[0] ** 3 / 3 + u[1]), -(1.0 / c) * (u[0] - a - b * u[1])]
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


def van_der_pol(u, du, /, stiffness_constant):
    """Van-der-Pol dynamics as a second-order differential equation."""
    return stiffness_constant * ((1.0 - u**2) * du - u)


van_der_pol_first_order = transform.second_to_first_order_vf_auto(
    van_der_pol,
    short_summary="""Van-der-Pol dynamics as a first-order differential equation.""",
)


def three_body_restricted(Y, dY, /, standardised_moon_mass):
    """Restricted three-body dynamics as a second-order differential equation."""
    mu, mp = standardised_moon_mass, 1.0 - standardised_moon_mass
    D1 = backend.numpy.linalg.norm(backend.numpy.asarray([Y[0] + mu, Y[1]])) ** 3.0
    D2 = backend.numpy.linalg.norm(backend.numpy.asarray([Y[0] - mp, Y[1]])) ** 3.0
    du0p = Y[0] + 2 * dY[1] - mp * (Y[0] + mu) / D1 - mu * (Y[0] - mp) / D2
    du1p = Y[1] - 2 * dY[0] - mp * Y[1] / D1 - mu * Y[1] / D2
    return backend.numpy.asarray([du0p, du1p])


_3bdocs = "Restricted three-body dynamics as a first-order differential equation."
three_body_restricted_first_order = transform.second_to_first_order_vf_auto(
    three_body_restricted,
    short_summary=_3bdocs,
)


def bratu(u, /, k):
    """Bratu's problem."""
    return -k * backend.numpy.exp(u)


def bratu_autonomous_api(u, _, /, k):
    """Bratu's problem with signature (u, u')."""
    return -k * backend.numpy.exp(u)


def pendulum(u, /, p):
    """Bratu's problem."""
    return -p * backend.numpy.sin(u)


def pendulum_autonomous_api(u, _, /, p):
    """Bratu's problem."""
    return -p * backend.numpy.sin(u)


# todo: merge with sir() vector field
def measles(t, u, /, mu, lmbda, eta, beta0):
    b = _beta(t, beta0)
    return backend.numpy.asarray(
        [
            mu - b * u[0] * u[2],
            b * u[0] * u[2] - u[1] / lmbda,
            u[1] / lmbda - u[2] / eta,
        ]
    )


def _beta(t, beta0):
    return beta0 * (1 + backend.numpy.cos(2 * backend.numpy.pi * t))
