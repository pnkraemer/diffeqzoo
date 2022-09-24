"""ODE problem descriptions.

Common descriptions of ODE problems that should appear in multiple docstrings
(such as the meaning of the equations, a reference, etc.), can be placed
here and prepended to the respective docstring via
`fun.__doc__ = <common_doc> + fun.__doc__`. This makes sure that
every occurrence of, e.g., the Pleiades problem, comes with
an appropriate description of the dynamics.
"""


PLEIADES = r"""Pleiades problem from celestial mechanics.

    The Pleiades problem describes the gravitational interaction(s) of seven stars
    (the "Pleiades", or "Seven Sisters") in a plane.
    It is a 14-dimensional, second-order differential equation
    and commonly solved as a 28-dimensional, first-order equation. [1]_
    Here, it is implemented in its original, second-order form,

    .. math::
        \ddot u(t) = f(u(t)),

    with nonlinear dynamics :math:`f: \mathbb{R}^{14} \rightarrow  \mathbb{R}^{14}`.

    The Pleiades problem is not stiff.
    It is a popular benchmark problem because
    it is not very difficult to solve numerically, but
    (a) it requires high accuracy in each ODE solver step, and
    (b) its 14 (or 28) dimensions start to expose those numerical solvers
    that do not scale well to high dimensions.

    References
    ----------
    .. [1] Hairer, E., Nørsett, S. P., and Wanner, G. (1993).
       Solving Ordinary Differential Equations I, Nonstiff Problems. Springer.
       Page 244.

    """
