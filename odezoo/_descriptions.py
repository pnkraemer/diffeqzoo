"""ODE problem descriptions.

Common descriptions of ODE problems that should appear in multiple docstrings
(such as the meaning of the equations, a reference, etc.), can be placed
here and prepended to the respective docstring via
`fun.__doc__ = <common_doc> + fun.__doc__`. This makes sure that
every occurrence of, e.g., the Pleiades problem, comes with
an appropriate description of the dynamics.
"""


PLEIADES = r"""

    The Pleiades problem from celestial mechanics describes the
    gravitational interaction(s) of seven stars
    (the "Pleiades", or "Seven Sisters") in a plane.
    It is a 14-dimensional, second-order differential equation
    and commonly solved as a 28-dimensional, first-order equation. [1]_
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

    References
    ----------
    .. [1] Hairer, E., Nørsett, S. P., and Wanner, G. (1993).
       Solving Ordinary Differential Equations I, Nonstiff Problems. Springer.
       Page 244.

    See Also
    --------
    odezoo.ivps.pleiades
    odezoo.ivps.pleiades_autonomous_api
    odezoo.ivps.pleiades_first_order
    odezoo.vector_fields.pleiades
    odezoo.vector_fields.pleiades_autonomous_api
    odezoo.vector_fields.pleiades_first_order
    """

FITZHUGH_NAGUMO = r"""


    The FitzHugh-Nagumo model is a simple example of an excitable system (for example a neuron).
    This simplified, 2d-version of the Hodgkin-Huxley model (which describes the spike generation in squid giant axons)
    was suggested by FitzHugh [1]_ and Nagumo et al. [2]_

    It no a non-stiff, first-order problem,

    .. math::
        \dot u(t) = f(u(t), \theta)

    and generally easy to solve by most ODE solvers.

    References
    ----------
    .. [1] FitzHugh R. (1961)
       Impulses and physiological states in theoretical models of nerve membrane. Biophysical J. 1:445-466

    .. [2] Nagumo J., Arimoto S., and Yoshizawa S. (1962)
       An active pulse transmission line simulating nerve axon. Proc IRE. 50:2061–2070.

    See Also
    --------
    odezoo.ivps.fitzhugh_nagumo
    odezoo.vector_fields.fitzhugh_nagumo


"""
