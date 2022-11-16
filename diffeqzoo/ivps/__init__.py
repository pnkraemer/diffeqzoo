r"""Initial value problem examples.

This module provides a number of example implementations
of initial value problems (IVPs).
IVPs are a combination of an ordinary differential equation

.. math:: \dot u(t) = f(u(t), t, \theta)

and an initial condition :math:`u(0) = u_0`.
The initial values :math:`u_0` and the vector field :math:`f`
are known, the parameters :math:`\theta` might be known,
and :math:`u` is unknown.

The functions in this module construct implementations of this
kind of problem. They (loosely) follow the input/output rule

.. code:: python

    f, u0, (t0, tmax), param = constructor()

where the constructor is, e.g., :code:`lotka_volterra()`
or :code:`sir()`.
This API specification is only loose, because every problem is different.
For example, second-order problems implement a second-order differential
equation

.. math:: \ddot u(t) = f(u(t), \dot u(t), t, \theta)

subject to the initial conditions
:math:`u(0) = u_0` and :math:`\dot u(0) = u_1`.
For these problems (e.g., :code:`three_body_restricted()` or :code:`van_der_pol()`),
there are two initial values:

.. code:: python

    f, (u0, u1), (t0, tmax), param = constructor()

We try to stick as closely as possible to the above signature,
but if problem-specific issues arise, we allow ourselves to deviate from
this specification.
When in doubt, consult the documentation of the respect constructor function.
"""

from diffeqzoo.ivps._chaos import lorenz63, lorenz96, roessler
from diffeqzoo.ivps._ivps import *
from diffeqzoo.ivps._misc import (
    affine_dependent,
    affine_independent,
    logistic,
    neural_ode_mlp,
    seir,
    sir,
    sird,
)
from diffeqzoo.ivps._nbody import (
    henon_heiles,
    henon_heiles_first_order,
    henon_heiles_with_unused_derivative_argument,
    pleiades,
    pleiades_first_order,
    pleiades_with_unused_derivative_argument,
    rigid_body,
    three_body_restricted,
    three_body_restricted_first_order,
)
from diffeqzoo.ivps._oscillators import (
    fitzhugh_nagumo,
    goodwin,
    lotka_volterra,
    van_der_pol,
    van_der_pol_first_order,
)
from diffeqzoo.ivps._reactions import (
    hires,
    nonlinear_chemical_reaction,
    oregonator,
    rober,
)
