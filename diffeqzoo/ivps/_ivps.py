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
from typing import Any, Callable, Iterable, NamedTuple, Union

from diffeqzoo import _vector_fields, backend, transform
from diffeqzoo.ivps import _ivp
