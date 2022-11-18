---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region -->
# Solve IVPs with JAX

JAX provides not only a linear algebra backend, automatic differentiation, and other useful function transformations, but also an initial value problem solver: `jax.experimental.ode.odeint()`.
Its API mirrors the API of `scipy.integrate.odeint` (which we cover in a different tutorial).



With the JAX backend, we can plug `diffeqzoo`'s initial value problems into this API as follows.

<!-- #endregion -->

```python
import inspect

import jax
import jax.experimental.ode
import matplotlib.pyplot as plt

from diffeqzoo import backend, ivps

backend.select("jax")
```

```python
print(inspect.signature(jax.experimental.ode.odeint))
```

Most ODEs are autonomous (which means that the vector field does not depend on the time variable), but just like most other ODE solvers, JAX' `odeint` expects a time-dependent vector field.
We can wrap the output of the `diffeqzoo` into the desired format easily.
Let's compute the solution of an example problem and plot the solution.

```python
f, y0, tspan, f_args = ivps.fitzhugh_nagumo()


@jax.jit
def fun(y, _, *args):
    return f(y, *args)


t = backend.numpy.linspace(*tspan, num=200)
y = jax.experimental.ode.odeint(fun, y0, t, *f_args)

plt.plot(t, y)
plt.show()
```

```python
(f, y0, tspan, f_args), info = ivps.heat_1d_dirichlet(num_gridpoints=100)
grid = info["grid"]


@jax.jit
def fun(y, _, *args):
    return f(y, *args)


t = backend.numpy.linspace(*tspan, num=200)
y = jax.experimental.ode.odeint(fun, y0, t, *f_args)

for i, ys in enumerate(y):
    # Reduce the opacity over time
    alpha = 2.0 * float(backend.numpy.mean(ys))
    plt.plot(grid, ys, alpha=alpha, color="C0")
plt.show()
```
