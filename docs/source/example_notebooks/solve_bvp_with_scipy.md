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
# Solve BVPs with SciPy


## Separable boundary conditions
<!-- #endregion -->

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate

from odezoo import backend, bvps

backend.select("numpy")
```

```python
def solve_two_point_bvp(bvp, **kwargs):
    f, (g0, g1), tspan, f_args = bvp

    def fun(_, y):
        u, du = backend.numpy.split(y, 2)
        ddu = f(u, *f_args)
        return backend.numpy.concatenate((du, ddu))

    def bcond(y0, y1):
        u0, du0 = backend.numpy.split(y0, 2)
        u1, du1 = backend.numpy.split(y1, 2)
        return backend.numpy.concatenate((g0(u0), g1(u1)))

    x = np.linspace(*tspan, 15)
    y = backend.numpy.zeros((2, x.shape[0]))
    y[0] += 3

    solution = scipy.integrate.solve_bvp(fun=fun, bc=bcond, x=x, y=y, **kwargs)

    plotgrid = np.linspace(*tspan)
    return plotgrid, solution.sol(plotgrid).T
```

```python
bvp_selection = (bvps.bratu(), bvps.pendulum())

fig, axes = plt.subplots(
    ncols=len(bvp_selection),
    figsize=(8, 2),
    tight_layout=True,
    sharey=True,
    sharex=False,
)

for ax, bvp in zip(axes, bvp_selection):
    xs, ys = solve_two_point_bvp(bvp)
    ax.plot(xs, ys)
plt.show()
```

```python
def solve_bvp(bvp, **kwargs):
    f, bcond, tspan, f_args = bvp

    def fun(t, y):
        return f(t, y, *f_args)

    x = np.linspace(*tspan, 50)
    y = backend.numpy.ones((3, x.shape[0]))

    solution = scipy.integrate.solve_bvp(fun=fun, bc=bcond, x=x, y=y, **kwargs)

    plotgrid = np.linspace(*tspan)
    return plotgrid, solution.sol(plotgrid).T
```

```python
bvp_selection = (bvps.measles(),)

fig, axes = plt.subplots(
    ncols=len(bvp_selection),
    figsize=(5, 2),
    tight_layout=True,
    sharey=True,
    sharex=False,
)

for ax, bvp in zip(backend.numpy.atleast_1d(axes), bvp_selection):
    xs, ys = solve_bvp(bvp)
    ax.plot(xs, ys)
plt.show()
```

```python

```
