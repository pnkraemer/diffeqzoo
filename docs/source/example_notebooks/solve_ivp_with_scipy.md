---
jupyter:
  jupytext:
    formats: ipynb,md,py:percent
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

# Solve IVPs with SciPy

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate

from odezoo import backend, ivps

backend.select("numpy")
```

```python
ivp_selection = [
    ivps.lotka_volterra(),
    ivps.lotka_volterra(parameters=(1.0, 0.2, 0.1, 1.0)),
    ivps.rigid_body(),
    ivps.lorenz96(),
]
```

```python
def solve_ivp(ivp, **kwargs):
    def fun(_, y, *args):
        return ivp.vector_field(y, *args)

    t_span = ivp.time_span
    y0 = ivp.initial_values
    args = ivp.vector_field_args

    solution = scipy.integrate.solve_ivp(
        fun=fun, t_span=t_span, y0=y0, args=args, **kwargs
    )

    plotgrid = np.linspace(*t_span)
    return plotgrid, solution.sol(plotgrid).T
```

```python
fig, axes = plt.subplots(ncols=len(ivp_selection), figsize=(8, 2), tight_layout=True)

for ax, ivp in zip(axes, ivp_selection):
    xs, ys = solve_ivp(ivp, dense_output=True)

    ax.plot(xs, ys)
plt.show()
```
