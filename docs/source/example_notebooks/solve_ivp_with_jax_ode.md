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

# Solve IVPs with JAX' ODE solver

```python
import jax.experimental.ode
import matplotlib.pyplot as plt

from odezoo import backend, ivps

backend.select("jax")
```

```python
ivp_selection = [
    ivps.van_der_pol_first_order(),
    ivps.lotka_volterra(),
    ivps.rigid_body(),
    ivps.lorenz96(),
]
```

```python
def solve_ivp(ivp, **kwargs):
    f, y0, tspan, f_args = ivp

    def fun(y, _, *args):
        return f(y, *args)

    t = backend.numpy.linspace(*tspan)
    y = jax.experimental.ode.odeint(fun, y0, t, *f_args, **kwargs)
    return t, y
```

```python
fig, axes = plt.subplots(ncols=len(ivp_selection), figsize=(8, 2), tight_layout=True)

for ax, ivp in zip(axes, ivp_selection):
    xs, ys = solve_ivp(ivp)

    ax.plot(xs, ys)
plt.show()
```
