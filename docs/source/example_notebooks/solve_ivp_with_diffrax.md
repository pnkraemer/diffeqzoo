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

# Solve IVPs with diffrax

```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffrax import Dopri5, ODETerm, PIDController, SaveAt, diffeqsolve

from diffeqzoo import backend, ivps

backend.select("jax")
```

```python
ivp = ivps.rigid_body()

f, y0, (t0, t1), f_args, *_ = ivp


@jax.jit
def vector_field(_, y, args):
    return f(y, *args)
```

```python
term = ODETerm(vector_field)
solver = Dopri5()
saveat = SaveAt(ts=jnp.linspace(t0, t1, num=250))
stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

sol = diffeqsolve(
    term,
    solver,
    t0=t0,
    t1=t1,
    dt0=0.1,
    y0=y0,
    args=f_args,
    saveat=saveat,
    stepsize_controller=stepsize_controller,
)
```

```python
plt.plot(sol.ts, sol.ys)
plt.show()
```
