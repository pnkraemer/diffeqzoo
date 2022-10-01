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

# Solve IVPs with Diffrax

Diffrax provides numerical differential equation solvers in JAX.
Its advantages over, e.g., JAX' solvers include a larger set of available solvers.
We can plug the `diffeqzoo`'s problems into diffrax as follows.



```python
import inspect

import diffrax
import jax
import matplotlib.pyplot as plt

from diffeqzoo import backend, ivps

backend.select("jax")
```

```python
print(inspect.signature(diffrax.diffeqsolve))
```

Most ODEs are autonomous (i.e., they do not depend on the time variable), and the `diffeqzoo` implements them as such. Just like most other ODE solvers in Python, Diffrax expects non-autonomous vector fields.
It further requires wrapping vector fields into `diffrax.ODETerm` objects, which can be achieved easily.

Let's plot the solution of an example initial value problem.

```python
f, y0, (t0, t1), args = ivps.seir()


@jax.jit
def vf(t, y, p):
    return f(y, *p)


term = diffrax.ODETerm(vf)
solver = diffrax.Dopri5()

ts = backend.numpy.linspace(t0, t1, num=200)
saveat = diffrax.SaveAt(ts=ts)

sol = diffrax.diffeqsolve(
    term,
    solver,
    t0=t0,
    t1=t1,
    dt0=0.1,
    y0=y0,
    args=args,
    saveat=saveat,
)

plt.plot(sol.ts, sol.ys)
plt.show()
```
