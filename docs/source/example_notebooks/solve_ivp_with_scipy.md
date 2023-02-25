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
# Solve IVPs with SciPy

SciPy is everyone's first stop for scientific computing in Python.



The initial value problems provided by `diffeqzoo` can be plugged into scipy's ordinary differential equation (ODE) solvers.


SciPy has two IVP solvers: `odeint` (wraps FORTRAN's `odepack`) and `solve_ivp` (native Python).
They require slightly different inputs: for example, `odeint` expects vector fields `f(y, t)` and `solve_ivp` expects vector fields `f(t, y)`.

`diffeqzoo` can be used for both.

<!-- #endregion -->

```python
import inspect  # to inspect function signatures

import matplotlib.pyplot as plt
import scipy.integrate

from diffeqzoo import backend, ivps

backend.select("numpy")
```

## Using solve_ivp

Let's start with `solve_ivp`, because that is the recommendation given in the docs of `scipy.integrate.odeint` ([link](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html)).

```python
print(inspect.signature(scipy.integrate.solve_ivp))
```

Here is how we solve ODEs from `diffeqzoo` with `solve_ivp`.
Most ODE test problems are autonomous, which means that the vector fields do not depend on time.
We can wrap them into a non-autonomous format and plug them into scipy.

```python
f, y0, t_span, args = ivps.lotka_volterra()
print(inspect.signature(f), args)


def fun(t, y, *args):
    return f(y, *args)


scipy.integrate.solve_ivp(fun=fun, t_span=t_span, y0=y0, args=args)
```

Let's plot the solution.

```python
t_eval = backend.numpy.linspace(*t_span, num=200)
sol = scipy.integrate.solve_ivp(fun=fun, t_span=t_span, t_eval=t_eval, y0=y0, args=args)
y_eval = sol.y.T
```

```python
plt.plot(t_eval, y_eval)
plt.show()
```

## Using odeint

The usage of `odeint` is very similar to that of `solve_ivp`.
We simply need to rename a few arguments and wrap the vector field slightly differently.


```python
print(inspect.signature(scipy.integrate.odeint))
```

Let's compute the ODE solution with `odeint` and plot the solution.

```python
f, y0, t_span, args = ivps.pleiades_first_order()
print(inspect.signature(f), args)


def func(y, t, *args):
    return f(y, *args)


t = backend.numpy.linspace(*t_span, num=300)
y = scipy.integrate.odeint(func=func, y0=y0, t=t, args=args)
colors = ["C" + str(i) for i in range(7)]
for x1, x2, color in zip(y[:, 0:7].T, y[:, 7:14].T, colors):
    plt.plot(x1, x2, color=color)
    plt.plot(x1[0], x2[0], marker=".", color=color)
    plt.plot(x1[-1], x2[-1], marker="*", markersize=10, color=color)
plt.show()
```
