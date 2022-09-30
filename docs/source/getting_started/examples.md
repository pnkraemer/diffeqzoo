# Quick example


To get started, import the `ivps` and `bvps` from `odezoo`.
You must also import the backend, because `odezoo` needs to now whether to build the ODEs in numpy or in jax.
```python
>>> from odezoo import ivps, bvps, backend
>>> backend.select("numpy")

```
Here, we chose the numpy backend. We could have also used `backend.select("jax")`.
Now with this backend, numpy-implementations are available via `backend.numpy`.
(Under the hood, this imports and exposes either `jax.numpy` or `numpy`).

Use this backend to create ODE problems.

```python
>>> f_lv, u0_lv, _, f_args_lv = ivps.lotka_volterra()
>>> x_lv = f_lv(u0_lv, *f_args_lv)
>>> print(x_lv)
[-10.  10.]

```
```python
>>> f_rb, u0_rb, _, f_args_rb = ivps.rigid_body()
>>> x_rb = f_rb(u0_rb, *f_args_rb)
>>> print(x_rb)
[-0.     1.125 -0.   ]

```
```python
>>> f_sir, u0_sir, _, f_args_sir = ivps.sir()
>>> x_sir = f_rb(u0_sir, *f_args_sir)
>>> print(backend.numpy.round(x_sir, 1))
[3.00000e-01 9.98000e+01 9.96004e+05]

```

While all IVP problem creators have a _similar_ API, the ODE functions are not necessarily unified.

```python
>>> import inspect
>>> f1, *_ = ivps.three_body_restricted()
>>>
>>> f2, *_ = ivps.sird()
>>>
>>> print(inspect.signature(f1))
(Y, dY, /, standardised_moon_mass)
>>>
>>> print(inspect.signature(f2))
(u, /, beta, gamma, eta, population_count)

```
Here, one IVP is a first-order problem with multiple parameters, the other one is a second-order problem with a single parameter.
Second-order ODEs are taken seriously in the `odezoo`, because the second-order form can be solved faster.
