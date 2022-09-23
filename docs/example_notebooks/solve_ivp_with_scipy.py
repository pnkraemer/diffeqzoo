# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import odezoo

odezoo.set_backend("numpy")

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate

import odezoo.ivps

# %%
ivps = [
    odezoo.ivps.lotka_volterra(),
    odezoo.ivps.lotka_volterra(parameters=(1.0, 0.1, 0.1, 1.0)),
    odezoo.ivps.rigid_body(),
    odezoo.ivps.lorenz96(),
]


# %%
def solve_ivp(ivp, **kwargs):
    def fun(_, y, *args):
        return ivp.vector_field(y, *args)

    t_span = ivp.time_span
    (y0,) = ivp.initial_values
    args = ivp.vector_field_args

    solution = scipy.integrate.solve_ivp(
        fun=fun, t_span=t_span, y0=y0, args=args, **kwargs
    )

    plotgrid = np.linspace(*t_span)
    return plotgrid, solution.sol(plotgrid).T


# %%
fig, axes = plt.subplots(ncols=len(ivps), figsize=(8, 2), tight_layout=True)

for ax, ivp in zip(axes, ivps):
    xs, ys = solve_ivp(ivp, dense_output=True)

    ax.plot(xs, ys)
plt.show()

# %%