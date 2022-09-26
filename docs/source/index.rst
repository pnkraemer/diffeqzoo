.. odezoo documentation master file, created by
   sphinx-quickstart on Fri Sep 23 12:39:13 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

odezoo
======

`So, what was the initial condition of the restricted three-body problem again?`


``ODEZoo`` is an either `NumPy <https://numpy.org/>`_- or `JAX <https://jax.readthedocs.io/en/latest/>`_-based library (your choice!) that collects ordinary differential equation models.


**Features include**

* Standard non-stiff benchmark problems (Lotka--Volterra, FitzHugh--Nagumo, Van-der-Pol, Rigid body, ...)
* Standard stiff benchmark problems (Stiff van-der-Pol, HIRES, ROBER, ...)
* Compartmental epidemiological models (SIR, SEIR, SIRD, ...)
* Chaotic systems (Lorenz63, Lorenz96)
* N-Body problems
* Boundary value problems (to appear)

**As well as**

* Flexibly NumPy and JAX-backends. Other than one of those two, there are 0 (zero!) dependencies.
* Mathematical descriptions of the ODE problems
* BibTex entries for each original reference, to be used in scientific publications

and many more goodies.


.. toctree::
   :maxdepth: 0
   :caption: Getting started

   getting_started/installation
   getting_started/rationale



.. toctree::
   :maxdepth: 1
   :caption: Examples

   example_notebooks/solve_ivp_with_scipy
   example_notebooks/solve_ivp_with_diffrax



.. toctree::
   :maxdepth: 1
   :caption: API documentation

   api_documentation/odezoo

.. toctree::
   :maxdepth: 1
   :caption: Developer documentation

   dev_documentation/contribution
   dev_documentation/design_choices
