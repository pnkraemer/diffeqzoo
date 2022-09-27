# odezoo


[![pypi](https://img.shields.io/pypi/v/odezoo.svg)](https://pypi.org/project/odezoo/)
[![python](https://img.shields.io/pypi/pyversions/odezoo.svg)](https://pypi.org/project/odezoo/)
[![Build Status](https://github.com/pnkraemer/odezoo/actions/workflows/dev.yml/badge.svg)](https://github.com/pnkraemer/odezoo/actions/workflows/dev.yml)
[![codecov](https://codecov.io/gh/pnkraemer/odezoo/branch/main/graphs/badge.svg)](https://codecov.io/github/pnkraemer/odezoo)



Ordinary differential equation problem definitions.

!!! warning

    This project is in development. Expect rough edges and rapidly changing APIs.


* Documentation: <https://pnkraemer.github.io/odezoo>
* GitHub: <https://github.com/pnkraemer/odezoo>
* PyPI: <https://pypi.org/project/odezoo/>
* Free software: MIT



## Features include

* Standard non-stiff benchmark problems (_Lotka--Volterra_, _FitzHugh--Nagumo_, _Van-der-Pol_, ...)
* Standard stiff benchmark problems (_HIRES_, _ROBER_, ...)
* Compartmental epidemiological models (_SIR_, _SEIR_, _SIRD_, ...)
* Chaotic systems (_Lorenz63_, _Lorenz96_)
* N-Body problems
* Second-order and first-order versions of ODE vector fields

**As well as**

* Flexible NumPy and JAX-backends.
* Mathematical descriptions of the ODE problems
* BibTex entries for each original reference, to be used in scientific publications

and many more goodies.


## Design goals

`ODEZoo` provides only ODE example problems (no solvers!).

`ODEZoo` has minimal implementation logic. Everything is constructed using python built-ins and either numpy or jax. A user does not have to learn any classes or interfaces.

`ODEZoo` must be trivial to learn, and it must be easy to copy & paste out of its source.





## Credits

* This package was initially created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [waynerv/cookiecutter-pypackage](https://github.com/waynerv/cookiecutter-pypackage) project template.
* Inspiration for the ``MkDocs`` usage has been taken from ``diffrax`` and ``fastapi``.
