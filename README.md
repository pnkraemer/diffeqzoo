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

* Standard non-stiff benchmark problems (Lotka--Volterra, FitzHugh--Nagumo, Van-der-Pol, Rigid body, ...)
* Standard stiff benchmark problems (Stiff van-der-Pol, HIRES, ROBER, ...)
* Compartmental epidemiological models (SIR, SEIR, SIRD, ...)
* Chaotic systems (Lorenz63, Lorenz96)
* N-Body problems


**As well as**

* Flexibly NumPy and JAX-backends. Other than one of those two, there are 0 (zero!) dependencies.
* Mathematical descriptions of the ODE problems
* BibTex entries for each original reference, to be used in scientific publications

and many more goodies.

* All sorts of ODEs
* in either NumPy or Jax (your choice!)
* etc.


## Disclaimers about tooling



## Credits

* This package was initially created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [waynerv/cookiecutter-pypackage](https://github.com/waynerv/cookiecutter-pypackage) project template.
* Inspiration for the ``MkDocs`` usage has been taken from ``diffrax`` and ``fastapi``.
