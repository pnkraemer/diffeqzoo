# Installation

## From PyPi

To install odezoo, run

```commandline
pip install odezoo
```

This is the preferred method to install odezoo, as it will always install the most recent stable release.

It assumes that you will install NumPy or JAX yourself.
If this is not the case, run

```commandline
pip install odezoo[numpy]
```
or
```commandline
pip install odezoo[jax]
```

!!! warning

    Currently, the package is not on pypi, so the above installations do not work.


## From GitHub

To get the most recent version from GitHub, run

```commandline
pip install git+https://github.com/pnkraemer/odezoo.git
```
