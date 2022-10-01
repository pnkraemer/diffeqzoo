# Installation



Get the most recent stable version from PyPi:

```commandline
pip install diffeqzoo
```

Or directly from GitHub:
```commandline
pip install git+https://github.com/pnkraemer/odezoo.git
```
This assumes that you have NumPy or JAX installed. It is best you do this yourself (especially for JAX),
but you can also install them with `diffeqzoo` via _either_ of the following:
```commandline
pip install diffeqzoo[jax]
pip install diffeqzoo[numpy]
pip install diffeqzoo[jax,numpy]
```
which installs the CPU version of JAX.
For the GPU version, install JAX yourself.



## Local installation
In a fork, you can install the project locally: go to the project's root and run:

```commandline
pip install .
```

or install in editable mode:
```commandline
pip install -e .
```

## Optional dependencies

`diffeqzoo` comes with optional dependencies. For example, NumPy or JAX (see above), but also `dev`, `doc`, `test`, `lint`, `example` groups, which are useful for the continuous integration.
See the explanation of the CI for more info.
