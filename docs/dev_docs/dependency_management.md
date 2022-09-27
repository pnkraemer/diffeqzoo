# Manage the dependencies

This project uses ``poetry`` to manage its packaging and its dependencies.
See poetry's [documentation](https://python-poetry.org/docs/configuration/) for some instructions.

!!! question "Can we use ``poetry`` in a better way?"

    If you know how to improve our usage of ``poetry``, please let us know!


## Install ``poetry``

Run

```commandline
pip install --upgrade pip
pip install poetry
```

## Use poetry to install dependencies

``ODEZoo`` has a different set of optional dependencies:

* `numpy`: if you want to use it with NumPy
* `jax`: if you want to use it with JAX
* `test`: for testing-related dependencies (pytest, etc.). This also includes linting with black, isort, and flake8
* `dev`: for all neither-testing-nor-documentation-related dev tools (tox, pre-commit, etc.)
*
