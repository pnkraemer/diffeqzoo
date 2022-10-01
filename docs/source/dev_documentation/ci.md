# Continuous integration


The continuous integration has multiple components,

* format: apply black, isort, etc.. Includes notebooks.
* lint: check flake8, and check whether black and isort are happy with the code. No formatting.
* test: Run the unittests with different backends, and run some doctests
* example: sync and execute the example notebooks. Bundles a bunch of bigger optional dependencies (scipy, diffrax, matplotlib, ...), so it is kept separate from the rest of the dependencies.
* doc: build the sphinx docs.

which appear in different parts of the specification

* Groups of optional dependencies: `pip install diffeqzoo[lint,test,example,doc]`. Formatting dependencies are a subset of the linting dependencies.
* In the makefile: `make format; make lint; make test; make example`
* In the workflows


There are also the `numpy` and the `jax` optional dependencies.
Combine any of the execution-based workflows (test, example) with either one, e.g.,
```
pip install diffeqzoo[jax]
pip install diffeqzoo[example,numpy]
pip install diffeqzoo[test,jax,numpy]
```


## The makefile

To apply _all_ {linting, formatting, testing, ...} operations, use the makefile.
In the root, run

```
make format
make lint
make test
```

To remove automatically generated files (caches, etc.), run ``make clean``.

To check conformity with the pre-commit file, run ``make pre-commit``.
This command also updates the versions in the pre-commit.
