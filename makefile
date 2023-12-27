sources = diffeqzoo

.PHONY: format lint test pre-commit doc clean

format:
	isort .
	black .
	nbqa black docs/
	nbqa isort docs/
	jupytext --sync docs/source/example_notebooks/*

lint:
	isort --check --diff .
	black --check --diff .
	nbqa isort --check --diff .
	nbqa black --check --diff .
	nbqa flake8 docs/
	# Opt-in for specific pylint checks. Exclude unused ones:
	#   invalid-names: maths-variables are not always snake-case
	#   fixme: todos can be scattered through the code base
	#   attribute-defined-outside-init: we change sth.__doc__ when transforming ODEs
	#   line-too-long: handled by black/flake8
	#   duplicate-code: a lot of functions wrap vector fields equally, but this is desired
	#   too-many-arguments: some vector fields have many parameters
	pylint diffeqzoo --disable=invalid-name,fixme,attribute-defined-outside-init,line-too-long,duplicate-code,too-many-arguments
test:
	BACKEND="NumPy" pytest
	BACKEND="Jax" JAX_PLATFORM_NAME="cpu" pytest
	JAX_PLATFORM_NAME="cpu" python -m doctest *.md
	JAX_PLATFORM_NAME="cpu" python -m doctest diffeqzoo/*.py
	JAX_PLATFORM_NAME="cpu" python -m doctest docs/source/getting_started/*.md

example:
	jupytext --sync docs/source/example_notebooks/*
	jupytext --execute docs/source/example_notebooks/*

pre-commit:
	pre-commit autoupdate
	pre-commit run --all-files

clean:
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -rf dist site
	rm -rf docs/source/api/
	rm -rf docs/_build/
	cd docs; make clean
	cd docs/source/example_notebooks; rm -rf .ipynb_checkpoints
	jupytext --sync docs/source/example_notebooks/*
	rm docs/source/example_notebooks/solve_bvp_with_scipy.ipynb
	rm docs/source/example_notebooks/solve_ivp_with_jax.ipynb
	rm docs/source/example_notebooks/solve_ivp_with_scipy.ipynb
	rm docs/source/example_notebooks/solve_ivp_with_diffrax.ipynb
doc:
	pip install -r docs/requirements-sphinx-build.txt
	cd docs; make html
