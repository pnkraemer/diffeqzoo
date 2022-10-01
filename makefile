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

test:
	BACKEND=NumPy pytest
	BACKEND=JAX pytest
	python -m doctest *.md
	python -m doctest diffeqzoo/*.py
	python -m doctest docs/source/getting_started/*.md

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
	rm docs/source/example_notebooks/solve_ivp_with_jax_ode.ipynb
	rm docs/source/example_notebooks/solve_ivp_with_scipy.ipynb
	rm docs/source/example_notebooks/solve_ivp_with_diffrax.ipynb
doc:
	cd docs; make html
