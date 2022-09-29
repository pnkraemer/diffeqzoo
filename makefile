sources = odezoo

.PHONY: format lint test pre-commit clean doc

format:
	isort .
	black .
	nbqa black docs/
	nbqa flake8 docs/
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
	python -m doctest odezoo/*.py

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
	cd docs; make clean
