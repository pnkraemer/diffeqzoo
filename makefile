sources = odezoo

.PHONY: test format lint unittest coverage pre-commit clean
test: format lint unittest

format:
	isort $(sources) tests
	black $(sources) tests
	nbqa black docs/example_notebooks/
	nbqa isort docs/example_notebooks/
	jupytext --sync docs/example_notebooks/*

lint:
	flake8 $(sources) tests
	# per-file does not work https://nbqa.readthedocs.io/en/latest/known-limitations.html
	nbqa flake8 docs/example_notebooks/  --ignore=D103,D100

unittest:
	pytest

coverage:
	pytest --cov=$(sources) --cov-branch --cov-report=term-missing tests

pre-commit:
	pre-commit run --all-files

clean:
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -rf .tox dist site
	rm -rf coverage.xml .coverage
	rm -rf docs/example_notebooks/.ipynb_checkpoints
