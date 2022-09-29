sources = odezoo

.PHONY: format lint test pre-commit clean
test: format lint unittest

format:
	isort .
	black .

lint:
	isort --check --diff .
    black --check --diff .
    flake8

test:
	pytest

pre-commit:
	pre-commit run --all-files

clean:
	rm -rf {% if cookiecutter.use_mypy == 'y' -%}.mypy_cache {% endif -%} .pytest_cache
	rm -rf *.egg-info
	rm -rf dist site
    cd docs
    make clean
    rm -rf source/api/
    cd ..
