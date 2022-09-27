# Working with the notebooks

``ODEZoo`` uses a lot of ``nbqa`` and ``jupytext`` to assure the code quality
in the notebooks, and to make ``MkDocs`` display them well.

What does this mean?


## Executing notebooks

The easiest way to execute the notebooks is to clone the repository, open jupyter {lab/notebook},
and run the notebooks.

To execute the notebooks from the command line,
make sure all dependencies are installed via `poetry install --extras execute-examples`, and run

```commandline
poetry run jupytext --execute docs/example_notebooks/*.ipynb
```

which executes all notebooks and fails if it does not work.

This also happens under the hood whenever we build the documentation with `poetry run mkdocs serve` (respectively `mkdocs serve`).
It is a crucial part of the CI.


## Formatting notebooks



* If you apply the rest of your linting and formatting via `poetry install --extras [test,dev,doc]`
  and then, e.g., `poetry run make format` or `poetry run tox -e format`,
  your notebooks are formatted automatically.
  Under the hood, we run something like `nbqa {black, isort flake8} docs/*.ipynb` as part of the makefile and in tox.
* The same applies to linting (replace `format` with `lint` in the above.)

## Adding a new notebook

To add a new notebook, create the notebook and fill it with content.
Then
* Pair it with a markdown file via jupytext (for version control)
* Add it to the ``nav`` tab in `mkdocs.yml` to display it in the docs
* Fix all errors that appear from formatting and linting (see above)

And afterwards, it is part of the team.
To edit an existing notebook, simply edit the notebook and let the CI do the rest.
