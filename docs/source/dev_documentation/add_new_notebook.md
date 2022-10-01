# Adding an example Jupyter notebook

First off: thank you for considering a contribution to `diffeqzoo`'s examples.
The more examples we have, the better! Here is how to add a notebook.

1. Create the .ipynb file in `docs/source/example_notebooks` and fill it with content
2. Sync the notebook to a markdown file via jupytext. Check out [these instructions](https://jupytext.readthedocs.io/en/latest/paired-notebooks.html). This is useful to keep the git diff clean and legible.
3. Make sure the `example` dependencies in `setup.cfg` are sufficient. We try to keep the optional dependencies small, but are not as strict with the `example` dependencies as with the remaining ones.
4. Add the notebook to `docs/source/index.rst`. If you skip this step, the notebook will not be rendered in the docs.
5. Add the .ipynb file to the .gitignore and to the `clean` job in the makefile. Do this after syncing it to a markdown version.
6. Check that everything passes the quality checks via `make format example doc lint`.
7. Make a pull request with your changes.
