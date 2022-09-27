# Build the documentation

This project uses ``mkdocs`` to generate the documentation. See the [MkDocs reference](https://www.mkdocs.org/) for some useful info.


!!! question "Can we use ``mkdocs`` in a better way?"

    If you know how to improve our usage of ``mkdocs``, please let us know!


## Install the dependencies

To install the library with all doc-related dependencies,
you can use ``poetry`` [as explained here](dependency_management.md).

```commandline
poetry install --extras doc
```

This installs all documentation-related dependencies, such as `mkdocs`, `mkdocstrings`, and so on.


## Edit the documentation

To edit the documentation, make the changes to the docstrings in the source code or to the Markdown files in the ``docs/`` folder.

If you create a new markdown file that is supposed to become a new site in the docs, reference it under the ``nav`` tab in ``mkdocs.yml``.

## Preview the documentation

To preview the changes, run:

```commandline
mkdocs serve
```

If you don't like what you're seeing, make some more changes.
Once you're happy with the result, build the docs.

## Build the documentation

To build the documentation, run

```commandline
mkdocs build
```

# Configure MkDocs

To configure the parameters of the documentation, edit ``mkdocs.yml``.
See the [MkDocs reference](https://www.mkdocs.org/).
