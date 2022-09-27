
# Development Setup

Ready to contribute? Here's how to set up `odezoo` for local development.

1. Fork the `odezoo` repo on GitHub.
2. Clone your fork locally

    ```
    $ git clone git@github.com:your_name_here/odezoo.git
    ```

3. Ensure [poetry](https://python-poetry.org/docs/) is installed.
4. Install dependencies and start your virtualenv:

    ```
    $ poetry install -E test -E doc -E dev
    ```

5. Create a branch for local development:

    ```
    $ git checkout -b name-of-your-bugfix-or-feature
    ```

    Now you can make your changes locally.

6. When you're done making changes, check that your changes pass the
   tests, including testing other Python versions, with tox:

    ```
    $ poetry run tox
    ```

7. Commit your changes and push your branch to GitHub:

    ```
    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature
    ```

8. Submit a pull request through the GitHub website.

## Running a Subset of Tests

To execute a specific test, run

```
$ poetry run pytest tests/test_odezoo.py
```

To quickly execute all unittests or apply formatters locally, use the makefile (make sure all dependencies are installed)

```commandline
poetry run make unittest
poetry run make lint
poetry run make format
poetry run make pre-commit
```

To clean up the repo (remove .tox, .pytest_cache, egg.info, ...), use the makefile
```commandline
make clean
```
