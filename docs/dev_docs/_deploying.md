
# Deploying



A reminder for the maintainers on how to deploy.

!!! warning

    This document is probably not of interest to you.

Make sure all your changes are committed (including an entry in CHANGELOG.md).
Then run:

```commandline
poetry run bump2version patch # possible: major / minor / patch
git push
git push --tags
```

GitHub Actions will then deploy to PyPI if tests pass.
