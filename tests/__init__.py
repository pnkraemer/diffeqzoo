"""Set the backend implementation for the tests."""
import os

import odezoo

if "BACKEND" not in os.environ:
    expected_command = 'BACKEND="NumPy" pytest'
    raise KeyError(
        "Choose an environment variable to set the backend. "
        "Otherwise the tests don't run. "
        f"For example, run the tests with '{expected_command}'"
    )

# Set the array backend for the tests.
odezoo.set_backend(os.environ["BACKEND"].lower())