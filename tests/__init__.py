"""Set the backend implementation for the tests."""
import os

import diffeqzoo

if "BACKEND" not in os.environ:
    expected_command_numpy = 'BACKEND="NumPy" pytest'
    expected_command_jax = 'BACKEND="JAX" pytest'
    raise KeyError(
        "Choose an environment variable to set the backend. "
        "Otherwise the tests don't run. "
        "For example, run the tests with "
        f"'{expected_command_numpy}' or '{expected_command_jax}"
    )

# Set the array backend for the tests.
diffeqzoo.backend.select(os.environ["BACKEND"].lower())
