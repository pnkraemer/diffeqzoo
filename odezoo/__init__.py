"""ODE zoo."""

BACKEND_HAS_BEEN_SET = False
"""We choose the backend (NumPy/JAX) exactly once. Not zero times, not twice."""

numpy_like = "None"
"""This will become the backend implementation."""

def set_backend(backend_name, /):
    """Set the backend to either JAX or NumPy."""
    global numpy_like, BACKEND_HAS_BEEN_SET

    # Check that the backend is still to be set.
    if BACKEND_HAS_BEEN_SET:
        raise RuntimeError("Backend has been set already. Can't change it anymore.")

    # Assert that the backend is one of the known ones.
    if backend_name.lower() not in ("jax", "numpy"):
        raise ValueError(f"Backend {backend_name} is unknown.")

    # Choose implementation
    if backend_name.lower() == "jax":
        import jax.numpy as numpy_like
    else:
        import numpy as numpy_like
    BACKEND_HAS_BEEN_SET = True
