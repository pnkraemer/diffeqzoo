"""Differential equation problem zoo."""
import warnings as _warnings  # don't expose warnings in the namespace


class backend:
    """Backend implementation of NumPy-like functions via either numpy or jax.

    >>> from diffeqzoo import backend  # one-stop-shop for numpy, scipy, etc.

    >>> # backend.numpy
    >>> backend.select("jax")
    >>> backend.numpy.asarray(2.)
    DeviceArray(2., dtype=float32, weak_type=True)

    >>> # Change backend; backend.select("numpy") is not allowed anymore!
    >>> backend.change_to("numpy")
    >>> backend.numpy.asarray(2.)
    array(2.)

    To learn about its properties and methods, run :code:`help(diffeqzoo.backend)`.


    Note
    ----
    If you want to change the backend, use
    :meth:`backend.change_to(choice)` instead of :meth:`backend.select`.
    The reason for this distinction is that we want to reduce the number
    of accidental backend-changes.
    Ideally, you select the backend once, and only once.
    """

    def __init__(self):
        self._backend_name = None
        self._numpy_backend = None
        self._np_import_cache = None
        self._jnp_import_cache = None

    @property
    def has_been_selected(self):
        """Whether a backend implementation has been selected.

        Once a backend implementation has been selected,
        :attr:`backend.numpy` is available, and backend.select() is disabled.

        If you want to change an existing backend from NumPy to JAX
        (uhm, why would you?), use :meth:`backend.change_to`.
        """
        return self._numpy_backend is not None

    def select(self, backend_name, /):
        """Select a backend implementation.

        This method is available exactly once.
        After calling :meth:`select` once, the backend implementation
        can be changed with :meth:`change_to`.
        """
        if self.has_been_selected:
            raise RuntimeError("A backend has been selected already.")
        self._select_backend(backend_name.lower())

    # todo: remove? Why would one want this
    def change_to(self, backend_name, /):
        """Change the backend implementation."""
        if not self.has_been_selected:
            raise RuntimeError(
                "The first backend-selection must be via `backend.select()`."
            )

        # If a user thinks they are changing the implementation,
        # but they are not, a warning is appropriate.
        if self._backend_name == backend_name.lower():
            _warnings.warn(
                "The desired backend implementation matches the current selection."
            )

        # After all those warnings, finally change the backend.
        self._select_backend(backend_name.lower())

    def _select_backend(self, backend_name, /):
        """Select a backend.

        This method is where the actual import-assign magic happens.

        If you're looking for global variables and dirty hacks, look here.
        """
        if backend_name == "jax":
            # Import the module (only now!
            # It should be usable if `jax` is not installed)
            import jax.numpy as jnp

            # Save the imported module. It might be needed later.
            self._jnp_import_cache = jnp

            # Assign the NumPy implementation.
            self._numpy_backend = jnp
            self._backend_name = backend_name

        elif backend_name == "numpy":

            # Import the module (only now! see comment above)
            import numpy as np

            # Save the imported module. It might be needed later.
            self._np_import_cache = np

            # Assign the NumPy implementation.
            self._numpy_backend = np
            self._backend_name = backend_name
        else:
            raise ValueError("Backend implementation not known.")

    @property
    def numpy(self):
        """Access to NumPy implementation."""
        if not self.has_been_selected:
            raise Exception("A backend implementation has not been selected yet.")
        return self._numpy_backend


_docs = backend.__doc__
# The variable has the same name as the class to enforce a singleton.
backend = backend()
backend.__doc__ = _docs
