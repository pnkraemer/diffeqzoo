"""Transform ODE models into equivalent versions."""

import inspect

from odezoo import backend


def second_to_first_order_auto(ivp_fn, /, short_summary=None):
    """Transform a second-order, autonomous differential equation \
    into an equivalent first-order form."""

    def ivp_fn_transformed(*, initial_values=None, **kwargs):

        if initial_values is not None:
            initial_values = backend.numpy.split(initial_values, 2)

        ivp_untransformed = ivp_fn(initial_values=initial_values, **kwargs)
        f_untransformed, u0s, tspan, f_args = ivp_untransformed

        # the new stuff
        if u0s[0].ndim == 0:
            initial_values = backend.numpy.stack(u0s)
        else:
            initial_values = backend.numpy.concatenate(u0s, axis=None)
        vector_field = second_to_first_order_vf_auto(
            f_untransformed, short_summary=short_summary
        )

        return ivp_untransformed._replace(
            initial_values=initial_values, vector_field=vector_field
        )

    # Update some problem-specific description.
    # This updates many of the things that functools.wraps updates,
    # but the signatures of the untransformed and the transformed functions differ,
    # which is why functools.wraps would generate the wrong docs.

    # Read the name of the current function (to be added to the disclaimer)
    this_function_name = inspect.currentframe().f_code.co_name
    disclaimer = _disclaimer(
        fun_original=ivp_fn.__name__, fun_wrapper=__name__ + f".{this_function_name}"
    )

    # Assign the transformed function to the same module as the
    # untransformed function (makes the transformed function appear in docs)
    ivp_fn_transformed.__module__ = ivp_fn.__module__

    # Add a disclaimer that the function has been transformed to first-order
    ivp_fn_transformed.__doc__ = ivp_fn.__doc__
    ivp_fn_transformed = long_description(disclaimer)(ivp_fn_transformed)

    # If the user desires, replace the short summary in the docstring
    if short_summary is not None:
        ivp_fn_transformed.__doc__ = replace_short_summary(
            ivp_fn_transformed.__doc__, short_summary=short_summary
        )

    return ivp_fn_transformed


def second_to_first_order_vf_auto(fn, /, short_summary=None):
    """Transform the vector-field of a second-order, \
    autonomous differential equation into an equivalent first-order form."""

    def fn_transformed(u, *args):
        u, du = backend.numpy.split(u, 2)
        ddu = fn(u, du, *args)
        if du.ndim == 0:
            return backend.numpy.stack((du, ddu))
        return backend.numpy.concatenate((du, ddu), axis=None)

    # Update some problem-specific description.
    # This updates many of the things that functools.wraps updates,
    # but the signatures of the untransformed and the transformed functions differ,
    # which is why functools.wraps would generate the wrong docs.

    # Read the name of the current function (to be added to the disclaimer)
    this_function_name = inspect.currentframe().f_code.co_name
    disclaimer = _disclaimer(
        fun_original=fn.__name__, fun_wrapper=__name__ + f".{this_function_name}"
    )

    # Assign the transformed function to the same module as the
    # untransformed function (makes the transformed function appear in docs)
    fn_transformed.__module__ = fn.__module__

    # Add a disclaimer that the function has been transformed to first-order
    fn_transformed = long_description(disclaimer)(fn_transformed)

    # If the user desires, replace the short summary in the docstring
    if short_summary is not None:
        fn_transformed.__doc__ = replace_short_summary(
            fn_transformed.__doc__, short_summary=short_summary
        )

    return fn_transformed


def _disclaimer(*, fun_original, fun_wrapper):
    return f"""


    Warning
    -------
    This problem has been generated by wrapping the function
    :func:`{fun_original}` through the function
    :func:`{fun_wrapper}`.

    The problem is not originally of first order.
    If you have access to solvers for second-order problems, it might
    be more efficient to solve the original problem.


    """


def long_description(description, /):
    """Add a long description to the docstring of a function.

    Use this function as a decorator.
    """

    def add_long_description(obj, /):
        """Add a long description to a docstring.

        This could be some mathematical content, or a warning about
        using a function in a specific way.
        """
        obj.__doc__ = construct_docstring(obj)
        return obj

    def construct_docstring(obj, /):
        if obj.__doc__ is None:
            return description
        n = obj.__doc__.find("\n")

        if n == -1:
            return obj.__doc__ + description
        return obj.__doc__[:n] + description + obj.__doc__[n:]

    return add_long_description


def replace_short_summary(docstring, /, *, short_summary):
    """Replace the short summary in a docstring with a new short summary."""
    if docstring is None:
        return short_summary
    n = docstring.find("\n")

    if n == -1:
        return short_summary
    return short_summary + docstring[n:]
