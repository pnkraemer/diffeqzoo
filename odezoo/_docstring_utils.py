"""Utility functions to manipulate docstrings."""


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
