"""Utility functions to manipulate docstrings."""


def add_long_description(docstring, /, *, long_description):
    """Add a long description to a docstring.


    This could be some mathematical content, or a warning about
    using a function in a specific way.
    """
    if docstring is None:
        return long_description
    n = docstring.find("\n")

    if n == -1:
        return docstring + long_description
    return docstring[:n] + long_description + docstring[n:]


def replace_short_summary(docstring, /, *, short_summary):
    if docstring is None:
        return short_summary
    n = docstring.find("\n")

    if n == -1:
        return short_summary
    return short_summary + docstring[n:]
