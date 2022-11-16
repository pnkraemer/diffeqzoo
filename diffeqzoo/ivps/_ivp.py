"""Initial value problem type."""

from typing import Any, Callable, Iterable, NamedTuple, Union


class InitialValueProblem(NamedTuple):
    """Initial value problem."""

    vector_field: Callable
    initial_values: Union[Iterable, Any]  # u0 or (u0, du0, ddu0, ...)
    time_span: Iterable  # (t0, t1)
    vector_field_args: Iterable = ()
