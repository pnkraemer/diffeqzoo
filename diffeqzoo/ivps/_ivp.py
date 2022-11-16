"""Initial value problem."""

from typing import Any, Callable, Iterable, NamedTuple, Union

from diffeqzoo import _vector_fields, backend, transform


class InitialValueProblem(NamedTuple):
    vector_field: Callable
    initial_values: Union[Iterable, Any]  # u0 or (u0, du0, ddu0, ...)
    time_span: Iterable  # (t0, t1)
    vector_field_args: Iterable = ()
