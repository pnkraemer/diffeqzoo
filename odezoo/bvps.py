"""Boundary value problems."""

from typing import Any, Callable, Iterable, NamedTuple, Union

from odezoo import _vector_fields, backend, transform


class _SeparableBoundaryValueProblem(NamedTuple):
    vector_field: Callable
    boundary_conditions: Iterable  # (g0, g1) or ((L, l), (R, r))
    time_span: Iterable  # (t0, t1)
    vector_field_args: Iterable = ()


def bratu(*, time_span=(0.0, 1.0), parameters=(1.0,)):

    # Boundary conditions
    eye_d = backend.numpy.asarray([1, 0])
    u0 = backend.numpy.asarray(0.0)
    umax = backend.numpy.asarray(0.0)

    return _SeparableBoundaryValueProblem(
        vector_field=_vector_fields.bratu,
        boundary_conditions=((eye_d, u0), (eye_d, umax)),
        vector_field_args=parameters,
        time_span=time_span,
    )
    return f_bratu, (t0, tmax), (u0, umax)


def bratu_autonomous_api(*, time_span=(0.0, 1.0), parameters=(1.0,)):

    # Boundary conditions
    eye_d = backend.numpy.asarray([1, 0])
    u0 = backend.numpy.asarray(0.0)
    umax = backend.numpy.asarray(0.0)

    return _SeparableBoundaryValueProblem(
        vector_field=_vector_fields.bratu_autonomous_api,
        boundary_conditions=((eye_d, u0), (eye_d, umax)),
        vector_field_args=parameters,
        time_span=time_span,
    )
    return f_bratu, (t0, tmax), (u0, umax)
