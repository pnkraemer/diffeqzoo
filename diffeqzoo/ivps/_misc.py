"""Miscellaneous problems."""


from diffeqzoo import _vector_fields, backend, transform
from diffeqzoo.ivps import _ivp


def logistic(*, initial_value=0.1, time_span=(0.0, 2.5), parameters=(1.0, 1.0)):
    """Construct the logistic ODE model.

    The logistic ODE is a differential equation model whose solution
    exhibits exponential growth early in the time interval,
    and approaches a constant value over time.

    It is a differential equation version of the sigmoid and the logistic function.
    The logistic ODE has a closed-form solution.

    .. note::
        **Help wanted!**

        If you know which paper/book to cite when the logistic ODE is used
        in a paper, please consider making a contribution.

    """
    initial_value = backend.numpy.asarray(initial_value)

    return _ivp.InitialValueProblem(
        vector_field=_vector_fields.logistic,
        vector_field_args=parameters,
        initial_values=initial_value,
        time_span=time_span,
    )
