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


def affine_independent(*, initial_values=1.0, time_span=(0.0, 1.0), a=1.0, b=0.0):
    r"""Construct an IVP with an affine vector field, \
    where each dimension is treated independently.

    In Python code, this means :code:`f(y, a, b)=a * y + b`.

    By default, this is a scalar problem.
    Change the initial value to make this a multidimensional problem.
    """
    initial_values = backend.numpy.asarray(initial_values)

    return _ivp.InitialValueProblem(
        vector_field=_vector_fields.affine_independent,
        vector_field_args=(a, b),
        initial_values=initial_values,
        time_span=time_span,
    )


def affine_dependent(
    *, initial_values=(1.0, 1.0), time_span=(0.0, 1.0), A=((1, 0), (0, 1)), b=(0, 0)
):
    r"""Construct an IVP with an affine vector field.

    In Python code, this means :code:`f(y, A, b) = A @ y + b`.

    By default, this is a 2d-problem.
    Change the initial value to make this a multidimensional problem.
    """

    initial_values = backend.numpy.asarray(initial_values)
    A = backend.numpy.asarray(A)
    b = backend.numpy.asarray(b)

    return _ivp.InitialValueProblem(
        vector_field=_vector_fields.affine_dependent,
        vector_field_args=(A, b),
        initial_values=initial_values,
        time_span=time_span,
    )


def neural_ode_mlp(
    *,
    initial_values=((0.0,)),
    time_span=(0.0, 1.0),
    scale=1.0,
    layer_sizes=(2, 20, 1),
    seed=1234,
):
    r"""Construct an IVP with a neural ODE vector field.

    The vector field is given by a neural network.

    The neural network is a multi-layer perceptron with `tanh` activation functions.

    We implement the dynamics used in the "implicit-layers" tutorial:
    ``http://implicit-layers-tutorial.org/neural_odes/``.

    Note
    ----
    The neural network is not trained in this function. This function only constructs the IVP.
    """
    initial_values = backend.numpy.asarray(initial_values)
    params = _init_random_params(scale, layer_sizes, seed)
    return _ivp.InitialValueProblem(
        vector_field=_vector_fields.neural_ode_mlp,
        vector_field_args=(params,),
        initial_values=initial_values,
        time_span=time_span,
    )


def _init_random_params(scale, layer_sizes, seed):
    if backend.name == "jax":  # pylint: disable=comparison-with-callable  # ??
        params = _init_random_params_jax(
            scale=scale, layer_sizes=layer_sizes, seed=seed
        )
    elif backend.name == "numpy":  # pylint: disable=comparison-with-callable  # ??
        params = _init_random_params_numpy(
            scale=scale, layer_sizes=layer_sizes, seed=seed
        )
    else:
        msg1 = f"Neural ODE is not compatible with the current backend {backend.name}. "
        msg2 = "Please use `jax` or `numpy` and/or consider raising an Issue."
        raise ValueError(msg1 + msg2)
    return params


def _init_random_params_numpy(*, scale, layer_sizes, seed):
    rng = backend.random.default_rng(seed=seed)
    return [
        (
            scale * rng.standard_normal(size=(m, n)),
            scale * rng.standard_normal(size=(n,)),
        )
        for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])
    ]


def _init_random_params_jax(*, scale, layer_sizes, seed):
    key = backend.random.PRNGKey(seed=seed)
    return [
        (
            scale * backend.random.normal(key, shape=(m, n)),
            scale * backend.random.normal(key, shape=(n,)),
        )
        for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])
    ]
