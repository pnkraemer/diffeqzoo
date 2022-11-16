"""Chaotic systems."""

from diffeqzoo import _vector_fields, backend, transform
from diffeqzoo.ivps import _ivp


def lorenz96(
    *,
    initial_values=None,
    time_span=(0.0, 30.0),
    num_variables=10,
    forcing=8.0,
    perturb=0.01,
):
    """Construct the Lorenz96 model.

    The Lorenz96 is a chaotic initial value problem, due to Lorenz (1996),
    and commonly used as a testproblem in data assimilation.

    .. collapse:: BibTex for Lorenz (1996)

        .. code-block:: tex

            @inproceedings{lorenz1996predictability,
                title={Predictability: A problem partly solved},
                author={Lorenz, Edward N},
                booktitle={Proceedings of the Seminar on Predictability},
                volume={1},
                number={1},
                year={1996}
            }

    """
    if initial_values is None:
        initial_values = _lorenz96_chaotic_u0(
            forcing=forcing, num_variables=num_variables, perturb=perturb
        )

    return _ivp.InitialValueProblem(
        vector_field=_vector_fields.lorenz96,
        vector_field_args=(forcing,),
        initial_values=initial_values,
        time_span=time_span,
    )


def _lorenz96_chaotic_u0(*, forcing, num_variables, perturb):
    u0_equilibrium = backend.numpy.ones(num_variables) * forcing
    return backend.numpy.concatenate(
        [backend.numpy.asarray([u0_equilibrium[0] + perturb]), u0_equilibrium[1:]]
    )


def lorenz63(
    *,
    initial_values=(0.0, 1.0, 1.05),
    time_span=(0.0, 20.0),
    parameters=(10.0, 28.0, 8.0 / 3.0),
):
    """Construct the Lorenz63 model.

    The Lorenz63 model, initially used for atmospheric convection,
    is a common example of an initial value problem that
    has a chaotic solution.

    It was proposed by Lorenz (1963).

    .. collapse:: BibTex for Lorenz (1963)

        .. code-block:: tex

            @article{lorenz1963deterministic,
                title={Deterministic nonperiodic flow},
                author={Lorenz, Edward N},
                journal={Journal of atmospheric sciences},
                volume={20},
                number={2},
                pages={130--141},
                year={1963}
            }
    """
    initial_values = backend.numpy.asarray(initial_values)

    return _ivp.InitialValueProblem(
        vector_field=_vector_fields.lorenz63,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
    )


def roessler(
    *,
    initial_values=(1.0, 0.0, 0.0),
    time_span=(0.0, 100.0),
    parameters=(0.1, 0.1, 14.0),
):
    """Construct the Roessler model.

    The Roessler model is a three-dimensional chaotic system,
    that was proposed by Roessler (1990).

    .. collapse:: BibTex for Roessler (1990)

        .. code-block:: tex

            @article{rossler1976equation,
                title={An equation for continuous chaos},
                author={R{\"o}ssler, Otto E},
                journal={Physics Letters A},
                volume={57},
                number={5},
                pages={397--398},
                year={1976},
                publisher={Elsevier}
            }
    """
    initial_values = backend.numpy.asarray(initial_values)

    return _ivp.InitialValueProblem(
        vector_field=_vector_fields.roessler,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
    )
