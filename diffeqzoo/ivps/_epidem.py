"""Epidemiological models."""


from diffeqzoo import _vector_fields, backend
from diffeqzoo.ivps import _ivp


def sir(
    *, initial_values=(998.0, 1.0, 1.0), time_span=(0.0, 200.0), beta=0.3, gamma=0.1
):
    """Construct the SIR model without vital dynamics.

    The SIR model describes the spread of a virus in a population.
    More specifically, it describes how populations move from being
    susceptible, to being infected, to being removed from the population.

    It was first proposed by Kermack and McKendrick (1927).

    .. collapse:: BibTex for Kermack and McKendrick (1927).

        .. code-block:: tex

            @article{kermack1927contribution,
                title={A contribution to the mathematical theory of epidemics},
                author={Kermack, William Ogilvy and McKendrick, Anderson G},
                journal={Proceedings of the Royal Society of London. Series A},
                volume={115},
                number={772},
                pages={700--721},
                year={1927},
                publisher={The Royal Society London}
            }


    See Also
    --------
    ivps.seir
    ivps.sird

    """

    initial_values = backend.numpy.asarray(initial_values)
    parameters = (beta, gamma, backend.numpy.sum(initial_values[0]))

    return _ivp.InitialValueProblem(
        vector_field=_vector_fields.sir,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
    )


def seir(
    *,
    initial_values=(998.0, 1.0, 1.0, 1.0),
    time_span=(0.0, 200.0),
    alpha=0.3,
    beta=0.3,
    gamma=0.1,
):
    """Construct the SEIR model.

    The SEIR model is a variant of the SIR model,
    but additionally includes a compartment of the population that
    has been exposed to the virus (but is not infected yet).
    See Hethcode (2000).

    .. collapse:: BibTex for Hethcote (2000).

        .. code-block:: tex

            @article{hethcote2000mathematics,
                title={The Mathematics of Infectious Diseases},
                author={Hethcote, Herbert W},
                journal={SIAM Review},
                volume={42},
                number={4},
                pages={599--653},
                year={2000},
                publisher={SIAM}
            }

    Note
    ----
    If you know a more suitable original reference, please make some noise!

    See Also
    --------
    ivps.sir
    ivps.sird

    """

    initial_values = backend.numpy.asarray(initial_values)
    parameters = (alpha, beta, gamma, backend.numpy.sum(initial_values[0]))

    return _ivp.InitialValueProblem(
        vector_field=_vector_fields.seir,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
    )


def sird(
    *,
    initial_values=(998.0, 1.0, 1.0, 0.0),
    time_span=(0.0, 200.0),
    beta=0.3,
    gamma=0.1,
    eta=0.005,
):
    """Construct the SIRD model.

    The SIRD model is a variant of the SIR model that
    distinguishes the recovered compartment from the deceased compartment
    in the population.
    See Hethcode (2000).

    .. collapse:: BibTex for Hethcote (2000).

        .. code-block:: tex

            @article{hethcote2000mathematics,
                title={The Mathematics of Infectious Diseases},
                author={Hethcote, Herbert W},
                journal={SIAM Review},
                volume={42},
                number={4},
                pages={599--653},
                year={2000},
                publisher={SIAM}
            }

    Note
    ----
    If you know a more suitable original reference, please make some noise!

    See Also
    --------
    ivps.sir
    ivps.seir
    """

    initial_values = backend.numpy.asarray(initial_values)
    parameters = (beta, gamma, eta, backend.numpy.sum(initial_values[0]))

    return _ivp.InitialValueProblem(
        vector_field=_vector_fields.sird,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
    )
