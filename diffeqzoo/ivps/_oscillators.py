"""Oscillators."""

from diffeqzoo import _vector_fields, backend, transform
from diffeqzoo.ivps import _ivp


def lotka_volterra(
    *,
    initial_values=(20.0, 20.0),
    time_span=(0.0, 20.0),
    parameters=(0.5, 0.05, 0.5, 0.05),
):
    r"""Construct the Lotka--Volterra / predator-prey model.

    The Lotka--Volterra equations describe the dynamics of biological systems
    in which two species, predators and prey, interact.

    The original version is due to Lotka (1910).
    Its application to predator-Prey dynamics is due to Lotka (1925).
    The same model was discovered by Volterra (1926).

    .. collapse:: BibTex for Lotka (1910)

        .. code-block:: tex

            @article{lotka1910contribution,
                title={Contribution to the theory of periodic reactions},
                author={Lotka, Alfred J},
                journal={The Journal of Physical Chemistry},
                volume={14},
                number={3},
                pages={271--274},
                year={1910},
                publisher={ACS Publications}
            }

    .. collapse:: BibTex for Lotka (1925)

        .. code-block:: tex

            @book{lotka1925elements,
                title={Elements of physical biology},
                author={Lotka, Alfred James},
                year={1925},
                publisher={Williams \& Wilkins}
            }

    .. collapse:: BibTex for Volterra (1926)

        .. code-block:: tex

            @book{volterra1926variazioni,
                title={Variazioni e fluttuazioni del numero d'individui in specie animali conviventi},
                author={Volterra, Vito},
                year={1926},
                publisher={Societ{\`a} anonima tipografica" Leonardo da Vinci"}
            }

    """
    initial_values = backend.numpy.asarray(initial_values)

    return _ivp.InitialValueProblem(
        vector_field=_vector_fields.lotka_volterra,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
    )


def fitzhugh_nagumo(
    *, initial_values=(-1.0, 1.0), time_span=(0.0, 20.0), parameters=(0.2, 0.2, 3.0)
):
    r"""Construct the FitzHugh-Nagumo model.

    The FitzHugh-Nagumo model is a simple example of an excitable system
    (for example: a neuron).
    This simplified, 2d-version of the Hodgkin-Huxley model
    (which describes the spike generation in squid giant axons)
    was suggested by FitzHugh (1961) and Nagumo et al. (1962)

    The following bibtex(s) point to the original papers about
    the FitzHugh-Nagumo models. (Source: Google Scholar).

    .. collapse:: BibTex for FitzHugh (1961)

        .. code-block:: tex

            @article{fitzhugh1961impulses,
                title={Impulses and physiological states in
                theoretical models of nerve membrane},
                author={FitzHugh, Richard},
                journal={Biophysical Journal},
                volume={1},
                number={6},
                pages={445--466},
                year={1961},
                publisher={Elsevier}
            }

    .. collapse:: BibTex for Nagumo et al. (1962)

        .. code-block:: tex

            @article{nagumo1962active,
                title={An active pulse transmission line simulating nerve axon},
                author={Nagumo, Jinichi and Arimoto, Suguru and Yoshizawa, Shuji},
                journal={Proceedings of the IRE},
                volume={50},
                number={10},
                pages={2061--2070},
                year={1962},
                publisher={IEEE}
            }

    """
    initial_values = backend.numpy.asarray(initial_values)

    return _ivp.InitialValueProblem(
        vector_field=_vector_fields.fitzhugh_nagumo,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
    )
