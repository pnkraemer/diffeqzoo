"""Oscillating systems."""

from diffeqzoo import backend, transform, vector_fields
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
        vector_field=vector_fields.lotka_volterra,
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
        vector_field=vector_fields.fitzhugh_nagumo,
        vector_field_args=parameters,
        initial_values=initial_values,
        time_span=time_span,
    )


def van_der_pol(
    *, stiffness_constant=1.0, initial_values=(2.0, 0.0), time_span=(0.0, 6.3)
):
    r"""Construct the Van-der-Pol system as a second-order differential equation.

    The Van-der-Pol system is a non-conservative oscillator subject to non-linear damping.
    It is a popular benchmark problem, because it involves a parameter :math:`\mu`
    (the "stiffness constant") which governs the stiffness of the problem.
    For :math:`\mu ~ 1`, the problen is not stiff.
    For large values (e.g. :math:`\mu ~ 10^6`) the problem is stiff.
    It was first published by Van der Pol (1920).

    .. collapse:: BibTex for Van der Pol (1920).

        .. code-block:: tex

            @article{van1920theory,
                title={Theory of the amplitude of free and forced triode vibrations},
                author={Van der Pol, Balthasar},
                journal={Radio Review},
                volume={1},
                pages={701--710},
                year={1920}
            }

    """

    u0, du0 = initial_values
    u0 = backend.numpy.asarray(u0)
    du0 = backend.numpy.asarray(du0)
    initial_values = (u0, du0)

    return _ivp.InitialValueProblem(
        vector_field=vector_fields.van_der_pol,
        vector_field_args=(stiffness_constant,),
        initial_values=initial_values,
        time_span=time_span,
    )


van_der_pol_first_order = transform.second_to_first_order_auto(
    van_der_pol,
    short_summary="Construct the Van-der-Pol system as a first-order differential equation.",
)


def goodwin(
    *,
    initial_values=(0.0, 0.0),
    time_span=(0.0, 25.0),
    r=10,
    a1=1.0,
    a2=3.0,
    alpha=0.5,
    k=(1.0,),
):
    r"""Construct the Goodwin Oscillator dynamics.

    Describes a mechanism for periodic protein expression described by Goodwin (1965).
    The first dimension describes the mRNA concentration,
    the last dimension the protein inhibiting mRNA production,
    and the remaining dimensions correspond to intermediate protein species.
    r > 8 leads to oscillatory behavior.
    It is a n-dimensional ODE initial value problem.
    The length of the `k` needs to be `len(initial_values)-1`.

    Common problem for parameter inference, where the posterior has a multimodal distribution.


    .. collapse:: BibTex for Goodwin (1965)

        .. code-block:: tex

            @article{goodwin1965oscillatory,
                title = {Oscillatory behavior in enzymatic control processes},
                journal = {Advances in Enzyme Regulation},
                volume = {3},
                pages = {425-437},
                year = {1965},
                author = {Brian C. Goodwin},
            }
    """

    initial_values = backend.numpy.asarray(initial_values)

    k = backend.numpy.asarray(k)
    return _ivp.InitialValueProblem(
        vector_field=vector_fields.goodwin,
        vector_field_args=(r, a1, a2, alpha, k),
        initial_values=initial_values,
        time_span=time_span,
    )
