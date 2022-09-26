"""ODE problem descriptions.

Common descriptions of ODE problems that should appear in multiple docstrings
(such as the meaning of the equations, a reference, etc.), can be placed
here and prepended to the respective docstring via
`fun.__doc__ = <common_doc> + fun.__doc__`. This makes sure that
every occurrence of, e.g., the Pleiades problem, comes with
an appropriate description of the dynamics.

A common (non-binding) template for this kind of information is

```

<long_description>

<maths_and/or_latex_code>

<why_it_is_interesting>

<references_and/or_bibtex>

<see_also>

```

but do what you like.


"""


PLEIADES = r"""

    The Pleiades problem from celestial mechanics describes the
    gravitational interaction(s) of seven stars
    (the "Pleiades", or "Seven Sisters") in a plane.
    It is a 14-dimensional, second-order differential equation
    and commonly solved as a 28-dimensional, first-order equation.
    In in its original, second-order form, it is

    .. math::
        \ddot u(t) = f(u(t)),

    with nonlinear dynamics :math:`f: \mathbb{R}^{14} \rightarrow  \mathbb{R}^{14}`.

    The Pleiades problem is not stiff.
    It is a popular benchmark problem because
    it is not very difficult to solve numerically, but
    (a) it requires high accuracy in each ODE solver step, and
    (b) its 14 (or 28) dimensions start to expose those numerical solvers
    that do not scale well to high dimensions.

    References
    ----------
    A common citation for the Pleiades problem is p. 245 in the book by Hairer et al.:

    .. code-block:: tex

        @book{hairer1993solving,
            title={Solving Ordinary Differential equations I, Nonstiff Problems},
            author={Hairer, Ernst and N{\o}rsett, Syvert P and Wanner, Gerhard},
            year={1993},
            publisher={Springer}
            edition={2}
        }

    If you know a better source, please make some noise!

    See Also
    --------
    odezoo.ivps.pleiades
    odezoo.ivps.pleiades_autonomous_api
    odezoo.ivps.pleiades_first_order
    odezoo.vector_fields.pleiades
    odezoo.vector_fields.pleiades_autonomous_api
    odezoo.vector_fields.pleiades_first_order
    """

FITZHUGH_NAGUMO = r"""


    The FitzHugh-Nagumo model is a simple example of an excitable system
    (for example: a neuron).
    This simplified, 2d-version of the Hodgkin-Huxley model
    (which describes the spike generation in squid giant axons)
    was suggested by FitzHugh and Nagumo et al.

    It no a non-stiff, first-order problem,

    .. math::
        \dot u(t) = f(u(t), \theta)

    and generally easy to solve by most ODE solvers.

    References
    ----------
    The following bibtex(s) point to the original papers about
    the FitzHugh-Nagumo models. (Source: Google Scholar).

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

    See Also
    --------
    odezoo.ivps.fitzhugh_nagumo
    odezoo.vector_fields.fitzhugh_nagumo


"""
