"""Initial value problems."""


from diffeqzoo.ivps._chaos import lorenz63, lorenz96, roessler
from diffeqzoo.ivps._ivps import *
from diffeqzoo.ivps._misc import (
    affine_dependent,
    affine_independent,
    logistic,
    neural_ode_mlp,
    seir,
    sir,
    sird,
)
from diffeqzoo.ivps._nbody import (
    henon_heiles,
    henon_heiles_first_order,
    henon_heiles_with_unused_derivative_argument,
    pleiades,
    pleiades_first_order,
    pleiades_with_unused_derivative_argument,
    rigid_body,
    three_body_restricted,
    three_body_restricted_first_order,
)
from diffeqzoo.ivps._oscillators import (
    fitzhugh_nagumo,
    goodwin,
    lotka_volterra,
    van_der_pol,
    van_der_pol_first_order,
)
from diffeqzoo.ivps._reactions import (
    hires,
    nonlinear_chemical_reaction,
    oregonator,
    rober,
)
