# Internal design choices

`diffeqzoo` is a database of ODE problems. As such, the following principles apply to the source:

It must be compatible with all numpy/jax-based ODE solvers.

It must be easy to copy/paste from, if desired.

It must not have opinions (we don't care whether your ODE variable is called ``u``, ``x``, or ``y``).

It must take non-standard dynamics seriously: if an ODE is of second order, autonomous, or in mass-matrix form, it is implemented as such (we trust the user to translate it to an appropriate first-order version).

It should provide all information that might be
relevant if the problem appears in a paper (citation, maths, meaning).

`diffeqzoo` must be extremely easy to maintain, even if it costs a tiny bit of user-friendliness.
