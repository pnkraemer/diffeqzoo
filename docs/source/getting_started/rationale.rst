Behind the scenes
=================

Here is why and `odezoo` delivers ODE example problem definitions.

Just the problem definitions? Yes, just the problem definitions. You solve them yourself.

**Why?**

On its own, it is probably quite useless.
`But if you're frequently working with ODE models, it might be just for you!`
Why?
Because it is the one-stop-shop for all benchmark problems, mathematical explanations, and bibtex references.
With ``ODEZoo``, you will never again have to copy and paste your implementation of, e.g.,
the FitzHugh--Nagumo model around multiple projects.
And ``ODEZoo`` can remind you of the initial condition of the restricted three-body problem.
It is all here.

**But not just that, the ODEZoo also has:**

* Either a NumPy or a JAX backend.
  The JAX backend makes the problem definitions fully differentiable, vectorisable, and GPU-compatible.
  The NumPy backend makes the implementations as lightweight as they can be.
  You switch between those with a single line of code.

* Compatibility with Scipy, Jax, Diffrax, ProbNum, tornadox, and basically all NumPy/JAX-based ODE solvers.
  See the example notebooks.


* Bibtex snippets for the original papers related to an ODE model.
  You cannot use an ODE example problem in a scientific publication without attributing who came up with the problem.
  But digging out the same old references over and over again is quite annoying -- we've got you covered!
