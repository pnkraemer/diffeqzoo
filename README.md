# diffeqzoo


_So, what was the initial condition of the restricted three-body problem again?_

``diffeqzoo`` delivers all differential equation test problems in one place. It works with numpy and jax.


**Features include**

* Standard non-stiff benchmark problems (Lotka--Volterra, FitzHugh--Nagumo, Van-der-Pol, rigid-body, ...)
* Standard stiff benchmark problems (HIRES, ROBER, ...)
* Compartmental epidemiological models (SIR, SEIR, SIRD, ...)
* Chaotic systems (Lorenz63, Lorenz96, ...)
* N-Body problems
* Boundary value problems

**As well as**

* Flexibly NumPy and JAX-backends. Other than one of those two, there are 0 (zero!) dependencies.
* Mathematical descriptions **and BibTex entries** of the ODE problems
* Compatibility with all NumPy/JAX-based ODE solvers: SciPy, JAX, Diffrax, ProbNum, Tornadox, etc..

and many more goodies.

* **DOCUMENTATION:** (todo: add link)
* **ISSUE TRACKER:** [click here](https://github.com/pnkraemer/diffeqzoo/issues)

## Quick example
```python 
>>> from diffeqzoo import ivps, backend
>>> backend.select("numpy")
>>>
>>> # Create test problems like this
>>> f, u0, t_span, f_args = ivps.lotka_volterra()
>>> x = f(u0, *f_args)
>>> print(x)
[-10.  10.]
>>>
>>> # The numpy backend determines the type of input/output
>>> print(type(x))
<class 'numpy.ndarray'>
>>>
>>> # All sorts of ODEs are available, e.g., Rigid-Body:
>>> f, u0, t_span, f_args = ivps.rigid_body()
>>> print(f(u0, *f_args))
[-0.     1.125 -0.   ]
>>>
>>> ## make it jax
>>> backend.change_to("jax")
>>> f, u0, t_span, f_args = ivps.rigid_body()
>>> x = f(u0, *f_args)
>>> print(x)
[-0.     1.125 -0.   ]
>>> print(type(x))
<class 'jaxlib.xla_extension.DeviceArray'>

```



## Related work

* F. Mazzia et al. published a ![test set for IVP solvers](https://archimede.uniba.it/~testset/testsetivpsolvers/?page_id=51) for Matlab and Fortran. 
  There is a similar ![test set for BVP solvers](https://archimede.uniba.it/~bvpsolvers/testsetbvpsolvers/). Neither one offers Python code, and both also run benchmarks, which `diffeqzoo` does not care about at all.
* E. Hairer et al. published their ![stiff ODE test set](https://www.unige.ch/~hairer/testset/testset.html), but there is no Python code
* ![NonlinearBenchmark](https://www.nonlinearbenchmark.org/) hosts datasets of nonlinear dynamical system observations. They are quite specialised problems, and don't contain the textbook problems like Lotka-Volterra, van der Pol, etc..
* DifferentialEquations.jl provides ![example ODE problems](https://diffeq.sciml.ai/stable/types/ode_types/#Example-Problems) in Julia.
* ![ProbNum's problem zoo](https://probnum.readthedocs.io/en/latest/api/problems/zoo.diffeq.html) offers a similar set of problems to `diffeqzoo` (no surprise -- the set of authors intersects) but tied to ProbNum's ODE solver interface. `diffeqzoo` is less of an API, switches more flexibly between numpy and jax (at the time of developing), and contains more problems.

Anything missing in this list? Please open an issue or make a pull request.
