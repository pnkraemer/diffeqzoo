# Behind the scenes

The two main assumptions behind the design of the `diffeqzoo` are:

1. A `diffeqzoo` dependency will be the least crucial dependency of any project, and the first one to be dropped; because:
2. Almost all users will use only a single function from `diffeqzoo` at a time. If necessary, this function can be copied/pasted from somewhere into each project.


What does this imply?
It implies that the `diffeqzoo`'s API should be instantaneous to learn,
easy to copy/paste out of (embrace the drop-ability), and extremely easy to maintain.

This is achieved by a pure-function API with minimal dependencies (either numpy or jax),
and no custom data structures: everything is a plain callable, tuple, array, list, or dict.
The source should be understandable by anyone that has spent time with Sections 1-5 in [this tutorial](https://docs.python.org/3/tutorial/). (If not, let us know!)

There are almost no nested function-calls, and the only non-trivial implementation is that of the flexible backend (which is, itself, much more minimal than comparable implementations).
If copying ODE examples from this project into your own project is the best thing for you,
please do so! We tried to make copy/pasting ODE code as easy as possible.
