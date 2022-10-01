# Adding a problem implementation

First off: thank you for considering a contribution to `diffeqzoo`'s problem test set.
Your example will be a valuable addition!
Here is how to proceed:

1. Add a test-case to `tests/test_bvps.py` or `tests/test_ivps.py`.
   This is important, because we need to verify that implementations work. You can use the existing test-cases as a guideline.
2. Implement the ODE vector field in `diffeqzoo/_vector_fields.py`. Implementing the vector field separately from the IVP/BVP constructor has the advantage that some vector fields may be used for multiple problems. For example, the SIR vector field powers both `ivps.sir` and `bvps.measles`.
3. Implement the IVP/BVP constructore in `diffeqzoo/ivps.py` or `diffeqzoo/bvps.py`. Again, use the existing code as a reference.
4. Add a short docstring that describes the problem, why it is interesting, and who came up with it (if possible). It would be great if you could add a bibtex snippet that points to the original paper, just like in the existing problems. The better the docs, the more useful the function will be to end-users, but if something is hard to find, don't sweat it too much.
5. Check that everything passes the quality checks via `make format test lint`.
6. Make a pull request with your changes.
