# Usage

To use odezoo, import one of its modules and continue with the resulting ODE.

```python
>>> from odezoo import ivps
>>>
>>> f, u0, tspan, f_args = ivps.lotka_volterra()
>>> f(u0, *f_args)
```
