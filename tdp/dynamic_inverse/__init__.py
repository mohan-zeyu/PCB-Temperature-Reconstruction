"""Dynamic thermal inversion from an IR time series, without an absolute energy scale.

The recording gives temperature only.  Dividing the thin-plate heat equation by
the areal heat capacity ``C = rho cp d`` removes every extensive constant from
the model and leaves exactly what a temperature trajectory can pin down::

    d(theta)/dt = div(alpha_eff grad(theta)) - beta theta + s

with ``alpha_eff = G/C`` (px^2/s), ``beta = h/C`` (1/s) and ``s = q/C`` (K/s).
Scaling ``C, G, h, q`` by one common constant leaves the trajectory unchanged,
so ``k``, ``G``, ``q`` and component wattage are *not* identifiable here and are
never emitted.  Names carry the distinction: ``effective``, ``normalized``,
``relative``, ``source_rate``.

See ``docs/dynamic_inverse_zh.md`` for the derivation and the limits.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
