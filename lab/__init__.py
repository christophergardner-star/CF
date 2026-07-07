"""The Cognitive Dynamics Laboratory.

Tools to *do science* on the organism rather than just watch it:

* :mod:`lab.observatory`   -- records a rich per-tick time series (a "Frame").
* :mod:`lab.metrics`       -- derived measures (an integrated-information proxy,
  sleep cycles, curiosity oscillation, memory-survival curves, the module
  interaction graph, novel-concept rate, structural-adaptation frequency).
* :mod:`lab.perturbations` -- lesion / ablate / starve / flood / suppress-sleep
  operators for controlled experiments.
* :mod:`lab.experiment`    -- run conditions, collect frames, compute a report.

The kernel itself is untouched: the lab only uses the organism's public surface
(the event bus, ``retire_module``, the memory stores, the energy budgets), so it
is a pure observer/experimenter, exactly like instrumentation around a physical
system.
"""
