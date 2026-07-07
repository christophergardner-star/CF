"""Core kernel of the thermodynamic continual-intelligence organism.

The ``core`` package holds the substrate every module lives on:

* :mod:`core.events`     -- the event types and the event bus (the only way
  subsystems talk to one another).
* :mod:`core.energy`     -- energy accounting (everything costs energy).
* :mod:`core.entropy`    -- internal uncertainty / thermodynamic temperature.
* :mod:`core.lifecycle`  -- the sleep/wake state machine of a module.
* :mod:`core.module`     -- the abstract ``Module`` base class.
* :mod:`core.scheduler`  -- decides which modules run each tick.
* :mod:`core.adaptation` -- simulated structural adaptation.
* :mod:`core.organism`   -- the kernel that owns everything and ticks.
"""
