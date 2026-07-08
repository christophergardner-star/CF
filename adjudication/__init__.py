"""The external adjudication run (paper/preregistration.md).

Split-MNIST, class-incremental, on standard baselines we did not design
(sequential fine-tuning, EWC, SI, replay), each instrumented with the
interventional probes of ``probes/``.  The unit under test is the
trichotomy of causes, not any particular system.  Criteria are frozen;
this package only executes them.
"""
