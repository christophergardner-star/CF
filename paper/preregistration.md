# Pre-registration: external adjudication of the STRATA failure taxonomy

**Status:** registered 2026-07-07, before any external-benchmark run has
been designed in detail, executed, or observed. This document is the
falsification criterion for the *taxonomy*, written in advance so that the
run can adjudicate the claim rather than the claim adapting to the run.

## What is under test

The unit under test is the **diagnostic method and the trichotomy of
causes**, not the STRATA artifact. The trichotomy claims that every
catastrophic-forgetting event decomposes into the joint presence of:

1. **Substrate overlap** — the parameters updated by new learning
   intersect the parameters computing the degraded old behaviour;
2. **Timescale collision** — those updates outpace whatever protection the
   system affords them;
3. **Basis correlation** — the representations of old and new experience
   overlap.

A STRATA-derived system failing on natural data for capability reasons
(weak learners, no spatial priors) does **not** bear on this claim and is
excluded in advance as a falsifier. Conversely, the taxonomy surviving by
narrative fit does not count as confirmation: attribution must be
**interventional**, not rhetorical (see §Vacuous survival).

## The experiment

A standard class-incremental or task-incremental benchmark **not designed
by us** (e.g., Split-MNIST / Split-CIFAR under the standard scenarios),
with standard baselines (sequential fine-tuning, EWC, SI, replay), each
instrumented with the three attribution probes below. A forgetting event
is a drop of more than 20% relative accuracy on a previously mastered task
after training on a later task.

## Registered predictions

- **(a) The trichotomy transfers.** Every fully diagnosed forgetting event
  in the baselines will be attributable to the three conditions under the
  interventional probes below.
- **(b) The mode-list grows.** We expect *new*, architecture-specific
  failure modes inside the trichotomy that our benches never produced.
  Named candidates, registered now: normalization-statistics drift, output
  head/bias calibration drift, and optimizer-state interference. Finding
  these confirms rather than falsifies the framework: the current
  seven-mode list is claimed as a first chart, not a complete map.
- **(c) What would falsify the trichotomy** — the criterion this document
  exists to fix in advance.

## (c) The falsification criterion

A forgetting event lies **outside the trichotomy** if all three exclusion
tests pass on the same event:

1. **Substrate exclusion.** Restoring every parameter that changed during
   the interfering phase to its pre-phase value recovers **less than 50%**
   of the lost performance. (The loss is then not carried by overwritten
   substrate.)
2. **Timescale exclusion.** Re-running the interfering phase with update
   magnitudes scaled down by 10x (equivalently: with the most-changed 1%
   of parameters frozen) reduces the forgetting by **less than 50%**.
   (The loss is then not a function of unprotected update speed.)
3. **Basis exclusion.** Measured representational overlap between the two
   tasks (CKA, computed at every shared layer on held-out data from both
   tasks, before the interfering phase) is **below 0.1 at every layer**,
   and the forgetting still occurs. (The loss is then not carried by
   correlated bases.)

**One** fully diagnosed event passing all three exclusions falsifies the
trichotomy as a complete decomposition. A **softer incompleteness
criterion** also registered: if, across the benchmark, more than 30% of
total forgetting magnitude remains unattributed after interventional
diagnosis, the decomposition is incomplete even if no single event passes
all three exclusions.

## Vacuous survival (the vagueness trap)

Attribution by elimination ("it must have been basis correlation, nothing
else fits") does not count. Each attribution must be demonstrated by the
corresponding intervention *reversing or preventing* the loss (parameter
restoration for substrate; slowed/frozen updates for timescale;
representational decorrelation or task-input orthogonalization for basis).
If the probes cannot be implemented for a given system, that system's
events are reported as *undiagnosed*, not as confirmations.

## Outcome commitments

- If (a) holds and (b) produces new modes: the taxonomy graduates from
  candidate to demonstrated-in-one-external-setting; the mode-list is
  versioned and extended.
- If (c) fires: the trichotomy is reported as falsified or incomplete in
  any subsequent write-up, with the offending event published in full.
- No re-definition of the criteria in this document after first contact
  with external-benchmark data. Amendments before the run are permitted
  and must be dated.

---

## Amendment 1 (2026-07-07, before any external run): CKA operationalization

The basis-exclusion threshold is the criterion most exposed to accidental
goalpost drift, because "CKA < 0.1" on toy unit-vector streams and across
deep-network layers are different animals, and layer-selection and
aggregation choices can silently move the line. Fixed now, adversarially:

- **Variant:** linear CKA (centered), no RBF kernel, no whitening.
- **Data:** held-out samples from both tasks, at least 512 per task, drawn
  before the interfering phase; the same samples reused for every layer.
- **Layer set:** defined *architecturally*: the outputs of every module
  with learnable parameters in the evaluated network, plus the final
  pre-readout (penultimate) representation, enumerated from the
  architecture definition alone. The set makes no reference to any other
  probe's output --- in particular not to the substrate probe's parameter
  diff --- so that no probe's parameters can widen or narrow another
  criterion's scope. Adding or removing layers after seeing CKA values is
  prohibited; if a layer is computationally infeasible, it must be
  excluded *by name, in a dated amendment, before the run*.
- **Aggregation:** the basis exclusion passes only if **every** layer in
  the set is below 0.1. The maximum-CKA layer must be reported alongside
  the verdict either way.
- **No substitution:** other similarity measures (SVCCA, cosine of means,
  probing accuracy) may be reported as context but cannot substitute for
  the registered criterion.

This amendment fixes measurement procedure only; it does not alter any
threshold value or the pass/fail logic of the registered criteria.

---

## Amendment 2 (2026-07-07, before any external run): the overlap measure

Registration catch-up disclosure: the probe instrument (`probes/`)
implemented this corrected measure in the repository's initial commit
(`07a7a04`); this registration text should have accompanied it in the same
change and lags it. The lag is disclosed rather than reordered.

**1. Measure.** Building the instrument surfaced a first-principles defect
in the registered basis measure, prior to any external data: *paired*
linear CKA between two different tasks' samples has an arbitrary sample
pairing, under which its expectation reduces to a chance floor
(approximately sqrt(p q)/n for feature widths p, q and sample count n)
**regardless of true subspace alignment**. The registered quantity
therefore carried no signal about representational overlap, and no
threshold on it was ever meaningful. The registered measure is corrected
to the pairing-free form --- **covariance alignment**:

    A(X, Y) = tr(S_X S_Y) / (||S_X||_F ||S_Y||_F)

over centered per-layer feature covariances, which is 0 for activity
confined to disjoint feature subspaces, 1 for identical second-order
structure, and invariant to sample pairing and count mismatch. This
argument references no architecture under test.

**2. Reporting.** The instrument must report, for every layer in the
architecturally-enumerated set: the alignment value, an empirical null
floor (alignment after destroying feature correspondence by column
permutation), and a **vacuous-layer flag** wherever the null floor reaches
the registered bar --- at such layers the exclusion cannot pass and its
verdict must not be read as evidence in either direction. A further
documented property: the measure is scale-invariant, so layers fed by
shared trained heads will genuinely align across tasks; this makes the
basis exclusion conservative (hard to falsify the trichotomy by accident),
which is the acceptable direction of lean.

**3. Threshold provenance.** The 0.1 threshold is retained unchanged, on
procedural grounds (no threshold edits by the hand that knows which way it
wants the criterion to fall). Its calibration was established against the
superseded paired measure and is therefore **unverified against the
covariance-alignment scale**; the per-layer null floor of part 2 is the
stringency reference against which 0.1's meaningfulness must be judged at
run time, layer by layer. Retention of the number is continuity of
procedure, not evidence of calibration.
