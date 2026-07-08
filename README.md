# Thermodynamic Kernel for Continual Intelligence

A research framework that treats intelligence as a **living organism** rather
than a neural network. Cognition emerges from many small, replaceable modules
that each carry their own **energy** and **entropy**, sleep when they have
nothing useful to do, wake when the world becomes interesting, and communicate
**only through messages** on a shared event bus.

This is *not* another transformer. There are no tensors, no gradients and no
global model. It is closer to an **operating-system kernel for continual
intelligence**: a scheduler, an energy budget, a memory hierarchy and a set of
event-driven "organs".

```
                         ┌──────────────────────────────┐
        world  ──sense──▶│  perception                  │
                         └──────────────┬───────────────┘
                     ObservationEvent   │
             ┌──────────────────────────┼───────────────────────┐
             ▼                          ▼                        ▼
        ┌─────────┐   CuriosityEvent ┌──────────┐  PlannerEvent ┌────────┐
        │curiosity│─────────────────▶│ planner  │──────────────▶│ critic │
        └────┬────┘                  └──────────┘               └───┬────┘
             │  ThoughtEvent  ┌──────────┐                          │
             └───────────────▶│ language │           LearningEvent  │
                              └──────────┘◀─────────────────────────┘
                         (all traffic flows through the EventBus;
                          the MemoryManager listens in and remembers)
```

## Design principles

* Everything has **energy**; every action (thinking, recalling, learning,
  communicating, planning) spends it, and it regenerates slowly with rest.
* Everything has **entropy** — internal uncertainty that drives exploration when
  high and exploitation when low, and that each entity self-regulates.
* Every module owns **local state** only. **No global variables.**
* Learning is **continual**; the loop never stops.
* Modules **sleep** and **awaken** based on energy, priority and events.
* Modules communicate **only through messages** — no module references another.
* **Event driven**, **loosely coupled**, and every component is **replaceable**.

## Layout

```
core/
  events.py       Event vocabulary + the EventBus (the only channel)
  energy.py       EnergyBudget + per-action costs
  entropy.py      EntropyField (self-regulating uncertainty)
  lifecycle.py    Sleep/wake state machine
  module.py       Abstract Module base class (template-method step)
  scheduler.py    Chooses who runs / wakes / sleeps each tick
  adaptation.py   Structural adaptation (spawn / prune / merge / re-weight)
  organism.py     The kernel: owns everything and drives the tick loop
memory/
  item.py         MemoryItem (importance, strength, decay, connections)
  working.py      WorkingMemory (small, volatile, ~7 items)
  episodic.py     EpisodicMemory (experiences that decay unless reinforced)
  semantic.py     SemanticMemory (consolidated concepts)
  consolidation.py  Replay + promotion + decay
  manager.py      Facade that forms memories from bus events
modules/
  perception.py   Samples the world -> ObservationEvent
  curiosity.py    Novelty / prediction error -> CuriosityEvent
  language.py     Narration + Markov prediction -> ThoughtEvent
  planner.py      Goals from curiosity -> PlannerEvent
  critic.py       Evaluates plans -> CritiqueEvent + LearningEvent
  hypothesis.py   A dynamic organ the organism grows for itself
  registry.py     Plugin discovery + dependency-injecting factory
utils/
  config.py       Typed, defaulted configuration (nested dataclasses)
  logger.py       Rich logging + the live console dashboard
  metrics.py      Rolling time series of vitals
main.py           Composition root + PatternedWorld environment
tests/            Unit tests for every subsystem
```

## Running

```bash
pip install -r requirements.txt      # rich (optional but recommended)

python main.py                       # live dashboard
python main.py --ticks 200           # run longer
python main.py --headless            # plain text (no rich needed)
python main.py --no-adapt            # disable structural adaptation
```

The dashboard shows current energy, entropy, active/sleeping modules, event and
memory counts, curiosity, learning rate, per-module health and the tick number.

## The tick

Every simulation tick the organism runs, in order:

1. `observe` (advance the world)
2. process events (deliver queued messages to inboxes and memory)
3. update energy (regenerate; modules rest)
4. update entropy (aggregate module uncertainty + surprise)
5. wake required modules
6. run active modules (`observe → think → learn → communicate`)
7. consolidate memories (heavier replay when idle)
8. allow learning to propagate
9. allow structural adaptation
10. sleep inactive modules
11. record metrics

## Thermodynamics: economics → physics

By default the organism runs on *economics* (energy is a variable that goes up
and down). Opt into `--thermo` and it runs on a small dynamical system of
**three coupled fields**, so behaviour becomes a consequence of constrained
physics rather than bookkeeping ([core/metabolism.py](core/metabolism.py)):

```
Energy   conserved capacity for work — drawn from a shared reservoir
  │      (the organism is an OPEN system, fed by an environmental influx),
  │      returned to the environment as dissipated heat. Never created.
  ▼
Work → Load   metabolic load / heat generated by computation; cannot do work,
  │           diffuses to wired neighbours, decays slowly (sleep cools ~3x).
  ▼
Information Entropy   driven UP by load; degrades fidelity, which throttles
  │                   retrieval, confidence, planning depth and consolidation.
  ▼
Behaviour → Recovery
```

`ClassicMetabolism` (the control / "Organism A") and `ThermodynamicMetabolism`
("Organism B") share one interface, so the substrate is swappable per module.

**Conservation law (unit-tested).** On the thermodynamic substrate, absent
structural growth, `d(reservoir + Σenergy + Σload) = influx − dissipated` holds
to floating-point tolerance — energy never appears from nowhere.

```bash
python main.py --thermo                 # live, on the physical substrate
python run_experiment.py constraint     # A (unlimited) vs B (thermodynamic)
python run_experiment.py overheat       # self-regulation across scarcity
```

**What the experiments actually found** (honest, single environment):

| claim | result |
|-------|--------|
| conservation of energy | **holds exactly** (test) |
| "B beats A on abstraction" | **falsified** — A forms concepts faster (10.9 vs 3.6 /100) |
| "B beats A on retention" | **supported** — B retains longer (142 vs 110 ticks)†|
| voluntary sleep from load | **emerges, but only in a scarcity band** |

† retention-up may be a churn artifact (B is less active, so memories are
evicted less), not "efficiency" — flagged, not yet resolved.

The emergent result worth chasing: planner **voluntary rest is non-monotonic in
free energy** — a scarcity band where the load↔energy feedback spontaneously
produces rest, with no `load→sleep` rule ever coded.

### Causal closure: load deforms the *action manifold*

Making load only degrade *outcomes* leaves a "scheduler with physics attached".
So load now also deforms **which actions are reachable**
([core/metabolism.py](core/metabolism.py) `capacity()`/`reachable()`): as load
rises the manifold shrinks and the most complex actions drop out first
(`learn → plan → think → communicate`). Planning depth is read straight off the
manifold, and a module that cannot even `think` *stalls* (does not refresh its
activity) and drifts to sleep — so rest can arise from action-space collapse,
not a rule.

Rerunning the **identical** influx sweep before vs after this one change:

| influx | novel/100 (before → after) | retention (before → after) | sleep (before → after) |
|---|---|---|---|
| 16 | 3.6 → **7.9** | 142 → 105 | 0 → 0 |
| 10 | 4.1 → **7.9** | 142 → 105 | 3 → 0 |
| 6  | 3.5 → **5.6** | 142 → 103 | 34 → 15 |
| 4  | 3.6 → 4.5 | 126 → 111 | 1 → 1 |

The phase diagram **changed shape** — abstraction ~doubled, load fell, the sleep
band narrowed — so decisions now genuinely re-encode physics. It also **resolved
the earlier retention artifact**: B's retention advantage collapsed to A's level
once B became productive, confirming it was low-activity churn, not efficiency.

But a diagnostic (`scratchpad/cause.py`) shows **100% of planner sleeps are still
energy-triggered, 0% action-collapse-triggered**. So *control* is now
thermodynamic, but *termination* is not — a clean split.

### Why termination cannot (here) move onto the load manifold

An extinction experiment (rest-dependent dissipation, so heat accumulates during
work and only sleep cools it) was run to try to force sleep from action collapse.
It **failed, informatively**: with dissipation proportional to accumulated load,
the manifold always settles at an *interior thermal equilibrium* — as actions
shed, heat generation falls, so load stabilises *above* THINK-extinction. `THINK`
is a **thermodynamic fixed point protected by the dissipation law itself**;
pushing harder just froze the planner at "can-think-can't-plan" and collapsed
novelty (7.6 → 2.1) without ever producing sleep. The change was reverted.

The consequence: load can never cleanly empty the action set, so the only
non-self-limiting gate is the *representational* (entropy) field — and that is
pinned in a hot basin (fidelity ~0.27, a "preferred informational temperature").
**So the two open problems are one:** manifold-driven termination is blocked by
the entropy saturation.

### The fixed point is a confusion trap, not the relaxation term

The natural fix — replace passive entropy relaxation with **prediction-earned
certainty** (entropy only falls when a module actually predicts well) — was
implemented and measured against one success condition: *does the single stable
entropy fixed point disappear?* Measured attractor (260 ticks, per influx):

| substrate | entropy mean | entropy std | across influx | band |
|---|---|---|---|---|
| baseline (relaxation) | 0.80 | 0.04 | identical | 99% high |
| prediction-earned | 0.86 | 0.05 | identical | 98% high |

**The fixed point survived and got hotter.** So the relaxation term was never
what created it — it was *masking* it. The real attractor is a **self-reinforcing
confusion trap**: high entropy → low fidelity → degraded prediction (language
dropout, critic noise) → misses → entropy stays high. The coherent (low-entropy)
basin is unreachable from a hot start because low fidelity sabotages the very
predictions that would earn certainty. That feedback loop is the Lyapunov
governor — not the baseline. (Change reverted; substrate restored.)

### Phase-structure audit: the system is a contraction homeostat

Rather than mutate the substrate a fourth time, an audit asked *what invariant
forces unimodality?* Two findings:

**1. Every operation in the entropy pathway is compressive; none is expansive.**
`perturb` saturates (`increase(|e|·(1−level))`, gain → 0 as level → 1); `relax`
contracts toward a baseline; the drivers are pre-compressed (`novelty =
1/(1+seen)`, binary language surprise, ×small constants); the organism readout is
an EMA of a spatial mean; everything is clamped to [0, 1]. There is **no place
where a difference is amplified** (no loop gain > 1).

**2. Impulse response confirms one dominant attractor.** Shocking a module's
entropy to 0.02 and to 0.98 mid-run, both return to the *same* value (0.924) in
≤ 5 ticks, monotonically, with no second resting point:

```
offset   cold-shock   hot-shock   control
     0        0.481       0.955     0.931
     2        0.761       0.925     0.917
     5        0.898       0.926     0.925
     8        0.927       0.930     0.930   ← fully converged from both sides
```

**Conclusion — the system class is a contraction-mapping homeostat, not a
phase-separating physical system.** A dynamical system built entirely from
contractions has, by Banach, exactly one fixed point; that is why *every*
perturbation (load, relaxation, prediction-earned entropy) restored the same
unimodal band. No parameter tuning can ever produce multiple phases — bifurcation
requires an **expansive nonlinearity (loop gain > 1)**, and the architecture
contains none. That missing gain is the single degree of freedom separating this
homeostat from a system that could exhibit regimes.

## Extending it

Adding an organ requires **no kernel changes**:

1. Subclass `core.module.Module`, declare `subscriptions`, and implement
   `observe`, `think`, `learn`.
2. Register it in `modules/registry.py` **or** point a `ModuleConfig.type` at a
   dotted `package.module:ClassName` path.
3. Add a `ModuleConfig` to the config.

The scheduler, energy system, memory and dashboard pick it up automatically.

## The Cognitive Dynamics Laboratory

Once the kernel runs, the interesting question stops being *"can I build it?"* and
becomes *"what properties does it have?"*. The `lab/` package treats the organism
as an ecosystem to be **measured and perturbed**, not just watched. It touches
only the organism's public surface, so it is pure instrumentation.

```
lab/
  observatory.py    Drives the organism tick-by-tick, taps the bus, records a
                    rich Frame per tick + every episodic memory's birth/death.
  metrics.py        Derived measures computed from Frames (pure functions):
                      · integrated-information proxy (Φ-like index)
                      · sleep cycles + mean period per module
                      · curiosity oscillation (autocorrelation → period)
                      · novel-concept creation rate
                      · memory-survival curves
                      · module interaction / energy-flow graph
                      · structural-adaptation frequency
  perturbations.py  Lesion · ablate · starve · flood · suppress-sleep operators.
  experiment.py     Build → observe → measure helpers.
run_experiment.py   The lab bench (CLI).
```

Run controlled experiments:

```bash
python run_experiment.py lesion      # remove the planner mid-run
python run_experiment.py ablation    # delete 30% of semantic memory
python run_experiment.py starve      # cut the energy supply
python run_experiment.py flood       # inject contradictory observations
python run_experiment.py sleep       # A/B: sleep vs no-sleep retention
python run_experiment.py all --ticks 200 --seed 1
```

Sample findings (200 ticks, seed 1 — illustrative, single environment):

* **Lesion** the planner → learning rate collapses (0.42 → 0.02) while curiosity
  *runs away* (0.50 → 0.93): the explore→plan→critique→learn loop loses its
  outlet, and the organism grows a new organ in response.
* **Ablate** 30% of semantic memory → concepts re-consolidate (13 → 11 → 20).
* **Starve** energy to 20% → the **planner shuts down first** (highest per-action
  cost); fraction asleep jumps 0.07 → 0.48.
* **Flood** with contradictions → entropy spikes +0.21 then self-relaxes in ~3
  ticks (the regulator holds; entropy does not explode).
* **Sleep** A/B (3 seeds) → enforced sleep gives a modest retention edge
  (median memory lifetime 106.7 vs 101.7 ticks).

> `integrated_information_proxy` is an approximation inspired by IIT, **not** a
> computation of Φ. Treat every measure here as a comparative index across runs.

## STRATA: a learnable substrate prototype

`strata/` is a self-contained prototype of **STRATA** (*Sparse Tiered Routing
with Anchored Two-timescale Adaptation*) — a continual-learning architecture
that answers the question the kernel raises: *what would an organ look like if
it learned rather than followed coded rules?* It is a next-input predictor
with **no transformer, no attention, no backprop-through-time and no replay
buffer**; every learning rule is local (delta rule + diagonal eligibility).
It depends on numpy and touches nothing in the kernel.

```
strata/
  encoder.py      Sparse kWTA pattern separation (interference shield #1)
  routing.py      Centroid routing over columns + novelty-driven spawning (#2)
  column.py       Liquid (novelty-modulated) diagonal SSM reservoir with
                  cascade-protected readouts; state bounded by construction
  cascade.py      Benna–Fusi diffusion chains: power-law memory (#3)
  fastweights.py  One-shot Hebbian episodic store, novelty-gated writes (#4)
  network.py      The tick loop: surprisal → CUSUM regime detection →
                  gated learning → calm-time consolidation (fast → slow)
  demo.py         The continual-learning bench (tasks that *collide*)
```

```bash
python -m strata.demo               # A→B→C→A bench, ablating each defense
python main.py --cortex             # run the organism WITH a learning organ
pytest tests/test_strata.py tests/test_strata_cortex.py
```

**The cortex** ([modules/strata_cortex.py](modules/strata_cortex.py), enabled
with `--cortex`) is the first organ whose behaviour is learned rather than
coded: a STRATA network living inside a regular `Module`. It predicts the next
percept (narrated as `ThoughtEvent`s, with `CuriosityEvent`s on surprisal and
`LearningEvent`s when it grows a column), and the kernel's physics gates its
plasticity — slow learning runs at the module's *fidelity* and stops entirely
when it cannot afford `LEARN`, while one-shot fast-weight capture keeps
working (a tired cortex stops consolidating but keeps taking notes). In a
300-tick default run it reaches ~0.65 recent prediction confidence against a
world with 12% injected novelty, grows 3 columns, and periodically starves,
stalls and sleeps under the same energy loop as every other organ.

**The thermal loop is closed both ways** on the thermodynamic substrate
(`--thermo --cortex`): every synaptic update is an outer product, so its
Frobenius norm `|Δθ|` is free to compute; the substrate reports it as `heat`
and the cortex pays `work(LEARN, scale ∝ |Δθ|)`, which the metabolism turns
into load. Learning literally warms the organ; load throttles the next tick's
plasticity (via fidelity) and can push `LEARN` out of the action manifold
(hard refractory). Measured results (400 ticks, seed 7):

* At the default battery (capacity 32), baseline THINK/COMMUNICATE heat alone
  pins load *above* the LEARN-reachability ceiling: the cortex falls into the
  kernel's known saturated attractor (fidelity ≈ 0.27), never learns again
  (accuracy 0.12) and never rests — the manifold self-limiting result from
  `cause.py`, reproduced in a learning organ.
* With a brain-sized battery (capacity 64, ceiling above baseline heat) a
  **burst/cool cadence emerges from pure physics**: LEARN reachability
  flickers (learnable ~61% of ticks, load crossing the ceiling ~86 times at
  ~4-tick spacing) and the organ self-organizes to the criticality boundary —
  learn → overheat → brief refractory → learn. No cooling rule is coded.
* The smooth throttle is *permanently* engaged: fidelity stays pinned ≈ 0.30
  (the documented saturated-entropy attractor), so thermodynamic learning runs
  at ~a third of classic speed (accuracy 0.34 vs 0.57). The cortex is now the
  clearest probe of that open thread: unpinning fidelity has a measurable
  learning-rate payoff.

**What the bench actually shows** (A→B→C→A, colliding vocabularies, 2 seeds):
after 1200 ticks of other tasks, full STRATA recalls task A at ~0.15–0.17
error vs ~0.31–0.37 without routing and ~0.80 for the dense baseline; a
frozen-circuit probe shows task A's circuit degrades 0.085 → 0.132 — graceful
power-law sag, not catastrophic loss. Honest caveats from building it:

* **Absorption is the failure mode, not erasure.** The first working version
  never spawned a column: a new task quietly dragged the old column's centroid
  onto itself. It took three hysteresis mechanisms (claim threshold ≫ route
  threshold, no centroid updates during transitions, fallback columns predict
  but never learn) to stop it — none of which the math "needed" on paper.
* **The encoder cannot separate what the input space doesn't.** At d=8 no
  routing threshold exists (measured); the s²/D orthogonality claim needs
  enough ambient input dimension.
* **On this bench, sparse routing does the heavy lifting**; the cascade's
  rebound-vs-sag roughly cancels for untouched sparse keys. Its value should
  appear on shared parameters and longer schedules — untested.
* Consolidation attribution matters: assigning fast-weight anchors to columns
  by key affinity silently distilled one task's memories into another task's
  column; ownership had to come from the routing context instead.

### The long-sequence bench: what actually got settled

`python -m strata.longbench` runs 24 tasks (8 families × 3 variants, related
vocabularies, correlated symbols) and measures the three claims that separate
"a retention demo" from continual learning — plus the frontier question of
learning the *shared encoder* without replay (500 ticks/task, seed 0):

| encoder | TTC v1 | TTC v2+ | fwd | ret old | ret new | cols |
|---|---|---|---|---|---|---|
| fixed random | 197 | 87 | 2.3× | **0.214** | 0.216 | 8 |
| fixed + matching pursuit | 298 | 159 | 1.9× | 0.253 | 0.255 | 8 |
| learnable, naive | 240 | 129 | 1.9× | 0.694 | 0.434 | 8 |
| learnable, consolidated | 220 | 96 | 2.3× | 0.217 | 0.224 | 8 |
| **consolidated + MP** | 232 | **88** | **2.6×** | 0.216 | **0.209** | 8 |

The last row is the campaign's headline: with matching-pursuit inference and
decoupled acquisition (below), the **learned dictionary now beats the fixed
encoder on the flagship bench** — best transfer, best recent-task retention,
old-task parity. And `fixed+mp` being *worse* than plain fixed is the
control that confirms the mechanism: explaining away only pays when the
atoms are learned to be worth explaining with.

**Settled, positive**: capacity grows with *diversity*, not task count
(8 columns / 24 tasks / 8 families); forward transfer is real (later family
variants learn 2.3× faster); retention is flat with age (no
forgetting-with-age gradient over 24 tasks). Replicates on seed 1.

**The frontier round-trip** — this took three failed mechanisms and one
architectural change, in that order:

1. Naive representation learning reintroduces catastrophic aging (old-task
   error 0.67 vs 0.21): code drift destroys the keys old columns depend on.
2. Per-unit consolidation + novelty recruitment rescued retention but killed
   transfer, across three mechanism variants (incremental, one-shot
   imprinting, residual imprinting). Measured root cause: learned features
   *inherit input correlations* that random projections destroy — within-task
   code separation collapsed from |cos| 0.21 (random) to 0.33–0.45 (learned).
   The textbook fix (continuous anti-Hebbian decorrelation) would perpetually
   move mature features — exactly the churn consolidation exists to prevent.
   **The stability–plasticity dilemma re-emerges at the representation
   layer, and consolidation alone does not solve it there.**
3. What solved it was architecture, not a better regularizer: **complementary
   learning systems for features**. Each column's centroid became its slow
   *expectation of the input* (routing moved to input space), and the sparse
   dictionary now codes only the **residual** — the deviation from the
   claiming column's expectation. The correlated (shared) component of a
   regime never reaches the dictionary at all, so there is nothing for
   learned features to inherit: predictive coding, structurally.

Post-change, the whole system improved (acquisition halved, transfer 1.6→2.3×,
retention slightly better), and the consolidated learnable encoder went from
losing on every metric to parity on task metrics **and better code geometry
than random** (within-task |cos| 0.036 vs 0.054) — the first evidence here of
representation learning *helping* under no-replay constraints.

**The low-rank round** (`--rank 3`: family residuals confined to a
3-dimensional subspace, where symbol directions are *inherently* correlated
and a dictionary must prove its worth). Two additions: **matching-pursuit
inference** (`encode_mode="mp"`) — explaining away as lateral decorrelation
done by *inference* rather than weight dynamics, so it cannot conflict with
consolidation — and **usage decay** (`encoder_usage_decay`), which makes
feature consolidation impermanent so a finite dictionary forgets gracefully,
least-used first. Findings (measured, 24 tasks):

* While capacity lasts, the learned dictionary + MP **beats random where
  theory says it must**: single-task error floor 0.168 vs 0.188, and
  within-task code separation 0.048 vs 0.137 (~3× better).
* Then it hits the wall nobody hits in papers with replay: **dictionary
  capacity exhaustion**. 288 symbols vs 128 units — all virgin rows consumed
  within 2 of 8 families, after which new symbols cannot get dedicated
  features and floors degrade 0.168 → 0.258 while random stays flat (random
  projections have nothing to exhaust; they are mediocre everywhere,
  forever). Usage decay converts the cliff into a graceful, usage-ranked
  tradeoff (transfer 0.9× → 1.2× at capacity 128).
* Acquisition was then **decoupled from allocation** (the readout-churn
  hypothesis, tested): (a) if an imprint event rotates the basis, the input
  is *re-encoded* before anything binds to it — keys are born stable, no
  orphaned fast-weight bindings; (b) slow readouts consolidate onto a key
  only as fast as its supporting features stop rotating
  (`support_maturity`) — novel structure rides the fast weights until the
  basis settles (hippocampus carries while cortex waits). This flipped the
  flagship full-rank bench (learned now wins: transfer 2.6× vs 2.3×) and
  gave the learned encoder its best rank-3 retention (0.357), but the
  rank-3 *transfer* gap vs fixed persists (1.2× vs 1.4×). With bindings and
  readout-chasing eliminated as causes, the remaining suspect is decoding
  itself: rank-limited keys cap what a static linear readout can express,
  pushing work onto the temporal readout, which favors a stationary basis.
  Next: structurally growing dictionaries (capacity that scales like
  columns), and per-column adaptive claim bands for the true scale-up.

Threshold policy for this phase, for the record: the novelty channel is
already self-normalizing (CUSUM on error variance–standardized surprisal);
the routing cosines were deliberately kept fixed so the dictionary-vs-random
comparison stayed single-variable. Per-column adaptive claim bands are the
plan for the true scale-up, where dimensionality compresses cosine gaps.

### Phase 2: depth, growth, and expressive readouts

Three mechanisms from the Phase 2 blueprint, each validated by its
pre-registered falsifier:

* **Conjunctive expansion** (`conjunctive_dim`): a fixed random kWTA
  expansion of code ⊕ liquid state, read by its own cascade — conjunctive
  symbol-in-context features lift effective key rank with zero churn risk
  (nothing rotates; maturity is inherited from the code). **Falsifier
  survived decisively**: the rank-3 transfer gap vanished — learned+MP went
  from 1.2× to 1.7× forward transfer, *matching* fixed, with TTC on later
  variants improving 2.4-fold (381 → 155). The low-rank residue was readout
  expressivity, as diagnosed.
* **Structurally growing dictionaries** (`encoder_grow_from`): capacity
  starts small and grows under allocation pressure (a leaky accumulator of
  recruitment failures — the same detector pattern as column spawning).
  The unit test caught the first design violating backward transparency:
  freshly grown random rows *won inference for old inputs*, churning
  existing codes — orphaned bindings sneaking back in through growth. Fix:
  grown capacity is **dormant** — masked out of inference entirely until
  the draft-and-imprint path allocates it. Growth is now *exactly*
  backward-transparent (bit-identical old codes, unit-tested).
* **Adaptive claim bands** (`adaptive_bands`): per-column thresholds
  tracked from claim statistics, with the load-bearing floor
  `θ_claim ≥ bg + 2σ_bg` that forbids any band from overlapping background
  similarity (without it, absorption returns *through the statistics* —
  threshold collapse). Its first 48-task scale run **failed its
  pre-registered falsifier** with a spawn storm (48 columns for 48 tasks):
  background statistics absorbed transition swings, inflating the floor and
  orphaning sibling variants. Fix — the identity-freeze law applied a third
  time (`stats_frozen` during transitions). Revalidation passed: **16
  columns for 16 families at 48 tasks** in both the static control and the
  fixed adaptive run, retention flat with age (0.306/0.310), transfer
  1.4–1.6×, adaptive now slightly beating static.

**Depth** ([strata/stack.py](strata/stack.py), `python -m strata.stack`):
a two-layer stack with event-driven ascent — Layer 2 ticks only at Layer 1's
regime boundaries, and its predictions decode into bounded top-down routing
priors plus context seeding at detected transitions. Priors bias inference
only, never learning (a hallucinated expectation cannot consolidate itself).
Three measured lessons from making it work:

* **Emit what arrived, not what confusion felt like.** Pooling the surprise
  burst itself fed L2 events that were identical across all boundaries; the
  event must pool the *arriving* regime's early post-settlement codes.
  Fixing this took the revisit-transient reduction from 18% to **31%**
  (seed 0; 15–31% across seeds).
* **Absorption reappears at every level, in level-specific costume.** With
  a short event-context window, L2's context and centroid are EMAs of the
  same stream and co-drift — affinity stayed 0.92 across a complete
  schedule change, claims never stopped. Mixtures of unit vectors crowd
  together, so schedule-level identity is intrinsically thin-margined.
* **The granularity/data trade-off**: L2 *can* differentiate regime-arrival
  types (tune context and spawn threshold to event granularity: 2–3 columns
  tracking schedule structure), but at tens of events per run, splitting
  experience across columns starves each successor map and prior quality
  *degrades* (31% → 10%). At small event budgets the best L2 is the least
  differentiated one. Diversity scaling at level N+1 requires orders of
  magnitude more time than level N — which is precisely why hierarchical
  timescales must be geometric, and why schedule-level abstraction belongs
  to a third layer, not a threshold tweak at the second.

## Testing

```bash
pytest
```

## The adjudication (stage 1: complete)

`python -m adjudication.run` executes the pre-registered external test
([paper/preregistration.md](paper/preregistration.md)) on Split-MNIST
(class-incremental) with standard baselines — naive fine-tuning, EWC, SI,
replay — each instrumented with the interventional probes in
[probes/](probes/). Verdict across two seeds, criteria frozen before the
run: **33/33 forgetting events attributed within the trichotomy; the (c)
criterion did not fire; unattributed forgetting 0.000** (bar: 30%).
Substrate recovery was 1.00 on every event (in these baselines, all
forgetting is parameter-carried); basis alignment ran 0.65–0.92 against
null floors ≤ 0.37; timescale was attributed on only 2/33 — at full data,
10×-slowed training still converges to the interfering solution.
Mandatory disclosure: every event had at least one vacuous layer, so the
basis route to (c) was partially jammed on this architecture — the
verdict rests chiefly on the fully-testable substrate exclusion. The
registered candidate modes (normalization-statistics drift, optimizer
state) were structurally untestable on plain-SGD MLPs; that is the
Split-CIFAR stage's question. Full per-event records:
[adjudication/results/](adjudication/results/).

## Provenance

This project was developed by Christopher Gardner in interactive research
sessions with Claude (Anthropic) — human-directed design and adjudication,
AI-assisted implementation and experimentation. The paper
([paper/strata.tex](paper/strata.tex)) discloses the collaboration; the
pre-registration ([paper/preregistration.md](paper/preregistration.md))
and the falsification record inside the paper document the process,
including the mechanisms that failed their own pre-registered tests and
were redesigned. The repo's claims are deliberately scoped: candidate
decomposition at prototype scale, not conquest — see the paper's
limitations section before citing any headline number.
