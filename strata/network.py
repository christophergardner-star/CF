"""The STRATA network: encoder -> router -> columns -> fast weights -> cascade.

The tick loop implements the two-timescale contract:

1. Score the previous tick's prediction; the standardized error is the
   *novelty* signal that gates everything else (no task labels anywhere).
2. A CUSUM detector on novelty declares regime shifts; during a shift the
   consolidated (mature) columns are protected while fast weights capture the
   new regime one-shot.
3. Slow learning is a local delta rule on the active columns only; inactive
   columns receive exactly zero gradient (structural isolation).
4. Every tick the cascades diffuse; during *calm* periods the fast store is
   distilled into the owning column's slow readout and allowed to decay --
   consolidation happens on the network's own schedule, from its own memory,
   never from stored raw data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from strata.column import LiquidColumn
from strata.encoder import SparseEncoder
from strata.fastweights import FastWeightMemory
from strata.routing import ColumnRouter


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(slots=True)
class StrataConfig:
    """All knobs, with ablation switches for the layered-defense experiment."""

    d_in: int = 8
    code_dim: int = 96
    code_sparsity: int = 8
    encoder_lr: float = 0.0         # 0 = fixed random encoder
    encoder_kappa: float = 0.2      # per-unit consolidation (0 = naive drift)
    encode_mode: str = "kwta"       # "kwta" | "mp" (matching pursuit)
    encoder_usage_decay: float = 0.0  # feature recycling (0 = permanent)
    encoder_grow_from: int | None = None  # start small, grow under pressure
    conjunctive_dim: int = 0        # fixed conjunctive expansion (0 = off)
    adaptive_bands: bool = False    # statistics-tracked routing thresholds
    n_state: int = 32
    top_k: int = 2
    theta_route: float = 0.65       # participate (and learn) above this
    theta_claim: float = 0.8        # absorb the input into the centroid
    spawn_rate: float = 0.15
    spawn_threshold: float = 0.8
    context_rho: float = 0.08       # EMA rate of the routing context signal
    max_columns: int = 12
    lr_state: float = 0.05          # C readout (dense liquid state)
    lr_code: float = 0.3            # D readout (sparse code)
    lr_distill: float = 0.1
    eta_fast: float = 0.6
    fast_write_threshold: float = 3.0   # novelty (in units of typical error)
    fast_write_gain: float = 2.0
    fast_blend_max: float = 0.5
    fast_decay_calm: float = 0.995
    fast_decay: float = 0.9995
    distill_per_tick: int = 2
    cusum_slack: float = 0.5
    cusum_threshold: float = 8.0
    transition_ticks: int = 15
    transition_plasticity: float = 0.15
    mature_usage: int = 150
    cascade_levels: int = 4
    cascade_tau1: float = 8.0
    seed: int = 0
    # ablation switches (the layered defenses, strippable one by one)
    use_routing: bool = True
    use_cascade: bool = True
    use_fast: bool = True
    use_sparse: bool = True


@dataclass(slots=True)
class _Pending:
    """What the last tick predicted, and with which circuit it predicted it."""

    prediction: np.ndarray
    code: np.ndarray
    actives: list  # (column, gate, state_used, learns)
    mu: np.ndarray  # the slow expectation the residual code was taken against


class StrataNetwork:
    """Online next-input predictor with stratified plasticity."""

    def __init__(self, config: StrataConfig | None = None) -> None:
        cfg = config or StrataConfig()
        self.cfg = cfg
        sparsity = cfg.code_sparsity if cfg.use_sparse else cfg.code_dim
        self.encoder = SparseEncoder(
            cfg.d_in, cfg.code_dim, sparsity, cfg.seed,
            lr=cfg.encoder_lr, kappa=cfg.encoder_kappa, mode=cfg.encode_mode,
            usage_decay=cfg.encoder_usage_decay,
            grow_from=cfg.encoder_grow_from)
        self.fast = FastWeightMemory(cfg.code_dim, cfg.d_in)
        self.router = ColumnRouter(
            factory=self._make_column,
            top_k=cfg.top_k if cfg.use_routing else 1,
            theta_route=cfg.theta_route if cfg.use_routing else -1.0,
            theta_claim=cfg.theta_claim,
            spawn_rate=cfg.spawn_rate,
            spawn_threshold=cfg.spawn_threshold,
            max_columns=cfg.max_columns if cfg.use_routing else 1,
            adaptive=cfg.adaptive_bands and cfg.use_routing,
        )
        self.context = np.zeros(cfg.d_in)
        self.pending: _Pending | None = None
        self.err_var = 1.0
        self.cusum = 0.0
        self.transition_left = 0
        self.anchor_cursor = 0
        self.tick = 0
        self._tick_heat = 0.0       # |dtheta| written this tick (synaptic work)

    def _make_column(self, index: int) -> LiquidColumn:
        cfg = self.cfg
        return LiquidColumn(
            d_in=cfg.d_in,
            d_out=cfg.d_in,
            key_dim=cfg.code_dim,
            n_state=cfg.n_state,
            cascade_levels=cfg.cascade_levels if cfg.use_cascade else 1,
            cascade_tau1=cfg.cascade_tau1,
            conjunctive_dim=cfg.conjunctive_dim,
            seed=cfg.seed + 1000 + index,
        )

    # ------------------------------------------------------------------ tick

    def step(
        self,
        x: np.ndarray,
        plasticity: float = 1.0,
        routing_prior: np.ndarray | None = None,
    ) -> dict:
        """One tick: score yesterday's prediction, learn, predict tomorrow.

        ``plasticity`` is an external gate on *slow* learning and
        consolidation (a host system's energy/fidelity signal, e.g. the
        organism's metabolism).  Fast-weight capture is deliberately not
        gated: one-shot episodic binding is the cheap path that keeps
        working when the host is too depleted to consolidate.

        ``routing_prior`` is an optional top-down expectation over columns
        from a higher layer; it biases routing inference (bounded) and never
        touches learning.
        """
        cfg = self.cfg
        self._tick_heat = 0.0
        error2, novelty_raw = self._score_and_learn(x, plasticity)
        novelty = novelty_raw / (1.0 + novelty_raw)

        # ---- forward pass -------------------------------------------------
        # route FIRST, on a slow input-space context, so columns form at task
        # granularity; the winning column's centroid is its slow expectation
        # of the input (the slow pathway of a two-pathway code)
        self.context = (1.0 - cfg.context_rho) * self.context + cfg.context_rho * x
        self.router.stats_frozen = self.transition_left > 0
        active = self.router.route(self.context, prior=routing_prior)
        top_column = active[0][0]
        mu = top_column.centroid

        # the sparse dictionary codes only the *deviation* from expectation:
        # the shared component of a regime (which learned features were
        # measured to inherit, collapsing code separation) never reaches it
        residual = x - mu
        code = self.encoder.encode(residual)
        # representation learning obeys the same plasticity gate as the
        # readouts, and its synaptic work heats the host like any other
        moved = self.encoder.learn(residual, code, plasticity)
        self._tick_heat += moved
        if moved > 0.5:
            # an imprint event just rotated the basis (allocation moves are
            # O(1); consolidated incremental moves are far smaller): re-encode
            # so every downstream consumer -- fast-weight bindings above all --
            # keys on the *settled* feature, never the pre-allocation guess
            code = self.encoder.encode(residual)

        prediction_res = np.zeros(cfg.d_in)
        actives_used = []
        for column, gate, learns in active:
            state = column.step(x, novelty)
            prediction_res += gate * column.predict(code)
            actives_used.append((column, gate, state.copy(), learns))
        if cfg.use_fast:
            recalled = self.fast.read(code)
            confidence = float(np.linalg.norm(recalled))
            blend = min(cfg.fast_blend_max, confidence / (confidence + 1.0))
            prediction_res = (1.0 - blend) * prediction_res + blend * recalled
        prediction = mu + prediction_res

        # a column absorbs an input into its identity only when it already
        # strongly owns the context, and never while a regime shift is in
        # progress -- this hysteresis is what prevents a new task from
        # silently dragging an old column's centroid onto itself (absorption)
        claim_affinity = top_column.affinity(self.context)   # raw: no prior
        if (
            self.transition_left == 0
            and claim_affinity >= self.router.claim_threshold(top_column)
        ):
            top_column.update_centroid(x)
            top_column.update_signature(code)
            top_column.update_claim_stats(claim_affinity)

        # ---- consolidation ------------------------------------------------
        if self.transition_left > 0:
            self.transition_left -= 1
        for column in self.router.columns:
            column.consolidate()
        calm = self.transition_left == 0 and novelty_raw < 1.5
        if cfg.use_fast:
            if calm:
                self._distill(top_column, plasticity)
                self.fast.decay(cfg.fast_decay_calm)
            else:
                self.fast.decay(cfg.fast_decay)

        self.pending = _Pending(prediction, code, actives_used, mu.copy())
        self.tick += 1
        return {
            "error": error2,
            "novelty": novelty_raw,
            "columns": len(self.router.columns),
            "spawns": self.router.spawn_count,
            "transition": self.transition_left > 0,
            # total |dtheta| written this tick (slow + fast + distilled);
            # cascade diffusion is excluded -- it moves existing information
            # between timescales, it does not write new information
            "heat": self._tick_heat,
        }

    # ------------------------------------------------------------- learning

    def _score_and_learn(
        self, x: np.ndarray, plasticity: float = 1.0
    ) -> tuple[float | None, float]:
        """Score the pending prediction against reality and apply learning."""
        cfg = self.cfg
        if self.pending is None:
            return None, 0.0
        error = x - self.pending.prediction
        error2 = float(error @ error)
        novelty_raw = error2 / max(self.err_var, 1e-6)
        self.err_var = 0.99 * self.err_var + 0.01 * error2

        self.cusum = max(0.0, self.cusum + novelty_raw - 1.0 - cfg.cusum_slack)
        if self.cusum > cfg.cusum_threshold:
            self.transition_left = cfg.transition_ticks
            self.cusum = 0.0

        # surprising samples are captured by the fast weights below, not by
        # consolidated readouts: gating slow learning off novelty closes the
        # pre-detection window in which a still-claiming old column would
        # learn the first ticks of a new regime at full rate
        novelty_gate = 1.0 / (1.0 + max(0.0, novelty_raw - 1.0))
        # slow readouts consolidate onto a key only as fast as its supporting
        # features stop rotating; until then the fast weights carry it
        # (hippocampus carries while cortex waits)
        maturity = self.encoder.support_maturity(self.pending.code)
        for column, gate, state_used, learns in self.pending.actives:
            if not learns or plasticity <= 0.0:
                continue        # fallback prediction: visible error, no damage
            rate = gate * novelty_gate * plasticity
            if self.transition_left > 0 and column.usage > cfg.mature_usage:
                rate *= cfg.transition_plasticity
            self._tick_heat += column.learn(
                state_used,
                self.pending.code,
                error,
                rate * cfg.lr_state,
                rate * maturity * cfg.lr_code,
            )

        if cfg.use_fast:
            write_rate = cfg.eta_fast * _sigmoid(
                cfg.fast_write_gain * (novelty_raw - cfg.fast_write_threshold)
            )
            # hard floor: the sigmoid tail would otherwise nibble at every
            # boring input forever (and churn the anchor ring); episodic
            # capture is for surprising events only
            if write_rate >= 0.05 * cfg.eta_fast:
                # values live in residual space (consistent with the keys and
                # the slow readouts); mu is added back at prediction time
                self._tick_heat += self.fast.write(
                    self.pending.code, x - self.pending.mu, write_rate
                )
        return error2, novelty_raw

    def _distill(self, owner, plasticity: float = 1.0) -> None:
        """Replay a few fast-weight anchors into the *current* owner's slow
        readout (calm only).  The owner is the column claiming the present
        context, not an affinity argmax over the anchor key: cross-task
        code-to-centroid affinities overlap, and key-based attribution was
        measured to distill one task's associations into another task's
        column (slow corruption of consolidated knowledge)."""
        anchors = self.fast.anchors
        if not anchors or plasticity <= 0.0:
            return
        for _ in range(self.cfg.distill_per_tick):
            key = anchors[self.anchor_cursor % len(anchors)]
            self.anchor_cursor += 1
            value = self.fast.read(key)
            if float(np.linalg.norm(value)) < 0.1:
                continue
            rate = (self.cfg.lr_distill * plasticity
                    * self.encoder.support_maturity(key))
            self._tick_heat += owner.distill(key, value, rate)
