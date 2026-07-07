"""Sparse pattern separation: a random expansion with k-winners-take-all,
optionally *learnable* under continual-learning constraints.

Dissimilar inputs land on nearly disjoint sets of active units, so everything
written downstream (Hebbian outer products, delta-rule updates) interferes
weakly *by construction*.  For two unrelated inputs the expected code overlap
is roughly ``s^2 / dim`` -- the structural shield against forgetting.

By default the projection is fixed and random (fly-hashing / cerebellum
style).  With ``lr > 0`` the rows become a *learned dictionary* trained by
competitive Hebbian learning: winning rows move toward the inputs they win.
Two mechanisms keep a learned encoder from destroying the codes that old
tasks (columns, centroids, fast-weight keys) depend on:

* **per-unit consolidation** -- each row's learning rate decays with its win
  count (``lr / (1 + kappa * usage)``), so features that already support
  consolidated knowledge become rigid while unallocated rows stay plastic;
* **novelty recruitment** -- when no row matches an input well, the least-used
  row is drafted into the winner set and pulled toward the input, so novel
  structure claims *free* capacity instead of repurposing consolidated rows.

Whether these two suffice -- or whether representation drift still corrupts
routing geometry -- is exactly what the long-sequence bench measures.
"""
from __future__ import annotations

import numpy as np


class SparseEncoder:
    """Expand ``d_in`` inputs into an ``s``-sparse code of size ``dim``."""

    def __init__(
        self,
        d_in: int,
        dim: int = 128,
        s: int = 6,
        seed: int = 0,
        lr: float = 0.0,
        kappa: float = 0.2,
        recruit_threshold: float = 0.5,
        mode: str = "kwta",
        usage_decay: float = 0.0,
        grow_from: int | None = None,
    ) -> None:
        if not 0 < s <= dim:
            raise ValueError("need 0 < s <= dim")
        if mode not in ("kwta", "mp"):
            raise ValueError("mode must be 'kwta' or 'mp'")
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.s = s
        self.lr = lr
        self.kappa = kappa
        self.recruit_threshold = recruit_threshold
        self.mode = mode
        self.usage_decay = usage_decay
        # structurally growing dictionary: rows beyond ``active`` are masked
        # out of inference.  Growth appends zero-initialised capacity, which
        # is *backward-transparent* -- every existing key produces identical
        # downstream readouts, because new dimensions contribute nothing
        # until written.  Growth is driven by allocation pressure (a leaky
        # accumulator of recruitment failures), the same detector pattern as
        # column spawning.
        self.active = dim if grow_from is None else max(s, min(grow_from, dim))
        self.grow_block = 4 * s
        self.pressure = 0.0
        self._recruit_failed = False
        # dormant rows are grown capacity not yet allocated: excluded from
        # inference entirely (never win, never activate) until drafted and
        # imprinted -- this is what makes growth *exactly* backward-transparent
        self.dormant = np.zeros(dim, dtype=bool)
        self.projection = rng.standard_normal((dim, d_in)) / np.sqrt(d_in)
        if lr > 0.0:
            # a learned dictionary compares by cosine: unit-norm rows
            self.projection /= np.linalg.norm(
                self.projection, axis=1, keepdims=True)
        self.usage = np.zeros(dim)

    def _grow(self) -> None:
        """Unmask a block of *dormant* capacity.  Grown rows cannot win
        inference for existing inputs (which would churn old codes --
        orphaned bindings through growth); they enter service only via the
        draft-and-imprint allocation path."""
        new_active = min(self.dim, self.active + self.grow_block)
        self.dormant[self.active:new_active] = True
        self.active = new_active
        self.pressure = 0.0

    def _pre(self, x: np.ndarray) -> np.ndarray:
        pre = self.projection[: self.active] @ x
        pre[self.dormant[: self.active]] = -np.inf
        return pre

    def _draft(self) -> int:
        """Pick the row to allocate; note a failure if none is virgin."""
        fresh = int(np.argmin(self.usage[: self.active]))
        if self.usage[fresh] >= 1.0:
            self._recruit_failed = True
        return fresh

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Return the unit-norm ``s``-sparse code for ``x``."""
        if self.mode == "mp":
            return self._encode_mp(x)
        s = min(self.s, self.active)
        pre = self._pre(x)
        winners = np.argpartition(pre, -s)[-s:]
        if self.lr > 0.0 and float(pre[winners].max()) < self.recruit_threshold:
            # nobody knows this input: draft the least-used row instead of
            # letting consolidated rows be dragged toward it
            weakest = winners[np.argmin(pre[winners])]
            fresh = self._draft()
            if fresh not in winners:
                winners = np.where(winners == weakest, fresh, winners)
        code = np.zeros(self.dim)
        code[winners] = np.maximum(pre[winners], 0.0)
        code[~np.isfinite(code)] = 0.0         # a drafted dormant row
        if not code.any():                     # all winners were negative
            code[winners] = 1.0
        if self.lr > 0.0:
            # a freshly drafted row may have pre <= 0: give it a real
            # activation so it participates downstream and learns to the input
            zeros = winners[code[winners] == 0.0]
            if zeros.size and code.any():
                code[zeros] = code[winners].max() * 0.5
        return code / np.linalg.norm(code)

    def _encode_mp(self, x: np.ndarray) -> np.ndarray:
        """Matching-pursuit inference: each chosen atom's contribution is
        *subtracted* before the next is chosen (explaining away).

        This is lateral decorrelation done by inference rather than by weight
        dynamics: when the dictionary contains a well-matched atom, one atom
        absorbs most of the signal and the code approaches one-hot -- even
        when the underlying inputs are mutually correlated (e.g. residuals
        confined to a low-rank subspace, where any dictionary's atoms must
        correlate).  Nothing moves, so it cannot conflict with consolidation.
        """
        x_norm = float(np.linalg.norm(x))
        if x_norm < 1e-12:
            code = np.zeros(self.dim)
            code[: self.s] = 1.0
            return code / np.linalg.norm(code)
        remainder = x.copy()
        code = np.zeros(self.dim)
        for step in range(min(self.s, self.active)):
            pre = self._pre(remainder)
            i = int(np.argmax(pre))
            rem_norm = float(np.linalg.norm(remainder))
            if (self.lr > 0.0 and step == 0
                    and pre[i] < self.recruit_threshold * rem_norm):
                # nobody explains this input: draft the least-used row; it
                # explains nothing yet, so it gets a nominal activation and
                # the remainder is left intact for the real atoms
                fresh = self._draft()
                code[fresh] = 0.5 * rem_norm
                continue
            a = float(pre[i])
            if a <= 0.05 * x_norm or rem_norm < 0.1 * x_norm:
                break
            code[i] += a
            remainder = remainder - a * self.projection[i]
        if not code.any():
            code[int(np.argmax(self._pre(x)))] = 1.0
        return code / np.linalg.norm(code)

    def learn(self, x: np.ndarray, code: np.ndarray, rate: float) -> float:
        """Competitive Hebbian step: active rows move toward the (normalised)
        input, each at its own consolidation-gated rate.  Returns the total
        weight change (synaptic work, for the host's heat accounting)."""
        if rate <= 0.0 or self.lr <= 0.0:
            return 0.0
        # consolidation is not permanent: unreinforced features slowly become
        # recyclable again (usage decays back below the virgin threshold), so
        # a finite dictionary forgets gracefully, least-used first, instead
        # of hitting a capacity cliff on unbounded symbol streams
        if self.usage_decay > 0.0:
            self.usage *= 1.0 - self.usage_decay
        # allocation pressure -> structural growth (budget-capped upstream)
        self.pressure += 0.05 * ((1.0 if self._recruit_failed else 0.0)
                                 - self.pressure)
        self._recruit_failed = False
        if self.pressure > 0.6 and self.active < self.dim:
            self._grow()
        norm = float(np.linalg.norm(x))
        if norm < 1e-12:
            return 0.0
        target = x / norm
        active = np.flatnonzero(code)
        if active.size == 0:
            return 0.0
        strength = code[active] / float(code[active].max())
        mature = [j for j in active if self.usage[j] >= 1.0]
        moved = 0.0
        for i, w in zip(active, strength):
            if self.usage[i] < 1.0:
                # One-shot feature allocation, in *residual* space: the virgin
                # row imprints only what its co-active mature features do not
                # already explain (Gram-Schmidt).  Imprinting the raw input
                # instead was measured to *inherit* input correlations --
                # every family symbol's unit fired for every sibling and code
                # separation collapsed (|cos| 0.45 vs 0.21 random).  Residual
                # allocation decorrelates the dictionary; consolidation below
                # keeps it stable afterwards.
                residual = target.copy()
                for j in mature:
                    if j != i:
                        residual -= (residual @ self.projection[j]) \
                            * self.projection[j]
                r_norm = float(np.linalg.norm(residual))
                if r_norm < 0.15:
                    continue        # fully explained: no new feature needed
                delta = residual / r_norm - self.projection[i]
                self.projection[i] = residual / r_norm
                self.dormant[i] = False        # allocated: enters service
            else:
                eta = rate * self.lr * w / (1.0 + self.kappa * self.usage[i])
                delta = eta * (target - self.projection[i])
                self.projection[i] += delta
                self.projection[i] /= np.linalg.norm(self.projection[i])
            self.usage[i] += w
            moved += float(np.linalg.norm(delta))
        return moved

    def support_maturity(self, code: np.ndarray, full: float = 10.0) -> float:
        """How settled the features under this code are, in [0, 1].

        Slow readouts should consolidate onto a key only as fast as that
        key's supporting rows stop rotating: an activation-weighted mean of
        per-row usage (saturating at ``full`` wins).  Fixed encoders are
        always fully mature."""
        if self.lr <= 0.0:
            return 1.0
        active = np.flatnonzero(code)
        if active.size == 0:
            return 1.0
        weights = code[active]
        maturity = np.minimum(1.0, self.usage[active] / full)
        return float((weights * maturity).sum() / weights.sum())

    @staticmethod
    def overlap(z_a: np.ndarray, z_b: np.ndarray) -> float:
        """Cosine overlap between two codes (both are already unit norm)."""
        return float(z_a @ z_b)
