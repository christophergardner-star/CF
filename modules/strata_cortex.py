"""The cortex: the first organ whose behaviour is *learned*, not coded.

Every other module follows hand-written rules.  The cortex wraps a STRATA
network (``strata/``) and learns, online and continually, to predict the next
percept.  It subscribes to :class:`ObservationEvent`, narrates its prediction
as a :class:`ThoughtEvent`, raises a :class:`CuriosityEvent` when its own
surprisal spikes, and announces structural growth (a new STRATA column) as a
:class:`LearningEvent`.

The kernel's physics gates the substrate's plasticity:

* slow learning and consolidation run at a rate set by the module's
  **fidelity** (metabolic load degrades learning before it degrades acting);
* when the module cannot afford :class:`~core.energy.Action.LEARN`, slow
  plasticity drops to zero and only the cheap one-shot fast-weight capture
  continues -- a tired cortex stops consolidating but keeps taking notes.

And the substrate's plasticity heats the kernel: every synaptic update has a
Frobenius norm ``|dtheta|`` (updates are outer products, so it comes for
free), and the cortex pays ``work(LEARN, scale ~ |dtheta|)`` for it.  On the
thermodynamic substrate (``--thermo``) that work becomes *load*: load raises
entropy, entropy lowers fidelity, fidelity throttles the next tick's
plasticity -- and under heavy load LEARN falls out of the reachable action
manifold entirely.  Learning bursts therefore carry their own refractory
period; no cooling rule is coded here, the loop closes through the kernel's
existing three-field physics.

Tokens are grounded as stable random unit vectors (hash-seeded), so the same
symbol always produces the same input and genuinely novel symbols produce
genuinely novel vectors.  This module is the only place the kernel touches
numpy; it is loaded via a dotted config path, never imported by the registry,
so the rest of the organism stays pure Python.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from core.energy import Action
from core.events import (
    CuriosityEvent,
    Event,
    LearningEvent,
    ObservationEvent,
    ThoughtEvent,
)
from core.module import Module
from strata.network import StrataConfig, StrataNetwork

_MAX_LEXICON = 512


def _token_vector(token: str, d: int) -> np.ndarray:
    """A stable, unique unit vector for a symbol (deterministic across runs)."""
    seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little")
    vector = np.random.default_rng(seed).standard_normal(d)
    return vector / np.linalg.norm(vector)


class StrataCortexModule(Module):
    subscriptions: tuple[type[Event], ...] = (ObservationEvent,)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.net = StrataNetwork(StrataConfig(
            d_in=int(self.options.get("d_in", 24)),
            code_dim=int(self.options.get("code_dim", 128)),
            seed=int(self.options.get("seed", 0)),
        ))
        self._lexicon: dict[str, np.ndarray] = {}
        self._predicted: str | None = None
        #: energy charged per unit of |dtheta| (as LEARN-action scale)
        self._heat_gain = float(self.options.get("heat_gain", 1.0))
        self.state.update(steps=0, hits=0, misses=0, columns=0, spawns=0,
                          heat=0.0)

    # -- cognition -------------------------------------------------------
    def observe(self) -> list[Event]:
        return self.drain_inbox()

    def think(self, observations: list[Event]) -> list[Event]:
        events: list[Event] = []
        for event in observations:
            if not isinstance(event, ObservationEvent):
                continue
            token = str(event.get("token"))
            vector = self._ground(token)

            # score the previous prediction before it is overwritten
            if self._predicted is not None:
                hit = self._predicted == token
                self.state["hits" if hit else "misses"] += 1
                self.confidence = 0.9 * self.confidence + 0.1 * float(hit)

            # the kernel's physics gates the substrate's plasticity ...
            plasticity = self.fidelity if self.can_afford(Action.LEARN) else 0.0
            report = self.net.step(vector, plasticity=plasticity)
            # ... and the substrate's plasticity heats the kernel: synaptic
            # work is paid for in energy, part of which becomes load
            heat = report["heat"]
            if heat > 1e-9:
                self.work(Action.LEARN, min(3.0, self._heat_gain * heat))
            self.state["heat"] = round(
                0.9 * self.state["heat"] + 0.1 * heat, 4)
            self.state["steps"] += 1
            self.state["columns"] = report["columns"]
            self._predicted = self._decode(self.net.pending.prediction)

            novelty = report["novelty"]
            self.entropy.perturb(min(1.0, novelty / 5.0) * 0.5)
            events.append(ThoughtEvent(
                source=self.name,
                payload={
                    "about": token,
                    "prediction": self._predicted,
                    "error": None if report["error"] is None
                    else round(report["error"], 4),
                    "novelty": round(novelty, 2),
                    "columns": report["columns"],
                },
                priority=0.4 + 0.4 * min(1.0, novelty / 5.0)))
            if novelty > 3.0:
                events.append(CuriosityEvent(
                    source=self.name,
                    payload={"token": token, "novelty": round(novelty, 2),
                             "reason": "surprisal"},
                    priority=min(1.0, 0.5 + novelty / 10.0)))
            if report["spawns"] > self.state["spawns"]:
                self.state["spawns"] = report["spawns"]
                events.append(LearningEvent(
                    source=self.name,
                    payload={"grew": "column", "columns": report["columns"],
                             "trigger": token},
                    priority=0.7))
        return events

    def learn(self, thoughts: list[Event]) -> None:
        # learning already happened inside the substrate; here we only keep
        # importance in step with how useful the cortex has proven to be
        scored = self.state["hits"] + self.state["misses"]
        if scored:
            accuracy = self.state["hits"] / scored
            self.importance = 0.95 * self.importance + 0.05 * (0.4 + 0.5 * accuracy)

    # -- token grounding ---------------------------------------------------
    def _ground(self, token: str) -> np.ndarray:
        vector = self._lexicon.get(token)
        if vector is None:
            vector = _token_vector(token, self.net.cfg.d_in)
            if len(self._lexicon) >= _MAX_LEXICON:      # forget oldest symbol
                self._lexicon.pop(next(iter(self._lexicon)))
            self._lexicon[token] = vector
        return vector

    def _decode(self, prediction: np.ndarray) -> str | None:
        """Read the prediction back out as the nearest known symbol."""
        norm = float(np.linalg.norm(prediction))
        if norm < 1e-9 or not self._lexicon:
            return None
        best_token, best_sim = None, -1.0
        for token, vector in self._lexicon.items():
            sim = float(vector @ prediction) / norm
            if sim > best_sim:
                best_token, best_sim = token, sim
        return best_token
