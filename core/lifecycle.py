"""Module lifecycle: the small sleep/wake state machine every module lives in.

Keeping the state machine separate from the module keeps both testable and lets
the scheduler reason about a module's state without knowing what the module
*does*.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class LifecycleState(Enum):
    DORMANT = "dormant"     # freshly created or a prune candidate
    SLEEPING = "sleeping"   # resting, recovering energy
    DROWSY = "drowsy"       # transitional, low activity
    ACTIVE = "active"       # actively doing work


@dataclass(slots=True)
class Transition:
    tick: int
    frm: LifecycleState
    to: LifecycleState


@dataclass(slots=True)
class Lifecycle:
    """Tracks a module's state, activity recency and transition history."""

    state: LifecycleState = LifecycleState.SLEEPING
    last_active: int = 0
    born_at: int = 0
    history: list[Transition] = field(default_factory=list)

    @property
    def sleeping(self) -> bool:
        return self.state in (LifecycleState.SLEEPING, LifecycleState.DORMANT)

    @property
    def awake(self) -> bool:
        return self.state in (LifecycleState.ACTIVE, LifecycleState.DROWSY)

    def transition_to(self, state: LifecycleState, tick: int) -> bool:
        if state is self.state:
            return False
        self.history.append(Transition(tick, self.state, state))
        self.state = state
        return True

    def mark_active(self, tick: int) -> None:
        self.last_active = tick
        self.transition_to(LifecycleState.ACTIVE, tick)

    def idle_for(self, tick: int) -> int:
        return max(0, tick - self.last_active)

    def age(self, tick: int) -> int:
        return max(0, tick - self.born_at)
