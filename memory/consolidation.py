"""Memory consolidation -- the organism's "sleep-time" housekeeping.

Each cycle it:

1. decays every store (unused memories weaken, dead ones are pruned);
2. replays the most salient episodic memories, reinforcing them; and
3. promotes frequently accessed episodic memories into semantic concepts.

More replay happens on idle cycles, mirroring how consolidation is strongest
during rest.
"""
from __future__ import annotations

from dataclasses import dataclass

from memory.episodic import EpisodicMemory
from memory.semantic import SemanticMemory
from memory.working import WorkingMemory


@dataclass(slots=True)
class ConsolidationStats:
    replayed: int = 0
    promoted: int = 0
    pruned: int = 0


class MemoryConsolidation:
    def __init__(self, replay_k: int = 3, promote_threshold: float = 3.0) -> None:
        self.replay_k = replay_k
        self.promote_threshold = promote_threshold

    def consolidate(self, working: WorkingMemory, episodic: EpisodicMemory,
                    semantic: SemanticMemory, idle: bool = False,
                    fidelity: float = 1.0) -> ConsolidationStats:
        """Consolidate memory.  ``fidelity`` (1 = unimpaired) scales how much
        replay happens and how easily concepts promote, so a metabolically
        stressed organism consolidates less and worse."""
        stats = ConsolidationStats()
        fidelity = max(0.05, min(1.0, fidelity))

        # 1. Decay everywhere; prune what has faded away.
        working.decay()
        stats.pruned += episodic.decay()
        stats.pruned += semantic.decay()

        # 2. Replay the strongest memories (more when resting, fewer when noisy).
        k = max(1, round(self.replay_k * (2 if idle else 1) * fidelity))
        replayed = episodic.recall(k=k, reinforce=True)
        stats.replayed = len(replayed)

        # 3. Promote well-worn episodic memories into durable concepts.  Low
        #    fidelity raises the bar (stressed brains generalise less reliably).
        threshold = self.promote_threshold / fidelity
        for item in episodic.frequently_accessed(threshold):
            key = item.label or str(item.content)
            semantic.integrate(key, importance=item.importance, weight=item.salience)
            stats.promoted += 1

        return stats
