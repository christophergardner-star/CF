"""Boot the thermodynamic kernel and let it live for a while.

``main.py`` is the *composition root*: it is the only place that knows how all
the pieces fit together.  It builds the event bus, the memory subsystem, the
cognitive modules and the organism, wires them with dependency injection, then
runs the tick loop under a live console dashboard.

Usage::

    python main.py                 # live dashboard for the default run
    python main.py --ticks 200     # run longer
    python main.py --headless      # plain-text output, no live dashboard
    python main.py --no-adapt      # disable structural adaptation
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import random
import sys

from core.adaptation import StructuralAdapter
from core.energy import EnergyBudget
from core.entropy import EntropyField
from core.events import EventBus
from core.metabolism import EnergyReservoir
from core.organism import Organism
from core.scheduler import Scheduler
from memory.consolidation import MemoryConsolidation
from memory.episodic import EpisodicMemory
from memory.manager import MemoryManager
from memory.semantic import SemanticMemory
from memory.working import WorkingMemory
from modules.registry import ModuleFactory
from utils.config import Config
from utils.logger import Dashboard, get_console, get_logger
from utils.metrics import Metrics


class PatternedWorld:
    """A tiny structured environment: a repeating pattern peppered with novelty.

    Perception samples :meth:`sense`; the organism advances it via :meth:`step`.
    The mix of predictable structure and surprise is what gives curiosity,
    language and the planner something to chew on.
    """

    def __init__(self, seed: int = 7, novelty_rate: float = 0.12) -> None:
        self.rng = random.Random(seed)
        self.pattern = ["red", "blue", "green", "circle", "square", "blue"]
        self.novelty_rate = novelty_rate
        self.t = 0
        self._current = self.pattern[0]

    def step(self) -> None:
        self.t += 1
        if self.rng.random() < self.novelty_rate:
            self._current = f"novel-{self.rng.randint(0, 9999)}"
        else:
            self._current = self.pattern[self.t % len(self.pattern)]

    def sense(self) -> str:
        return self._current


def build_organism(config: Config, logger: logging.Logger) -> Organism:
    """Wire every subsystem together (the dependency-injection graph)."""
    bus = EventBus()

    # Memory subsystem.
    working = WorkingMemory(capacity=config.memory.working_capacity)
    episodic = EpisodicMemory(capacity=config.memory.episodic_capacity)
    semantic = SemanticMemory()
    consolidation = MemoryConsolidation(replay_k=config.memory.replay_k,
                                        promote_threshold=config.memory.promote_threshold)
    memory = MemoryManager(bus, working, episodic, semantic, consolidation, logger)

    # The external world + the module factory (perception gets the sensor).
    world = PatternedWorld(seed=config.seed)
    mode = "thermodynamic" if config.thermo.enabled else "classic"
    factory = ModuleFactory(bus, injectors={"perception": {"sensor": world.sense}},
                            metabolism=mode, thermo_params=config.thermo.params(),
                            seed=config.seed)
    modules = [factory.build(spec) for spec in config.modules]
    reservoir = EnergyReservoir(level=config.thermo.reservoir_level,
                                capacity=config.thermo.reservoir_capacity,
                                influx=config.thermo.influx)

    scheduler = Scheduler(
        energy_reserve=config.scheduler.energy_reserve,
        wake_threshold=config.scheduler.wake_threshold,
        sleep_after_idle=config.scheduler.sleep_after_idle,
        max_active=config.scheduler.max_active,
        logger=logger)
    adapter = StructuralAdapter(
        enabled=config.adaptation.enabled,
        prune_idle_ticks=config.adaptation.prune_idle_ticks,
        spawn_curiosity=config.adaptation.spawn_curiosity,
        max_dynamic=config.adaptation.max_dynamic,
        weight_decay=config.adaptation.weight_decay,
        merge_idle=config.adaptation.merge_idle,
        logger=logger)

    energy = EnergyBudget(config.organism.energy.capacity,
                          config.organism.energy.level,
                          config.organism.energy.regen_rate)
    entropy = EntropyField(config.organism.entropy.level,
                           config.organism.entropy.baseline,
                           config.organism.entropy.relaxation)

    return Organism(
        name=config.organism.name, bus=bus, scheduler=scheduler, memory=memory,
        energy=energy, entropy=entropy, metrics=Metrics(), adapter=adapter,
        module_factory=factory, logger=logger, modules=modules, world=world,
        reservoir=reservoir, thermo=config.thermo.enabled,
        diffusion=config.thermo.diffusion,
        run_cost=config.organism.run_cost, tick_interval=config.organism.tick_interval)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thermodynamic kernel for continual intelligence")
    parser.add_argument("--ticks", type=int, default=None, help="number of ticks to run")
    parser.add_argument("--interval", type=float, default=None, help="seconds between ticks")
    parser.add_argument("--headless", action="store_true", help="plain output, no live dashboard")
    parser.add_argument("--no-adapt", action="store_true", help="disable structural adaptation")
    parser.add_argument("--thermo", action="store_true",
                        help="run on the thermodynamic substrate (energy+load+entropy)")
    parser.add_argument("--cortex", action="store_true",
                        help="add the STRATA cortex: an organ that learns to "
                             "predict percepts instead of following coded rules "
                             "(requires numpy)")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    return parser.parse_args(argv)


def _use_utf8_stdout() -> None:
    """Best-effort UTF-8 output so the dashboard's block/box glyphs render on
    Windows consoles (whose default cp1252 codec cannot encode them)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main(argv: list[str] | None = None) -> None:
    _use_utf8_stdout()
    args = parse_args(argv)
    config = Config.default()
    if args.ticks is not None:
        config.ticks = args.ticks
    if args.interval is not None:
        config.organism.tick_interval = args.interval
    if args.seed is not None:
        config.seed = args.seed
    if args.no_adapt:
        config.adaptation.enabled = False
    if args.thermo:
        config.thermo.enabled = True
    if args.cortex:
        from utils.config import EnergyConfig, ModuleConfig
        # the cortex gets a brain-sized battery: learning is the most complex
        # action (complexity 0.8), so its reachability ceiling scales with
        # energy capacity -- at the default 32 the module's own baseline
        # THINK/COMMUNICATE heat already pins load above the LEARN ceiling
        # on the thermodynamic substrate, and the organ can never learn
        config.modules.append(ModuleConfig(
            name="cortex", type="modules.strata_cortex:StrataCortexModule",
            base_priority=0.6, importance=0.65,
            energy=EnergyConfig(capacity=64.0, level=64.0, regen_rate=2.6)))

    console = get_console()
    dashboard = Dashboard(console)
    live_mode = dashboard.rich and not args.headless
    # Keep the live dashboard readable by quietening info logs while it is up.
    logger = get_logger("organism", logging.WARNING if live_mode else logging.INFO, console)

    organism = build_organism(config, logger)
    ticks = config.ticks

    if live_mode:
        from rich.live import Live

        def on_tick(org: Organism) -> None:
            live.update(dashboard.render(org.metrics.latest, org.reports()))

        with Live(console=console, refresh_per_second=12, screen=False) as live:
            asyncio.run(organism.run(ticks, on_tick=on_tick))
        console.print(dashboard.render(organism.metrics.latest, organism.reports()))
    else:
        step = max(1, ticks // 12)

        def on_tick(org: Organism) -> None:
            if org.age % step == 0 or org.age == ticks:
                print(dashboard.plain(org.metrics.latest, org.reports()))

        asyncio.run(organism.run(ticks, on_tick=on_tick))

    _print_summary(organism)


def _print_summary(organism: Organism) -> None:
    summary = organism.metrics.summary()
    mem = organism.memory.snapshot()
    lines = [
        "",
        "Run complete.",
        f"  ticks           : {int(summary['ticks'])}",
        f"  avg energy      : {summary['avg_energy']:.1f}",
        f"  avg entropy     : {summary['avg_entropy']:.2f}",
        f"  avg curiosity   : {summary['avg_curiosity']:.2f}",
        f"  avg learn/tick  : {summary['avg_learning_rate']:.2f}",
        f"  memories        : working={mem['working']} episodic={mem['episodic']} "
        f"semantic={mem['semantic']}",
        f"  modules alive   : {len(organism.modules)}",
        f"  events processed: {organism.bus.processed}",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
