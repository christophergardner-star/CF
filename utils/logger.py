"""Logging and the live console dashboard.

Uses `rich` for colourful logging and a live vitals dashboard when it is
available, and degrades gracefully to the standard library otherwise so the
kernel always runs.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

try:  # rich is optional; the kernel still runs without it.
    from rich import box
    from rich.console import Console, Group
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    _RICH = True
except Exception:  # pragma: no cover - exercised only when rich is missing
    _RICH = False
    Console = Any  # type: ignore


def rich_available() -> bool:
    return _RICH


def get_console() -> Any:
    return Console() if _RICH else None


def get_logger(name: str = "organism", level: int = logging.INFO,
               console: Any = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    if _RICH:
        handler: logging.Handler = RichHandler(
            console=console, rich_tracebacks=True, show_path=False, markup=True)
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="%H:%M:%S"))
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(name)-10s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _bar(fraction: float, width: int = 14) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return "█" * filled + "░" * (width - filled)


class Dashboard:
    """Renders organism vitals + per-module status as a console panel."""

    def __init__(self, console: Any = None) -> None:
        self.console = console or get_console()
        self.rich = _RICH and self.console is not None

    # -- rich rendering -------------------------------------------------------
    def render(self, snap: dict[str, Any], reports: Sequence[dict[str, Any]]) -> Any:
        if not self.rich:
            return self.plain(snap, reports)

        vitals = Table(box=box.SIMPLE, expand=True, pad_edge=False, show_header=False)
        vitals.add_column("vital", style="bold cyan", no_wrap=True)
        vitals.add_column("value")
        energy_f = _safe(snap, "energy_fraction")
        entropy = _safe(snap, "entropy")
        vitals.add_row("Tick", f"{snap.get('tick', 0)}   (age {snap.get('age', 0)})")
        vitals.add_row("Energy", f"[green]{_bar(energy_f)}[/] {snap.get('energy', 0):.0f}")
        vitals.add_row("Entropy", f"[magenta]{_bar(entropy)}[/] {entropy:.2f}")
        vitals.add_row("Curiosity", f"[yellow]{_bar(_safe(snap, 'curiosity'))}[/] "
                                     f"{_safe(snap, 'curiosity'):.2f}")
        vitals.add_row("Learning rate", f"{_safe(snap, 'learning_rate'):.2f} /tick")
        vitals.add_row("Health", f"[red]{_bar(_safe(snap, 'health'))}[/] "
                                 f"{_safe(snap, 'health'):.2f}")
        vitals.add_row("Active / Sleeping",
                       f"{snap.get('active', 0)} / {snap.get('sleeping', 0)}")
        vitals.add_row("Events processed", f"{snap.get('events', 0)}")
        vitals.add_row("Memories", f"{snap.get('memories', 0)}")
        thermo = bool(snap.get("thermo"))
        if thermo:
            vitals.add_row("Metabolic load",
                           f"[red]{_bar(min(1.0, _safe(snap, 'load') / 30))}[/] "
                           f"{_safe(snap, 'load'):.1f}")
            vitals.add_row("Fidelity", f"[cyan]{_bar(_safe(snap, 'fidelity'))}[/] "
                                       f"{_safe(snap, 'fidelity'):.2f}")
            vitals.add_row("Free-energy reservoir", f"{_safe(snap, 'reservoir'):.0f}")

        columns = ["module", "state", "energy", "entropy", "conf", "idle"]
        if thermo:
            columns[5:5] = ["load", "fidel"]
        mods = Table(box=box.SIMPLE, expand=True, pad_edge=False)
        for col in columns:
            mods.add_column(col, no_wrap=True)
        for r in reports:
            state = r.get("state", "?")
            colour = {"active": "green", "drowsy": "yellow",
                      "sleeping": "dim", "dormant": "dim"}.get(state, "white")
            cells = [r.get("name", "?"), f"[{colour}]{state}[/]",
                     f"{r.get('energy', 0):.0f}", f"{r.get('entropy', 0):.2f}",
                     f"{r.get('confidence', 0):.2f}"]
            if thermo:
                cells += [f"{r.get('load', 0):.1f}", f"{r.get('fidelity', 1):.2f}"]
            cells.append(f"{r.get('idle', 0)}")
            mods.add_row(*cells)
        title = f"[bold]Thermodynamic Kernel :: {snap.get('name', 'organism')}[/bold]"
        return Panel(Group(vitals, Text("Modules", style="bold cyan"), mods),
                     title=title, border_style="cyan", box=box.ROUNDED)

    # -- plain-text fallback --------------------------------------------------
    def plain(self, snap: dict[str, Any], reports: Sequence[dict[str, Any]]) -> str:
        lines = [
            "=" * 60,
            f" Kernel :: {snap.get('name', 'organism')}   tick={snap.get('tick', 0)} "
            f"age={snap.get('age', 0)}",
            "-" * 60,
            f" Energy    : {snap.get('energy', 0):7.1f}  ({_safe(snap, 'energy_fraction'):.0%})",
            f" Entropy   : {_safe(snap, 'entropy'):7.2f}",
            f" Curiosity : {_safe(snap, 'curiosity'):7.2f}",
            f" Learn/t   : {_safe(snap, 'learning_rate'):7.2f}",
            f" Health    : {_safe(snap, 'health'):7.2f}",
            f" Active/Sleep: {snap.get('active', 0)}/{snap.get('sleeping', 0)}"
            f"   Events: {snap.get('events', 0)}   Memories: {snap.get('memories', 0)}",
        ]
        if snap.get("thermo"):
            lines.append(
                f" Load      : {_safe(snap, 'load'):7.2f}   "
                f"Fidelity: {_safe(snap, 'fidelity'):.2f}   "
                f"Reservoir: {_safe(snap, 'reservoir'):.0f}")
        lines.append("-" * 60)
        for r in reports:
            lines.append(
                f" {r.get('name', '?'):<12} {r.get('state', '?'):<9} "
                f"E={r.get('energy', 0):5.0f} H={r.get('entropy', 0):.2f} "
                f"c={r.get('confidence', 0):.2f} idle={r.get('idle', 0)}")
        lines.append("=" * 60)
        return "\n".join(lines)


def _safe(snap: dict[str, Any], key: str) -> float:
    value = snap.get(key, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0
