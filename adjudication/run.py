"""Execute the pre-registered adjudication on Split-MNIST.

For each baseline: train the five tasks sequentially; before each new task
B, snapshot the system and run the full interventional diagnosis
(substrate / timescale / basis) for every previously *mastered* task A
against B.  Criteria are frozen in paper/preregistration.md; this harness
only executes and reports them:

  * event         = >20% relative drop on a mastered task
  * attribution   = whichever exclusions FAIL name the carrying conditions
  * (c) fires     = one event passing all three exclusions
  * incompleteness= >30% of total forgetting magnitude unattributed

Run:  python -m adjudication.run  [--seed 0] [--max-per-class N]
Writes a JSON record to adjudication/results/ and prints the verdict.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from adjudication.split_mnist import load_split_mnist
from adjudication.targets import MLPTarget
from probes import diagnose
from probes.target import clone

MASTERY_BAR = 0.7


def run_baseline(method: str, tasks, seed: int) -> dict:
    target = MLPTarget(method=method, seed=seed)
    record = {"method": method, "events": [], "task_accuracies": []}
    mastered: list[int] = []

    for t, task_b in enumerate(tasks):
        if t > 0:
            snapshot = clone(target)
            for i in mastered:
                task_a = tasks[i]
                report = diagnose(
                    clone(snapshot), task_a, task_b,
                    heldout_a=task_a.x_test[:512],
                    heldout_b=task_b.x_test[:512],
                )
                report["pair"] = f"{task_a.name} after {task_b.name}"
                record["events"].append(_summarise(report))
        target.train(task_b)
        acc = target.evaluate(task_b)
        record["task_accuracies"].append(
            {"task": task_b.name, "acc_after_training": acc})
        if acc >= MASTERY_BAR:
            mastered.append(t)

    record["final_accuracies"] = [
        {"task": task.name, "acc": target.evaluate(task)} for task in tasks]
    return record


def _summarise(report: dict) -> dict:
    return {
        "pair": report["pair"],
        "event": report["event"],
        "drop": round(report["substrate"]["acc_before"]
                      - report["substrate"]["acc_after"], 4),
        "attributed_to": report["attributed_to"],
        "outside_trichotomy": report["outside_trichotomy"],
        "substrate_recovery": round(
            report["substrate"].get("recovery_fraction", float("nan")), 3),
        "timescale_reduction": round(
            report["timescale"].get("reduction_fraction", float("nan")), 3),
        "basis_max_cka": round(report["basis"]["max_cka"], 3),
        "basis_max_layer": report["basis"]["max_layer"],
        "basis_null_floor": {k: round(v, 3) for k, v
                             in report["basis"]["null_floor"].items()},
        "vacuous_layers": report["basis"]["vacuous_layers"],
    }


def verdict(records: list[dict]) -> dict:
    events = [e for r in records for e in r["events"] if e["event"]]
    outside = [e for e in events if e["outside_trichotomy"]]
    total_drop = sum(e["drop"] for e in events)
    unattributed_drop = sum(e["drop"] for e in outside)
    return {
        "forgetting_events": len(events),
        "events_outside_trichotomy": len(outside),
        "unattributed_fraction": (unattributed_drop / total_drop
                                  if total_drop > 0 else 0.0),
        "c_criterion_fired": bool(outside),
        "incompleteness_criterion_fired":
            total_drop > 0 and unattributed_drop / total_drop > 0.30,
        "attribution_counts": {
            cond: sum(1 for e in events if cond in e["attributed_to"])
            for cond in ("substrate", "timescale", "basis")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-per-class", type=int, default=None)
    args = parser.parse_args()

    tasks = load_split_mnist(max_per_class=args.max_per_class)
    records = []
    for method in ("naive", "ewc", "si", "replay"):
        start = time.time()
        record = run_baseline(method, tasks, args.seed)
        record["seconds"] = round(time.time() - start, 1)
        records.append(record)
        n_events = sum(1 for e in record["events"] if e["event"])
        print(f"{method:<7} events={n_events:2d} "
              f"final accs=" + " ".join(
                  f"{a['acc']:.2f}" for a in record["final_accuracies"])
              + f"  ({record['seconds']}s)")

    result = {"seed": args.seed, "verdict": verdict(records),
              "baselines": records}
    out = Path("adjudication/results")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"split_mnist_seed{args.seed}.json"
    path.write_text(json.dumps(result, indent=1))

    v = result["verdict"]
    print(f"\n=== VERDICT (criteria frozen in paper/preregistration.md) ===")
    print(f"forgetting events: {v['forgetting_events']}")
    print(f"attributed to: {v['attribution_counts']}")
    print(f"events outside trichotomy: {v['events_outside_trichotomy']}"
          f"   -> (c) fired: {v['c_criterion_fired']}")
    print(f"unattributed forgetting fraction: "
          f"{v['unattributed_fraction']:.3f}"
          f"   -> incompleteness fired: "
          f"{v['incompleteness_criterion_fired']}")
    print(f"full record: {path}")


if __name__ == "__main__":
    main()
