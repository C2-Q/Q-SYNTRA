from __future__ import annotations

import csv
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import qiskit
from mqt.bench import BenchmarkLevel, get_benchmark
from qiskit import transpile
from qiskit.transpiler import CouplingMap, Layout, PassManager
from qiskit.transpiler.passes import ApplyLayout, SabreSwap, SetLayout

OUT = Path("experiments/qroute_budget_go_no_go/results")
OUT.mkdir(parents=True, exist_ok=True)

FAMILIES = ["ghz", "qft", "graphstate", "qaoa"]
SIZES = [8, 12, 16]
HEURISTICS = ["basic", "lookahead", "decay"]
MAX_SEEDS = 64
BUDGETS = [1, 2, 4, 8, 16, 32, 64]


def topology(name: str, n: int) -> CouplingMap:
    if name == "path":
        edges = [(i, i + 1) for i in range(n - 1)]
    elif name == "ring":
        edges = [(i, (i + 1) % n) for i in range(n)]
    elif name == "grid":
        cols = int(np.ceil(np.sqrt(n)))
        edges = []
        for i in range(n):
            r, c = divmod(i, cols)
            if c + 1 < cols and i + 1 < n:
                edges.append((i, i + 1))
            if i + cols < n:
                edges.append((i, i + cols))
    else:
        raise ValueError(name)
    undirected = edges + [(b, a) for a, b in edges]
    return CouplingMap(undirected)


def route_once(qc, cmap: CouplingMap, heuristic: str, seed: int):
    layout = Layout.from_intlist(list(range(qc.num_qubits)), *qc.qregs)
    pm = PassManager([
        SetLayout(layout),
        ApplyLayout(),
        SabreSwap(cmap, heuristic=heuristic, seed=seed, trials=1),
    ])
    t0 = time.perf_counter()
    out = pm.run(qc)
    elapsed = time.perf_counter() - t0
    ops = out.count_ops()
    swaps = int(ops.get("swap", 0))
    return swaps, int(out.depth()), elapsed


def main() -> None:
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "qiskit": qiskit.__version__,
        "families": FAMILIES,
        "sizes": SIZES,
        "heuristics": HEURISTICS,
        "max_seeds": MAX_SEEDS,
        "budgets": BUDGETS,
        "go_thresholds": {
            "late_improvement_after_16_fraction": 0.25,
            "mean_swap_regret_at_16_percent": 3.0,
            "ranking_reversal_fraction": 0.10,
            "coverage_min_families": 3,
            "coverage_min_topologies": 2,
            "independent_router_required_for_final_go": True,
        },
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))

    raw = []
    failures = []
    for family in FAMILIES:
        for n in SIZES:
            try:
                source = get_benchmark(family, BenchmarkLevel.ALG, n)
                qc = transpile(source, basis_gates=["u", "cx"], optimization_level=0)
                qc.remove_final_measurements(inplace=True)
            except Exception as exc:
                failures.append({"family": family, "size": n, "stage": "benchmark", "error": repr(exc)})
                continue
            for topo in ["path", "ring", "grid"]:
                cmap = topology(topo, n)
                for heuristic in HEURISTICS:
                    for seed in range(MAX_SEEDS):
                        try:
                            swaps, depth, elapsed = route_once(qc, cmap, heuristic, seed)
                            raw.append({
                                "family": family,
                                "size": n,
                                "topology": topo,
                                "heuristic": heuristic,
                                "seed": seed,
                                "swaps": swaps,
                                "depth": depth,
                                "seconds": elapsed,
                            })
                        except Exception as exc:
                            failures.append({
                                "family": family, "size": n, "topology": topo,
                                "heuristic": heuristic, "seed": seed,
                                "stage": "routing", "error": repr(exc),
                            })

    fields = ["family", "size", "topology", "heuristic", "seed", "swaps", "depth", "seconds"]
    with (OUT / "raw_trials.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(raw)
    (OUT / "failures.json").write_text(json.dumps(failures, indent=2))

    groups = defaultdict(list)
    for r in raw:
        groups[(r["family"], r["size"], r["topology"], r["heuristic"])].append(r)
    curves = []
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda x: x["seed"])
        cumulative_time = 0.0
        best_swaps = 10**9
        best_depth = 10**9
        by_budget = {}
        for i, r in enumerate(rows, start=1):
            cumulative_time += r["seconds"]
            pair = (r["swaps"], r["depth"])
            if pair < (best_swaps, best_depth):
                best_swaps, best_depth = pair
            if i in BUDGETS:
                by_budget[i] = (best_swaps, best_depth, cumulative_time)
        if MAX_SEEDS not in by_budget:
            continue
        final_swaps, final_depth, final_time = by_budget[MAX_SEEDS]
        for budget in BUDGETS:
            if budget not in by_budget:
                continue
            swaps, depth, secs = by_budget[budget]
            regret = 0.0 if final_swaps == 0 else 100.0 * (swaps - final_swaps) / final_swaps
            curves.append({
                "family": key[0], "size": key[1], "topology": key[2], "heuristic": key[3],
                "budget": budget, "best_swaps": swaps, "best_depth": depth,
                "cumulative_seconds": secs, "final_swaps": final_swaps,
                "final_depth": final_depth, "final_seconds": final_time,
                "swap_regret_percent": regret,
            })
    curve_fields = list(curves[0].keys()) if curves else []
    with (OUT / "budget_curves.csv").open("w", newline="") as f:
        if curve_fields:
            w = csv.DictWriter(f, fieldnames=curve_fields); w.writeheader(); w.writerows(curves)

    at16 = [r for r in curves if r["budget"] == 16]
    late = [r for r in at16 if r["best_swaps"] > r["final_swaps"]]
    late_fraction = len(late) / len(at16) if at16 else 0.0
    mean_regret16 = float(np.mean([r["swap_regret_percent"] for r in at16])) if at16 else 0.0

    by_cell_budget = defaultdict(dict)
    for r in curves:
        cell = (r["family"], r["size"], r["topology"])
        by_cell_budget[(cell, r["budget"])][r["heuristic"]] = (r["best_swaps"], r["best_depth"])
    reversals = []
    comparable = 0
    for cell in sorted({k[0] for k in by_cell_budget}):
        small = by_cell_budget.get((cell, 16), {})
        final = by_cell_budget.get((cell, 64), {})
        if len(small) != len(HEURISTICS) or len(final) != len(HEURISTICS):
            continue
        comparable += 1
        rank16 = sorted(HEURISTICS, key=lambda h: (small[h], h))
        rank64 = sorted(HEURISTICS, key=lambda h: (final[h], h))
        if rank16 != rank64:
            reversals.append({"family": cell[0], "size": cell[1], "topology": cell[2], "rank16": rank16, "rank64": rank64})
    reversal_fraction = len(reversals) / comparable if comparable else 0.0

    late_families = sorted({r["family"] for r in late})
    late_topologies = sorted({r["topology"] for r in late})
    gates = {
        "late_improvement": late_fraction >= 0.25,
        "material_regret": mean_regret16 >= 3.0,
        "ranking_reversal": reversal_fraction >= 0.10,
        "coverage": len(late_families) >= 3 and len(late_topologies) >= 2,
        "independent_router": False,
    }
    official_qiskit_gate = all(gates[k] for k in ["late_improvement", "material_regret", "ranking_reversal", "coverage"])
    decision = "HOLD" if official_qiskit_gate else "NO_GO"
    reason = (
        "Official LightSABRE evidence passes the phenomenon gate, but a second independently maintained router is still required for final GO."
        if official_qiskit_gate else
        "The phenomenon did not pass the frozen official-LightSABRE prevalence, consequence, ranking, and coverage gates."
    )
    summary = {
        "decision": decision,
        "reason": reason,
        "official_qiskit_gate": official_qiskit_gate,
        "final_go": False,
        "n_trial_rows": len(raw),
        "n_failures": len(failures),
        "n_budget16_groups": len(at16),
        "late_improvement_after_16_fraction": late_fraction,
        "mean_swap_regret_at_16_percent": mean_regret16,
        "ranking_reversal_fraction_16_to_64": reversal_fraction,
        "ranking_reversal_count": len(reversals),
        "comparable_cells": comparable,
        "late_families": late_families,
        "late_topologies": late_topologies,
        "gates": gates,
        "reversals": reversals,
    }
    (OUT / "decision.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
