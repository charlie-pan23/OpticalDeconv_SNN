"""Small deterministic Pareto-front and design-point selector."""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping


def pareto_front(rows: Iterable[Mapping[str, Any]], objectives: List[str]) -> List[Mapping[str, Any]]:
    data = list(rows)
    out: List[Mapping[str, Any]] = []
    for i, row in enumerate(data):
        dominated = False
        values_i = [float(row.get(k, 0.0)) for k in objectives]
        for j, other in enumerate(data):
            if i == j:
                continue
            values_j = [float(other.get(k, 0.0)) for k in objectives]
            no_worse = all(b <= a for a, b in zip(values_i, values_j))
            strictly_better = any(b < a for a, b in zip(values_i, values_j))
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            out.append(row)
    return out


def select_balanced(rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Pick a reproducible balanced point from a feasible Pareto set."""

    data = list(rows)
    if not data:
        raise ValueError("no feasible design points")
    front = pareto_front(data, ["area_mm2", "energy_uJ_per_image", "latency_us_per_image", "p99_queue"])

    def norm(key: str, row: Mapping[str, Any]) -> float:
        vals = [float(r.get(key, 0.0)) for r in front]
        lo, hi = min(vals), max(vals)
        return 0.0 if hi == lo else (float(row.get(key, 0.0)) - lo) / (hi - lo)

    return min(
        front,
        key=lambda r: (
            norm("energy_uJ_per_image", r)
            + norm("latency_us_per_image", r)
            + norm("area_mm2", r)
            + norm("p99_queue", r),
            int(r.get("hapr_group_size", 0)),
            int(r.get("adc_macros", 0)),
        ),
    )
