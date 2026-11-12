"""Bracket the first physical violation; never turn numerical failures into HC labels."""
from __future__ import annotations

import time

import numpy as np


def find_capacity(evaluate, *, initial_kw, ceiling_kw, tolerance_kw, audit_points):
    started = time.perf_counter()
    trials = {}

    def trial(kw):
        kw = float(kw)
        if kw not in trials:
            trials[kw] = evaluate(kw)
        return trials[kw]

    def finish(status, lower=None, upper=None):
        return {"status": status, "hc_kw": lower if status == "ok" else None,
                "lower_bound_kw": lower, "upper_bound_kw": upper,
                "solve_count": len(trials), "elapsed_seconds": time.perf_counter() - started,
                "trials": [{"kw": kw, **result} for kw, result in sorted(trials.items())]}

    baseline = trial(0)
    if baseline["status"] != "ok":
        return finish(baseline["status"])
    if not baseline["feasible"]:
        return finish("baseline_violation")
    lower, upper = 0.0, float(initial_kw)
    while True:
        result = trial(upper)
        if result["status"] != "ok":
            return finish(result["status"], lower)
        if not result["feasible"]:
            break
        lower = upper
        if upper >= ceiling_kw:
            return finish("ceiling_reached", lower)
        upper = min(upper * 2, ceiling_kw)
    # Audit the entire discovered interval. This detects sampled nonmonotonicity;
    # it is not a mathematical proof of monotonicity or a search beyond this bracket.
    for kw in np.linspace(0, upper, audit_points):
        result = trial(kw)
        if result["status"] != "ok":
            return finish(result["status"])
    infeasible_seen = False
    for kw, result in sorted(trials.items()):
        if result["feasible"] and infeasible_seen:
            return finish("nonmonotonic")
        infeasible_seen |= not result["feasible"]
    lower = max(kw for kw, result in trials.items() if result["feasible"])
    upper = min(kw for kw, result in trials.items() if not result["feasible"])
    while upper - lower > tolerance_kw:
        middle = (lower + upper) / 2
        result = trial(middle)
        if result["status"] != "ok":
            return finish(result["status"], lower, upper)
        if result["feasible"]:
            lower = middle
        else:
            upper = middle
    return finish("ok", lower, upper)
