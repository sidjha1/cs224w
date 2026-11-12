import pytest

from solar_hc.search import find_capacity

SETTINGS = dict(initial_kw=5, ceiling_kw=100, tolerance_kw=0.5, audit_points=17)


def response(feasible, status="ok"):
    return {"status": status, "feasible": feasible}


def test_brackets_conservative_boundary_and_caches_trials():
    calls = []
    def evaluate(kw):
        calls.append(kw)
        return response(kw <= 23.12)
    result = find_capacity(evaluate, **SETTINGS)
    assert result["status"] == "ok"
    assert result["hc_kw"] <= 23.12 < result["upper_bound_kw"]
    assert result["upper_bound_kw"] - result["hc_kw"] <= 0.5
    assert len(calls) == len(set(calls)) == result["solve_count"]


@pytest.mark.parametrize("status", ["solver_failure", "missing_ratings"])
def test_unknown_feasibility_never_becomes_a_label(status):
    result = find_capacity(lambda kw: response(True) if kw < 10 else response(None, status), **SETTINGS)
    assert result["status"] == status
    assert result["hc_kw"] is None


def test_baseline_violation_is_not_zero_capacity():
    result = find_capacity(lambda kw: response(False), **SETTINGS)
    assert result["status"] == "baseline_violation"
    assert result["hc_kw"] is None
    assert result["solve_count"] == 1


def test_ceiling_is_censored():
    result = find_capacity(lambda kw: response(True), **SETTINGS)
    assert result["status"] == "ceiling_reached"
    assert result["hc_kw"] is None
    assert result["lower_bound_kw"] == 100


def test_audit_detects_sampled_nonmonotonicity():
    # Exponential bracketing sees 0/5/10/20 feasible, 40 infeasible;
    # audit reveals an earlier violation and subsequent return to feasibility.
    result = find_capacity(lambda kw: response(not (12 <= kw <= 15 or kw > 30)), **SETTINGS)
    assert result["status"] == "nonmonotonic"
    assert result["hc_kw"] is None


def test_zero_can_be_a_valid_bracketed_label():
    result = find_capacity(lambda kw: response(kw == 0), **SETTINGS)
    assert result["status"] == "ok"
    assert result["hc_kw"] == 0
    assert 0 < result["upper_bound_kw"] <= 0.5
