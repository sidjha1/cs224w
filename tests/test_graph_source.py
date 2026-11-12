import pytest

from solar_hc.graph import orient_tree
from solar_hc.pipeline import split_feeders
from solar_hc.source import dependency_key, references


@pytest.mark.parametrize("edges", [
    [{"a": "r", "b": "a"}, {"a": "a", "b": "b"}, {"a": "b", "b": "r"}],
    [{"a": "r", "b": "a"}, {"a": "r", "b": "a"}],
    [{"a": "r", "b": "a"}],
])
def test_rejects_cycle_parallel_and_disconnected_topology(edges):
    with pytest.raises(ValueError, match="radial"):
        orient_tree(["r", "a", "b"], edges, "r")


def test_parent_order_is_valid_for_both_sweeps():
    order, parents, depths, edges = orient_tree(["leaf", "r", "a"], [
        {"a": "leaf", "b": "a"}, {"a": "a", "b": "r"}], "r")
    assert order == ["r", "a", "leaf"]
    assert parents == {"r": None, "a": "r", "leaf": "a"}
    assert depths["leaf"] == 2
    assert edges[0]["parent"] == "r"


def test_profiles_and_redirects_are_discovered():
    assert references('Redirect "Loads.dss"\nnew loadshape.x mult=(file=../../profiles/p.csv) '
                      'qmult=(file="../../profiles/q.csv") ! file=ignored\nBuscoords Buscoords.dss') == [
        "Loads.dss", "../../profiles/p.csv", "../../profiles/q.csv", "Buscoords.dss"]
    assert dependency_key("scenarios/base/opendss/LoadShapes.dss", "../../../profiles/p.csv") == "profiles/p.csv"
    with pytest.raises(ValueError, match="escapes"):
        dependency_key("Master.dss", "../private.txt")


def test_splits_keep_physical_feeders_together_and_are_reproducible():
    ids = [f"feeder_{i}" for i in range(20)] * 3
    a, b = split_feeders(ids), split_feeders(list(reversed(ids)))
    assert a == b
    assert [len(a[key]) for key in ("train", "val", "test")] == [14, 3, 3]
    assert len(set(a["train"] + a["val"] + a["test"])) == 20
    assert split_feeders(["one"])["test"] == []
