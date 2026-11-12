import pytest
import torch
from torch_geometric.data import Data

from solar_hc.common import paths
from solar_hc.export import EDGE_FEATURES, NODE_FEATURES, export_graph
from solar_hc.graph import extract_graph
from solar_hc.simulator import Simulator, names


@pytest.fixture
def config(tmp_path):
    config = {"data_dir": str(tmp_path), "feeder_id": "fixture", "master_key": "Master.dss",
              "voltage_min_pu": 0.95, "voltage_max_pu": 1.05, "daytime_hours": [11, 14]}
    raw, _ = paths(config)
    raw.mkdir(parents=True)
    # A split-phase service transformer: three windings, two distinct buses.
    (raw / "Master.dss").write_text('''clear
new circuit.test bus1=root basekv=12.47 pu=1.0 phases=3
new transformer.service phases=1 windings=3 normhkva=27.5 xhl=2.4 xht=1.6 xlt=2.4
~ wdg=1 bus=root.1 kv=7.2 kva=25 %r=0.266272
~ wdg=2 bus=secondary.1.0 kv=0.12 kva=25 %r=0.532544
~ wdg=3 bus=secondary.0.2 kv=0.12 kva=25 %r=0.532544
new line.service phases=2 bus1=secondary.1.2 bus2=customer.1.2 units=km length=0.02
~ rmatrix=[0.7 | 0.28 0.7] xmatrix=[0.3 | 0.19 0.3] normamps=115 units=km length=0.02
new loadshape.day npts=96 interval=0.25 mult=(''' + ' '.join(['0.5'] * 96) + ''')
new load.leg1 phases=1 conn=wye bus1=customer.1 kv=0.12 kw=4 kvar=0.5 yearly=day
new load.leg2 phases=1 conn=wye bus1=customer.2 kv=0.12 kw=4 kvar=0.5 yearly=day
set voltagebases=[12.47,0.208]
calcvoltagebases
solve mode=yearly number=35040
export voltages
''')
    return config


def make_snapshot(simulator):
    return {"loads": [{**record, "kw": 2.0, "kvar": 0.25} for record in simulator.loads]}


def test_snapshot_applies_load_values_and_does_not_launch_yearly_run(config):
    s = Simulator(config)
    s.apply_snapshot(make_snapshot(s))
    assert s.baseline["feasible"]
    for name in names(s.dss.Loads):
        s.dss.Loads.Name(name)
        assert s.dss.Loads.kW() == pytest.approx(2)
        assert s.dss.Loads.kvar() == pytest.approx(0.25)
        assert not s.dss.Loads.Yearly()
    assert s.dss.Solution.Mode() == 0
    assert s.dss.Solution.ControlMode() == -1
    assert not list(paths(config)[1].glob("*_EXP_VOLTAGES.csv"))


def test_phase_matched_pv_is_total_bus_kw_and_resets_independent_additions(config):
    s = Simulator(config)
    s.apply_snapshot(make_snapshot(s))
    baseline = s.check()
    s.set_injection_bus("customer")
    assert len(s.injections) == 2
    assert s.evaluate(5)["feasible"]
    for name in s.injections:
        s.dss.Generators.Name(name)
        assert s.dss.Generators.kW() == pytest.approx(2.5)
        # Actual terminal injection, not just the requested setpoint.
        assert sum(s.dss.CktElement.Powers()[::2]) == pytest.approx(-2.5, abs=1e-5)
    assert not s.evaluate(80)["feasible"]
    assert s.evaluate(0)["voltage_min_pu"] == pytest.approx(baseline["voltage_min_pu"], abs=1e-7)
    s.set_injection_bus("customer")
    assert s.evaluate(0)["max_loading_pu"] == pytest.approx(baseline["max_loading_pu"], abs=1e-7)


def test_split_phase_graph_and_safe_tensor_export(config):
    s = Simulator(config)
    snapshot = make_snapshot(s)
    s.apply_snapshot(snapshot)
    graph = extract_graph(s, snapshot)
    assert graph["topology"]["node_count"] == 3
    assert graph["topology"]["edge_count"] == 2
    transformer = next(e for e in graph["edges"] if e["kind"] == "transformer")
    assert len(transformer["windings"]) == 3
    assert transformer["terminal_nodes"][2] == [0, 2]
    _, output = paths(config)
    data = export_graph(graph, {"customer": {"status": "ceiling_reached", "hc_kw": None}}, output)
    assert not data.label_mask.any()
    assert data.candidate_mask.sum() == 1
    data = export_graph(graph, {"customer": {"status": "ok", "hc_kw": 0}}, output)
    loaded = Data.from_dict(torch.load(output / "graph.pt", weights_only=True))
    assert loaded.x.shape == (3, len(NODE_FEATURES))
    assert loaded.edge_attr.shape == (4, len(EDGE_FEATURES))
    assert loaded.label_mask.sum() == 1
    assert loaded.y[loaded.label_mask].item() == 0
    assert loaded.parent_index.tolist() == [-1, 0, 1]


def test_missing_declared_rating_is_not_replaced_with_opendss_default(config):
    raw, _ = paths(config)
    path = raw / "Master.dss"
    path.write_text(path.read_text().replace(" normamps=115", ""))
    s = Simulator(config)
    s.apply_snapshot(make_snapshot(s))
    assert s.baseline["status"] == "missing_ratings"
    assert s.baseline["feasible"] is None


def test_snapshot_selection_uses_coincident_profile_values(config):
    s = Simulator(config)
    snapshot = s.select_snapshot()
    assert snapshot["hour_of_day"] == 11
    assert snapshot["profile_index"] == 43
    assert snapshot["aggregate_load_kw"] == 4
    assert s.baseline["feasible"]


def test_switching_candidate_buses_removes_previous_injection(config):
    raw, _ = paths(config)
    path = raw / "Master.dss"
    extra = '''new line.other phases=1 bus1=secondary.1 bus2=other.1 r1=0.7 x1=0.3 units=km length=0.01 normamps=115
new load.other phases=1 conn=wye bus1=other.1 kv=0.12 kw=4 kvar=0.5 yearly=day
'''
    path.write_text(path.read_text().replace("set voltagebases", extra + "set voltagebases"))
    s = Simulator(config)
    s.apply_snapshot(make_snapshot(s))
    baseline = s.baseline
    s.set_injection_bus("customer")
    s.evaluate(20)
    s.set_injection_bus("other")
    assert len(s.injections) == 1
    restored = s.evaluate(0)
    assert restored["max_loading_pu"] == pytest.approx(baseline["max_loading_pu"], abs=1e-7)
    assert restored["voltage_min_pu"] == pytest.approx(baseline["voltage_min_pu"], abs=1e-7)
    s.dss.Generators.Name("hc_candidate_1")
    assert not s.dss.CktElement.Enabled()


def prepared_config(config):
    from solar_hc.common import sha256, write_json
    raw, _ = paths(config)
    config.update(source_prefix="fixture/", initial_kw=5, ceiling_kw=200, tolerance_kw=0.5, audit_points=17, seed=224)
    write_json(raw / "manifest.json", {"source_prefix": "fixture/", "master_key": "Master.dss", "files": {
        "Master.dss": {"bytes": (raw / "Master.dss").stat().st_size, "sha256": sha256(raw / "Master.dss")}}})
    return config


def test_resume_preserves_labels_without_repeating_search(config, monkeypatch):
    from solar_hc.pipeline import label
    from solar_hc.common import read_json
    config = prepared_config(config)
    first = label(config)
    _, output = paths(config)
    checkpoint = read_json(output / "labels.json")
    assert first["valid_labels"] == 1
    def unexpected_solve(*args):
        pytest.fail("Completed HC search was repeated")
    monkeypatch.setattr(Simulator, "evaluate", unexpected_solve)
    second = label(config)
    assert first == second
    assert read_json(output / "labels.json") == checkpoint


def test_changed_settings_cannot_overwrite_prepared_run(config):
    from solar_hc.pipeline import label, prepare
    from solar_hc.common import sha256
    config = prepared_config(config)
    label(config)
    _, output = paths(config)
    before = sha256(output / "snapshot.json")
    with pytest.raises(ValueError, match="different settings"):
        prepare({**config, "tolerance_kw": 0.1})
    assert sha256(output / "snapshot.json") == before


def test_modified_graph_cannot_reuse_checkpoint(config):
    from solar_hc.pipeline import label
    from solar_hc.common import read_json, write_json
    config = prepared_config(config)
    label(config)
    _, output = paths(config)
    graph = read_json(output / "graph.json")
    graph["nodes"][0]["load_kw"] += 1
    write_json(output / "graph.json", graph)
    with pytest.raises(ValueError, match="modified"):
        label(config)


def test_independent_boundary_validation(config):
    from solar_hc.pipeline import label
    from solar_hc.validation import validate
    config = prepared_config(config)
    label(config)
    result = validate(config)
    assert result["status"] == "ok"
    assert result["validated_buses"] == 1
    assert result["pending_buses"] == 0


def add_existing_pv(config, inverter_control=False):
    raw, _ = paths(config)
    path = raw / "Master.dss"
    irradiance = ["1"] * 96
    irradiance[43] = "0.6"
    extra = 'new loadshape.sun npts=96 interval=0.25 mult=(' + ' '.join(irradiance) + ')\n'
    extra += 'new tshape.module_temperature npts=96 interval=0.25 temp=(' + ' '.join(['20'] * 96) + ')\n'
    extra += '''new xycurve.temperature_factor npts=2 xarray=[0 50] yarray=[1 0.5]
new pvsystem.a phases=1 bus1=customer.1 kv=0.12 kva=5 pmpp=4 irradiance=1 yearly=sun tyearly=module_temperature p-tcurve=temperature_factor kvar=0.25
new pvsystem.b phases=1 bus1=customer.2 kv=0.12 kva=2.5 pmpp=2 irradiance=1 yearly=sun pf=1
'''
    if inverter_control:
        extra += '''new xycurve.vv npts=3 xarray=[0.95 1.0 1.05] yarray=[0.5 0.2 -0.5]
new invcontrol.existing derlist=[pvsystem.a] mode=voltvar vvc_curve1=vv voltage_curvex_ref=rated deltaq_factor=0.5
'''
    path.write_text(path.read_text().replace("set voltagebases", extra + "set voltagebases"))


def test_existing_pv_uses_coincident_irradiance_temperature_and_reactive_output(config):
    add_existing_pv(config)
    s = Simulator(config)
    snapshot = s.select_snapshot()
    assert snapshot["profile_index"] == 43
    assert len(snapshot["pvs"]) == 2
    assert snapshot["existing_pv_kw"] == pytest.approx(1.92 + 1.2, abs=1e-5)
    assert snapshot["existing_pv_kvar"] == pytest.approx(0.25, abs=1e-5)
    assert snapshot["existing_pv_dc_nameplate_kw"] == 6
    graph = extract_graph(s, snapshot)
    customer = next(n for n in graph["nodes"] if n["name"] == "customer")
    assert customer["existing_pv_kw"] == pytest.approx(snapshot["existing_pv_kw"])
    assert customer["existing_pv_kvar"] == pytest.approx(0.25, abs=1e-5)
    data = export_graph(graph, {}, paths(config)[1])
    assert data.x[2, NODE_FEATURES.index("existing_pv_kw")] == pytest.approx(3.12, abs=1e-5)
    assert data.x[2, NODE_FEATURES.index("existing_pv_kvar")] == pytest.approx(0.25, abs=1e-5)


def test_existing_pv_dispatch_survives_trials_and_snapshot_replay(config):
    add_existing_pv(config)
    s = Simulator(config)
    snapshot = s.select_snapshot()
    original = snapshot["existing_pv_kw"], snapshot["existing_pv_kvar"]
    baseline = s.baseline
    s.set_injection_bus("customer")
    for kw in [5, 50, 0]:
        s.evaluate(kw)
        for proxy, point in zip(s.pv.proxies, snapshot["pvs"]):
            s.dss.Generators.Name(proxy)
            powers = s.dss.CktElement.Powers()
            assert -sum(powers[::2]) == pytest.approx(point["kw"], abs=1e-5)
            assert -sum(powers[1::2]) == pytest.approx(point["kvar"], abs=1e-5)
    # Restoring a snapshot also removes any candidate injection left in this engine.
    s.evaluate(10)
    s.apply_snapshot(snapshot)
    assert s.baseline["voltage_min_pu"] == pytest.approx(baseline["voltage_min_pu"], abs=1e-7)
    assert (snapshot["existing_pv_kw"], snapshot["existing_pv_kvar"]) == original
    fresh = Simulator(config)
    fresh.apply_snapshot(snapshot)
    assert fresh.baseline["voltage_min_pu"] == pytest.approx(baseline["voltage_min_pu"], abs=1e-7)


def test_inverter_controls_settle_before_existing_pv_dispatch_is_frozen(config):
    add_existing_pv(config, inverter_control=True)
    s = Simulator(config)
    snapshot = s.select_snapshot()
    assert snapshot["inverter_controls"][0]["name"] == "existing"
    assert snapshot["inverter_controls"][0]["enabled"]
    assert abs(snapshot["pvs"][0]["kvar"] - 0.25) > 0.01
    s.set_injection_bus("customer")
    s.evaluate(10)
    s.dss.Generators.Name(s.pv.proxies[0])
    assert -sum(s.dss.CktElement.Powers()[1::2]) == pytest.approx(snapshot["pvs"][0]["kvar"], abs=1e-5)


def test_storage_scenarios_are_still_explicitly_rejected(config):
    raw, _ = paths(config)
    path = raw / "Master.dss"
    path.write_text(path.read_text().replace("set voltagebases", "new storage.battery bus1=customer.1 phases=1 kv=0.12 kwrated=2 kwhrated=4\nset voltagebases"))
    with pytest.raises(ValueError, match="storage"):
        Simulator(config)


def test_existing_pv_checkpoint_replay_and_independent_validation(config):
    from solar_hc.pipeline import label
    from solar_hc.validation import validate
    from solar_hc.common import read_json
    add_existing_pv(config, inverter_control=True)
    config = prepared_config(config)
    first = label(config)
    assert first["existing_pv_systems"] == 2
    assert first["valid_labels"] == 1
    _, output = paths(config)
    before = read_json(output / "snapshot.json")
    assert label(config) == first
    assert read_json(output / "snapshot.json") == before
    assert validate(config)["validated_buses"] == 1
