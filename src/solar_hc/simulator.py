from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from opendssdirect import dss as default_dss

from .common import paths, write_json
from .pv import ExistingPV


def bus_name(connection):
    return connection.split(".")[0].lower()


def names(interface):
    return [name for name in interface.AllNames() if name.lower() != "none"]


def declared_ratings(raw):
    """Distinguish source ratings from OpenDSS's implicit default ampacities."""
    declared = {}
    for path in sorted(raw.rglob("*.dss")):
        commands = re.sub(r"\n\s*~\s*", " ", path.read_text())
        for line in commands.splitlines():
            line = line.split("!", 1)[0]
            match = re.match(r"(?i)\s*(?:new|edit)\s+(linecode|line|transformer)\.([^\s]+)\s+(.*)", line)
            if match:
                kind, name, body = match.groups()
                properties = dict((key.lower(), float(value)) for key, value in
                                  re.findall(r"(?i)\b(normamps|normhkva)\s*=\s*([\d.eE+\-]+)", body))
                declared.setdefault((kind.lower(), name.strip('"').lower()), {}).update(properties)
    return declared


class Simulator:
    def __init__(self, config):
        self.config = config
        self.dss = default_dss.NewContext()
        self.dss.Basic.AllowChangeDir(False)
        self.dss.Basic.AllowForms(False)
        self.dss.Basic.AllowEditor(False)
        raw, output = paths(config)
        self.declared = declared_ratings(raw)
        output.mkdir(parents=True, exist_ok=True)
        master = raw / config["master_key"]
        # Never compile the upstream master: it launches 35,040 solves and plots.
        safe = []
        for line in master.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("!"):
                continue
            if re.match(r"(?i)solve\b", stripped):
                break
            if re.match(r"(?i)new\s+(monitor|energymeter)\.", stripped):
                continue
            if re.match(r"(?i)(redirect|buscoords)\s+", stripped):
                command, relative = stripped.split(maxsplit=1)
                line = f'{command} "{(master.parent / relative).resolve()}"'
            safe.append(line)
        # Redirected files contain relative CSV references. DSS resolves these
        # against the redirect file directory even with AllowChangeDir disabled.
        entry = output / "snapshot_master.dss"
        entry.write_text("\n".join(safe) + "\n")
        self.dss.Text.Command(f'compile "{entry}"')
        if names(self.dss.Generators) or names(self.dss.Storages):
            raise ValueError("Pre-existing non-PV generators and storage are not supported; choose a batteries_none scenario")
        self.pv = ExistingPV(self.dss)
        self.dss.Text.Command(f'set datapath="{output}"')
        self.dss.Text.Command("set mode=snapshot controlmode=static maxiterations=100 maxcontroliter=100 tolerance=1e-8")
        self.loads = self.load_records()

    def load_records(self):
        records = []
        for name in names(self.dss.Loads):
            self.dss.Loads.Name(name)
            if not self.dss.CktElement.Enabled():
                continue
            records.append({"name": name, "connection": self.dss.CktElement.BusNames()[0].lower(),
                            "phases": self.dss.CktElement.NumPhases(), "kv": self.dss.Loads.kV(),
                            "delta": self.dss.Loads.IsDelta(), "base_kw": self.dss.Loads.kW(),
                            "base_kvar": self.dss.Loads.kvar(), "shape": self.dss.Loads.Yearly()})
        return records

    def select_snapshot(self):
        total = None
        shapes = {}
        for record in self.loads:
            shape = record["shape"]
            if shape not in shapes:
                self.dss.LoadShape.Name(shape)
                if self.dss.LoadShape.UseActual() or abs(self.dss.LoadShape.SInterval() - 900) > 1e-6:
                    raise ValueError("Expected normalized 15-minute SMART-DS profiles")
                p = np.asarray(self.dss.LoadShape.PMult(), dtype=float)
                q = np.asarray(self.dss.LoadShape.QMult(), dtype=float)
                # Classic API represents an unallocated Q array as [0.0].
                if q.size == 0 or (q.size == 1 and q[0] == 0 and p.size > 1):
                    q = p.copy()
                if len(p) != len(q) or not np.isfinite(p).all() or not np.isfinite(q).all():
                    raise ValueError(f"Invalid shape {shape}")
                shapes[shape] = p, q
            p, _ = shapes[shape]
            if total is None:
                total = np.zeros_like(p)
            if len(total) != len(p):
                raise ValueError("Profiles must have a common time axis")
            total += record["base_kw"] * p
        if total is None:
            raise ValueError("No enabled loads")
        # OpenDSS fixed-interval shapes: first sample occurs at interval=0.25 h.
        hours = ((np.arange(len(total)) + 1) * 0.25) % 24
        lo, hi = self.config["daytime_hours"]
        candidates = np.flatnonzero((hours >= lo) & (hours <= hi))
        rejected = []
        for index in sorted(candidates, key=lambda i: (total[i], i)):
            index = int(index)
            loads = []
            for record in self.loads:
                p, q = shapes[record["shape"]]
                loads.append({**record, "kw": record["base_kw"] * float(p[index]),
                              "kvar": record["base_kvar"] * float(q[index])})
            snapshot = {"profile_index": index, "model_hour": (index + 1) * 0.25,
                        "model_day": int((index + 1) * 0.25 // 24) + 1,
                        "hour_of_day": float(hours[index]), "aggregate_load_kw": float(total[index]),
                        "selection": f"minimum baseline-feasible coincident load in [{lo}, {hi}] model hours",
                        "pvs": self.pv.at(index, len(total)),
                        "solar_assumption": "new PV delivers its tested AC kW at unity power factor; no weather certification",
                        "loads": loads}
            self.apply_snapshot(snapshot)
            if self.baseline["status"] != "ok":
                _, output = paths(self.config)
                write_json(output / "baseline_audit.json", {"status": self.baseline["status"], "baseline": self.baseline})
                raise ValueError(f"Cannot validate baseline: {self.baseline}")
            if self.baseline["feasible"]:
                snapshot["rejected_lower_load_snapshots"] = rejected
                _, output = paths(self.config)
                write_json(output / "baseline_audit.json", {"status": "ok", "tested_snapshots": len(rejected) + 1,
                                                          "selected_profile_index": index, "rejected": rejected})
                return snapshot
            rejected.append({"profile_index": index, "aggregate_load_kw": float(total[index]),
                             "baseline": self.baseline})
            if len(rejected) % 100 == 0:
                print(f"Rejected {len(rejected)} baseline-infeasible daytime snapshots", flush=True)
        _, output = paths(self.config)
        write_json(output / "baseline_audit.json", {"status": "no_feasible_snapshot", "rejected": rejected})
        raise ValueError("No baseline-feasible daytime snapshot exists; see baseline_audit.json")

    def apply_snapshot(self, snapshot):
        for name in getattr(self, "injections", []):
            self.dss.Text.Command(f"edit generator.{name} enabled=no")
        self.pv.restore_native(snapshot.get("pvs", []))
        for record in snapshot["loads"]:
            self.dss.Loads.Name(record["name"])
            self.dss.Loads.Yearly("")
            self.dss.Loads.Daily("")
            self.dss.Loads.Duty("")
            self.dss.Loads.kW(record["kw"])
            self.dss.Loads.kvar(record["kvar"])
            if abs(self.dss.Loads.kW() - record["kw"]) > 1e-8 or abs(self.dss.Loads.kvar() - record["kvar"]) > 1e-8:
                raise ValueError(f'Failed to set snapshot load {record["name"]}')
        points = snapshot.get("pvs", [])
        if "control_state" in snapshot and all("kw" in point and "kvar" in point for point in points):
            # Resume the recorded electrical state, without depending on inverter
            # hysteresis or a second control-iteration convergence trajectory.
            self.dss.Text.Command("set mode=snapshot controlmode=off")
            self.restore_control_state(snapshot["control_state"])
            dispatch = self.pv.freeze(points, from_snapshot=True)
            self.dss.Solution.Solve()
        else:
            self.dss.Text.Command("set mode=snapshot controlmode=static")
            self.dss.Solution.Solve()
            if not self.dss.Solution.Converged() or not self.dss.Solution.ControlActionsDone():
                raise ValueError("Baseline power flow or controls did not converge")
            native_voltages = np.asarray(self.dss.Circuit.AllBusMagPu())
            snapshot["control_state"] = self.control_state()
            self.dss.Text.Command("set controlmode=off")
            dispatch = self.pv.freeze(points)
            self.dss.Solution.Solve()
            if dispatch and (not self.dss.Solution.Converged() or not np.allclose(
                    native_voltages, self.dss.Circuit.AllBusMagPu(), rtol=0, atol=1e-5)):
                raise ValueError("Fixed-P/Q PV equivalents do not reproduce the native baseline")
        # Preserve saved values on resume, rather than changing fingerprinted data.
        snapshot.setdefault("pvs", dispatch)
        if dispatch and "kw" not in snapshot["pvs"][0]:
            snapshot["pvs"] = dispatch
        snapshot.setdefault("existing_pv_kw", sum(point["kw"] for point in dispatch))
        snapshot.setdefault("existing_pv_kvar", sum(point["kvar"] for point in dispatch))
        snapshot.setdefault("existing_pv_dc_nameplate_kw", sum(point["pmpp_kw"] for point in dispatch))
        snapshot.setdefault("existing_pv_inverter_kva", sum(point["inverter_kva"] for point in dispatch))
        snapshot.setdefault("inverter_controls", self.pv.controls)
        snapshot.setdefault("pv_control_policy", "native controls settle at baseline; existing PV P/Q is then fixed during HC search")
        self.baseline = self.check()
        self.injections = []

    def control_state(self):
        state = {"transformer_taps": [], "capacitors": []}
        for name in names(self.dss.Transformers):
            self.dss.Transformers.Name(name)
            taps = []
            for winding in range(1, self.dss.Transformers.NumWindings() + 1):
                self.dss.Transformers.Wdg(winding)
                taps.append(self.dss.Transformers.Tap())
            state["transformer_taps"].append({"name": name, "taps": taps})
        for name in names(self.dss.Capacitors):
            self.dss.Capacitors.Name(name)
            state["capacitors"].append({"name": name, "states": self.dss.Capacitors.States()})
        return state

    def restore_control_state(self, state):
        for transformer in state["transformer_taps"]:
            self.dss.Transformers.Name(transformer["name"])
            for winding, tap in enumerate(transformer["taps"], 1):
                self.dss.Transformers.Wdg(winding)
                self.dss.Transformers.Tap(tap)
        for capacitor in state["capacitors"]:
            self.dss.Capacitors.Name(capacitor["name"])
            self.dss.Capacitors.States(capacitor["states"])

    def thermal_limits(self):
        """Per energized winding/conductor normal current limits, in A."""
        records = []
        for name in names(self.dss.Lines):
            self.dss.Lines.Name(name)
            if not self.dss.CktElement.Enabled():
                continue
            explicit = self.declared.get(("line", name.lower()), {}).get("normamps")
            inherited = self.declared.get(("linecode", self.dss.Lines.LineCode().lower()), {}).get("normamps")
            limit = self.dss.CktElement.NormalAmps() if explicit is not None or inherited is not None else 0
            records.append({"element": "Line." + name, "limits_a": [limit] * 2,
                            "rating_source": "line" if explicit is not None else "linecode" if inherited is not None else "missing"})
        for name in names(self.dss.Transformers):
            self.dss.Transformers.Name(name)
            if not self.dss.CktElement.Enabled():
                continue
            phases = self.dss.CktElement.NumPhases()
            normal_kva = (float(self.dss.Properties.Value("normhkva"))
                          if "normhkva" in self.declared.get(("transformer", name.lower()), {}) else 0)
            self.dss.Transformers.Wdg(1)
            rated_primary = self.dss.Transformers.kVA()
            factor = normal_kva / rated_primary if rated_primary > 0 else 0
            limits = []
            for winding in range(1, self.dss.Transformers.NumWindings() + 1):
                self.dss.Transformers.Wdg(winding)
                kva, kv = self.dss.Transformers.kVA(), self.dss.Transformers.kV()
                # SMART-DS uses 1- or 3-phase transformers; kV is LL for 3 phase.
                if phases not in (1, 3):
                    raise ValueError(f"Unsupported transformer phase count: {name}")
                denominator = kv * (math.sqrt(3) if phases == 3 else 1)
                limits.append(factor * kva / denominator if denominator > 0 else 0)
            records.append({"element": "Transformer." + name, "limits_a": limits,
                            "rating_source": "transformer_normhkva" if normal_kva else "missing"})
        return records

    def check(self):
        if not self.dss.Solution.Converged():
            return {"status": "solver_failure", "feasible": None}
        violations, missing = [], []
        voltages = []
        for name in self.dss.Circuit.AllBusNames():
            self.dss.Circuit.SetActiveBus(name)
            if self.dss.Bus.kVBase() <= 0:
                missing.append(f"voltage_base:{name}")
                continue
            for phase, value in zip(self.dss.Bus.Nodes(), self.dss.Bus.puVmagAngle()[::2]):
                if phase == 0:
                    continue
                voltages.append(float(value))
                if not np.isfinite(value):
                    missing.append(f"nonfinite_voltage:{name}.{phase}")
                elif value < self.config["voltage_min_pu"] - 1e-7 or value > self.config["voltage_max_pu"] + 1e-7:
                    violations.append({"kind": "voltage", "element": f"{name}.{phase}", "value_pu": value})
        maximum_loading = 0.0
        if not hasattr(self, "ratings"):
            self.ratings = self.thermal_limits()
        for record in self.ratings:
            self.dss.Circuit.SetActiveElement(record["element"])
            nc = self.dss.CktElement.NumConductors()
            nodes = self.dss.CktElement.NodeOrder()
            currents = self.dss.CktElement.CurrentsMagAng()[::2]
            for terminal, limit in enumerate(record["limits_a"]):
                if not np.isfinite(limit) or limit <= 0:
                    missing.append(f'{record["element"]}:terminal{terminal + 1}')
                    continue
                for conductor in range(nc):
                    offset = terminal * nc + conductor
                    if nodes[offset] == 0:
                        continue
                    value = currents[offset]
                    if not np.isfinite(value):
                        missing.append(f'nonfinite_current:{record["element"]}')
                        continue
                    ratio = value / limit
                    maximum_loading = max(maximum_loading, ratio)
                    if ratio > 1 + 1e-7:
                        violations.append({"kind": "thermal", "element": record["element"],
                                           "terminal": terminal + 1, "conductor": conductor + 1,
                                           "current_a": value, "limit_a": limit})
        if not voltages:
            missing.append("no_energized_voltages")
        return {"status": "missing_ratings" if missing else "ok", "feasible": None if missing else not violations,
                "voltage_min_pu": min(voltages) if voltages else None,
                "voltage_max_pu": max(voltages) if voltages else None,
                "max_loading_pu": maximum_loading, "violations": violations, "missing": missing}

    def set_injection_bus(self, bus):
        for name in self.injections:
            self.dss.Text.Command(f'edit generator.{name} enabled=no')
        self.injections = []
        connections = {}
        for record in self.loads:
            if bus_name(record["connection"]) == bus:
                key = record["connection"], record["phases"], record["delta"], record["kv"]
                connections[key] = record
        if not connections:
            raise ValueError(f"No load connections at {bus}")
        for i, record in enumerate(connections.values()):
            if record["phases"] not in (1, 3):
                raise ValueError("Only 1- and 3-phase customer connections supported")
            name = f"hc_candidate_{i}"
            command = "edit" if name in names(self.dss.Generators) else "new"
            self.dss.Text.Command(f'{command} generator.{name} bus1={record["connection"]} '
                                  f'phases={record["phases"]} kv={record["kv"]} '
                                  f'conn={"delta" if record["delta"] else "wye"} '
                                  'model=1 kw=0 kvar=0 vminpu=0.01 vmaxpu=2 enabled=yes')
            self.injections.append(name)

    def evaluate(self, kw):
        for name in self.injections:
            self.dss.Text.Command(f'edit generator.{name} kw={kw / len(self.injections):.12g} kvar=0')
        # Controls remain frozen; a solver failure is never counted as a physical violation.
        self.dss.Solution.Solve()
        return self.check()


def prepare_snapshot(config):
    simulator = Simulator(config)
    snapshot = simulator.select_snapshot()
    snapshot["baseline"] = simulator.baseline
    snapshot["engine_version"] = simulator.dss.Basic.Version()
    snapshot["thermal_limits"] = simulator.ratings
    _, output = paths(config)
    write_json(output / "snapshot.json", snapshot)
    return simulator, snapshot
