"""Native PV weather/control evaluation followed by fixed-P/Q snapshot dispatch."""
from __future__ import annotations

import json
import re

import numpy as np


def class_json(dss, name):
    dss.Circuit.SetActiveClass(name)
    return json.loads(dss.ActiveClass.ToJSON())


class ExistingPV:
    def __init__(self, dss):
        self.dss = dss
        self.records, self.proxies, self.shape_cache = [], [], {}
        for settings in class_json(dss, "PVSystem"):
            dss.PVsystems.Name(settings["Name"])
            if not dss.CktElement.Enabled():
                continue
            modes = [(key.lower(), value) for key, value in settings.items() if key.lower() in ("pf", "kvar")]
            self.records.append({
                "name": dss.PVsystems.Name(), "connection": dss.CktElement.BusNames()[0].lower(),
                "phases": dss.CktElement.NumPhases(), "kv": float(dss.Properties.Value("kv")),
                "conn": dss.Properties.Value("conn"), "pmpp_kw": dss.PVsystems.Pmpp(),
                "inverter_kva": dss.PVsystems.kVARated(), "base_irradiance": dss.PVsystems.Irradiance(),
                "base_temperature": float(dss.Properties.Value("temperature")),
                "irradiance_shape": dss.PVsystems.yearly(), "temperature_shape": dss.PVsystems.Tyearly(),
                "reactive_mode": modes[-1] if modes else ("pf", 1.0), "source_settings": settings,
            })
        self.controls = []
        for kind in ("InvControl", "ExpControl"):
            for settings in class_json(dss, kind):
                dss.ActiveClass.Name(settings["Name"])
                self.controls.append({"class": kind, "name": settings["Name"],
                                      "enabled": dss.Properties.Value("enabled").lower() in ("yes", "true"),
                                      "source_settings": settings})

    def profile(self, name, npts, temperature=False):
        key = name, temperature
        if key not in self.shape_cache:
            if temperature:
                self.dss.Circuit.SetActiveClass("TShape")
                if name.lower() not in [value.lower() for value in self.dss.ActiveClass.AllNames()]:
                    raise ValueError(f"Unknown PV temperature shape: {name}")
                self.dss.ActiveClass.Name(name)
                interval = float(self.dss.Properties.Value("interval"))
                values = np.asarray([float(value) for value in re.findall(
                    r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?", self.dss.Properties.Value("temp"))])
            else:
                self.dss.LoadShape.Name(name)
                if self.dss.LoadShape.UseActual():
                    raise ValueError(f"PV irradiance must use normalized multipliers: {name}")
                interval = self.dss.LoadShape.SInterval() / 3600
                values = np.asarray(self.dss.LoadShape.PMult(), dtype=float)
            if abs(interval - 0.25) > 1e-6 or len(values) != npts or not np.isfinite(values).all():
                raise ValueError(f"PV profile must share the loads' 15-minute time axis: {name}")
            if not temperature and (values < 0).any():
                raise ValueError(f"Negative PV irradiance: {name}")
            self.shape_cache[key] = values
        return self.shape_cache[key]

    def at(self, index, npts):
        points = []
        for record in self.records:
            irradiance = record["base_irradiance"]
            if record["irradiance_shape"]:
                irradiance *= float(self.profile(record["irradiance_shape"], npts)[index])
            temperature = record["base_temperature"]
            if record["temperature_shape"]:
                temperature = float(self.profile(record["temperature_shape"], npts, temperature=True)[index])
            points.append({**record, "irradiance_kw_m2": irradiance, "temperature_c": temperature})
        return points

    def restore_native(self, points):
        if {point["name"] for point in points} != {record["name"] for record in self.records}:
            raise ValueError("Snapshot PV systems do not match source model")
        for name in self.proxies:
            self.dss.Text.Command(f"edit generator.{name} enabled=no")
        for control in self.controls:
            self.dss.Text.Command(f'edit {control["class"]}.{control["name"]} enabled={"yes" if control["enabled"] else "no"}')
        for point in points:
            self.dss.PVsystems.Name(point["name"])
            self.dss.CktElement.Enabled(True)
            # Snapshot mode ignores dispatch shapes. Preserve their source
            # definitions and set the sampled weather explicitly instead.
            self.dss.PVsystems.Pmpp(point["pmpp_kw"])
            self.dss.Properties.Value("%Pmpp", str(point["source_settings"].get("pctPmpp", 100)))
            self.dss.PVsystems.Irradiance(point["irradiance_kw_m2"])
            self.dss.Properties.Value("temperature", str(point["temperature_c"]))
            mode, value = point["reactive_mode"]
            getattr(self.dss.PVsystems, mode)(value)

    def freeze(self, points, from_snapshot=False):
        """Retain native baseline dispatch while all controls are frozen.

        Constant-P/Q equivalents prevent voltage-dependent native inverter modes
        from quietly changing existing dispatch between independent HC trials.
        They are baseline sources, never part of the added candidate kW.
        """
        dispatch = []
        for index, point in enumerate(points):
            self.dss.PVsystems.Name(point["name"])
            if from_snapshot:
                measured_kw, measured_kvar = point["kw"], point["kvar"]
            else:
                powers = self.dss.CktElement.Powers()
                measured_kw, measured_kvar = -sum(powers[::2]), -sum(powers[1::2])
            if not np.isfinite([measured_kw, measured_kvar]).all():
                raise ValueError(f'Nonfinite baseline PV dispatch: {point["name"]}')
            for key, measured in (("kw", measured_kw), ("kvar", measured_kvar)):
                if key in point and abs(point[key] - measured) > 1e-4 * max(1, abs(measured)):
                    raise ValueError(f'Native PV baseline differs from saved snapshot: {point["name"]} {key}')
            record = {**point, "kw": point.get("kw", measured_kw), "kvar": point.get("kvar", measured_kvar)}
            self.dss.CktElement.Enabled(False)
            name = f"hc_existing_pv_{index}"
            command = "edit" if name in self.proxies else "new"
            self.dss.Text.Command(
                f'{command} generator.{name} bus1={point["connection"]} phases={point["phases"]} '
                f'kv={point["kv"]} conn={point["conn"]} model=1 '
                f'kw={record["kw"]:.12g} kvar={record["kvar"]:.12g} '
                'vminpu=0.01 vmaxpu=2 enabled=yes')
            if name not in self.proxies:
                self.proxies.append(name)
            dispatch.append(record)
        return dispatch
