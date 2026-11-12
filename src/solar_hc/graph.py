from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np

from .simulator import bus_name, names

# DSS LineUnits: none, mi, kft, km, m, ft, in, cm, mm.
METERS = {1: 1609.344, 2: 304.8, 3: 1000.0, 4: 1.0,
          5: 0.3048, 6: 0.0254, 7: 0.01, 8: 0.001}


def orient_tree(nodes, edges, root):
    """Do not silently replace a meshed or disconnected circuit with a tree."""
    graph = nx.MultiGraph()
    graph.add_nodes_from(nodes)
    for index, edge in enumerate(edges):
        graph.add_edge(edge["a"], edge["b"], key=index)
    if root not in graph or not nx.is_tree(graph):
        raise ValueError(f"Expected one radial connected feeder: nodes={len(graph)}, "
                         f"edges={graph.number_of_edges()}, components={nx.number_connected_components(graph)}")
    order = list(nx.bfs_tree(graph, root))
    parents = {root: None}
    depths = {root: 0}
    oriented = []
    for parent, child in nx.bfs_edges(graph, root):
        index = next(iter(graph[parent][child]))
        parents[child] = parent
        depths[child] = depths[parent] + 1
        oriented.append({**edges[index], "parent": parent, "child": child})
    return order, parents, depths, oriented


def extract_graph(simulator, snapshot):
    dss = simulator.dss
    loads = defaultdict(list)
    for load in snapshot["loads"]:
        loads[bus_name(load["connection"])].append(load)
    pvs = defaultdict(list)
    for pv in snapshot.get("pvs", []):
        pvs[bus_name(pv["connection"])].append(pv)
    sources = names(dss.Vsources)
    if len(sources) != 1:
        raise ValueError("Pilot requires one voltage source")
    dss.Vsources.Name(sources[0])
    root = bus_name(dss.CktElement.BusNames()[0])
    nodes = {}
    for name in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(name)
        phases = [int(p) for p in dss.Bus.Nodes() if p != 0]
        voltages = [float(v) for p, v in zip(dss.Bus.Nodes(), dss.Bus.puVmagAngle()[::2]) if p != 0]
        if any(phase not in (1, 2, 3) for phase in phases):
            raise ValueError(f"Unsupported phase identifier at {name}")
        nodes[name] = {"name": name, "kv_base_ln": dss.Bus.kVBase(), "phases": phases,
                       "load_kw": sum(load["kw"] for load in loads[name]),
                       "load_kvar": sum(load["kvar"] for load in loads[name]),
                       "existing_pv_kw": sum(pv["kw"] for pv in pvs[name]),
                       "existing_pv_kvar": sum(pv["kvar"] for pv in pvs[name]),
                       "pv_connections": pvs[name],
                       "voltage_min_pu": min(voltages), "voltage_max_pu": max(voltages),
                       "is_load": bool(loads[name]), "is_root": name == root,
                       "is_transformer_terminal": False, "is_regulator_terminal": False,
                       "load_connections": loads[name]}
    regulator_transformers = set()
    for name in names(dss.RegControls):
        dss.RegControls.Name(name)
        regulator_transformers.add(dss.RegControls.Transformer().lower())
    edges = []
    for name in names(dss.Lines):
        dss.Lines.Name(name)
        if not dss.CktElement.Enabled():
            continue
        if any(dss.CktElement.IsOpen(t, c) for t in (1, 2)
               for c in range(1, dss.CktElement.NumConductors() + 1)):
            # An explicit failure avoids silently dropping partially open phase paths.
            raise ValueError(f"Open/partially open line needs phase-aware extraction: {name}")
        connections = dss.CktElement.BusNames()
        nc = dss.CktElement.NumConductors()
        terminal_nodes = np.asarray(dss.CktElement.NodeOrder()).reshape(-1, nc).tolist()
        unit = dss.Lines.Units()
        is_switch = dss.Lines.IsSwitch()
        if unit not in METERS and not is_switch:
            raise ValueError(f"Line {name} lacks physical length units")
        length = dss.Lines.Length()
        r = np.asarray(dss.Lines.RMatrix()).reshape(dss.Lines.Phases(), -1) * length
        x = np.asarray(dss.Lines.XMatrix()).reshape(dss.Lines.Phases(), -1) * length
        edges.append({"name": "Line." + name, "a": bus_name(connections[0]), "b": bus_name(connections[1]),
                      "kind": "switch" if is_switch else "line",
                      "connections": connections, "terminal_nodes": terminal_nodes,
                      "length_m": 0.0 if is_switch else length * METERS[unit],
                      "r_matrix_ohm": r.tolist(), "x_matrix_ohm": x.tolist(),
                      "r_self_mean_ohm": float(np.diag(r).mean()), "x_self_mean_ohm": float(np.diag(x).mean()),
                      "normal_amps": dss.CktElement.NormalAmps(), "windings": []})
    for name in names(dss.Transformers):
        dss.Transformers.Name(name)
        if not dss.CktElement.Enabled():
            continue
        connections = dss.CktElement.BusNames()
        nc = dss.CktElement.NumConductors()
        terminal_nodes = np.asarray(dss.CktElement.NodeOrder()).reshape(-1, nc).tolist()
        endpoints = list(dict.fromkeys(bus_name(value) for value in connections))
        if len(endpoints) != 2:
            raise ValueError(f"Transformer {name} has {len(endpoints)} distinct bus terminals; needs auxiliary component node")
        windings = []
        for winding, connection in enumerate(connections, 1):
            dss.Transformers.Wdg(winding)
            windings.append({"connection": connection, "kv": dss.Transformers.kV(),
                             "kva": dss.Transformers.kVA(), "r_percent": dss.Transformers.R(),
                             "tap": dss.Transformers.Tap(), "delta": dss.Transformers.IsDelta()})
        normal_kva = float(dss.Properties.Value("normhkva"))
        kind = "regulator" if name.lower() in regulator_transformers else "transformer"
        for endpoint in endpoints:
            nodes[endpoint]["is_transformer_terminal"] = True
            nodes[endpoint]["is_regulator_terminal"] |= kind == "regulator"
        edges.append({"name": "Transformer." + name, "a": endpoints[0], "b": endpoints[1], "kind": kind,
                      "connections": connections, "terminal_nodes": terminal_nodes, "windings": windings,
                      "normal_kva": normal_kva, "xhl_percent": dss.Transformers.Xhl(),
                      "xht_percent": dss.Transformers.Xht(), "xlt_percent": dss.Transformers.Xlt(),
                      "length_m": 0.0, "normal_amps": 0.0,
                      "r_self_mean_ohm": 0.0, "x_self_mean_ohm": 0.0})
    supported = {edge["name"].lower() for edge in edges}
    i = dss.PDElements.First()
    while i:
        name = dss.CktElement.Name()
        if dss.CktElement.Enabled() and dss.CktElement.NumTerminals() > 1 and name.lower() not in supported:
            raise ValueError(f"Unrepresented series component: {name}")
        i = dss.PDElements.Next()
    order, parents, depths, edges = orient_tree(nodes, edges, root)
    for name in nodes:
        nodes[name].update(parent=parents[name], depth=depths[name])
    return {"feeder_id": simulator.config["feeder_id"],
            "source_master_key": simulator.config.get("master_key"),
            "profile_index": snapshot.get("profile_index"), "root": root,
            "nodes": [nodes[name] for name in order], "edges": edges,
            "topology": {"is_tree": True, "node_count": len(nodes), "edge_count": len(edges),
                         "max_depth": max(depths.values())}}
