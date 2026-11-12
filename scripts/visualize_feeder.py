#!/usr/bin/env python3
"""Generate an offline feeder explorer using only Python's standard library.

    python3 scripts/visualize_feeder.py --open
    python3 scripts/visualize_feeder.py --data-dir data/processed/ANOTHER_FEEDER
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import webbrowser

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FEEDER = "SFO_P1U_p1uhs0_1247--p1udt942"


def make_data(directory):
    graph = json.loads((directory / "graph.json").read_text())
    labels_path = directory / "labels.json"
    labels = json.loads(labels_path.read_text())["buses"] if labels_path.exists() else {}
    nodes, limits = [], {}
    for node in graph["nodes"]:
        result = labels.get(node["name"], {})
        hc = result.get("hc_kw") if result.get("status") == "ok" else None
        role = ("root" if node["is_root"] else "customer" if node["is_load"] else
                "transformer terminal" if node["is_transformer_terminal"] else "junction")
        nodes.append(dict(name=node["name"], parent=node["parent"], depth=node["depth"], role=role,
                          kv=node["kv_base_ln"], kw=node["load_kw"], vmin=node["voltage_min_pu"],
                          vmax=node["voltage_max_pu"], hc=hc,
                          pv_kw=node.get("existing_pv_kw", 0), pv_kvar=node.get("existing_pv_kvar", 0),
                          pv_systems=len(node.get("pv_connections", []))))
        if hc is not None:
            upper = next((trial for trial in result.get("trials", []) if trial["kw"] == result["upper_bound_kw"]), {})
            violations = upper.get("violations", [])
            limits[node["name"]] = dict(upper=result["upper_bound_kw"], kind=violations[0]["kind"] if violations else "unknown")
    if not nodes:
        raise ValueError("Graph has no buses")
    capacities = [node["hc"] for node in nodes if node["hc"] is not None]
    edges = [{key: edge.get(key) for key in ("name", "parent", "child", "kind", "normal_kva")} for edge in graph["edges"]]
    return dict(feeder=graph["feeder_id"], nodes=nodes, edges=edges,
                maxDepth=max(node["depth"] for node in nodes),
                maxHC=max(capacities) if capacities else 0, minHC=min(capacities) if capacities else 0,
                maxPV=max(node["pv_kw"] for node in nodes),
                limits=limits, roles=dict(Counter(node["role"] for node in nodes)),
                components=dict(Counter(edge["kind"] for edge in edges)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / DEFAULT_FEEDER,
                        help="Processed feeder directory containing graph.json and optional labels.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "visualizations" / "feeder-local.html")
    parser.add_argument("--open", action="store_true", help="Open the generated visualization in your default browser")
    args = parser.parse_args()
    try:
        data = make_data(args.data_dir.resolve())
    except (OSError, ValueError, KeyError) as error:
        parser.error(f"Cannot load processed feeder: {error}")
    template = Path(__file__).with_name("feeder_viewer.html").read_text()
    if template.count("__FEEDER_DATA__") != 1:
        raise ValueError("Viewer template must contain exactly one data placeholder")
    # Escape HTML-significant characters so data stays inside its JSON script tag.
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False)
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    output = args.output.resolve()
    if output in {args.data_dir.resolve() / "graph.json", args.data_dir.resolve() / "labels.json"}:
        parser.error("Output must not overwrite source graph or labels")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__FEEDER_DATA__", payload))
    print(f"Created {output}")
    print(f"{len(data['nodes'])} buses, {len(data['edges'])} components, {len(data['limits'])} valid HC labels")
    if args.open and not webbrowser.open(output.as_uri()):
        print("Browser did not open automatically. Open the HTML file above manually.")


if __name__ == "__main__":
    main()
