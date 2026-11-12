from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import load_config


def main():
    parser = argparse.ArgumentParser(description="SMART-DS snapshot HC pilot")
    parser.add_argument("command", choices=["download", "prepare", "label", "run", "validate"])
    parser.add_argument("--config", default="configs/sfo_pilot.json")
    parser.add_argument("--data-dir", help="Use a separate experiment data directory, relative to the current directory")
    parser.add_argument("--max-buses", type=int, help="Process this many additional candidate buses, then checkpoint/export")
    args = parser.parse_args()
    if args.max_buses is not None and args.max_buses < 0:
        parser.error("--max-buses must be nonnegative")
    config = load_config(args.config)
    if args.data_dir:
        config["data_dir"] = str(Path(args.data_dir).resolve())
    if args.command == "download":
        from .source import download
        download(config)
    elif args.command == "prepare":
        from .pipeline import prepare
        _, _, graph, _ = prepare(config)
        print(json.dumps(graph["topology"], indent=2))
    elif args.command == "validate":
        from .validation import validate
        print(json.dumps(validate(config), indent=2))
    else:
        from .pipeline import label, run
        summary = (run if args.command == "run" else label)(config, max_buses=args.max_buses)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
