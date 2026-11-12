"""Train SweepGNN on masked node labels; imports never start a training run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..common import read_json, write_json
from .data import collate_trees, fit_feature_stats, load_splits
from .model import SweepGNN, load_checkpoint


class RegressionMetrics:
    """Accumulate over valid nodes, rather than averaging batch/feeder averages."""

    def __init__(self, requests):
        self.count = 0
        self.absolute_error = 0.0
        self.squared_error = 0.0
        self.requests = {float(r): {"approvals": 0, "false_approvals": 0} for r in requests}

    @torch.no_grad()
    def update(self, prediction, target):
        error = prediction.detach() - target
        self.count += target.numel()
        self.absolute_error += error.abs().sum().item()
        self.squared_error += error.square().sum().item()
        for request, counts in self.requests.items():
            approved = prediction.detach() >= request
            counts["approvals"] += int(approved.sum())
            counts["false_approvals"] += int((approved & (target < request)).sum())

    def compute(self):
        if not self.count:
            return None
        return {
            "labeled_nodes": self.count,
            "mae_kw": self.absolute_error / self.count,
            "rmse_kw": (self.squared_error / self.count) ** 0.5,
            "pv_requests": {str(request): {
                **counts,
                "false_approval_rate": counts["false_approvals"] / counts["approvals"] if counts["approvals"] else None,
            } for request, counts in self.requests.items()},
        }


def choose_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    return torch.device(name)


def train_epoch(model, loader, optimizer, device, requests, gradient_clip_norm):
    model.train()
    metrics = RegressionMetrics(requests)
    for graph in loader:
        graph = graph.to(device)
        if not graph.label_mask.any():
            continue
        optimizer.zero_grad(set_to_none=True)
        prediction = model(graph)
        selected = prediction[graph.label_mask]
        target = graph.y[graph.label_mask]
        # A fixed positive scale changes gradient magnitude, not the MAE optimum.
        loss = nn.functional.l1_loss(selected, target) / model.target_scale_kw
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm, error_if_nonfinite=True)
        optimizer.step()
        metrics.update(selected, target)
    return metrics.compute()


@torch.no_grad()
def evaluate(model, loader, device, requests):
    model.eval()
    metrics = RegressionMetrics(requests)
    for graph in loader:
        graph = graph.to(device)
        if graph.label_mask.any():
            prediction = model(graph)
            metrics.update(prediction[graph.label_mask], graph.y[graph.label_mask])
    return metrics.compute()


def save_checkpoint(path, *, model, optimizer, epoch, config, schema, records, metrics):
    temporary = path.with_suffix(".pt.tmp")
    torch.save({
        "model_config": model.config, "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(), "epoch": epoch,
        "training_config": config, "feature_schema": schema,
        "graphs": records, "metrics": metrics,
    }, temporary)
    temporary.replace(path)


def load_config(path, output_dir=None):
    path = Path(path).resolve()
    config = read_json(path)
    config["graphs"] = [str((path.parent / value).resolve()) for value in config["graphs"]]
    config["split_file"] = str((path.parent / config["split_file"]).resolve())
    config["output_dir"] = str(Path(output_dir).resolve() if output_dir else (path.parent / config["output_dir"]).resolve())
    for key in ("epochs", "batch_size", "hidden_dim"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["learning_rate"] <= 0 or config["weight_decay"] < 0 or config["gradient_clip_norm"] <= 0:
        raise ValueError("Invalid optimizer settings")
    if config["patience"] is not None and (not isinstance(config["patience"], int) or config["patience"] <= 0):
        raise ValueError("patience must be null or a positive integer")
    if not config["pv_request_sizes_kw"] or any(request <= 0 for request in config["pv_request_sizes_kw"]):
        raise ValueError("PV request sizes must be positive")
    return config


def train(config):
    """Explicit entry point. No dataset generation or simulation occurs here."""
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    device = choose_device(config["device"])
    groups, schema, records = load_splits(config["graphs"], config["split_file"])
    output = Path(config["output_dir"])
    # Keep separate experiments reviewable instead of overwriting prior weights.
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}. Choose --output-dir for a new run.")
    model = SweepGNN(node_dim=len(schema["node_features"]), edge_dim=len(schema["edge_features"]),
                     hidden_dim=config["hidden_dim"], dropout=config["dropout"],
                     target_scale_kw=config["target_scale_kw"])
    model.set_feature_stats(fit_feature_stats(groups["train"]))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(config["seed"])
    loaders = {split: DataLoader(graphs, batch_size=config["batch_size"],
                                shuffle=split == "train", collate_fn=collate_trees,
                                generator=generator if split == "train" else None)
               for split, graphs in groups.items() if graphs}
    has_validation = "val" in loaders
    manifest = {"config": config, "device": str(device), "torch_version": str(torch.__version__),
                "feature_schema": schema, "graphs": records,
                "counts": {split: {"graphs": len(graphs), "feeders": len({g.feeder_id for g in graphs}),
                                   "labeled_nodes": sum(int(g.label_mask.sum()) for g in graphs)}
                           for split, graphs in groups.items()},
                "false_approval_definition": "predicted HC >= request but conservative HC label < request; denominator = all approvals",
                "seed_note": "Seeds control initialization and data order; device kernels may remain nondeterministic."}
    write_json(output / "run.json", manifest)
    print(f"Device: {device}; {manifest['counts']}", flush=True)
    if not has_validation:
        print("No validation feeders: early stopping and best-model selection are disabled.", flush=True)
    history, best_mae, stale_epochs = [], float("inf"), 0
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_epoch(model, loaders["train"], optimizer, device,
                                    config["pv_request_sizes_kw"], config["gradient_clip_norm"])
        val_metrics = evaluate(model, loaders["val"], device, config["pv_request_sizes_kw"]) if has_validation else None
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics,
               "elapsed_seconds": time.perf_counter() - epoch_start}
        history.append(row)
        checkpoint_args = dict(model=model, optimizer=optimizer, epoch=epoch, config=config,
                               schema=schema, records=records, metrics=row)
        save_checkpoint(output / "last.pt", **checkpoint_args)
        if has_validation:
            if val_metrics["mae_kw"] < best_mae:
                best_mae, stale_epochs = val_metrics["mae_kw"], 0
                save_checkpoint(output / "best.pt", **checkpoint_args)
            else:
                stale_epochs += 1
        write_json(output / "history.json", history)
        val_text = f" val MAE={val_metrics['mae_kw']:.3f} kW" if has_validation else ""
        print(f"Epoch {epoch}: train MAE={train_metrics['mae_kw']:.3f} kW{val_text}", flush=True)
        if has_validation and config["patience"] is not None and stale_epochs >= config["patience"]:
            print(f"Early stopping after {stale_epochs} epochs without validation improvement.", flush=True)
            break
    chosen = output / ("best.pt" if has_validation else "last.pt")
    model, checkpoint = load_checkpoint(chosen, device)
    # Held-out test data is evaluated only once, after selecting the checkpoint.
    final = {split: evaluate(model, loader, device, config["pv_request_sizes_kw"])
             for split, loader in loaders.items()}
    result = {"selected_checkpoint": str(chosen), "selected_epoch": checkpoint["epoch"],
              "epochs_completed": len(history), "elapsed_seconds": time.perf_counter() - started,
              "metrics": final}
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Train SweepGNN on curated feeder snapshots")
    parser.add_argument("--config", default="configs/sweepgnn.json")
    parser.add_argument("--output-dir", help="New experiment directory, relative to the current directory")
    args = parser.parse_args()
    print(json.dumps(train(load_config(args.config, args.output_dir)), indent=2))


if __name__ == "__main__":
    main()
