from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fashion_mlp.data import load_fashion_mnist
from fashion_mlp.model import MLPClassifier
from fashion_mlp.trainer import TrainConfig, fit


def _parse_csv_values(text: str, cast):
    return [cast(part.strip()) for part in text.split(",") if part.strip()]


def _parse_hidden_specs(text: str):
    specs = []
    for spec in text.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        dims = [int(part.strip()) for part in spec.split(",") if part.strip()]
        if len(dims) == 1:
            dims = [dims[0], dims[0]]
        if len(dims) != 2:
            raise ValueError(f"Hidden spec must be one or two integers, got {spec!r}")
        specs.append(",".join(str(dim) for dim in dims))
    return specs


def _parse_activation_specs(text: str):
    specs = []
    for spec in text.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        names = [part.strip().lower() for part in spec.split(",") if part.strip()]
        if len(names) == 1:
            names = [names[0], names[0]]
        if len(names) != 2:
            raise ValueError(f"Activation spec must be one or two names, got {spec!r}")
        specs.append(",".join(names))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search for Fashion-MNIST MLP hyperparameters.")
    parser.add_argument("--data-dir", default="data/fashion-mnist")
    parser.add_argument("--output-dir", default="outputs/hyperparam_search")
    parser.add_argument("--learning-rates", default="0.1,0.05,0.02")
    parser.add_argument("--hidden-dims", default="128,64;256,128")
    parser.add_argument("--weight-decays", default="0,0.0001,0.001")
    parser.add_argument("--activations", default="relu,relu;relu,tanh;tanh,tanh;sigmoid,sigmoid")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr-decay", type=float, default=0.95)
    parser.add_argument("--validation-size", type=int, default=10000)
    parser.add_argument("--max-train", type=int, default=12000)
    parser.add_argument("--max-val", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    train_data, val_data, _ = load_fashion_mnist(
        data_dir=args.data_dir,
        validation_size=args.validation_size,
        seed=args.seed,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=1,
    )

    learning_rates = _parse_csv_values(args.learning_rates, float)
    hidden_specs = _parse_hidden_specs(args.hidden_dims)
    weight_decays = _parse_csv_values(args.weight_decays, float)
    activation_specs = _parse_activation_specs(args.activations)
    results = []

    for run_id, (lr, hidden_spec, wd, activation_spec) in enumerate(
        itertools.product(learning_rates, hidden_specs, weight_decays, activation_specs),
        start=1,
    ):
        safe_hidden = hidden_spec.replace(",", "-")
        safe_activation = activation_spec.replace(",", "-")
        run_dir = out_root / f"run_{run_id:03d}_lr{lr}_h{safe_hidden}_wd{wd}_{safe_activation}"
        config = TrainConfig(
            data_dir=args.data_dir,
            output_dir=str(run_dir),
            hidden_dims=hidden_spec,
            activations=activation_spec,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=lr,
            lr_decay=args.lr_decay,
            weight_decay=wd,
            validation_size=args.validation_size,
            seed=args.seed + run_id,
            max_train=args.max_train,
            max_val=args.max_val,
            max_test=1,
        )
        print(f"\n=== search run {run_id}: lr={lr}, hidden={hidden_spec}, wd={wd}, activations={activation_spec} ===")
        hidden_dims = [int(dim) for dim in hidden_spec.split(",")]
        model = MLPClassifier(hidden_dims=hidden_dims, activations=activation_spec, seed=args.seed + run_id)
        history = fit(model, train_data, val_data, config, run_dir)
        best = max(history, key=lambda row: row["val_accuracy"])
        results.append(
            {
                "run_id": run_id,
                "learning_rate": lr,
                "hidden_dims": hidden_spec,
                "weight_decay": wd,
                "activations": activation_spec,
                "best_epoch": int(best["epoch"]),
                "best_val_accuracy": float(best["val_accuracy"]),
                "best_val_loss": float(best["val_loss"]),
                "run_dir": str(run_dir),
            }
        )

    results.sort(key=lambda row: row["best_val_accuracy"], reverse=True)
    with (out_root / "hyperparam_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with (out_root / "hyperparam_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print("\nTop results:")
    for row in results[:5]:
        print(row)


if __name__ == "__main__":
    main()
