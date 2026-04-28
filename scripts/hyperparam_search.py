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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search for Fashion-MNIST MLP hyperparameters.")
    parser.add_argument("--data-dir", default="data/fashion-mnist")
    parser.add_argument("--output-dir", default="outputs/hyperparam_search")
    parser.add_argument("--learning-rates", default="0.1,0.05,0.02")
    parser.add_argument("--hidden-dims", default="64,128,256")
    parser.add_argument("--weight-decays", default="0,0.0001,0.001")
    parser.add_argument("--activations", default="relu,tanh")
    parser.add_argument("--epochs", type=int, default=5)
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
    hidden_dims = _parse_csv_values(args.hidden_dims, int)
    weight_decays = _parse_csv_values(args.weight_decays, float)
    activations = _parse_csv_values(args.activations, str)
    results = []

    for run_id, (lr, hidden, wd, activation) in enumerate(
        itertools.product(learning_rates, hidden_dims, weight_decays, activations),
        start=1,
    ):
        run_dir = out_root / f"run_{run_id:03d}_lr{lr}_h{hidden}_wd{wd}_{activation}"
        config = TrainConfig(
            data_dir=args.data_dir,
            output_dir=str(run_dir),
            hidden_dim=hidden,
            activation=activation,
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
        print(f"\n=== search run {run_id}: lr={lr}, hidden={hidden}, wd={wd}, activation={activation} ===")
        model = MLPClassifier(hidden_dim=hidden, activation=activation, seed=args.seed + run_id)
        history = fit(model, train_data, val_data, config, run_dir)
        best = max(history, key=lambda row: row["val_accuracy"])
        results.append(
            {
                "run_id": run_id,
                "learning_rate": lr,
                "hidden_dim": hidden,
                "weight_decay": wd,
                "activation": activation,
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

