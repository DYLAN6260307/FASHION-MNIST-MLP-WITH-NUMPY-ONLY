from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fashion_mlp.data import CLASS_NAMES
from fashion_mlp.metrics import confusion_matrix, per_class_accuracy
from fashion_mlp.reporting import build_report
from fashion_mlp.trainer import TrainConfig, run_training
from fashion_mlp.visualization import create_standard_visualizations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NumPy MLP on Fashion-MNIST.")
    parser.add_argument("--data-dir", default="data/fashion-mnist")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--activation", choices=["relu", "sigmoid", "tanh"], default="relu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--lr-decay", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--github-url", default="TODO: replace with Public GitHub Repo URL")
    parser.add_argument("--weights-url", default="TODO: replace with Google Drive model weights URL")
    parser.add_argument("--skip-report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or f"outputs/run_{timestamp}"
    config = TrainConfig(
        data_dir=args.data_dir,
        output_dir=output_dir,
        hidden_dim=args.hidden_dim,
        activation=args.activation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_decay=args.lr_decay,
        weight_decay=args.weight_decay,
        validation_size=args.validation_size,
        seed=args.seed,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
    )
    result = run_training(config)
    model = result["model"]
    test_x, test_y = result["test_data"]
    test_pred = model.predict(test_x, batch_size=args.batch_size)
    cm = confusion_matrix(test_y, test_pred, num_classes=len(CLASS_NAMES))
    test_accuracy = float((test_pred == test_y).mean())
    metrics = {
        "test_accuracy": test_accuracy,
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": per_class_accuracy(cm).tolist(),
        "class_names": CLASS_NAMES,
        "best_model": str(Path(output_dir) / "best_model.npz"),
    }
    with (Path(output_dir) / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    create_standard_visualizations(result["history"], model, cm, test_x, test_y, test_pred, output_dir)
    if not args.skip_report:
        report_path = build_report(output_dir, github_url=args.github_url, weights_url=args.weights_url)
        print(f"Report written to {report_path}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("Confusion matrix:")
    print(cm)


if __name__ == "__main__":
    main()

