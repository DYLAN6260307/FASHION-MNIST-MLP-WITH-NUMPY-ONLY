from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fashion_mlp.data import CLASS_NAMES, load_fashion_mnist
from fashion_mlp.metrics import confusion_matrix, per_class_accuracy
from fashion_mlp.model import MLPClassifier
from fashion_mlp.visualization import create_standard_visualizations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained NumPy MLP.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-dir", default="data/fashion-mnist")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--validation-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-test", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = MLPClassifier.load(args.weights)
    _, _, test_data = load_fashion_mnist(
        data_dir=args.data_dir,
        validation_size=args.validation_size,
        seed=args.seed,
        max_test=args.max_test,
    )
    test_x, test_y = test_data
    pred = model.predict(test_x, batch_size=args.batch_size)
    cm = confusion_matrix(test_y, pred, num_classes=len(CLASS_NAMES))
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.weights).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "test_accuracy": float((pred == test_y).mean()),
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": per_class_accuracy(cm).tolist(),
        "class_names": CLASS_NAMES,
        "best_model": str(args.weights),
    }
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    history_path = output_dir / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    if history:
        create_standard_visualizations(history, model, cm, test_x, test_y, pred, output_dir)
    print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
    print("Confusion matrix:")
    print(cm)


if __name__ == "__main__":
    main()

