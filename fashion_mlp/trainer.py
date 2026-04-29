from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .autodiff import cross_entropy_with_logits
from .data import load_fashion_mnist
from .metrics import accuracy
from .model import MLPClassifier
from .optimizer import SGD


@dataclass
class TrainConfig:
    data_dir: str = "data/fashion-mnist"
    output_dir: str = "outputs/run"
    hidden_dims: str = "256,128"
    activations: str = "relu,tanh"
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 0.05
    lr_decay: float = 0.95
    weight_decay: float = 1e-4
    validation_size: int = 10_000
    seed: int = 42
    max_train: int | None = None
    max_val: int | None = None
    max_test: int | None = None


def evaluate_loss_accuracy(
    model: MLPClassifier,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = 1024,
) -> Tuple[float, float]:
    losses: List[float] = []
    preds: List[np.ndarray] = []
    counts: List[int] = []
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size]
        yb = y[start : start + batch_size]
        logits = model.forward(xb)
        loss, _ = cross_entropy_with_logits(logits.copy(), yb)
        losses.append(loss)
        counts.append(xb.shape[0])
        preds.append(np.argmax(logits, axis=1))
    weighted_loss = float(np.average(losses, weights=counts))
    return weighted_loss, accuracy(y, np.concatenate(preds))


def fit(
    model: MLPClassifier,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    config: TrainConfig,
    output_dir: str | Path,
) -> List[Dict[str, float]]:
    x_train, y_train = train_data
    x_val, y_val = val_data
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    best_model_path = out_path / "best_model.npz"
    optimizer = SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    rng = np.random.default_rng(config.seed)

    history: List[Dict[str, float]] = []
    best_val_acc = -1.0
    n_train = x_train.shape[0]

    for epoch in range(1, config.epochs + 1):
        started = time.time()
        order = rng.permutation(n_train)
        batch_losses: List[float] = []
        batch_counts: List[int] = []
        for start in range(0, n_train, config.batch_size):
            idx = order[start : start + config.batch_size]
            xb, yb = x_train[idx], y_train[idx]
            loss = model.loss_and_backward(xb, yb, weight_decay=config.weight_decay)
            optimizer.step()
            batch_losses.append(loss)
            batch_counts.append(xb.shape[0])

        train_loss = float(np.average(batch_losses, weights=batch_counts))
        train_eval_loss, train_acc = evaluate_loss_accuracy(model, x_train, y_train, config.batch_size)
        val_loss, val_acc = evaluate_loss_accuracy(model, x_val, y_val, config.batch_size)
        row = {
            "epoch": float(epoch),
            "lr": float(optimizer.lr),
            "train_loss": train_loss,
            "train_eval_loss": train_eval_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "seconds": float(time.time() - started),
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{config.epochs} "
            f"lr={optimizer.lr:.5f} train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(best_model_path)
        optimizer.decay_lr(config.lr_decay)

    save_history(history, out_path)
    with (out_path / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)
    return history


def save_history(history: List[Dict[str, float]], output_dir: str | Path) -> None:
    out_path = Path(output_dir)
    with (out_path / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    if history:
        with (out_path / "history.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)


def run_training(config: TrainConfig) -> Dict[str, object]:
    train_data, val_data, test_data = load_fashion_mnist(
        data_dir=config.data_dir,
        validation_size=config.validation_size,
        seed=config.seed,
        max_train=config.max_train,
        max_val=config.max_val,
        max_test=config.max_test,
    )
    hidden_dims = [int(dim.strip()) for dim in config.hidden_dims.split(",") if dim.strip()]
    model = MLPClassifier(hidden_dims=hidden_dims, activations=config.activations, seed=config.seed)
    history = fit(model, train_data, val_data, config, config.output_dir)
    best_model = MLPClassifier.load(Path(config.output_dir) / "best_model.npz")
    return {
        "history": history,
        "model": best_model,
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "config": config,
    }
