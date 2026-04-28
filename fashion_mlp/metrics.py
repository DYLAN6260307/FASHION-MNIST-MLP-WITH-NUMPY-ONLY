from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        matrix[int(true), int(pred)] += 1
    return matrix


def per_class_accuracy(matrix: np.ndarray) -> np.ndarray:
    totals = matrix.sum(axis=1)
    return np.divide(np.diag(matrix), totals, out=np.zeros_like(totals, dtype=np.float64), where=totals != 0)

