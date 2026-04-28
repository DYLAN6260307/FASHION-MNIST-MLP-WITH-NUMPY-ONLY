from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .data import CLASS_NAMES, unstandardize_flat_images
from .model import MLPClassifier


def _font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _nice_color(index: int) -> tuple[int, int, int]:
    colors = [
        (36, 94, 179),
        (214, 90, 49),
        (31, 135, 83),
        (142, 73, 173),
        (190, 148, 38),
    ]
    return colors[index % len(colors)]


def draw_curve(
    history: List[Dict[str, float]],
    series: Dict[str, str],
    path: str | Path,
    title: str,
    y_label: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 900, 560
    left, right, top, bottom = 82, 42, 52, 78
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = _font(16)
    title_font = _font(22)

    xs = np.array([row["epoch"] for row in history], dtype=np.float64)
    values = [np.array([row[key] for row in history], dtype=np.float64) for key in series]
    y_min = min(float(v.min()) for v in values)
    y_max = max(float(v.max()) for v in values)
    if abs(y_max - y_min) < 1e-9:
        y_max = y_min + 1.0
    pad = 0.08 * (y_max - y_min)
    y_min -= pad
    y_max += pad

    plot_w = width - left - right
    plot_h = height - top - bottom
    draw.rectangle((left, top, width - right, height - bottom), outline=(30, 30, 30), width=2)
    draw.text((left, 16), title, fill=(20, 20, 20), font=title_font)
    draw.text((16, top + plot_h // 2), y_label, fill=(20, 20, 20), font=font)
    draw.text((left + plot_w // 2 - 30, height - 42), "Epoch", fill=(20, 20, 20), font=font)

    for tick in range(6):
        frac = tick / 5
        y = top + plot_h - frac * plot_h
        value = y_min + frac * (y_max - y_min)
        draw.line((left, y, width - right, y), fill=(225, 225, 225))
        draw.text((12, y - 8), f"{value:.2f}", fill=(80, 80, 80), font=_font(12))

    x_min, x_max = float(xs.min()), float(xs.max())
    if x_max == x_min:
        x_max = x_min + 1

    for i, (key, label) in enumerate(series.items()):
        ys = np.array([row[key] for row in history], dtype=np.float64)
        points = []
        for x, y in zip(xs, ys):
            px = left + (x - x_min) / (x_max - x_min) * plot_w
            py = top + plot_h - (y - y_min) / (y_max - y_min) * plot_h
            points.append((px, py))
        color = _nice_color(i)
        if len(points) > 1:
            draw.line(points, fill=color, width=4)
        for px, py in points:
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color)
        legend_x = width - right - 210
        legend_y = top + 18 + i * 28
        draw.line((legend_x, legend_y + 8, legend_x + 30, legend_y + 8), fill=color, width=4)
        draw.text((legend_x + 38, legend_y), label, fill=(40, 40, 40), font=font)

    img.save(path)


def visualize_first_layer_weights(model: MLPClassifier, path: str | Path, max_units: int = 64) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = model.fc1.weight.data.T[:max_units]
    n = weights.shape[0]
    cols = 8
    rows = int(np.ceil(n / cols))
    cell = 58
    margin = 8
    img = Image.new("RGB", (cols * cell + margin, rows * cell + margin), "white")
    for i, w in enumerate(weights):
        patch = w.reshape(28, 28)
        patch = patch - patch.min()
        patch = patch / (patch.max() + 1e-8)
        tile = Image.fromarray((patch * 255).astype(np.uint8), mode="L").resize((50, 50), Image.Resampling.NEAREST)
        x = margin + (i % cols) * cell
        y = margin + (i // cols) * cell
        img.paste(Image.merge("RGB", (tile, tile, tile)), (x, y))
    img.save(path)


def plot_confusion_matrix(matrix: np.ndarray, labels: Iterable[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(labels)
    n = matrix.shape[0]
    cell = 62
    left, top = 150, 64
    width = left + n * cell + 30
    height = top + n * cell + 120
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = _font(12)
    title_font = _font(22)
    draw.text((left, 18), "Confusion Matrix", fill=(20, 20, 20), font=title_font)
    max_value = max(int(matrix.max()), 1)
    for r in range(n):
        draw.text((18, top + r * cell + 22), labels[r][:16], fill=(35, 35, 35), font=font)
        draw.text((left + r * cell + 10, top + n * cell + 8), labels[r][:8], fill=(35, 35, 35), font=font)
        for c in range(n):
            value = int(matrix[r, c])
            shade = int(255 - 220 * value / max_value)
            color = (shade, shade + min(25, 255 - shade), 255)
            x0 = left + c * cell
            y0 = top + r * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=color, outline=(245, 245, 245))
            text = str(value)
            text_box = draw.textbbox((0, 0), text, font=font)
            tx = x0 + (cell - (text_box[2] - text_box[0])) / 2
            ty = y0 + (cell - (text_box[3] - text_box[1])) / 2
            draw.text((tx, ty), text, fill=(20, 20, 20), font=font)
    draw.text((left + n * cell // 2 - 45, height - 38), "Predicted label", fill=(20, 20, 20), font=font)
    draw.text((18, top - 28), "True label", fill=(20, 20, 20), font=font)
    img.save(path)


def save_error_examples(
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    path: str | Path,
    max_examples: int = 12,
) -> List[Dict[str, object]]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrong = np.where(y_true != y_pred)[0][:max_examples]
    images = unstandardize_flat_images(x[wrong])
    cols = 4
    rows = int(np.ceil(max(1, len(wrong)) / cols))
    cell_w, cell_h = 150, 150
    img = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(img)
    font = _font(12)
    examples: List[Dict[str, object]] = []
    for i, idx in enumerate(wrong):
        row, col = divmod(i, cols)
        tile = Image.fromarray(images[i], mode="L").resize((84, 84), Image.Resampling.NEAREST)
        x0 = col * cell_w + 33
        y0 = row * cell_h + 8
        img.paste(Image.merge("RGB", (tile, tile, tile)), (x0, y0))
        true_label = labels[int(y_true[idx])]
        pred_label = labels[int(y_pred[idx])]
        draw.text((col * cell_w + 8, row * cell_h + 98), f"T: {true_label}", fill=(30, 30, 30), font=font)
        draw.text((col * cell_w + 8, row * cell_h + 116), f"P: {pred_label}", fill=(170, 40, 40), font=font)
        examples.append(
            {
                "index": int(idx),
                "true": int(y_true[idx]),
                "pred": int(y_pred[idx]),
                "true_name": true_label,
                "pred_name": pred_label,
            }
        )
    img.save(path)
    return examples


def create_standard_visualizations(
    history: List[Dict[str, float]],
    model: MLPClassifier,
    matrix: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    test_pred: np.ndarray,
    output_dir: str | Path,
) -> Dict[str, str]:
    out = Path(output_dir)
    assets = out / "figures"
    assets.mkdir(parents=True, exist_ok=True)
    loss_path = assets / "loss_curve.png"
    acc_path = assets / "accuracy_curve.png"
    weights_path = assets / "first_layer_weights.png"
    cm_path = assets / "confusion_matrix.png"
    err_path = assets / "error_examples.png"
    draw_curve(
        history,
        {"train_loss": "Train Loss", "val_loss": "Validation Loss"},
        loss_path,
        "Training and Validation Loss",
        "Cross-Entropy",
    )
    draw_curve(history, {"val_accuracy": "Validation Accuracy"}, acc_path, "Validation Accuracy", "Accuracy")
    visualize_first_layer_weights(model, weights_path)
    plot_confusion_matrix(matrix, CLASS_NAMES, cm_path)
    save_error_examples(test_x, test_y, test_pred, CLASS_NAMES, err_path)
    return {
        "loss_curve": str(loss_path),
        "accuracy_curve": str(acc_path),
        "first_layer_weights": str(weights_path),
        "confusion_matrix": str(cm_path),
        "error_examples": str(err_path),
    }

