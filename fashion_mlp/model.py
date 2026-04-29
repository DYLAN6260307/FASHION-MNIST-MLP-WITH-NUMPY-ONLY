from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from .autodiff import Linear, Module, Parameter, collect_parameters, cross_entropy_with_logits, make_activation


class MLPClassifier(Module):
    """MLP classifier with two hidden layers by default."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: Iterable[int] | int | str = (256, 128),
        output_dim: int = 10,
        activations: Iterable[str] | str = "relu,tanh",
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dims = self._parse_hidden_dims(hidden_dims)
        self.output_dim = output_dim
        self.activation_names = self._parse_activations(activations, len(self.hidden_dims))
        rng = np.random.default_rng(seed)
        self.hidden_layers: List[Linear] = []
        self.activation_layers: List[Module] = []
        self.layers: List[Module] = []

        prev_dim = input_dim
        for i, hidden_dim in enumerate(self.hidden_dims, start=1):
            linear = Linear(prev_dim, hidden_dim, rng, f"fc{i}")
            activation = make_activation(self.activation_names[i - 1])
            self.hidden_layers.append(linear)
            self.activation_layers.append(activation)
            self.layers.extend([linear, activation])
            prev_dim = hidden_dim

        self.output_layer = Linear(prev_dim, output_dim, rng, f"fc{len(self.hidden_dims) + 1}")
        self.layers.append(self.output_layer)

    @staticmethod
    def _parse_hidden_dims(hidden_dims: Iterable[int] | int | str) -> List[int]:
        if isinstance(hidden_dims, int):
            return [hidden_dims, hidden_dims]
        if isinstance(hidden_dims, str):
            hidden_dims = [part.strip() for part in hidden_dims.split(",") if part.strip()]
        dims = [int(dim) for dim in hidden_dims]
        if len(dims) != 2:
            raise ValueError("This homework version expects exactly two hidden layers, e.g. 256,128")
        if any(dim <= 0 for dim in dims):
            raise ValueError("Hidden dimensions must be positive integers")
        return dims

    @staticmethod
    def _parse_activations(activations: Iterable[str] | str, count: int) -> List[str]:
        if isinstance(activations, str):
            names = [part.strip().lower() for part in activations.split(",") if part.strip()]
        else:
            names = [str(part).strip().lower() for part in activations]
        if len(names) == 1:
            names = names * count
        if len(names) != count:
            raise ValueError(f"Expected one activation or {count} activations, got {names}")
        return names

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self) -> List[Parameter]:
        return collect_parameters(self.layers)

    def l2_penalty(self) -> float:
        return float(sum(0.5 * np.sum(p.data * p.data) for p in self.parameters() if p.decay))

    def loss_and_backward(self, x: np.ndarray, y: np.ndarray, weight_decay: float = 0.0) -> float:
        logits = self.forward(x)
        loss, grad = cross_entropy_with_logits(logits, y)
        loss += weight_decay * self.l2_penalty()
        self.zero_grad()
        self.backward(grad)
        return loss

    def predict(self, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        preds = []
        for start in range(0, x.shape[0], batch_size):
            logits = self.forward(x[start : start + batch_size])
            preds.append(np.argmax(logits, axis=1))
        return np.concatenate(preds).astype(np.int64)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "input_dim": np.array(self.input_dim),
            "hidden_dims": np.array(self.hidden_dims, dtype=np.int64),
            "output_dim": np.array(self.output_dim),
            "activations": np.array(",".join(self.activation_names)),
        }
        for i, layer in enumerate(self.hidden_layers, start=1):
            arrays[f"fc{i}_weight"] = layer.weight.data
            arrays[f"fc{i}_bias"] = layer.bias.data
        output_index = len(self.hidden_layers) + 1
        arrays[f"fc{output_index}_weight"] = self.output_layer.weight.data
        arrays[f"fc{output_index}_bias"] = self.output_layer.bias.data
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "MLPClassifier":
        data = np.load(path, allow_pickle=True)
        if "hidden_dims" in data:
            hidden_dims = [int(dim) for dim in data["hidden_dims"].tolist()]
            activations = str(data["activations"])
        else:
            raise ValueError(
                "This checkpoint was created by the old one-hidden-layer model. "
                "Please retrain to create a two-hidden-layer checkpoint."
            )
        model = cls(
            input_dim=int(data["input_dim"]),
            hidden_dims=hidden_dims,
            output_dim=int(data["output_dim"]),
            activations=activations,
        )
        for i, layer in enumerate(model.hidden_layers, start=1):
            if f"fc{i}_weight" in data:
                layer.weight.data[...] = data[f"fc{i}_weight"]
                layer.bias.data[...] = data[f"fc{i}_bias"]
        output_index = len(model.hidden_layers) + 1
        if f"fc{output_index}_weight" in data:
            model.output_layer.weight.data[...] = data[f"fc{output_index}_weight"]
            model.output_layer.bias.data[...] = data[f"fc{output_index}_bias"]
        return model

    def summary(self) -> Dict[str, int | str]:
        return {
            "input_dim": self.input_dim,
            "hidden_dims": ",".join(str(dim) for dim in self.hidden_dims),
            "output_dim": self.output_dim,
            "activations": ",".join(self.activation_names),
            "num_parameters": int(sum(p.data.size for p in self.parameters())),
        }

    @property
    def fc1(self) -> Linear:
        return self.hidden_layers[0]
