from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from .autodiff import Linear, Module, Parameter, collect_parameters, cross_entropy_with_logits, make_activation


class MLPClassifier(Module):
    """A three-layer MLP counted as input-hidden-output."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 128,
        output_dim: int = 10,
        activation: str = "relu",
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation_name = activation.lower()
        rng = np.random.default_rng(seed)
        self.fc1 = Linear(input_dim, hidden_dim, rng, "fc1")
        self.activation = make_activation(self.activation_name)
        self.fc2 = Linear(hidden_dim, output_dim, rng, "fc2")
        self.layers: List[Module] = [self.fc1, self.activation, self.fc2]

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
        np.savez_compressed(
            path,
            input_dim=np.array(self.input_dim),
            hidden_dim=np.array(self.hidden_dim),
            output_dim=np.array(self.output_dim),
            activation=np.array(self.activation_name),
            fc1_weight=self.fc1.weight.data,
            fc1_bias=self.fc1.bias.data,
            fc2_weight=self.fc2.weight.data,
            fc2_bias=self.fc2.bias.data,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MLPClassifier":
        data = np.load(path, allow_pickle=True)
        activation = str(data["activation"])
        model = cls(
            input_dim=int(data["input_dim"]),
            hidden_dim=int(data["hidden_dim"]),
            output_dim=int(data["output_dim"]),
            activation=activation,
        )
        model.fc1.weight.data[...] = data["fc1_weight"]
        model.fc1.bias.data[...] = data["fc1_bias"]
        model.fc2.weight.data[...] = data["fc2_weight"]
        model.fc2.bias.data[...] = data["fc2_bias"]
        return model

    def summary(self) -> Dict[str, int | str]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "activation": self.activation_name,
            "num_parameters": int(sum(p.data.size for p in self.parameters())),
        }

