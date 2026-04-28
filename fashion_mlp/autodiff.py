from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


@dataclass
class Parameter:
    data: np.ndarray
    name: str
    decay: bool = True

    def __post_init__(self) -> None:
        self.grad = np.zeros_like(self.data, dtype=np.float32)

    def zero_grad(self) -> None:
        self.grad.fill(0.0)


class Module:
    def parameters(self) -> List[Parameter]:
        return []

    def zero_grad(self) -> None:
        for param in self.parameters():
            param.zero_grad()


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator, name: str) -> None:
        scale = np.sqrt(2.0 / in_features).astype(np.float32)
        weight = rng.normal(0.0, scale, size=(in_features, out_features)).astype(np.float32)
        bias = np.zeros(out_features, dtype=np.float32)
        self.weight = Parameter(weight, f"{name}.weight", decay=True)
        self.bias = Parameter(bias, f"{name}.bias", decay=False)
        self._input: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        return x @ self.weight.data + self.bias.data

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._input is None:
            raise RuntimeError("Linear.backward called before forward")
        x = self._input
        self.weight.grad += x.T @ grad_output
        self.bias.grad += grad_output.sum(axis=0)
        return grad_output @ self.weight.data.T

    def parameters(self) -> List[Parameter]:
        return [self.weight, self.bias]


class ReLU(Module):
    def __init__(self) -> None:
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return np.maximum(x, 0)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._mask is None:
            raise RuntimeError("ReLU.backward called before forward")
        return grad_output * self._mask


class Sigmoid(Module):
    def __init__(self) -> None:
        self._output: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))
        self._output = out.astype(np.float32)
        return self._output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._output is None:
            raise RuntimeError("Sigmoid.backward called before forward")
        return grad_output * self._output * (1.0 - self._output)


class Tanh(Module):
    def __init__(self) -> None:
        self._output: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._output = np.tanh(x).astype(np.float32)
        return self._output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._output is None:
            raise RuntimeError("Tanh.backward called before forward")
        return grad_output * (1.0 - self._output**2)


def make_activation(name: str) -> Module:
    lowered = name.lower()
    if lowered == "relu":
        return ReLU()
    if lowered == "sigmoid":
        return Sigmoid()
    if lowered == "tanh":
        return Tanh()
    raise ValueError(f"Unsupported activation {name!r}; use relu, sigmoid, or tanh")


def cross_entropy_with_logits(logits: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    n = logits.shape[0]
    loss = -np.log(probs[np.arange(n), targets] + 1e-12).mean()
    grad = probs
    grad[np.arange(n), targets] -= 1.0
    grad /= n
    return float(loss), grad.astype(np.float32)


def collect_parameters(modules: Iterable[Module]) -> List[Parameter]:
    params: List[Parameter] = []
    for module in modules:
        params.extend(module.parameters())
    return params

