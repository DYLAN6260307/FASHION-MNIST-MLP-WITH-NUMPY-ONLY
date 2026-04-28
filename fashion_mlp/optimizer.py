from __future__ import annotations

from typing import Iterable

from .autodiff import Parameter


class SGD:
    def __init__(self, parameters: Iterable[Parameter], lr: float, weight_decay: float = 0.0) -> None:
        self.parameters = list(parameters)
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self) -> None:
        for param in self.parameters:
            grad = param.grad
            if self.weight_decay > 0 and param.decay:
                grad = grad + self.weight_decay * param.data
            param.data -= self.lr * grad

    def zero_grad(self) -> None:
        for param in self.parameters:
            param.zero_grad()

    def decay_lr(self, factor: float) -> None:
        self.lr *= factor

