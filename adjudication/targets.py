"""A NumPy MLP implementing the ProbeTarget protocol, with four standard
continual-learning strategies as training methods:

    naive    plain SGD, sequential (the catastrophic-forgetting control)
    ewc      Elastic Weight Consolidation (diagonal Fisher penalty)
    si       Synaptic Intelligence (online path-integral importance)
    replay   reservoir of past examples mixed into every batch

All strategies live inside ``train`` so the probes' interventions apply to
the whole update (penalty gradients included): ``update_scale`` multiplies
the entire parameter step, exactly as the registered timescale
intervention requires.  Non-parameter state (SI traces, EWC anchors,
replay buffers) survives parameter restoration by construction --- which
is precisely what the substrate probe exists to detect.
"""
from __future__ import annotations

import numpy as np

N_CLASSES = 10


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLPTarget:
    """784 -> hidden -> 10 shared-head classifier (class-incremental)."""

    def __init__(self, method: str = "naive", hidden: int = 256,
                 lr: float = 0.1, epochs: int = 3, batch: int = 64,
                 ewc_lambda: float = 100.0, si_c: float = 0.5,
                 replay_per_task: int = 200, grad_clip: float = 10.0,
                 seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.method = method
        self.lr, self.epochs, self.batch = lr, epochs, batch
        self.ewc_lambda, self.si_c = ewc_lambda, si_c
        self.replay_per_task = replay_per_task
        self.grad_clip = grad_clip
        self.rng = rng
        self.p = {
            "W1": rng.standard_normal((784, hidden)) * (2.0 / 784) ** 0.5,
            "b1": np.zeros(hidden),
            "W2": rng.standard_normal((hidden, N_CLASSES)) * (2.0 / hidden) ** 0.5,
            "b2": np.zeros(N_CLASSES),
        }
        self._keys = ("W1", "b1", "W2", "b2")
        # strategy state (deliberately NOT parameters)
        self.anchors: list[tuple[dict, dict]] = []      # EWC (theta*, fisher)
        self.omega = {k: np.zeros_like(v) for k, v in self.p.items()}  # SI
        self.si_w = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.si_start = None
        self.buffer_x: list[np.ndarray] = []
        self.buffer_y: list[np.ndarray] = []

    # ---------------------------------------------------- ProbeTarget API
    def get_params(self) -> np.ndarray:
        return np.concatenate([self.p[k].ravel() for k in self._keys])

    def set_params(self, flat: np.ndarray) -> None:
        i = 0
        for k in self._keys:
            n = self.p[k].size
            self.p[k] = flat[i:i + n].reshape(self.p[k].shape).copy()
            i += n

    def evaluate(self, task) -> float:
        logits = self._forward(task.x_test)[1]
        return float(np.mean(logits.argmax(axis=1) == task.y_test))

    def activations(self, inputs: np.ndarray) -> dict[str, np.ndarray]:
        hidden, logits = self._forward(inputs)
        return {"fc1": hidden, "fc2": logits}

    def train(self, task, update_scale: float = 1.0) -> None:
        x, y = task.x_train, task.y_train
        if self.method == "si":
            self.si_start = {k: v.copy() for k, v in self.p.items()}
            self.si_w = {k: np.zeros_like(v) for k, v in self.p.items()}
        for _ in range(self.epochs):
            order = self.rng.permutation(len(x))
            for start in range(0, len(x), self.batch):
                idx = order[start:start + self.batch]
                bx, by = x[idx], y[idx]
                if self.method == "replay" and self.buffer_x:
                    rx = np.concatenate(self.buffer_x)
                    ry = np.concatenate(self.buffer_y)
                    pick = self.rng.integers(0, len(rx), len(idx))
                    bx = np.concatenate([bx, rx[pick]])
                    by = np.concatenate([by, ry[pick]])
                grads, task_grads = self._grads(bx, by)
                # global-norm clipping: keeps every method (EWC's penalty
                # gradients especially) numerically sane; a diverged model
                # produces fake "forgetting" events that contaminate the
                # diagnosis
                norm = np.sqrt(sum(float((g ** 2).sum())
                                   for g in grads.values()))
                if norm > self.grad_clip:
                    grads = {k: g * (self.grad_clip / norm)
                             for k, g in grads.items()}
                for k in self._keys:
                    step = -self.lr * update_scale * grads[k]
                    if self.method == "si":
                        self.si_w[k] += -task_grads[k] * step
                    self.p[k] += step
        self._consolidate(task)

    # -------------------------------------------------------- internals
    def _forward(self, x: np.ndarray):
        hidden = np.maximum(x @ self.p["W1"] + self.p["b1"], 0.0)
        return hidden, hidden @ self.p["W2"] + self.p["b2"]

    def _grads(self, x: np.ndarray, y: np.ndarray):
        hidden, logits = self._forward(x)
        d_logits = (_softmax(logits) - np.eye(N_CLASSES)[y]) / len(x)
        d_hidden = (d_logits @ self.p["W2"].T) * (hidden > 0)
        task_grads = {
            "W1": x.T @ d_hidden, "b1": d_hidden.sum(axis=0),
            "W2": hidden.T @ d_logits, "b2": d_logits.sum(axis=0),
        }
        grads = {k: v.copy() for k, v in task_grads.items()}
        if self.method == "ewc":
            for theta_star, fisher in self.anchors:
                for k in self._keys:
                    grads[k] += self.ewc_lambda * fisher[k] \
                        * (self.p[k] - theta_star[k])
        elif self.method == "si":
            for k in self._keys:
                if self.si_start is not None:
                    grads[k] += self.si_c * self.omega[k] \
                        * (self.p[k] - self.si_start[k])
        return grads, task_grads

    def _consolidate(self, task) -> None:
        if self.method == "ewc":
            fisher = self._diag_fisher(task)
            self.anchors.append(
                ({k: v.copy() for k, v in self.p.items()}, fisher))
        elif self.method == "si" and self.si_start is not None:
            for k in self._keys:
                delta = self.p[k] - self.si_start[k]
                self.omega[k] += np.maximum(self.si_w[k], 0.0) \
                    / (delta ** 2 + 0.1)
        elif self.method == "replay":
            pick = self.rng.permutation(len(task.x_train))[
                : self.replay_per_task]
            self.buffer_x.append(task.x_train[pick].copy())
            self.buffer_y.append(task.y_train[pick].copy())

    def _diag_fisher(self, task, n_samples: int = 300) -> dict:
        fisher = {k: np.zeros_like(v) for k, v in self.p.items()}
        pick = self.rng.permutation(len(task.x_train))[:n_samples]
        for i in pick:
            x = task.x_train[i:i + 1]
            y = task.y_train[i:i + 1]
            _, task_grads = self._grads(x, y)
            for k in self._keys:
                fisher[k] += task_grads[k] ** 2
        for k in fisher:
            fisher[k] /= max(len(pick), 1)
        return fisher
