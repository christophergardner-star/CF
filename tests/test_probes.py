"""The probe instrument must discriminate in both directions:

* ordinary weight-overwrite forgetting -> attributed (no false (c) alarm);
* forgetting carried by a non-parameter buffer (a head-calibration-drift
  analog, one of the registered candidate modes) -> all three exclusions
  pass and the (c) criterion fires.

If either test fails, the instrument cannot adjudicate the trichotomy.
"""
import numpy as np
import pytest

pytest.importorskip("numpy")

from probes import diagnose, linear_cka


def make_task(proto_dims, labels, d=20, n=40, noise=0.05, seed=0,
              noise_dims=None):
    """``noise_dims`` restricts noise to a subspace: tasks that claim to be
    orthogonal must actually be orthogonal, or a shared trained head will
    genuinely (and correctly) couple their representations."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for i, dim in enumerate(proto_dims):
        for _ in range(n // len(proto_dims)):
            x = np.zeros(d)
            scope = noise_dims if noise_dims is not None else slice(None)
            x[scope] = noise * rng.standard_normal(len(np.zeros(d)[scope]))
            x[dim] += 1.0
            xs.append(x)
            ys.append(labels[i])
    return np.array(xs), np.array(ys)


class LinearTarget:
    """Plain delta-rule classifier: forgetting is pure weight overwrite."""

    n_classes = 8

    def __init__(self, d=20, seed=0):
        rng = np.random.default_rng(seed)
        self.W = 0.01 * rng.standard_normal((self.n_classes, d))

    def get_params(self):
        return self.W.ravel().copy()

    def set_params(self, flat):
        self.W = flat.reshape(self.W.shape).copy()

    def _logits(self, X):
        return X @ self.W.T

    def evaluate(self, task):
        X, y = task
        return float(np.mean(self._logits(X).argmax(axis=1) == y))

    def train(self, task, update_scale=1.0):
        X, y = task
        for _ in range(6):
            logits = self._logits(X)
            targets = np.eye(self.n_classes)[y]
            self.W += 0.3 * update_scale * (targets - logits).T @ X / len(X)

    def activations(self, inputs):
        return {"penultimate": inputs, "linear": self._logits(inputs)}


class BufferTarget(LinearTarget):
    """Forgetting carried by a *buffer*: a class-bias vector tracking the
    label distribution (as batch-norm statistics track activations),
    updated during any training regardless of update scale, and not part
    of the parameter vector.  Old-task loss survives full parameter
    restoration and 10x-slowed training: the registered (c) shape."""

    def __init__(self, d=20, seed=0):
        super().__init__(d, seed)
        # zero init: untrained head rows must contribute exactly nothing,
        # or their numerically-negligible random responses register as
        # covariance alignment (the measure is scale-invariant) and the
        # rig stops being the clean orthogonal case it claims to be
        self.W = np.zeros_like(self.W)
        self.bias = np.zeros(self.n_classes)

    def _logits(self, X):
        return X @ self.W.T + 3.0 * self.bias

    def train(self, task, update_scale=1.0):
        X, y = task
        for _ in range(6):
            logits = X @ self.W.T          # weights learn without the bias
            targets = np.eye(self.n_classes)[y]
            self.W += 0.3 * update_scale * (targets - logits).T @ X / len(X)
            label_mass = np.eye(self.n_classes)[y].mean(axis=0)
            self.bias += 0.5 * (label_mass - self.bias)   # never lr-scaled


def test_cka_bounds():
    # linear CKA between independent features has a chance floor of
    # roughly sqrt(p*q)/n -- sample counts must dominate feature dims for
    # an absolute threshold to be meaningful (n=512, d=12: floor ~0.023)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((512, 12))
    assert linear_cka(x, x) > 0.999
    assert linear_cka(x, rng.standard_normal((512, 12))) < 0.1


def test_overwrite_forgetting_is_attributed_not_outside():
    # A and B share inputs; B's labels conflict -> classic overwrite
    task_a = make_task([0, 1, 2, 3], [0, 1, 2, 3], seed=2)
    task_b = make_task([0, 1, 2, 3], [1, 2, 3, 0], seed=3)
    target = LinearTarget()
    target.train(task_a)
    report = diagnose(target, task_a, task_b,
                      heldout_a=make_task([0, 1, 2, 3], [0, 1, 2, 3],
                                          seed=4)[0],
                      heldout_b=make_task([0, 1, 2, 3], [1, 2, 3, 0],
                                          seed=5)[0])
    assert report["event"]
    assert not report["outside_trichotomy"]
    assert "substrate" in report["attributed_to"]      # restoration recovers
    assert "basis" in report["attributed_to"]          # shared inputs: CKA high


def test_buffer_forgetting_fires_the_c_criterion():
    # genuinely disjoint input subspaces (noise confined to each task's own
    # dims) and disjoint class ids; the forgetting lives in the bias buffer
    lo, hi = slice(0, 10), slice(10, 20)
    task_a = make_task([0, 1, 2, 3], [0, 1, 2, 3], seed=6, noise_dims=lo)
    task_b = make_task([10, 11, 12, 13], [4, 5, 6, 7], seed=7, noise_dims=hi)
    target = BufferTarget()
    target.train(task_a)
    assert target.evaluate(task_a) > 0.9
    report = diagnose(target, task_a, task_b,
                      heldout_a=make_task([0, 1, 2, 3], [0, 1, 2, 3], n=400,
                                          seed=8, noise_dims=lo)[0],
                      heldout_b=make_task([10, 11, 12, 13], [4, 5, 6, 7],
                                          n=400, seed=9, noise_dims=hi)[0])
    assert report["event"]
    assert report["substrate"]["passes"]       # restoring W does not help
    assert report["timescale"]["passes"]       # slow updates: same bias drift
    assert report["basis"]["passes"]           # orthogonal task subspaces
    assert report["outside_trichotomy"]        # (c) fires -- not vacuous
