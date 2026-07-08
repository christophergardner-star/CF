"""The adjudication target must honor the ProbeTarget protocol exactly:
parameter get/set is a lossless round-trip, update_scale scales the whole
step, and strategy state (buffers, anchors, traces) survives restoration
--- the very property the substrate probe exists to detect."""
import numpy as np
import pytest

pytest.importorskip("numpy")

from adjudication.targets import MLPTarget


class TinyTask:
    def __init__(self, cls_a, cls_b, seed):
        rng = np.random.default_rng(seed)
        n = 120
        protos = rng.standard_normal((2, 784)) * 0.5
        x = np.repeat(protos, n // 2, axis=0) \
            + 0.1 * rng.standard_normal((n, 784))
        y = np.repeat([cls_a, cls_b], n // 2)
        order = rng.permutation(n)
        self.x_train, self.y_train = x[order], y[order]
        self.x_test, self.y_test = x[order][:60], y[order][:60]
        self.name = f"{cls_a}v{cls_b}"


def test_params_roundtrip_and_learning():
    target = MLPTarget(hidden=32, epochs=2, seed=0)
    task = TinyTask(0, 1, seed=1)
    flat = target.get_params()
    target.set_params(flat)
    assert np.allclose(target.get_params(), flat)
    target.train(task)
    assert target.evaluate(task) > 0.9


def test_update_scale_scales_the_step():
    task = TinyTask(2, 3, seed=2)
    big = MLPTarget(hidden=32, epochs=1, seed=0)
    small = MLPTarget(hidden=32, epochs=1, seed=0)
    start = big.get_params()
    big.train(task, update_scale=1.0)
    small.train(task, update_scale=0.1)
    moved_big = np.linalg.norm(big.get_params() - start)
    moved_small = np.linalg.norm(small.get_params() - start)
    assert moved_small < 0.5 * moved_big


def test_strategy_state_survives_parameter_restoration():
    target = MLPTarget(method="replay", hidden=32, epochs=1, seed=0)
    task = TinyTask(4, 5, seed=3)
    before = target.get_params()
    target.train(task)
    target.set_params(before)
    assert len(target.buffer_x) == 1      # buffer intact: not a parameter
