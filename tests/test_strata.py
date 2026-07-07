"""Tests for the STRATA prototype: each defense layer, then the whole slice."""
import numpy as np
import pytest

pytest.importorskip("numpy")

from strata.cascade import CascadeChain
from strata.column import LiquidColumn
from strata.demo import CyclicTask, make_interference_tasks, run_condition
from strata.encoder import SparseEncoder
from strata.fastweights import FastWeightMemory
from strata.network import StrataConfig, StrataNetwork


# --------------------------------------------------------------- encoder

def test_encoder_codes_are_sparse_and_deterministic():
    enc = SparseEncoder(d_in=8, dim=96, s=8, seed=1)
    x = np.random.default_rng(0).standard_normal(8)
    z1, z2 = enc.encode(x), enc.encode(x)
    assert np.count_nonzero(z1) <= 8
    assert np.allclose(z1, z2)
    assert np.isclose(np.linalg.norm(z1), 1.0)


def test_encoder_separates_dissimilar_inputs():
    enc = SparseEncoder(d_in=8, dim=96, s=8, seed=1)
    rng = np.random.default_rng(2)
    overlaps = []
    for _ in range(50):
        a, b = rng.standard_normal(8), rng.standard_normal(8)
        overlaps.append(SparseEncoder.overlap(enc.encode(a), enc.encode(b)))
    assert np.mean(overlaps) < 0.3          # near-orthogonal on average


def test_matching_pursuit_explains_away_correlated_atoms():
    """With a dictionary containing the exact input direction, MP inference
    concentrates the code on that atom even when sibling atoms correlate --
    decorrelation by inference, with no weight movement."""
    enc = SparseEncoder(d_in=8, dim=32, s=6, seed=3, mode="mp")
    rng = np.random.default_rng(4)
    proto = rng.standard_normal(8)
    proto /= np.linalg.norm(proto)
    fresh = rng.standard_normal(8)
    fresh -= (fresh @ proto) * proto
    fresh /= np.linalg.norm(fresh)
    enc.projection[0] = proto                        # the exact input atom
    enc.projection[1] = 0.7 * proto + (1 - 0.7**2) ** 0.5 * fresh
    code = enc.encode(proto)
    assert np.count_nonzero(code) <= 6
    assert np.isclose(np.linalg.norm(code), 1.0)
    assert code[0] > 0.9                       # own atom dominates ...
    assert code[1] < 0.35                      # ... sibling explained away


# --------------------------------------------------------------- cascade

def test_cascade_rebounds_after_brief_overwrite():
    """Deep levels pull the surface back toward consolidated knowledge."""

    def run(levels: int) -> float:
        chain = CascadeChain((1,), levels=levels, tau1=8.0)
        for _ in range(400):                      # consolidate value +1
            chain.write(np.array([0.2 * (1.0 - chain.effective[0])]))
            chain.diffuse()
        for _ in range(15):                       # brief opposing pressure
            chain.write(np.array([0.2 * (-1.0 - chain.effective[0])]))
            chain.diffuse()
        after_overwrite = chain.effective[0]
        for _ in range(60):                       # idle: diffusion only
            chain.diffuse()
        return chain.effective[0] - after_overwrite

    assert run(levels=4) > 0.05                   # rebound toward old value
    assert abs(run(levels=1)) < 1e-9              # single level: no memory


# ----------------------------------------------------------- fast weights

def test_fastweights_one_shot_binding_with_low_interference():
    mem = FastWeightMemory(key_dim=96, value_dim=8)
    rng = np.random.default_rng(3)
    k1, k2 = np.zeros(96), np.zeros(96)
    k1[:8] = 1.0 / np.sqrt(8)                     # disjoint sparse keys
    k2[8:16] = 1.0 / np.sqrt(8)
    v1, v2 = rng.standard_normal(8), rng.standard_normal(8)
    mem.write(k1, v1, rate=1.0)
    mem.write(k2, v2, rate=1.0)
    assert np.allclose(mem.read(k1), v1, atol=1e-9)
    assert np.allclose(mem.read(k2), v2, atol=1e-9)


# ---------------------------------------------------------------- column

def test_column_state_is_bounded_by_construction():
    col = LiquidColumn(d_in=8, d_out=8, key_dim=96, n_state=32, seed=4)
    x = 100.0 * np.ones(8)                        # violent input
    bound = np.abs(col.W_B @ x)
    for _ in range(200):
        col.step(x, novelty=0.5)
    assert np.all(np.abs(col.h) <= bound + 1e-9)
    assert np.all(np.isfinite(col.h))


def test_novelty_stretches_the_integration_window():
    """Surprised columns integrate more slowly (they 'dwell')."""
    calm = LiquidColumn(d_in=8, d_out=8, key_dim=96, seed=5)
    surprised = LiquidColumn(d_in=8, d_out=8, key_dim=96, seed=5)
    x = np.ones(8)
    calm.step(x, novelty=0.0)
    surprised.step(x, novelty=1.0)
    # from h = 0, one step moves h by (1 - decay) * W_B x; a longer time
    # constant means a smaller step toward the input
    assert np.linalg.norm(surprised.h) < np.linalg.norm(calm.h)


def test_dictionary_growth_is_backward_transparent():
    """Growth appends masked capacity: existing keys decode identically."""
    enc = SparseEncoder(d_in=24, dim=128, s=8, seed=5, lr=0.15, grow_from=32)
    rng = np.random.default_rng(6)
    x = rng.standard_normal(24)
    before = enc.encode(x)
    grown_at = enc.active
    enc._grow()                                              # force growth
    after = enc.encode(x)
    assert enc.active > grown_at
    assert np.allclose(before, after)      # old codes unchanged exactly


def test_adaptive_claim_band_never_crosses_background_floor():
    from strata.routing import ColumnRouter
    from strata.column import LiquidColumn

    router = ColumnRouter(factory=lambda i: LiquidColumn(24, 24, 128),
                          adaptive=True)
    column = LiquidColumn(24, 24, 128)
    column.claim_mean, column.claim_var = 0.3, 0.5   # collapsed statistics
    router.bg_mean, router.bg_var = 0.4, 0.01
    floor = router.bg_mean + 2.0 * router.bg_var ** 0.5
    assert router.claim_threshold(column) >= floor


def test_stack_runs_and_emits_events():
    from strata.stack import StrataStack
    from strata.demo import make_interference_tasks

    stack = StrataStack(StrataConfig(d_in=24, code_dim=128, seed=0))
    (a, b) = make_interference_tasks(24, 8, 3, task_seeds=[1, 2],
                                     shared_seed=9)
    for cycle in range(3):
        for task in (a, b):
            for t in range(150):
                report = stack.step(task.sample(t))
    assert report["l2_events"] >= 2            # boundaries reached layer 2
    assert np.isfinite(report["error"])


# --------------------------------------------------------------- network

def test_network_spawns_columns_for_orthogonal_regimes():
    cfg = StrataConfig(d_in=24, code_dim=128, seed=0)
    net = StrataNetwork(cfg)
    task_a = CyclicTask(24, 6, seed=11)
    task_b = CyclicTask(24, 6, seed=12)
    for t in range(300):
        net.step(task_a.sample(t))
    for t in range(300):
        net.step(task_b.sample(t))
    assert len(net.router.columns) >= 2           # B earned its own column


def test_retention_full_strata_beats_dense_baseline():
    """The whole point: task A survives learning task B."""
    def schedule():
        a1, b = make_interference_tasks(
            24, n_unique=8, n_shared=3, task_seeds=[21, 22], shared_seed=77
        )
        (a2,) = make_interference_tasks(
            24, n_unique=8, n_shared=3, task_seeds=[21], shared_seed=77
        )
        return [("A", a1, 400), ("B", b, 400), ("A", a2, 400)]
    full = run_condition(StrataConfig(d_in=24, code_dim=128, seed=0), schedule())
    dense = run_condition(
        StrataConfig(
            d_in=24, code_dim=128, seed=0,
            use_routing=False, use_cascade=False,
            use_fast=False, use_sparse=False,
        ),
        schedule(),
    )
    assert full["A revisit"] < dense["A revisit"]
