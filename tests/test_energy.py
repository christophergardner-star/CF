from core.energy import Action, EnergyBudget
from core.entropy import EntropyField


def test_consume_respects_cost_and_balance():
    budget = EnergyBudget(capacity=10, level=10, regen_rate=1)
    assert budget.consume(Action.THINK) is True      # cost 1.0
    assert budget.level == 9.0
    assert budget.can_afford(Action.LEARN) is True    # cost 2.0


def test_cannot_overspend():
    budget = EnergyBudget(capacity=10, level=1, regen_rate=1)
    assert budget.consume(Action.LEARN) is False      # needs 2.0
    assert budget.level == 1.0                        # unchanged


def test_regenerate_clamps_to_capacity():
    budget = EnergyBudget(capacity=10, level=9, regen_rate=5)
    budget.regenerate()
    assert budget.level == 10.0
    assert budget.is_high()


def test_is_low_threshold():
    budget = EnergyBudget(capacity=10, level=1)
    assert budget.is_low()


def test_entropy_relaxes_toward_baseline():
    field = EntropyField(level=0.9, baseline=0.3, relaxation=0.5)
    field.relax()
    assert 0.3 < field.level < 0.9


def test_entropy_perturb_saturates():
    field = EntropyField(level=0.5, baseline=0.3)
    for _ in range(100):
        field.perturb(1.0)
    assert field.level <= 1.0
    assert field.is_high
