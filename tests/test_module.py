from core.energy import EnergyBudget
from core.entropy import EntropyField
from core.events import Event, EventBus, ObservationEvent, ThoughtEvent
from core.lifecycle import LifecycleState
from core.module import Module
from core.scheduler import Scheduler


class Echo(Module):
    """Minimal module: turns each observation into a thought."""

    subscriptions = (ObservationEvent,)

    def observe(self):
        return self.drain_inbox()

    def think(self, observations):
        return [ThoughtEvent(source=self.name, payload={"about": e.get("token")})
                for e in observations]

    def learn(self, thoughts):
        self.confidence = min(1.0, self.confidence + 0.1 * len(thoughts))


def make_module(bus, name="echo", **kw):
    return Echo(name, bus, energy=EnergyBudget(50, 50, 3),
                entropy=EntropyField(), **kw)


def test_module_does_not_receive_its_own_events():
    bus = EventBus()
    mod = make_module(bus)
    bus.publish(ObservationEvent(source=mod.name, payload={"token": "x"}))
    bus.pump()
    assert not mod.has_work()  # own event ignored


def test_step_consumes_energy_and_emits():
    bus = EventBus()
    mod = make_module(bus)
    bus.publish(ObservationEvent(source="world", payload={"token": "x"}))
    bus.pump()
    before = mod.energy.level
    mod.step(tick=1)
    assert mod.energy.level < before
    assert mod.lifecycle.state is LifecycleState.ACTIVE
    produced = bus.pump()
    assert any(isinstance(e, ThoughtEvent) for e in produced)


def test_receiving_events_strengthens_connection_weights():
    bus = EventBus()
    mod = make_module(bus)
    bus.publish(ObservationEvent(source="world", payload={"token": "x"}))
    bus.pump()
    assert mod.connections.get("world", 0) > 0


def test_scheduler_runs_module_with_work_and_sleeps_idle_one():
    bus = EventBus()
    mod = make_module(bus, idle_sleep_ticks=1)
    mod.wake(0)
    scheduler = Scheduler(sleep_after_idle=1, wake_threshold=0.0)
    bus.publish(ObservationEvent(source="world", payload={"token": "x"}))
    bus.pump()
    decision = scheduler.plan([mod], EnergyBudget(100, 100), tick=1, run_cost=1.0)
    assert mod in decision.to_run

    mod.drain_inbox()  # no more work
    decision = scheduler.plan([mod], EnergyBudget(100, 100), tick=5, run_cost=1.0)
    assert mod in decision.to_sleep


def test_detach_unsubscribes_from_bus():
    bus = EventBus()
    mod = make_module(bus)
    mod.detach()
    bus.publish(ObservationEvent(source="world", payload={"token": "x"}))
    bus.pump()
    assert not mod.has_work()
