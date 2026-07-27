"""Unit test of the priority scheduler's ordering logic (no GPU/model needed).
Bypasses Scheduler.__init__ and drives the real _priority / _pick_waiting methods with fake seqs."""
from types import SimpleNamespace
from nanovllm.engine.scheduler import Scheduler


def make_sched(deadline, max_priority=2):
    s = object.__new__(Scheduler)
    s.priority_deadline = deadline
    s.max_priority = max_priority
    s.waiting = []
    return s


def seq(prio, arrival, started=False):
    return SimpleNamespace(priority=prio, arrival_time=arrival,
                           block_table=[1] if started else [])


def test_static_order():
    """Higher priority served first; FCFS within a tier. deadline off (no aging)."""
    s = make_sched(deadline=0)
    now = 100.0
    s.waiting = [seq(0, 0.0), seq(2, 5.0), seq(1, 1.0), seq(2, 3.0)]
    order = []
    picked = []
    for _ in range(len(s.waiting)):
        p = s._pick_waiting(now, picked)
        picked.append(p); order.append((p.priority, p.arrival_time))
    # P2 first (earliest-arrival P2 = t3 then t5), then P1, then P0
    assert order == [(2, 3.0), (2, 5.0), (1, 1.0), (0, 0.0)], order
    print("test_static_order PASS:", order)


def test_started_first():
    """A mid-chunk (started) low-priority seq outranks a fresh high-priority one."""
    s = make_sched(deadline=0)
    now = 100.0
    started_low = seq(0, 0.0, started=True)
    fresh_high = seq(2, 1.0)
    s.waiting = [fresh_high, started_low]
    assert s._pick_waiting(now, []) is started_low
    print("test_started_first PASS")


def test_aging_promotes():
    """A starving P0 ages up: after >D it beats a fresh P1; after >2D it reaches the top tier."""
    s = make_sched(deadline=4.0, max_priority=2)
    old_p0 = seq(0, 0.0)          # has waited a long time
    fresh_p1 = seq(1, 105.0)      # just arrived at now -> no aging yet
    # at now=105 (waited 105s -> steps=26 -> capped at 2): eff=2 > fresh P1 eff=1
    assert s._priority(old_p0, 105.0) == 2
    assert s._priority(fresh_p1, 105.0) == 1
    s.waiting = [fresh_p1, old_p0]
    assert s._pick_waiting(105.0, []) is old_p0, "aged P0 should be picked over fresh P1"
    # just after one deadline step (waited 5s: steps=1 -> eff=1), ties P1 on priority but older arrival wins
    assert s._priority(old_p0, 5.0) == 1
    print("test_aging_promotes PASS")


def test_aging_cap():
    """Aging never exceeds max_priority."""
    s = make_sched(deadline=1.0, max_priority=2)
    old = seq(1, 0.0)
    assert s._priority(old, 1000.0) == 2   # 1 + 1000 steps, capped at 2
    print("test_aging_cap PASS")


def test_deadline_off():
    """deadline <= 0 disables aging: effective priority == static priority forever."""
    s = make_sched(deadline=0)
    old_p0 = seq(0, 0.0)
    assert s._priority(old_p0, 1e9) == 0
    print("test_deadline_off PASS")


if __name__ == "__main__":
    test_static_order()
    test_started_first()
    test_aging_promotes()
    test_aging_cap()
    test_deadline_off()
    print("\nALL PRIORITY-ORDER TESTS PASS")
