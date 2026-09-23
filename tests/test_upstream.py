from monitoring.execution_graph import ExecutionGraph
from schemas.events import ExecutionEvent


def ev(task, step, ok):
    return ExecutionEvent(
        task_id=task, agent_id="a1", step_id=step,
        event_type="tool_call", content="x", success=ok,
    )


g = ExecutionGraph()
g.add_event(ev("t1", 1, False))   # upstream failure
g.add_event(ev("t1", 2, True))
g.add_event(ev("t1", 3, True))
g.add_event(ev("t2", 1, True))    # different task, clean

assert g.get_upstream_failed("t1", 3) is True     # ancestor failed
assert g.get_upstream_failed("t1", 2) is True
assert g.get_upstream_failed("t1", 1) is False    # own failure doesn't count
assert g.get_upstream_failed("t2", 1) is False    # other task unaffected
assert g.get_upstream_failed("t9", 1) is False    # unknown node
print("get_upstream_failed: all checks passed")