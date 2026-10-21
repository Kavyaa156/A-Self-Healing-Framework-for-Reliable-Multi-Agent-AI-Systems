
"""
AgentHeal integration entry point.

Runs one clean scenario and then tests F1, F2, F3 and F4.
"""

import json
import uuid
from typing import List

from schemas.events import TaskSpec, RawEvent
from agents.workflow import MultiAgentWorkflow
from tests.fault_injection import run_task_with_fault

try:
    from reliability.seed_validation_store import seed_store
    HAS_VALIDATION_STORE_DEPS = True
except ImportError:
    HAS_VALIDATION_STORE_DEPS = False


def save_events_to_jsonl(
    events: List[RawEvent],
    filepath: str = "telemetry_events.jsonl",
):
    """Append raw telemetry events to a JSONL file."""
    with open(filepath, "a", encoding="utf-8") as file:
        for event in events:
            file.write(json.dumps(event.model_dump()) + "\n")

    print(f"Saved {len(events)} events to {filepath}")


def print_result(label: str, result: dict):
    """Print reliability scores, detection status and diagnosis."""
    score = result["score"]

    print(f"\n--- {label} ---")
    print(
        f"C={score.C:.3f}  "
        f"S={score.S:.3f}  "
        f"E={score.E:.3f}  "
        f"R={score.R:.3f}  "
        f"(threshold={score.threshold})"
    )
    print(f"Detected as failure: {result['detected']}")

    diagnosis = result["diagnosis"]

    if diagnosis:
        print(f"Diagnosed fault: {diagnosis.failure_type}")
        print(f"Root cause agent: {diagnosis.root_cause_agent}")
        print(f"Description: {diagnosis.description}")
    else:
        print("Diagnosed fault: None")


def create_task(label: str) -> TaskSpec:
    """Create a separate task for each scenario."""
    return TaskSpec(
        task_id=f"task_{label}_{uuid.uuid4().hex[:8]}",
        description=(
            "Audit access control policies for unauthorized "
            "privilege escalation risks"
        ),
        initial_input={
            "target_system": "Authentication Gateway",
            "log_sample_id": "LOG_2026_08_28",
        },
    )


def main():
    print("=" * 62)
    print(" AgentHeal - Multi-Agent Failure Detection and Self-Healing ")
    print("=" * 62)

    workflow = MultiAgentWorkflow(
        model_name="openai/gpt-oss-20b",
        temperature=0.0,
    )

    # F1 and F3 emphasize semantic accuracy and consistency.
    # F2 and F4 emphasize execution failures.
    fault_weights = {
        "F1": (0.4, 0.4, 0.2),
        "F2": (0.2, 0.3, 0.5),
        "F3": (0.4, 0.4, 0.2),
        "F4": (0.2, 0.3, 0.5),
    }

    THETA = 0.65
    K = 3

    validation_store = None

    if HAS_VALIDATION_STORE_DEPS:
        try:
            validation_store = seed_store()
        except Exception as error:
            print(
                "[WARNING] Could not initialize validation store: "
                f"{error}"
            )
    else:
        print(
            "[WARNING] chromadb/sentence-transformers are unavailable. "
            "Semantic accuracy may be stubbed at 1.0, so F1 may not "
            "be diagnosed correctly."
        )

    # ---------------------------------------------------------
    # Scenario 1: Clean run
    # ---------------------------------------------------------
    clean_result = run_task_with_fault(
        workflow=workflow,
        task=create_task("clean"),
        fault_type=None,
        k=K,
        weights=(0.2, 0.3, 0.5),
        theta=THETA,
        validation_store=validation_store,
    )

    save_events_to_jsonl(
        [RawEvent(**event.model_dump())
         for event in clean_result["events"]]
    )
    print_result("CLEAN RUN (no fault injected)", clean_result)

    # ---------------------------------------------------------
    # Scenarios 2-5: Test all four fault types
    # ---------------------------------------------------------
    print("\n" + "=" * 62)
    print(" TESTING ALL FOUR FAULT TYPES ")
    print("=" * 62)

    for fault_type in ("F1", "F2", "F3", "F4"):
        print(f"\nInjected fault type: {fault_type}")

        result = run_task_with_fault(
            workflow=workflow,
            task=create_task(fault_type.lower()),
            fault_type=fault_type,
            k=K,
            weights=fault_weights[fault_type],
            theta=THETA,
            validation_store=validation_store,
            seed=1,
        )

        save_events_to_jsonl(
            [RawEvent(**event.model_dump())
             for event in result["events"]]
        )

        print_result(f"FAULT TEST: {fault_type}", result)

        diagnosis = result["diagnosis"]

        if not result["detected"]:
            print("CHECK: Failure was NOT detected.")
        elif diagnosis is None:
            print("CHECK: Failure detected, but no diagnosis was produced.")
        elif diagnosis.failure_type == fault_type:
            print("CHECK: PASS - diagnosed type matches injected type.")
        else:
            print(
                "CHECK: MISMATCH - injected "
                f"{fault_type}, diagnosed {diagnosis.failure_type}."
            )

    print("\n" + "=" * 62)
    print(" Integration run complete.")
    print("=" * 62)


if __name__ == "__main__":
    main()
