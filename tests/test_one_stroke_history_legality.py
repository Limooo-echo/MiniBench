import json
import unittest
from dataclasses import asdict, replace

from minibench.core.one_stroke_checkpoint import one_stroke_result_from_dict
from minibench.datasets.one_stroke.dataset import (
    one_stroke_task_from_dict,
    simulate_one_stroke_history,
)
from minibench.datasets.one_stroke.evaluation import (
    audit_one_stroke_history,
    evaluate_one_stroke_tasks,
    summarize_one_stroke,
)


class StateSequenceAgent:
    def __init__(self, states, final_path):
        self.states = iter(states)
        self.final_path = final_path

    def generate_messages_for_phase(self, messages, task, *, phase, **kwargs):
        if phase == "final":
            return json.dumps({"path": self.final_path})
        output = next(self.states)
        return output if isinstance(output, str) else json.dumps(output)


def chain_task():
    return one_stroke_task_from_dict({
        "id": "ordered-history", "capability": "history_memory",
        "vertices": ["A", "B", "C", "D"],
        "edges": [["A", "B"], ["B", "C"], ["C", "D"]],
        "start": "A", "end": "D",
        "history_events": [
            {"action": "move", "edge_id": "e01", "from": "A", "to": "B"},
            {"action": "move", "edge_id": "e02", "from": "B", "to": "C"},
            {"action": "move", "edge_id": "e03", "from": "C", "to": "D"},
        ],
    })


def exact_outputs(task):
    outputs = []
    for step in range(1, len(task.history_events) + 1):
        state = simulate_one_stroke_history(
            replace(task, history_events=task.history_events[:step])
        )
        outputs.append({
            "current_vertex": state.current_vertex,
            "used_edges": list(state.used_edge_ids),
            "remaining_edges": list(state.remaining_edge_ids),
        })
    return outputs


class OneStrokeHistoryLegalityTests(unittest.TestCase):
    def test_disconnected_order_stays_illegal_after_final_state_recovers(self):
        task = chain_task()
        states = exact_outputs(task)
        # A-B, C-D, B-C has the right final edge set, but cannot be walked in order.
        states[1] = {
            "current_vertex": "D", "used_edges": ["e01", "e03"],
            "remaining_edges": ["e02"],
        }
        result = evaluate_one_stroke_tasks(
            [task], StateSequenceAgent(states, ["D"]),
            memory_modes=("incremental_state",),
        )[0]
        self.assertTrue(result.success)
        self.assertEqual(result.score, 1.0)
        self.assertFalse(result.is_legal)
        self.assertAlmostEqual(result.history_process_score, 2 / 3)
        self.assertEqual([a["is_legal"] for a in result.history_step_audits],
                         [True, False, False])
        self.assertIn("disconnected_move:e03:B->D",
                      result.history_step_audits[1]["legality_reasons"])
        self.assertEqual(summarize_one_stroke([result])["legality_rate"], 0.0)

    def test_edge_list_order_is_not_traversal_order(self):
        task = chain_task()
        states = exact_outputs(task)
        for state in states:
            state["used_edges"].reverse()
            state["remaining_edges"].reverse()
        result = evaluate_one_stroke_tasks(
            [task], StateSequenceAgent(states, ["D"]),
            memory_modes=("incremental_state",),
        )[0]
        self.assertTrue(result.is_legal)
        self.assertEqual(result.history_process_score, 1.0)

    def test_parallel_edges_and_latest_move_undo(self):
        task = one_stroke_task_from_dict({
            "id": "parallel-undo", "capability": "history_memory",
            "vertices": ["A", "B", "C"],
            "edges": [["A", "B"], ["A", "B"], ["A", "C"]],
            "start": "A", "end": "C",
            "history_events": [
                {"action": "move", "edge_id": "e01", "from": "A", "to": "B"},
                {"action": "move", "edge_id": "e02", "from": "B", "to": "A"},
                {"action": "undo", "edge_id": "e02", "from": "A", "to": "B"},
            ],
        })
        states = exact_outputs(task)
        audits = audit_one_stroke_history(task, [json.dumps(s) for s in states])
        self.assertTrue(all(a["is_legal"] for a in audits))
        states[-1]["used_edges"] = ["e02"]
        states[-1]["remaining_edges"] = ["e01", "e03"]
        audits = audit_one_stroke_history(task, [json.dumps(s) for s in states])
        self.assertFalse(audits[-1]["is_legal"])
        self.assertIn("undo_not_latest_move", audits[-1]["legality_reasons"])

    def test_wrong_but_continuous_state_is_legal(self):
        task = one_stroke_task_from_dict({
            "id": "alternative-continuous", "capability": "history_memory",
            "vertices": ["A", "B", "C"],
            "edges": [["A", "B"], ["B", "C"], ["C", "A"]],
            "start": "A", "end": "A",
            "history_events": [
                {"action": "move", "edge_id": "e01", "from": "A", "to": "B"},
            ],
        })
        audit = audit_one_stroke_history(task, [json.dumps({
            "current_vertex": "C", "used_edges": ["e03"],
            "remaining_edges": ["e01", "e02"],
        })])[0]
        self.assertTrue(audit["is_legal"])
        self.assertFalse(audit["state_exact"])

    def test_invalid_json_does_not_imply_observed_discontinuity(self):
        task = chain_task()
        states = exact_outputs(task)
        states[0] = "not json"
        result = evaluate_one_stroke_tasks(
            [task], StateSequenceAgent(states, ["D"]),
            memory_modes=("incremental_state",),
        )[0]
        self.assertIsNone(result.is_legal)
        self.assertAlmostEqual(result.history_process_score, 2 / 3)
        self.assertEqual(summarize_one_stroke([result])["legality_denominator"], 0)

    def test_ack_only_history_has_no_observed_state_metrics(self):
        result = evaluate_one_stroke_tasks(
            [chain_task()], StateSequenceAgent([{"step": i} for i in (1, 2, 3)], ["D"]),
            memory_modes=("step_history_only",),
        )[0]
        self.assertTrue(result.success)
        self.assertIsNone(result.is_legal)
        self.assertIsNone(result.history_process_score)
        self.assertEqual(result.history_step_audits, ())

    def test_non_history_result_has_no_legality_or_process_score(self):
        task = replace(chain_task(), capability="direct", history_events=())

        class WrongPathAgent:
            def generate(self, prompt, task):
                return '{"path":["A","D"]}'

        result = evaluate_one_stroke_tasks([task], WrongPathAgent())[0]
        self.assertFalse(result.success)
        self.assertIsNone(result.is_legal)
        self.assertIsNone(result.history_process_score)

    def test_audits_round_trip_and_legacy_records_remain_loadable(self):
        task = chain_task()
        result = evaluate_one_stroke_tasks(
            [task], StateSequenceAgent(exact_outputs(task), ["D"]),
            memory_modes=("incremental_state",),
        )[0]
        payload = json.loads(json.dumps(asdict(result)))
        self.assertEqual(one_stroke_result_from_dict(payload), result)
        for field in ("is_legal", "history_process_score", "history_step_audits"):
            del payload[field]
        legacy = one_stroke_result_from_dict(payload)
        self.assertIsNone(legacy.is_legal)
        self.assertIsNone(legacy.history_process_score)
        self.assertEqual(legacy.history_step_audits, ())


if __name__ == "__main__":
    unittest.main()
