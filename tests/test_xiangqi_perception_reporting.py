"""Recognition gates remain separate from the seven primary JSON labels."""
from __future__ import annotations
import contextlib, csv, importlib.util, io, json
from pathlib import Path
import tempfile, unittest, warnings
from minibench.core.checkpoint import atomic_write_json
from minibench.datasets.xiangqi.outcome_labels import summarize_outcome_labels
from minibench.visualization.xiangqi_suite import LABEL_KEYS, _bar_label, comparison_rows, plot_suite
SPEC=importlib.util.spec_from_file_location("xiangqi_perception_web_test",Path(__file__).resolve().parents[1]/"scripts/xiangqi_web_test.py")
web=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(web)


def records():
    return [{"success":m,"is_legal":True,"recognition_exact":r} for m,r in [(True,True),(False,True),(True,False),(True,False)]]

def summary(rows=None):
    return summarize_outcome_labels(records() if rows is None else rows,recognition_applicable=True,multimove=False)

class PerceptionReportingTests(unittest.TestCase):
    @staticmethod
    def capture(callback,*args):
        out=io.StringIO()
        with contextlib.redirect_stdout(out):callback(*args)
        return out.getvalue()
    def test_web_preserves_mating_move_with_failed_transcription(self):
        record={"task_id":"paired-board","success":True,"is_legal":True,"recognition_exact":False,"agent_uci":"a0a1"}
        for family in ("d3","m2"):
            payload=json.loads(self.capture(web._print_item_result,family,record).removeprefix("RESULT "))
            self.assertEqual((payload["labels"]["mate"],payload["labels"]["legal"]),(1,1))
            if family == "m2":
                self.assertEqual(payload["labels"]["recognition_exact"],0)
                self.assertEqual(payload["labels"]["mate_given_recognition_and_legal"],"NA")
            else:
                self.assertEqual(len(payload["labels"]),5)
            self.assertEqual(payload["diagnostics"]["moves"],"agent:a0a1")
    def test_web_summaries_show_gates_with_actual_counts(self):
        for family in ("m2",):
            payload=json.loads(self.capture(web._print_family_summary,family,summary()).splitlines()[0].removeprefix("SUMMARY "))
            self.assertEqual((payload["labels"]["mate"],payload["labels"]["legal"],payload["labels"]["mate_given_recognition_and_legal"]),(.75,1,.5))
            self.assertEqual((payload["conditional_success_rate"],payload["conditional_success_count"],payload["conditional_eligible_count"]),(.5,1,2))
        payload=json.loads(self.capture(web._print_family_summary,"m2",summary([{"success":True,"is_legal":True,"recognition_exact":False}])).removeprefix("SUMMARY "))
        self.assertEqual(payload["conditional_success_rate"],"NA")
        self.assertEqual(payload["conditional_eligible_count"],0)
        self.assertEqual(payload["labels"]["mate_given_recognition_and_legal"],"NA")
    def test_unknown_evidence_is_not_a_fabricated_zero(self):
        values=summary([{"success":True,"is_legal":True,"recognition_exact":True},{"success":True,"is_legal":True}])
        payload=json.loads(self.capture(web._print_family_summary,"m2",values).removeprefix("SUMMARY "))
        self.assertEqual(payload["conditional_success_rate"],"NA")
        self.assertEqual(payload["conditional_unknown_count"],1)
        self.assertEqual(payload["labels"]["mate_given_recognition_and_legal"],"NA")
        with tempfile.TemporaryDirectory() as temporary:
            suite=Path(temporary)
            atomic_write_json(suite/"plan.json",{"agents":["direct"],"experiments":{"m2":{"family":"xiangqi-multimodal","condition_counts":{"chinese-piece-image":2}}}})
            atomic_write_json(suite/"results/m2/direct/results.json",{"by_input_mode":{"chinese-piece-image":values}})
            row=next(r for r in comparison_rows(suite) if r["metric"]=="mate_given_recognition_and_legal")
            self.assertIsNone(row["value"])
            self.assertEqual((row["numerator"],row["denominator"],row["unknown_gate_count"]),(1,1,1))
            self.assertIn("known 1; unknown 1",_bar_label(row))
    def test_comparison_uses_exact_transcription_not_partial_piece_accuracy(self):
        with tempfile.TemporaryDirectory() as temporary:
            suite=Path(temporary)
            atomic_write_json(suite/"plan.json",{"agents":["direct"],"experiments":{
                "d3":{"family":"xiangqi-mate-in-one","condition_counts":{"native":4}},
                "m2":{"family":"xiangqi-multimodal","condition_counts":{"chinese-piece-image":4}}}})
            atomic_write_json(suite/"results/d3/direct/results.json",summary())
            atomic_write_json(suite/"results/m2/direct/results.json",{"by_input_mode":{"chinese-piece-image":summary()}})
            for task in ("d3","m2"):
                by={r["metric"]:r for r in comparison_rows(suite) if r["task"]==task}
                self.assertEqual(by["mate"]["value"],.75)
                if task == "m2":
                    self.assertEqual(by["recognition_exact"]["value"],.5)
                    self.assertEqual((by["mate_given_recognition_and_legal"]["value"],by["mate_given_recognition_and_legal"]["denominator"]),(.5,2))
                else:
                    self.assertNotIn("recognition_exact",by)
                self.assertNotIn("recognition_accuracy",by)
    def test_legacy_or_zero_eligible_never_becomes_zero_conditional_accuracy(self):
        with tempfile.TemporaryDirectory() as temporary:
            suite=Path(temporary)
            atomic_write_json(suite/"plan.json",{"agents":["legacy","empty"],"experiments":{"d3":{"family":"xiangqi-mate-in-one","condition_counts":{"native":1}}}})
            atomic_write_json(suite/"results/d3/legacy/results.json",{"mate_rate":1.,"legality_rate":1.})
            atomic_write_json(suite/"results/d3/empty/results.json",summary([{"success":True,"is_legal":True,"recognition_exact":False}]))
            rows=comparison_rows(suite)
            conditional=[r for r in rows if r["metric"]=="conditional_success_rate"]
            self.assertTrue(all(r["value"] is None for r in conditional))
            with warnings.catch_warnings():
                warnings.simplefilter("error",UserWarning)
                charts=plot_suite(suite)
            self.assertEqual(len(list(charts.glob("*.png"))),5)
            with (charts/"comparison.csv").open(newline="") as handle:
                data=list(csv.DictReader(handle))
            self.assertTrue(all(r["metric"]!="conditional_success_rate" for r in data))

if __name__=="__main__": unittest.main()
