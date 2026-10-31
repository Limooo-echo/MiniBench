"""Manual web output carries a seven-label JSON vector and separate diagnostics."""
from __future__ import annotations
import contextlib, importlib.util, io, json
from pathlib import Path
import unittest
from minibench.datasets.xiangqi.outcome_labels import summarize_outcome_labels
SPEC=importlib.util.spec_from_file_location("xiangqi_web_test",Path(__file__).resolve().parents[1]/"scripts/xiangqi_web_test.py")
web=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(web)

class XiangqiWebOutputTests(unittest.TestCase):
    def capture(self,callback,*args):
        out=io.StringIO()
        with contextlib.redirect_stdout(out): callback(*args)
        return out.getvalue().strip()
    def test_h2_item_reports_binary_labels_and_move_trace_without_loss(self):
        record={"task_id":"h2-example","history_mode":"full-state","success":False,
                "extended_mate_success":True,"mate_within_shortest_plus_two_steps":True,
                "legality_rate":1.,"avg_cp_loss":12.5,
                "steps":[{"actor":"agent","uci":"a0a1","is_legal":True},{"actor":"pikafish","uci":"a9a8"}]}
        text=self.capture(web._print_item_result,"h2",record)
        payload=json.loads(text.removeprefix("RESULT "))
        self.assertEqual((payload["labels"]["mate"],payload["labels"]["legal"]),(0,1))
        self.assertEqual(payload["diagnostics"]["moves"],"agent:a0a1,pikafish:a9a8")
        self.assertNotIn("loss",text);self.assertNotIn("plus_two",text)
    def test_one_ply_items_use_unit_labels_and_na_without_recognition(self):
        record={"task_id":"d3-example","success":True,"is_legal":True,"agent_uci":"a0a1"}
        text=self.capture(web._print_item_result,"d3",record)
        labels=json.loads(text.removeprefix("RESULT "))["labels"]
        self.assertEqual((labels["mate"],labels["legal"]),(1,1))
        self.assertEqual(len(labels),5)
        self.assertNotIn("%",text);self.assertNotIn("loss",text)
    def test_c2_has_no_loss_or_unsupported_recognition(self):
        record={"task_id":"c2-example","ruleset":"standard","success":True,"avg_variant_loss":8,
                "steps":[{"actor":"agent","uci":"a0a1","is_legal":True},{"actor":"depth-opponent","uci":"a9a8"}]}
        text=self.capture(web._print_item_result,"c2",record)
        payload=json.loads(text.removeprefix("RESULT "))
        self.assertEqual((payload["labels"]["mate"],payload["labels"]["legal"]),(1,1))
        self.assertEqual(len(payload["labels"]),6)
        self.assertNotIn("recognition_exact",payload["labels"])
        self.assertNotIn("loss",text)
    def test_h2_summary_is_json_with_conditional_counts(self):
        rows=[{"success":True,"recognition_exact":True,"steps":[{"actor":"agent","uci":"a0a1","is_legal":True}]},
              {"success":False,"recognition_exact":True,"steps":[{"actor":"agent","uci":"a0a1","is_legal":True}]}]
        summary=summarize_outcome_labels(rows,recognition_applicable=True,multimove=True)
        text=self.capture(web._print_family_summary,"h2",{"by_mode":{"move-history-only":summary}})
        payload=json.loads(text.removeprefix("SUMMARY "))
        self.assertEqual((payload["labels"]["mate"],payload["labels"]["legal"]),(.5,1))
        self.assertEqual((payload["conditional_success_rate"],payload["conditional_eligible_count"]),(.5,2))
        self.assertNotIn("loss",text);self.assertNotIn("%",text)

if __name__=="__main__":unittest.main()
