"""The primary Xiangqi export is an exact seven-dimensional JSON vector."""
from __future__ import annotations
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minibench.core.checkpoint import atomic_write_json
from minibench.datasets.xiangqi.outcome_labels import summarize_outcome_labels, task_label_keys
from minibench.visualization.xiangqi_suite import (
    LABEL_KEYS, comparison_rows, outcome_label_rows, outcome_summary_rows, plot_suite,
)
SPEC = importlib.util.spec_from_file_location("seven_dim_web", Path(__file__).resolve().parents[1]/"scripts/xiangqi_web_test.py")
web = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(web)


def row(mate=True, legal=True, recognition=True, *, ident="board", **extra):
    return {"id":ident, "success":mate, "status":"ok", "is_legal":legal,
            "recognition_exact":recognition, "steps":[{"actor":"agent", "uci":"a0a1", "is_legal":legal, "recognition_exact":recognition}],
            "metrics":{"llm_calls":1, "usage_missing_calls":0,
                       "token_usage":{"prompt_tokens":5,"completion_tokens":5,"total_tokens":10},
                       "cost":{"amount":.01,"complete":True,"currency":"CNY"},"task_elapsed_seconds":2}, **extra}


def summary(records, *, family="d3"):
    return summarize_outcome_labels(records, recognition_applicable=family!="c2", multimove=family in {"h2","c2"})


class SevenDimReportingTests(unittest.TestCase):
    def capture(self, callback, *args):
        out=io.StringIO()
        with contextlib.redirect_stdout(out): callback(*args)
        return out.getvalue()

    def test_web_item_is_binary_json_and_has_no_loss(self):
        record=row(recognition=False,avg_cp_loss=99,avg_variant_loss=88)
        text=self.capture(web._print_item_result,"m2",record)
        payload=json.loads(text.removeprefix("RESULT "))
        self.assertEqual(tuple(payload["labels"]),task_label_keys("m2"))
        self.assertEqual(payload["labels"],{"mate":1,"legal":1,"recognition_exact":0,"mate_given_recognition_and_legal":"NA","tokens":10,"money":.01,"time":2})
        self.assertNotIn("loss",text)
        self.assertNotIn("%",text)

    def test_c2_keeps_unsupported_recognition_and_strict_as_na(self):
        payload=json.loads(self.capture(web._print_item_result,"c2",row()).removeprefix("RESULT "))
        self.assertNotIn("recognition_exact",payload["labels"])
        self.assertEqual(tuple(payload["labels"]),task_label_keys("c2"))

    def test_primary_summary_preserves_raw_mate_and_conditional_denominator(self):
        values=summary([row(),row(False),row(recognition=False)])
        text=self.capture(web._print_family_summary,"m2",values)
        payload=json.loads(text.removeprefix("SUMMARY "))
        self.assertAlmostEqual(payload["labels"]["mate"],2/3)
        self.assertEqual((payload["conditional_success_rate"],payload["conditional_success_count"],payload["conditional_eligible_count"]),(.5,1,2))
        self.assertEqual((payload["labels"]["tokens"],payload["labels"]["money"],payload["labels"]["time"]),(30,.03,6))

    def test_lagged_web_turns_always_use_fresh_browser_chat(self):
        with tempfile.TemporaryDirectory() as temporary:
            agent=web.ManualWebAgent(Path(temporary))
            messages=[{"role":"system","content":"rules"},{"role":"user","content":"Mode: LAGGED STATE. previous board and last round."}]
            with patch("builtins.input",return_value='{"move":"a0a1","pieces":{}}'):
                first=self.capture(agent.generate_messages,messages,{"id":"same-board"})
                second=self.capture(agent.generate_messages,messages,{"id":"same-board"})
            self.assertIn("NEW browser chat",first)
            self.assertIn("NEW browser chat",second)
            self.assertNotIn("Continue the SAME",second)
            records=[json.loads(line) for line in agent.transcript_path.read_text().splitlines()]
            self.assertEqual([r["transport"] for r in records],["lagged-round-stateless"]*2)

    def test_label_exports_do_not_duplicate_c2_standard_and_variant_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            suite=Path(temporary)
            atomic_write_json(suite/"plan.json",{"agents":["direct"],"experiments":{"c2":{"family":"xiangqi-rule-variants","condition_counts":{"all":2,"standard":1,"variant":1}}}})
            a,b=row(ruleset="standard",ident="a"),row(False,ruleset="horse-no-leg-block",ident="b")
            atomic_write_json(suite/"results/c2/direct/results.json",{"by_condition":{"standard":summary([a],family="c2"),"variant":summary([b],family="c2")}})
            (suite/"results/c2/direct/predictions.jsonl").write_text(json.dumps(a)+"\n"+json.dumps(b)+"\n")
            labels=outcome_label_rows(suite)
            self.assertEqual(len(labels),2)
            self.assertEqual({r["group"] for r in labels},{"standard","variant"})
            self.assertTrue(all(set(r["labels"])==set(task_label_keys("c2")) for r in labels))
            groups=outcome_summary_rows(suite)
            self.assertEqual({r["group"]:r["labels"]["mate"] for r in groups},{"standard":1,"variant":0})

    def test_physical_billing_overrides_replayed_logical_usage_and_preserves_unknown_money(self):
        with tempfile.TemporaryDirectory() as temporary:
            suite=Path(temporary)
            atomic_write_json(suite/"plan.json",{"agents":["direct"],"experiments":{"c2":{"family":"xiangqi-rule-variants","condition_counts":{"all":2,"standard":1,"variant":1,"horse-no-leg-block":1}}}})
            a,b=row(ruleset="standard"),row(ruleset="horse-no-leg-block")
            atomic_write_json(suite/"results/c2/direct/results.json",{"by_condition":{"standard":summary([a],family="c2"),"variant":summary([b],family="c2")},"by_ruleset":{"horse-no-leg-block":summary([b],family="c2")}})
            atomic_write_json(suite/"summary.json",{"by_experiment_agent_condition_resources":{"c2":{"direct":{
                "standard":{"token_usage":{"total_tokens":40},"cost":{"amount":None,"complete":False}},
                "variant":{"token_usage":{"total_tokens":50},"cost":{"amount":.2,"complete":True,"currency":"CNY"}}}}}})
            groups={r["group"]:r for r in outcome_summary_rows(suite)}
            self.assertEqual(groups["standard"]["labels"]["tokens"],40)
            self.assertEqual(groups["standard"]["labels"]["money"],"NA")
            self.assertEqual(groups["variant"]["labels"]["tokens"],50)
            self.assertEqual(groups["variant"]["labels"]["money"],.2)
            self.assertEqual(groups["standard"]["labels"]["time"],2)
            table={(r["mode"],r["metric"]):r for r in comparison_rows(suite)}
            self.assertEqual(table[("all","tokens")]["value"],90)
            self.assertIsNone(table[("all","money")]["value"])
            self.assertIsNone(table[("horse-no-leg-block","tokens")]["value"])

    def test_plot_and_tables_use_zero_one_and_exclude_loss(self):
        with tempfile.TemporaryDirectory() as temporary:
            suite=Path(temporary)
            atomic_write_json(suite/"plan.json",{"agents":["direct"],"experiments":{"d3":{"family":"xiangqi-mate-in-one","condition_counts":{"native":2}}}})
            records=[row(),row(False,recognition=False)]
            atomic_write_json(suite/"results/d3/direct/results.json",summary(records))
            (suite/"results/d3/direct/predictions.jsonl").write_text("\n".join(map(json.dumps,records))+"\n")
            rows=comparison_rows(suite)
            self.assertEqual({r["metric"] for r in rows},set(task_label_keys("d3")))
            self.assertEqual(next(r["value"] for r in rows if r["metric"]=="mate"),.5)
            charts=plot_suite(suite)
            exported=json.loads((charts/"outcome-labels.json").read_text())
            self.assertEqual(exported,outcome_label_rows(suite))
            with (charts/"outcome-labels.csv").open(newline="") as handle:
                data=list(csv.DictReader(handle))
            self.assertEqual((data[0]["mate"],data[1]["mate"]),("1","0"))
            self.assertFalse(any("loss" in p.name for p in charts.glob("*.png")))
            self.assertTrue((charts/"outcome-summary.json").exists())
            self.assertIn("0.500",(charts/"d3-native-mate.svg").read_text())


if __name__=="__main__": unittest.main()
