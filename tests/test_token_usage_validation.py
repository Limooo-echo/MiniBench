"""Offline accounting regressions; no real model calls or billing."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minibench.core.metrics import extract_token_usage, start_task_metrics, finish_task_metrics, summarize_metrics, summary_metrics_line
from minibench.core.rate_limit import RateGate
from minibench.core.streaming import JournalClient, call_attempts
from minibench.core.token_usage import assess_token_usage, known_total_tokens
from minibench.datasets.xiangqi.resources import (
    add_physical_attempt, finish_physical_bucket, physical_bucket, estimate_cost,
)
from minibench.factory.providers import OpenAICompatibleClient
from tests import test_cloud_streaming as streaming_fixtures

GOOD = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30,
        "completion_tokens_details": {"reasoning_tokens": 15}}
SCHOOL = {"prompt_tokens": 1058, "completion_tokens": 0, "total_tokens": 1058,
          "completion_tokens_details": {"reasoning_tokens": 100872}}
PRICE = {"currency": "CNY", "input_per_million": 2, "output_per_million": 8}


class UsageValidationTests(unittest.TestCase):
    def test_school_anomaly_quarantined_without_mutating_raw_data(self):
        before = deepcopy(SCHOOL)
        checked = assess_token_usage(SCHOOL)
        self.assertEqual(checked["status"], "invalid")
        self.assertIn("count_exceeds_parent:reasoning_tokens", checked["issues"])
        self.assertIsNone(checked["total_tokens"])
        self.assertIsNone(extract_token_usage(SCHOOL))
        self.assertEqual(SCHOOL, before)
        self.assertIsNone(estimate_cost(SCHOOL, PRICE)["amount"])

    def test_valid_reasoning_is_never_added_again(self):
        self.assertEqual(extract_token_usage(GOOD)["total_tokens"], 30)
        duplicate = {**GOOD, "reasoning_tokens": 15}
        self.assertEqual(extract_token_usage(duplicate)["reasoning_tokens"], 15)

    def test_malformed_and_contradictory_counts_rejected(self):
        for value in (True, -1, 1.5, "30", None):
            with self.subTest(value=value):
                self.assertIsNone(known_total_tokens({**GOOD, "total_tokens": value}))
        for usage in ({**GOOD, "total_tokens": 31},
                      {**GOOD, "reasoning_tokens": 16},
                      {**GOOD, "prompt_tokens_details": {"cached_tokens": 11}},
                      {**GOOD, "prompt_cache_hit_tokens": 6, "prompt_cache_miss_tokens": 5}):
            self.assertEqual(assess_token_usage(usage)["status"], "invalid")

    def test_missing_breakdown_not_synthesized_as_zero(self):
        self.assertIsNone(extract_token_usage({"total_tokens": 30}))
        self.assertEqual(known_total_tokens({"total_tokens": 30}), 30)
        self.assertIsNone(known_total_tokens({}))
        self.assertIsNone(known_total_tokens(None))
        self.assertIsNone(known_total_tokens({"total_tokens":10,"reasoning_tokens":100}))

    def test_zero_usage_with_nonempty_output_is_invalid(self):
        zero = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.assertEqual(assess_token_usage(zero, content_chars=1)["status"], "invalid")
        self.assertEqual(assess_token_usage(zero, reasoning_chars=1)["status"], "invalid")

    def test_partial_parent_bounds_quarantine_impossible_breakdowns(self):
        for usage in (
            {"prompt_tokens":10,"total_tokens":10,"reasoning_tokens":5},
            {"completion_tokens":10,"total_tokens":10,"cached_tokens":1},
            {"completion_tokens":8,"total_tokens":10,
             "prompt_cache_hit_tokens":2,"prompt_cache_miss_tokens":1},
            {**GOOD,"prompt_audio_tokens":11},
            {**GOOD,"completion_audio_tokens":21},
        ):
            with self.subTest(usage=usage):
                self.assertEqual(assess_token_usage(usage)["status"],"invalid")
                self.assertIsNone(known_total_tokens(usage))
        self.assertEqual(assess_token_usage({"prompt_tokens":10,"total_tokens":10},
                                           content_chars=1)["status"],"invalid")
        partial={"prompt_tokens":10,"total_tokens":20,"reasoning_tokens":5}
        checked=assess_token_usage(partial)
        self.assertEqual(checked["status"],"partial")
        self.assertNotIn("completion_tokens",checked["tokens"])
        self.assertEqual(known_total_tokens(partial),20)

    def test_different_input_and_output_audio_details_are_not_conflicting_aliases(self):
        usage={**GOOD,"prompt_tokens_details":{"audio_tokens":2},
               "completion_tokens_details":{"audio_tokens":3,"reasoning_tokens":15}}
        self.assertEqual(extract_token_usage(usage)["prompt_audio_tokens"],2)
        self.assertEqual(extract_token_usage(usage)["completion_audio_tokens"],3)

    def test_serialized_runtime_trace_marks_missing_usage_unknown(self):
        from minibench.core.runtime import AgentRuntime
        from tests.test_agent_runtime import LegacyClient
        runtime=AgentRuntime(LegacyClient(['{"ok":true}']),trace_mode="summary")
        span=runtime.begin_span("unit")
        runtime.complete_result("fixture")
        trace=runtime.end_span(span).to_dict()
        self.assertIsNone(trace["metrics"]["token_usage"]["total_tokens"])
        self.assertIsNone(trace["stages"][0]["token_usage"]["total_tokens"])
        self.assertFalse(trace["stages"][0]["usage_complete"])

    def test_provider_and_public_task_metrics_do_not_publish_false_zero(self):
        client = OpenAICompatibleClient(model="fixture", base_url="https://invalid.example/v1", api_key_env="UNIT_KEY")
        start = start_task_metrics(client)
        client._record_completion_metrics(1.0, SCHOOL)
        result = finish_task_metrics(client, start)
        self.assertEqual(result["usage_missing_calls"], 1)
        self.assertIsNone(result["token_usage"]["total_tokens"])
        self.assertIsNone(result["known_token_usage"]["total_tokens"])
        self.assertFalse(result["usage_complete"])
        summary = summarize_metrics([{"metrics": result}])
        self.assertIsNone(summary["total"]["token_usage"]["total_tokens"])
        self.assertIn("total_tokens=unknown", summary_metrics_line(summary))

    def test_mixed_call_totals_unknown_but_validated_subtotal_retained(self):
        client = OpenAICompatibleClient(model="fixture", base_url="https://invalid.example/v1", api_key_env="UNIT_KEY")
        start = start_task_metrics(client)
        client._record_completion_metrics(1.0, GOOD)
        client._record_completion_metrics(1.0, SCHOOL)
        result = finish_task_metrics(client, start)
        summary = summarize_metrics([{"metrics": result}])
        self.assertIsNone(result["token_usage"]["total_tokens"])
        self.assertEqual(result["known_token_usage"]["total_tokens"], 30)
        self.assertIsNone(summary["average_per_task"]["token_usage"]["total_tokens"])
        self.assertEqual(summary["total"]["known_token_usage"]["total_tokens"], 30)

    def test_legacy_bad_record_revalidated_when_summarizing(self):
        summary = summarize_metrics([{"metrics": {"llm_calls": 1, "usage_missing_calls": 0,
                                                  "token_usage": {"prompt_tokens": 1058,
                                                  "completion_tokens": 0, "total_tokens": 1058,
                                                  "reasoning_tokens": 100872}}}])
        self.assertIsNone(summary["total"]["token_usage"]["total_tokens"])
        self.assertEqual(summary["total"]["usage_missing_calls"], 1)

    def test_quota_release_and_restore_keep_unknown_reservation(self):
        gate = RateGate({"a": {"reserve_tokens": 40000}})
        gate.acquire("a")
        gate.release("a", SCHOOL)
        gate.restore("a", SCHOOL)
        state = gate.snapshot()["a"]
        self.assertEqual(state["unknown_usage_calls"], 2)
        self.assertEqual(state["accounted_or_reserved_tokens"], 80000)
        self.assertIsNone(state["total_tokens"])
        self.assertIsNone(state["known_total_tokens"])
        self.assertFalse(state["usage_complete"])

    def test_physical_unknown_totals_and_cost_not_1058(self):
        bucket = physical_bucket()
        add_physical_attempt(bucket, {"usage": SCHOOL, "status": "failed"}, PRICE)
        report = finish_physical_bucket(bucket)
        self.assertIsNone(report["token_usage"]["total_tokens"])
        self.assertIsNone(report["known_token_usage"]["total_tokens"])
        self.assertIsNone(report["cost"]["amount"])
        self.assertEqual(report["attempts_with_unknown_usage"], 1)
        self.assertEqual(report["attempts_with_invalid_usage"], 1)

    def test_physical_mixed_usage_preserves_only_known_subtotal(self):
        bucket = physical_bucket()
        for usage in (GOOD, SCHOOL, None):
            add_physical_attempt(bucket, {"usage": usage, "status": "failed"}, PRICE)
        report = finish_physical_bucket(bucket)
        self.assertEqual(report["known_token_usage"]["total_tokens"], 30)
        self.assertIsNone(report["token_usage"]["total_tokens"])
        self.assertEqual(report["attempts_with_unknown_usage"], 2)

    def test_total_only_physical_usage_does_not_publish_zero_breakdowns(self):
        bucket=physical_bucket()
        add_physical_attempt(bucket,{"usage":{"total_tokens":30},"status":"success"},PRICE)
        report=finish_physical_bucket(bucket)
        self.assertEqual(report['known_token_usage']['total_tokens'],30)
        self.assertIsNone(report['known_token_usage']['prompt_tokens'])
        self.assertIsNone(report['known_token_usage']['completion_tokens'])
        self.assertFalse(report['usage_complete'])
        self.assertIsNone(report['cost']['amount'])

    def test_cloud_report_and_resume_reject_school_anomaly(self):
        from minibench import cloud
        from minibench.core.checkpoint import atomic_write_json
        from minibench.core.streaming import now
        from tests.test_cloud import CloudPlanTests
        with tempfile.TemporaryDirectory() as temporary:
            suite,plan=CloudPlanTests().fixture(Path(temporary))
            job=plan["jobs"][0]
            attempt=suite/"jobs"/job["key"]/"calls/attempts/0000-0000"
            atomic_write_json(attempt/"started.json",{"at":now()})
            atomic_write_json(attempt/"accounting.json",{"at":now(),"status":"failed","usage":SCHOOL})
            summary=cloud.report(suite,plan,{})
            self.assertIsNone(summary["usage"]["total_tokens"])
            self.assertIsNone(summary["usage"]["known_total_tokens"])
            self.assertEqual(summary["usage"]["attempts_without_usage"],1)
            gate=RateGate(plan["limits"])
            cloud.restore_accounting(suite,plan,gate)
            account=next(iter(gate.snapshot().values()))
            self.assertEqual(account["unknown_usage_calls"],1)
            self.assertIsNone(account["total_tokens"])

    def test_nonstream_zero_usage_is_not_reintroduced_by_runtime(self):
        from minibench.core.runtime import AgentRuntime
        from tests.test_openai_compatible_agent import FakeHTTPResponse
        zero={"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}
        payload={"choices":[{"finish_reason":"stop","message":{"content":'{"ok":true}'}}],"usage":zero}
        client=OpenAICompatibleClient(model="fixture",base_url="https://invalid.example/v1",api_key_env="UNIT_KEY")
        runtime=AgentRuntime(client,trace_mode="summary")
        start=start_task_metrics(runtime)
        with patch.dict(os.environ,{"UNIT_KEY":"fixture"}),\
                patch("urllib.request.urlopen",return_value=FakeHTTPResponse(payload)):
            response=runtime.complete_result("fixture")
        self.assertEqual(response.usage,zero)
        metrics=finish_task_metrics(runtime,start)
        self.assertIsNone(metrics["token_usage"]["total_tokens"])
        self.assertEqual(metrics["usage_missing_calls"],1)

    def test_stream_preserves_raw_invalid_usage_and_still_accepts_valid_answer(self):
        fixture = streaming_fixtures.XiangqiStreamingTests()
        body = fixture.body([fixture.chunk({"reasoning_content": "checking"}),
                             fixture.chunk({"content": '{"move":"a0a1"}'}, finish="stop"),
                             fixture.chunk(usage=SCHOOL)])
        with tempfile.TemporaryDirectory() as temporary:
            journal = Path(temporary)
            client = fixture.client(journal)
            gate = RateGate({"default": {"reserve_tokens": 40000}})
            client.gate = gate
            with patch.dict(os.environ, {"UNIT_TEST_TRANSIENT_KEY": "fixture"}):
                with patch("urllib.request.urlopen", return_value=streaming_fixtures.StreamingResponse(body)) as network:
                    result = client.complete_result("fixture")
                self.assertEqual(network.call_count, 1)
                with patch("urllib.request.urlopen", side_effect=AssertionError("no rebilling")):
                    fixture.client(journal).complete_result("fixture")
            self.assertEqual(result.usage, SCHOOL)
            self.assertEqual(result.metadata["usage_validation"]["status"], "invalid")
            self.assertFalse(result.metadata["usage_known"])
            self.assertEqual(client.metrics_snapshot()["usage_missing_calls"], 1)
            self.assertEqual(gate.snapshot()["default"]["unknown_usage_calls"], 1)
            attempts = list(call_attempts(journal))
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0]["usage"], SCHOOL)


if __name__ == "__main__":
    unittest.main()
