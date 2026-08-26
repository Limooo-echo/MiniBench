import hashlib
import unittest

from minibench.agents.cot import CoTAgent
from minibench.core.agent import CompletionResult, ReasoningConfig
from minibench.core.metrics import (
    finish_task_metrics,
    start_task_metrics,
    summarize_metrics,
)
from minibench.core.runtime import (
    AgentRuntime,
    ExecutionBudget,
    ExecutionBudgetExceeded,
    StrictJSONObjectError,
    parse_single_json_object,
)


class LegacyClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, prompt, **options):
        self.calls.append(("prompt", prompt, options))
        if not self.responses:
            raise AssertionError("legacy client ran out of responses")
        return self.responses.pop(0)

    def complete_messages(self, messages, **options):
        self.calls.append(("messages", list(messages), options))
        if not self.responses:
            raise AssertionError("legacy client ran out of responses")
        return self.responses.pop(0)


class RichClient:
    def __init__(self):
        self.calls = []

    def complete(self, prompt, **options):
        raise AssertionError("rich client should use complete_result")

    def complete_result(self, prompt, **options):
        self.calls.append((prompt, options))
        return CompletionResult(
            content='{"answer":"C"}',
            reasoning="private provider reasoning",
            finish_reason="stop",
            usage={
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8,
            },
            model="test-model",
            response_id="response-1",
            elapsed_seconds=0.25,
            metadata={"provider": "unit"},
        )


class WrapperAgent:
    def __init__(self, client):
        self.client = client


class AgentRuntimeTests(unittest.TestCase):
    def test_legacy_complete_fallback_records_summary_trace(self):
        client = LegacyClient(['{"ok":true}'])
        runtime = AgentRuntime(client, trace_mode="summary")
        span_id = runtime.begin_span("unit")

        result = runtime.complete_result("Question")
        run = runtime.end_span(span_id, output=result.content)

        self.assertEqual(result.content, '{"ok":true}')
        self.assertIs(runtime.last_run, run)
        self.assertEqual(run.metrics["llm_calls"], 1)
        self.assertEqual(run.metrics["usage_missing_calls"], 1)
        self.assertEqual(len(run.stages), 1)
        self.assertEqual(run.budget["calls_used"], 1)
        self.assertIsNone(run.budget["max_calls"])
        stage = run.stages[0]
        self.assertEqual(stage.stage_name, "completion")
        self.assertEqual(
            stage.prompt_sha256,
            hashlib.sha256(b"Question").hexdigest(),
        )
        self.assertIsNone(stage.prompt)
        self.assertIsNone(stage.output)
        self.assertEqual(
            run.output_sha256,
            hashlib.sha256(result.content.encode()).hexdigest(),
        )
        self.assertIsNone(run.output)

    def test_rich_result_usage_and_full_trace_are_preserved(self):
        client = RichClient()
        runtime = AgentRuntime(client, trace_mode="full")
        span_id = runtime.begin_span("rich")

        result = runtime.complete_result("Question", stage_name="final")
        run = runtime.end_span(span_id, output=result.content)

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(run.metrics["token_usage"]["total_tokens"], 8)
        self.assertEqual(run.metrics["usage_missing_calls"], 0)
        stage = run.stages[0]
        self.assertEqual(stage.model_elapsed_seconds, 0.25)
        self.assertEqual(stage.finish_reason, "stop")
        self.assertEqual(stage.model, "test-model")
        self.assertEqual(stage.response_id, "response-1")
        self.assertEqual(stage.prompt, "Question")
        self.assertEqual(stage.output, '{"answer":"C"}')
        self.assertEqual(stage.reasoning, "private provider reasoning")
        self.assertEqual(stage.metadata, {"provider": "unit"})
        self.assertEqual(run.output, '{"answer":"C"}')

    def test_complete_messages_uses_legacy_fallback(self):
        client = LegacyClient(['{"ok":true}'])
        runtime = AgentRuntime(client)

        output = runtime.complete_messages(
            [{"role": "user", "content": "Question"}],
        )

        self.assertEqual(output, '{"ok":true}')
        self.assertEqual(client.calls[0][0], "messages")

    def test_strict_json_repairs_format_at_most_once(self):
        client = LegacyClient(["answer: C", '{"answer":"C"}'])
        runtime = AgentRuntime(client, trace_mode="full")
        span_id = runtime.begin_span("repair")

        result = runtime.complete_result(
            "Return an answer object.",
            strict_json=True,
        )
        run = runtime.end_span(span_id, output=result.content)

        self.assertTrue(result.format_repaired)
        self.assertEqual(result.parsed_json, {"answer": "C"})
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(len(run.stages), 2)
        self.assertFalse(run.stages[0].json_valid)
        self.assertTrue(run.stages[1].json_valid)
        self.assertTrue(run.stages[1].is_format_repair)
        self.assertEqual(run.stages[1].metadata["repairs_stage"], "completion")

    def test_second_invalid_json_fails_without_a_third_call(self):
        client = LegacyClient(["answer: C", "still not JSON"])
        runtime = AgentRuntime(client)
        span_id = runtime.begin_span("repair-failure")

        with self.assertRaisesRegex(
            StrictJSONObjectError,
            "format repair did not produce",
        ):
            runtime.complete_result("Question", strict_json=True)
        run = runtime.end_span(span_id)

        self.assertEqual(len(client.calls), 2)
        self.assertEqual(run.metrics["llm_calls"], 2)
        self.assertEqual(len(run.stages), 2)

    def test_call_budget_is_checked_before_invocation(self):
        client = LegacyClient(['{"one":1}', '{"two":2}'])
        budget = ExecutionBudget(max_calls=1)
        runtime = AgentRuntime(client, budget=budget)
        span_id = runtime.begin_span("budget")

        runtime.complete("First")
        with self.assertRaisesRegex(ExecutionBudgetExceeded, "budget exhausted"):
            runtime.complete("Second")
        runtime.end_span(span_id)

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(budget.calls_used, 1)
        self.assertEqual(budget.remaining_calls, 0)
        self.assertEqual(runtime.last_run.budget["max_calls"], 1)
        self.assertEqual(runtime.last_run.budget["calls_used"], 1)

    def test_exact_json_parser_rejects_extra_text_and_non_objects(self):
        self.assertEqual(parse_single_json_object('  {"ok":true}\n'), {"ok": True})
        for response in (
            '{"ok":true} trailing',
            '{"a":1} {"b":2}',
            '{"a":1,"a":2}',
            '{"a":NaN}',
            "[]",
            "",
        ):
            with self.subTest(response=response):
                with self.assertRaises(StrictJSONObjectError):
                    parse_single_json_object(response)

    def test_task_metrics_open_and_close_runtime_span(self):
        client = LegacyClient(['{"ok":true}'])
        runtime = AgentRuntime(client, trace_mode="summary")
        agent = WrapperAgent(runtime)

        metrics_start = start_task_metrics(agent)
        runtime.complete("Question", stage_name="direct")
        metrics = finish_task_metrics(agent, metrics_start)

        self.assertEqual(metrics["llm_calls"], 1)
        self.assertIn("trace", metrics)
        self.assertEqual(metrics["trace"]["name"], "task")
        self.assertEqual(metrics["trace"]["stages"][0]["stage_name"], "direct")
        self.assertIsNotNone(runtime.last_run)
        self.assertIsNone(runtime.active_span_id)

        summarized = summarize_metrics([{"metrics": metrics}])
        self.assertEqual(summarized["total"]["llm_calls"], 1)
        self.assertNotIn("trace", summarized["total"])

    def test_existing_reasoning_agent_can_use_runtime_as_chat_client(self):
        client = LegacyClient(["Reasoning says C.", '{"answer":"C"}'])
        runtime = AgentRuntime(client, trace_mode="summary")
        agent = CoTAgent(runtime, ReasoningConfig())

        metrics_start = start_task_metrics(agent)
        output = agent.generate("Question", object())
        metrics = finish_task_metrics(agent, metrics_start)

        self.assertEqual(output, '{"answer":"C"}')
        self.assertEqual(metrics["llm_calls"], 2)
        self.assertEqual(len(metrics["trace"]["stages"]), 2)
        self.assertEqual(len(client.calls), 2)

    def test_off_trace_keeps_last_run_but_not_task_metric_trace(self):
        client = LegacyClient(['{"ok":true}'])
        runtime = AgentRuntime(client, trace_mode="off")
        agent = WrapperAgent(runtime)

        metrics_start = start_task_metrics(agent)
        runtime.complete("Question")
        metrics = finish_task_metrics(agent, metrics_start)

        self.assertNotIn("trace", metrics)
        self.assertIsNotNone(runtime.last_run)
        self.assertEqual(runtime.last_run.stages, ())

    def test_summary_hides_reasoning_and_records_prompt_version(self):
        client = RichClient()
        runtime = AgentRuntime(
            client,
            trace_mode="summary",
            prompt_version="v2",
        )
        span_id = runtime.begin_span("summary")

        runtime.complete_result("Question")
        run = runtime.end_span(span_id)

        self.assertIsNone(run.stages[0].reasoning)
        self.assertEqual(run.stages[0].metadata["prompt_version"], "v2")

    def test_soft_token_budget_stops_before_the_next_call(self):
        client = RichClient()
        budget = ExecutionBudget(max_total_tokens=8)
        runtime = AgentRuntime(client, budget=budget)
        span_id = runtime.begin_span("tokens")

        runtime.complete("First")
        with self.assertRaisesRegex(ExecutionBudgetExceeded, "token budget"):
            runtime.complete("Second")
        run = runtime.end_span(span_id)

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(budget.total_tokens_used, 8)
        self.assertEqual(budget.remaining_tokens, 0)
        self.assertEqual(run.budget["max_total_tokens"], 8)
        self.assertEqual(run.budget["total_tokens_used"], 8)


if __name__ == "__main__":
    unittest.main()
