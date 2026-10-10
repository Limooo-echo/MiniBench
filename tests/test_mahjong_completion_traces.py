from dataclasses import replace
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minibench.agents.cot import CoTAgent
from minibench.agents.direct import DirectAgent
from minibench.agents.tree_of_thought import TreeOfThoughtAgent
from minibench.core.agent import ReasoningConfig
from minibench.datasets.mahjong.dataset import load_mahjong_tasks
from minibench.datasets.mahjong.evaluation import (
    evaluate_mahjong_tasks,
    write_mahjong_run,
)
from minibench.factory.providers import OpenAICompatibleAgent


def response(content, *, finish_reason="stop", native_reasoning=None, tokens=20):
    return io.BytesIO(
        json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": content,
                            "reasoning_content": native_reasoning,
                        },
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": tokens,
                    "total_tokens": 100 + tokens,
                    "completion_tokens_details": {"reasoning_tokens": 7},
                },
            }
        ).encode("utf-8")
    )


class MahjongCompletionTraceTests(unittest.TestCase):
    def test_text_and_image_runs_save_stage_outputs_and_truncation_per_task(self):
        for mode, filename in (("text", "tasks.jsonl"), ("image", "visual_tasks.jsonl")):
            with self.subTest(mode=mode):
                template = next(
                    task
                    for task in load_mahjong_tasks(Path("data/mahjong") / filename)
                    if task.id == "mj-easy-visible12-wait-008"
                )
                tasks = [replace(template, id=f"trace-{index}") for index in (1, 2)]
                answer = {"winning_tiles": ["4s"], "live_winning_copies": {"4s": 4}}
                if mode == "image":
                    answer.update(hand=list(template.hand), visible_tiles=list(template.visible_tiles))
                final = json.dumps(answer)
                client = OpenAICompatibleAgent(
                    model="qwen3.8-max",
                    base_url="https://example.invalid/v1",
                    api_key_env="TRACE_TEST_API_KEY",
                    extra_body={"enable_thinking": False},
                )
                agent = CoTAgent(client, ReasoningConfig(max_reasoning_tokens=1024, final_max_tokens=1024))
                replies = [
                    response("Incomplete draft 1", finish_reason="length", native_reasoning="Native 1", tokens=1024),
                    response(final),
                    response("Complete draft 2", native_reasoning="Native 2", tokens=80),
                    response(final),
                ]
                with patch.dict(os.environ, {"TRACE_TEST_API_KEY": "test-only"}), patch(
                    "urllib.request.urlopen", side_effect=replies
                ) as transport:
                    results = evaluate_mahjong_tasks(tasks, agent, input_modes=[mode])
                self.assertEqual(transport.call_count, 4)
                self.assertTrue(all(result.success for result in results))
                self.assertEqual([result.metrics["llm_calls"] for result in results], [2, 2])
                with tempfile.TemporaryDirectory() as directory:
                    run_dir = write_mahjong_run(results, directory, "traces")
                    saved = [
                        json.loads(line)
                        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
                    ]
                for index, record in enumerate(saved):
                    trace = record["reasoning_trace"]
                    self.assertEqual(trace["architecture"], "cot")
                    self.assertEqual(trace["reasoning"], ["Incomplete draft 1", "Complete draft 2"][index])
                    self.assertEqual(trace["final"], final)
                    self.assertEqual(trace["stage_metrics"]["reasoning"]["token_usage"]["completion_tokens"], [1024, 80][index])
                    calls = record["completion_traces"]
                    self.assertEqual(len(calls), 2)
                    self.assertEqual(calls[0]["finish_reason"], ["length", "stop"][index])
                    self.assertEqual(calls[0]["content"], trace["reasoning"])
                    self.assertEqual(calls[0]["reasoning_content"], f"Native {index + 1}")
                    self.assertEqual(calls[0]["usage"]["completion_tokens_details"]["reasoning_tokens"], 7)
                    self.assertEqual(calls[0]["max_tokens"], 1024)
                    self.assertEqual(calls[1]["content"], final)
                    self.assertEqual(calls[1]["finish_reason"], "stop")
                    self.assertGreaterEqual(calls[0]["model_elapsed_seconds"], 0)
                    self.assertNotIn("test-only", json.dumps(record))

    def test_direct_and_tot_keep_their_outputs_and_capture_all_responses(self):
        task = next(
            task
            for task in load_mahjong_tasks("data/mahjong/tasks.jsonl")
            if task.id == "mj-easy-visible12-wait-008"
        )
        final = json.dumps({"winning_tiles": ["4s"], "live_winning_copies": {"4s": 4}})
        for agent_type, drafts in (
            (DirectAgent, []),
            (TreeOfThoughtAgent, ["Candidate 1", "Candidate 2", "Candidate 3"]),
        ):
            with self.subTest(agent=agent_type.name):
                client = OpenAICompatibleAgent(
                    model="qwen3.8-max",
                    base_url="https://example.invalid/v1",
                    api_key_env="TRACE_TEST_API_KEY",
                )
                agent = agent_type(client, ReasoningConfig(samples=3))
                with patch.dict(os.environ, {"TRACE_TEST_API_KEY": "test-only"}), patch(
                    "urllib.request.urlopen",
                    side_effect=[response(text) for text in [*drafts, final]],
                ):
                    result = evaluate_mahjong_tasks([task], agent)[0]
                self.assertTrue(result.success)
                self.assertEqual(result.raw_output, final)
                self.assertEqual(
                    [call["content"] for call in result.completion_traces],
                    [*drafts, final],
                )
                if drafts:
                    self.assertEqual(result.reasoning_trace["candidates"], drafts)

    def test_capture_does_not_record_other_calls_and_is_restored_after_error(self):
        client = OpenAICompatibleAgent(
            model="qwen3.8-max",
            base_url="https://example.invalid/v1",
            api_key_env="TRACE_TEST_API_KEY",
        )
        replies = [
            response("captured"),
            response("outside"),
            response("", finish_reason="length"),
            response("outside after error"),
        ]
        with patch.dict(os.environ, {"TRACE_TEST_API_KEY": "test-only"}), patch(
            "urllib.request.urlopen", side_effect=replies
        ):
            with client.capture_completion_traces() as captured:
                self.assertEqual(client.complete("first"), "captured")
            self.assertEqual(client.complete("second"), "outside")
            with self.assertRaisesRegex(RuntimeError, "empty message content"):
                with client.capture_completion_traces() as failed:
                    client.complete("third")
            self.assertEqual(client.complete("fourth"), "outside after error")
        self.assertEqual([call["content"] for call in captured], ["captured"])
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["finish_reason"], "length")


if __name__ == "__main__":
    unittest.main()
