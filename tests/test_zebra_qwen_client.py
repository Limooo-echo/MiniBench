from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import urllib.error

from minibench.agents.direct import DirectAgent
from minibench.core.agent import ReasoningConfig
from minibench.core.runtime import StrictJSONObjectError
from minibench.datasets.zebra.qwen_client import (
    ZebraQwenClient, ZEBRA_FINAL_SYSTEM_PROMPT, ZEBRA_FINAL_USER_PROMPT,
)
from minibench.factory.providers import OpenAICompatibleClient


class Stream(BytesIO):
    headers = {"Content-Type": "text/event-stream"}


def stream(content='{"reasoning":"Done","solution":{}}', *, reasoning="draft", finish="stop", done=True):
    chunks = [
        {"id": "test", "model": "qwen3.8-flash", "choices": [
            {"index": 0, "delta": {"reasoning_content": reasoning}, "finish_reason": None}]},
        {"id": "test", "model": "qwen3.8-flash", "choices": [
            {"index": 0, "delta": {"content": content}, "finish_reason": finish}]},
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}},
    ]
    text = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
    return Stream((text + ("data: [DONE]\n\n" if done else "")).encode())


def abort_stream(code="invalid_parameter_error"):
    error = {"code": code, "message": (
        "<400> InternalError.Algo.InvalidParameter: Model output became abnormal "
        "while generating a JSON response for response_format; partial output "
        "may be incomplete or invalid JSON. Please retry."
    )}
    chunks = [
        {"choices": [{"index": 0, "delta": {"reasoning_content": "partial reasoning"}, "finish_reason": None}]},
        {"error": error},
    ]
    return Stream("".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks).encode())


class ZebraQwenClientTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.environment = patch.dict(os.environ, {"UNIT_TEST_ZEBRA_KEY": "secret-fixture-never-save"})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.directory.cleanup()

    def client(self, **kwargs):
        return ZebraQwenClient(self.root, base_url="https://invalid.example/v1",
                               api_key_env="UNIT_TEST_ZEBRA_KEY", **kwargs)

    def test_budget_prompt_replay_and_secret_free_evidence(self):
        gate = Mock()
        client = self.client(before_request=gate)
        with patch("urllib.request.urlopen", side_effect=[stream("partial", finish="length"), stream()]) as network:
            result = client.complete_result("Zebra puzzle and JSON schema")
        requests = [json.loads(call.args[0].data) for call in network.call_args_list]
        first, final = requests
        self.assertEqual((first["max_tokens"], first["thinking_budget"], first["enable_thinking"]), (7168, 7168, True))
        self.assertEqual((final["max_tokens"], final["enable_thinking"]), (1024, False))
        self.assertNotIn("thinking_budget", final)
        self.assertEqual(first["messages"][1], final["messages"][1])
        self.assertIn("complete solution grid", final["messages"][-1]["content"])
        self.assertIn("exact attribute names", final["messages"][0]["content"])
        self.assertEqual(result.usage["total_tokens"], 24)
        self.assertEqual(client.metrics_snapshot()["llm_calls"], 2)
        self.assertEqual(gate.call_count, 2)
        all_text = "\n".join(p.read_text(encoding="utf-8") for p in self.root.rglob("*.json*"))
        self.assertNotIn("secret-fixture-never-save", all_text)
        self.assertNotIn("Authorization", all_text)
        with patch.dict(os.environ, {}, clear=True), patch("urllib.request.urlopen", side_effect=AssertionError("must replay")):
            replay = self.client(before_request=gate).complete_result("Zebra puzzle and JSON schema")
        self.assertTrue(replay.metadata["first"]["replayed"])
        self.assertTrue(replay.metadata["second"]["replayed"])
        self.assertEqual(replay.content, result.content)
        self.assertEqual(gate.call_count, 2)

    def test_resume_rejects_prompt_and_endpoint_drift_before_network(self):
        with patch("urllib.request.urlopen", side_effect=[stream(), stream()]):
            self.client().complete_result("original JSON puzzle")
        with patch("urllib.request.urlopen", side_effect=AssertionError("must reject")):
            with self.assertRaisesRegex(ValueError, "Resume request differs"):
                self.client().complete_result("changed JSON puzzle")
            with self.assertRaisesRegex(ValueError, "Resume request differs"):
                ZebraQwenClient(self.root, base_url="https://other.invalid/v1").complete_result("original JSON puzzle")

    def test_incomplete_first_is_not_scored_or_replayed(self):
        client = self.client()
        with patch("urllib.request.urlopen", return_value=stream(done=False)) as network:
            with self.assertRaisesRegex(RuntimeError, "without.*DONE"):
                client.complete_result("JSON puzzle")
        self.assertEqual(network.call_count, 1)
        self.assertFalse((self.root / "00-thinking.json").exists())
        self.assertFalse((self.root / "01-final.json").exists())
        partial = json.loads(next(self.root.glob("attempts/*/partial.json")).read_text())
        self.assertEqual(partial["status"], "interrupted")
        self.assertFalse(partial["done_received"])
        self.assertEqual(client.metrics_snapshot()["usage_missing_calls"], 1)

    def test_second_failure_replays_only_successful_first(self):
        with patch("urllib.request.urlopen", side_effect=[stream(), urllib.error.URLError("offline")]) as network:
            with self.assertRaisesRegex(RuntimeError, "connection failed"):
                self.client().complete_result("JSON puzzle")
        self.assertEqual(network.call_count, 2)
        self.assertTrue((self.root / "00-thinking.json").exists())
        self.assertFalse((self.root / "01-final.json").exists())
        with patch("urllib.request.urlopen", return_value=stream()) as resumed:
            result = self.client().complete_result("JSON puzzle")
        self.assertEqual(resumed.call_count, 1)
        self.assertTrue(result.metadata["first"]["replayed"])

    def test_exact_provider_abort_is_retained_and_finalized(self):
        with patch("urllib.request.urlopen", side_effect=[abort_stream(), stream()]) as network:
            result = self.client().complete_result("JSON puzzle")
        self.assertEqual(network.call_count, 2)
        self.assertIsNone(result.usage)
        saved = json.loads((self.root / "00-thinking.json").read_text())
        self.assertEqual(saved["status"], "provider_json_abort")
        self.assertIsNone(saved["response"]["usage"])
        with patch("urllib.request.urlopen", side_effect=AssertionError("must replay")):
            replay = self.client().complete_result("JSON puzzle")
        self.assertTrue(replay.metadata["first"]["provider_json_abort"])

    def test_other_provider_errors_do_not_fall_through_to_final(self):
        with patch("urllib.request.urlopen", return_value=abort_stream("insufficient_quota")) as network:
            with self.assertRaisesRegex(RuntimeError, "Provider stream error"):
                self.client().complete_result("JSON puzzle")
        self.assertEqual(network.call_count, 1)
        self.assertFalse((self.root / "00-thinking.json").exists())

    def test_direct_agent_keeps_malformed_final_for_scoring_without_repair(self):
        config = ReasoningConfig(final_max_tokens=8192, max_format_repairs=0, trace="full")
        agent = DirectAgent(self.client(), config)
        with patch("urllib.request.urlopen", side_effect=[stream("", finish="length"), stream('{"solution":', finish="length")]) as network:
            with self.assertRaises(StrictJSONObjectError) as captured:
                agent.generate("Return JSON for this Zebra puzzle", None)
        self.assertEqual(captured.exception.raw_output, '{"solution":')
        self.assertEqual(network.call_count, 2)
        final = json.loads((self.root / "01-final.json").read_text())
        self.assertEqual(final["response"]["finish_reason"], "length")
        self.assertEqual(agent.last_run.metrics["llm_calls"], 2)

    def test_completed_empty_final_is_saved_and_returned_as_model_failure(self):
        with patch("urllib.request.urlopen", side_effect=[stream(""), stream("", reasoning="")]):
            result = self.client().complete_result("JSON puzzle")
        self.assertEqual(result.content, "")
        self.assertTrue((self.root / "01-final.json").exists())

    def test_generic_provider_and_other_tasks_are_unchanged(self):
        generic = OpenAICompatibleClient(model="other", base_url="https://invalid.example", api_key_env="UNUSED")
        original = generic.build_payload("other task")
        self.assertFalse(original["stream"])
        self.assertNotIn("enable_thinking", original)
        self.assertNotIn("Zebra", original["messages"][0]["content"])
        for unwanted in ("board", "move", "棋盘", "走法"):
            self.assertNotIn(unwanted, ZEBRA_FINAL_SYSTEM_PROMPT + ZEBRA_FINAL_USER_PROMPT)

    def test_wrong_budget_and_corrupt_journal_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            self.client().complete_result("JSON puzzle", max_tokens=4096)
        with patch("urllib.request.urlopen", side_effect=[stream(), stream()]):
            self.client().complete_result("JSON puzzle")
        saved = json.loads((self.root / "00-thinking.json").read_text())
        saved["response"]["content"] = "tampered"
        (self.root / "00-thinking.json").write_text(json.dumps(saved))
        with patch("urllib.request.urlopen", side_effect=AssertionError("must reject")):
            with self.assertRaisesRegex(ValueError, "Corrupt stage journal"):
                self.client().complete_result("JSON puzzle")


if __name__ == "__main__":
    unittest.main()
