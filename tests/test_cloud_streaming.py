"""Offline SSE and durable successful-call replay protocol tests."""
from dataclasses import MISSING, fields
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from minibench.core import streaming as runner
from minibench.core.agent import CompletionResult
from minibench.datasets.xiangqi.mate_in_one import MateInOneResult
from minibench.factory.providers import OpenAICompatibleClient
from minibench.core.rate_limit import RateGate


class ResumableXiangqiTests(unittest.TestCase):
    def client(self, path):
        return runner.JournalClient(
            path, model="unit-test-model", base_url="https://invalid.example/v1",
            api_key_env="UNIT_TEST_TRANSIENT_KEY", extra_body={"enable_thinking": True, "chat_template_kwargs": {"enable_thinking": True}},
        )

    def test_all_request_token_limits_are_omitted(self):
        with tempfile.TemporaryDirectory() as temporary:
            client = self.client(Path(temporary))
            for limit in (1, 64, 512, 4096, None):
                payload = client.build_payload("fixture", max_tokens=limit)
                self.assertNotIn("max_tokens", payload)
                self.assertNotIn("max_completion_tokens", payload)
                self.assertTrue(payload["enable_thinking"])
                self.assertTrue(payload["chat_template_kwargs"]["enable_thinking"])

    def test_call_resume_does_not_contact_server_or_store_credential(self):
        with tempfile.TemporaryDirectory() as temporary:
            journal = Path(temporary)
            result = CompletionResult(
                content='{"move":"a0a1"}', reasoning="fixture",
                usage={"total_tokens": 7}, elapsed_seconds=1,
            )
            with patch.dict(os.environ, {"UNIT_TEST_TRANSIENT_KEY": "unit-test-secret"}):
                with patch.object(runner.JournalClient, "_stream_completion_once",
                                  return_value=result) as transport:
                    first = self.client(journal).complete_result("same fixture")
                    second = self.client(journal).complete_result("same fixture")
                    self.assertEqual(first, second)
                    self.assertEqual(transport.call_count, 1)
                for file in journal.rglob("*.json"):
                    self.assertNotIn("unit-test-secret", file.read_text())

    def test_changed_resume_request_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            journal = Path(temporary)
            result = CompletionResult(content='{"move":"a0a1"}')
            with patch.dict(os.environ, {"UNIT_TEST_TRANSIENT_KEY": "fixture"}):
                with patch.object(runner.JournalClient, "_stream_completion_once",
                                  return_value=result) as transport:
                    self.client(journal).complete_result("original")
                    with self.assertRaisesRegex(RuntimeError, "Resume request differs"):
                        self.client(journal).complete_result("changed")
                    self.assertEqual(transport.call_count, 1)

    def test_h2_opponent_checkpoint_replays_exact_move(self):
        from minibench.datasets.xiangqi.variants.board import VariantBoard
        from minibench.datasets.xiangqi import history
        board = VariantBoard([[0] * 9 for _ in range(10)], [])
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(history, "_pikafish_analysis",
                              return_value=("d8d7", 120.0)) as analyse:
                with runner.replay_h2_engine(Path(temporary)):
                    first = history._pikafish_analysis(
                        None, board, -1, perspective_side=1, depth=16)
                with runner.replay_h2_engine(Path(temporary)):
                    second = history._pikafish_analysis(
                        None, board, -1, perspective_side=1, depth=16)
                self.assertEqual(first, second)
                self.assertEqual(analyse.call_count, 1)



class StreamingResponse(BytesIO):
    def __init__(self, body, media_type="text/event-stream; charset=utf-8"):
        super().__init__(body)
        self.headers = {"Content-Type": media_type}


class XiangqiStreamingTests(unittest.TestCase):
    def body(self, chunks, *, done=True):
        text = ": heartbeat\r\n\r\n" + "".join(
            "data: " + json.dumps(chunk, ensure_ascii=False) + "\r\n\r\n"
            for chunk in chunks
        )
        if done:
            text += "data: [DONE]\r\n\r\n"
        return text.encode("utf-8")

    def chunk(self, delta=None, *, finish=None, usage=None, response_id="fixture"):
        return {
            "id": response_id, "model": "qwen", "usage": usage,
            "choices": ([{"index": 0, "delta": delta or {},
                          "finish_reason": finish}] if delta is not None or finish else []),
        }

    def client(self, journal, events=None):
        return runner.JournalClient(
            journal, events.append if events is not None else None,
            model="qwen", base_url="https://invalid.example/v1",
            api_key_env="UNIT_TEST_TRANSIENT_KEY", extra_body={"enable_thinking": True, "chat_template_kwargs": {"enable_thinking": True}}, timeout=3600, max_retries=0,
        )

    def test_sse_framing_multiline_bom_and_comments(self):
        body = b'\xef\xbb\xbf: keepalive\r\ndata: {"x":\r\ndata: 1}\r\n\r\nevent: message\ndata: [DONE]'
        self.assertEqual(list(runner.sse_data(BytesIO(body))), ['{"x":\n1}', '[DONE]'])

    def test_complete_stream_preserves_reasoning_answer_and_usage_once(self):
        chunks = [
            self.chunk({"role": "assistant", "reasoning_content": "先看棋盘。"}),
            self.chunk({"reasoning_content": "再检查将军。"}, usage={"total_tokens": 9}),
            self.chunk({"content": '{"move":'}),
            self.chunk({"content": '"a0a1"}'}, finish="stop"),
            self.chunk(usage={"prompt_tokens": 5, "completion_tokens": 7,
                              "total_tokens": 12}),
        ]
        events = []
        with tempfile.TemporaryDirectory() as temporary:
            journal = Path(temporary)
            client = self.client(journal, events)
            with patch.dict(os.environ, {"UNIT_TEST_TRANSIENT_KEY": "fixture-secret"}):
                with patch("urllib.request.urlopen", return_value=StreamingResponse(self.body(chunks))) as network:
                    result = client.complete_result("unchanged fixture")
                    request = network.call_args.args[0]
                    sent = json.loads(request.data)
                    self.assertTrue(sent["stream"])
                    self.assertEqual(sent["stream_options"], {"include_usage": True})
                    self.assertNotIn("max_tokens", sent)
                    self.assertEqual(network.call_args.kwargs["timeout"], 3600)
                    self.assertEqual(network.call_count, 1)
                with patch("urllib.request.urlopen", side_effect=AssertionError("replay must stay offline")):
                    replay = self.client(journal).complete_result("unchanged fixture")
                self.assertEqual(replay, result)
            self.assertEqual(result.content, '{"move":"a0a1"}')
            self.assertEqual(result.reasoning, "先看棋盘。再检查将军。")
            self.assertEqual(client.metrics_snapshot()["token_usage"]["total_tokens"], 12)
            self.assertEqual(client.metrics_snapshot()["llm_calls"], 1)
            self.assertEqual(sum(e["kind"] == "new_call" for e in events), 1)
            partial = json.loads((journal / "attempts/0000-0000/partial.json").read_text())
            self.assertEqual(partial["status"], "completed")
            self.assertTrue(partial["done_received"])
            for path in journal.rglob("*"):
                if path.is_file():
                    self.assertNotIn("fixture-secret", path.read_text())

    def test_missing_usage_is_unknown_not_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            client = self.client(Path(temporary))
            body = self.body([self.chunk({"content": "answer"}, finish="stop")])
            with patch.dict(os.environ, {"UNIT_TEST_TRANSIENT_KEY": "fixture"}):
                with patch("urllib.request.urlopen", return_value=StreamingResponse(body)):
                    result = client.complete_result("fixture")
            self.assertIsNone(result.usage)
            self.assertFalse(result.metadata["usage_known"])
            self.assertEqual(client.metrics_snapshot()["usage_missing_calls"], 1)

    def test_interrupted_stream_is_saved_not_scored_or_replayed(self):
        first = self.body([self.chunk({"reasoning_content": "unfinished"})], done=False)
        complete = self.body([self.chunk({"content": "final"}, finish="stop")])
        with tempfile.TemporaryDirectory() as temporary:
            journal = Path(temporary)
            with patch.dict(os.environ, {"UNIT_TEST_TRANSIENT_KEY": "fixture"}):
                with patch("urllib.request.urlopen", return_value=StreamingResponse(first)) as network:
                    with self.assertRaisesRegex(RuntimeError, "partial output retained"):
                        self.client(journal).complete_result("fixture")
                    self.assertEqual(network.call_count, 1)
                self.assertFalse((journal / "0000.json").exists())
                partial = json.loads((journal / "attempts/0000-0000/partial.json").read_text())
                self.assertEqual(partial["reasoning"], "unfinished")
                self.assertEqual(partial["status"], "interrupted")
                with patch("urllib.request.urlopen", return_value=StreamingResponse(complete)) as network:
                    self.assertEqual(self.client(journal).complete_result("fixture").content, "final")
                    self.assertEqual(network.call_count, 1)
                self.assertTrue((journal / "attempts/0000-0001/request.json").exists())
                self.assertEqual(json.loads((journal / "attempts/0000-0000/partial.json").read_text()), partial)

    def test_abnormal_streams_fail_closed_without_retry(self):
        cases = {
            "missing_finish": self.body([self.chunk({"content": "answer"})]),
            "length": self.body([self.chunk({"content": "cut"}, finish="length")]),
            "reasoning_only": self.body([self.chunk({"reasoning": "thought"}, finish="stop")]),
            "server_error": self.body([{"error": {"code": "quota_exceeded"}}]),
            "changed_id": self.body([self.chunk({"content": "a"}),
                                     self.chunk({"content": "b"}, finish="stop", response_id="other")]),
            "bad_json": b'data: not-json\n\ndata: [DONE]\n\n',
        }
        for label, body in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                journal = Path(temporary)
                with patch.dict(os.environ, {"UNIT_TEST_TRANSIENT_KEY": "fixture"}):
                    with patch("urllib.request.urlopen", return_value=StreamingResponse(body)) as network:
                        with self.assertRaises((RuntimeError, json.JSONDecodeError)):
                            self.client(journal).complete_result("fixture")
                        self.assertEqual(network.call_count, 1)
                self.assertFalse((journal / "0000.json").exists())
                self.assertTrue((journal / "failed_attempts.jsonl").exists())

    def test_non_sse_response_does_not_trigger_fallback_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, {"UNIT_TEST_TRANSIENT_KEY": "fixture"}):
                with patch("urllib.request.urlopen", return_value=StreamingResponse(b'{}', "application/json")) as network:
                    with self.assertRaisesRegex(RuntimeError, "silent transport fallback"):
                        self.client(Path(temporary)).complete_result("fixture")
                    self.assertEqual(network.call_count, 1)

    def test_failed_answer_keeps_received_usage_and_manual_retry_counts_both(self):
        failed = self.body([
            self.chunk({"reasoning_content": "unfinished thought"}, finish="stop"),
            self.chunk(usage={"total_tokens": 10944}),
        ])
        success = self.body([
            self.chunk({"content": "answer"}, finish="stop"),
            self.chunk(usage={"total_tokens": 7}),
        ])
        with tempfile.TemporaryDirectory() as temporary:
            journal = Path(temporary)
            gate = RateGate({"default": {"reserve_tokens": 32768}})
            with patch.dict(os.environ, {"UNIT_TEST_TRANSIENT_KEY": "fixture"}):
                client = self.client(journal)
                client.gate = gate
                with patch("urllib.request.urlopen", return_value=StreamingResponse(failed)):
                    with self.assertRaisesRegex(RuntimeError, "no final answer"):
                        client.complete_result("fixture")
                self.assertEqual(gate.snapshot()["default"]["accounted_or_reserved_tokens"], 10944)
                self.assertEqual(gate.snapshot()["default"]["unknown_usage_calls"], 0)
                client = self.client(journal)
                client.gate = gate
                with patch("urllib.request.urlopen", return_value=StreamingResponse(success)):
                    client.complete_result("fixture")
                with patch("urllib.request.urlopen", side_effect=AssertionError("replay is offline")):
                    self.client(journal).complete_result("fixture")
            attempts = list(runner.call_attempts(journal))
            self.assertEqual([a["status"] for a in attempts], ["failed", "success"])
            self.assertEqual(sum(a["usage"]["total_tokens"] for a in attempts), 10951)
            self.assertEqual(gate.snapshot()["default"]["network_calls"], 2)

    def test_started_without_partial_is_unknown_and_queued_is_not_a_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            journal = Path(temporary)
            started = journal / "attempts/0000-0000"
            queued = journal / "attempts/0001-0000"
            started.mkdir(parents=True)
            queued.mkdir()
            runner.atomic_write_json(started / "started.json", {"at": runner.now()})
            runner.atomic_write_json(queued / "request.json", {"messages": []})
            attempts = list(runner.call_attempts(journal))
            self.assertEqual(len(attempts), 1)
            self.assertIsNone(attempts[0]["usage"])

    def test_usage_arriving_before_malformed_chunk_is_still_accounted(self):
        body = self.body([self.chunk(usage={"total_tokens": 19})], done=False)
        body += b'data: not-json\n\n'
        with tempfile.TemporaryDirectory() as temporary:
            journal = Path(temporary)
            client = self.client(journal)
            client.gate = RateGate({"default": {}})
            with patch.dict(os.environ, {"UNIT_TEST_TRANSIENT_KEY": "fixture"}):
                with patch("urllib.request.urlopen", return_value=StreamingResponse(body)):
                    with self.assertRaises(json.JSONDecodeError):
                        client.complete_result("fixture")
            self.assertEqual(client.gate.snapshot()["default"]["accounted_or_reserved_tokens"], 19)
            self.assertEqual(list(runner.call_attempts(journal))[0]["usage"]["total_tokens"], 19)


if __name__ == "__main__":
    unittest.main()
