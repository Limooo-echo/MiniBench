import json
from unittest.mock import patch
import unittest

from minibench.factory.providers import OpenAICompatibleAgent, resolve_provider


class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class OpenAICompatibleAgentTests(unittest.TestCase):
    def test_deepseek_provider_defaults(self):
        model, base_url, api_key_env = resolve_provider(
            "deepseek",
            model=None,
            base_url=None,
            api_key_env=None,
        )

        self.assertEqual(model, "deepseek-v4-flash")
        self.assertEqual(base_url, "https://api.deepseek.com")
        self.assertEqual(api_key_env, "DEEPSEEK_API_KEY")

    def test_qwen_provider_defaults(self):
        model, base_url, api_key_env = resolve_provider(
            "qwen",
            model=None,
            base_url=None,
            api_key_env=None,
        )

        self.assertEqual(model, "qwen3.6-plus")
        self.assertEqual(base_url, "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.assertEqual(api_key_env, "DASHSCOPE_API_KEY")

    def test_siliconflow_provider_requires_model(self):
        with self.assertRaisesRegex(ValueError, "siliconflow provider requires --model"):
            resolve_provider(
                "siliconflow",
                model=None,
                base_url=None,
                api_key_env=None,
            )

    def test_siliconflow_provider_uses_model_and_defaults(self):
        model, base_url, api_key_env = resolve_provider(
            "siliconflow",
            model="Qwen/Qwen3-32B",
            base_url=None,
            api_key_env=None,
        )

        self.assertEqual(model, "Qwen/Qwen3-32B")
        self.assertEqual(base_url, "https://api.siliconflow.cn/v1")
        self.assertEqual(api_key_env, "SILICONFLOW_API_KEY")

    def test_endpoint_appends_chat_completions(self):
        agent = OpenAICompatibleAgent(
            model="test-model",
            base_url="https://example.com/v1",
            api_key_env="TEST_KEY",
        )

        self.assertEqual(agent.endpoint, "https://example.com/v1/chat/completions")

    def test_endpoint_accepts_full_chat_completions_url(self):
        agent = OpenAICompatibleAgent(
            model="test-model",
            base_url="https://example.com/v1/chat/completions",
            api_key_env="TEST_KEY",
        )

        self.assertEqual(agent.endpoint, "https://example.com/v1/chat/completions")

    def test_payload_uses_chat_completions_shape(self):
        agent = OpenAICompatibleAgent(
            model="test-model",
            base_url="https://example.com/v1",
            api_key_env="TEST_KEY",
            json_mode=True,
            extra_body={"reasoning_effort": "high"},
        )

        payload = agent.build_payload("Question?")

        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["messages"][1]["content"], "Question?")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["reasoning_effort"], "high")

    def test_payload_combines_default_and_phase_system_prompts(self):
        agent = OpenAICompatibleAgent(
            model="test-model",
            base_url="https://example.com/v1",
            api_key_env="TEST_KEY",
            default_system_prompt="Mahjong rules prompt.",
        )

        payload = agent.build_payload(
            "Question?",
            system_prompt="Finalize as JSON.",
        )

        system_content = payload["messages"][0]["content"]
        self.assertIn("Mahjong rules prompt.", system_content)
        self.assertIn("Finalize as JSON.", system_content)

    def test_complete_uses_reasoning_content_when_visible_content_is_empty(self):
        agent = OpenAICompatibleAgent(
            model="test-model",
            base_url="https://example.com/v1",
            api_key_env="TEST_KEY",
        )
        payload = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "content": "",
                        "reasoning_content": "answer: C",
                    },
                }
            ]
        }

        with patch.dict("os.environ", {"TEST_KEY": "test-key"}):
            with patch("urllib.request.urlopen", return_value=FakeHTTPResponse(payload)):
                output = agent.complete("Question?")

        self.assertEqual(output, "answer: C")

    def test_complete_records_usage_metrics(self):
        agent = OpenAICompatibleAgent(
            model="test-model",
            base_url="https://example.com/v1",
            api_key_env="TEST_KEY",
        )
        payload = {
            "choices": [{"message": {"content": '{"answer":"C"}'}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
                "prompt_tokens_details": {"cached_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        }

        with patch.dict("os.environ", {"TEST_KEY": "test-key"}):
            with patch("urllib.request.urlopen", return_value=FakeHTTPResponse(payload)):
                output = agent.complete("Question?")

        metrics = agent.metrics_snapshot()
        self.assertEqual(output, '{"answer":"C"}')
        self.assertEqual(metrics["llm_calls"], 1)
        self.assertEqual(metrics["usage_missing_calls"], 0)
        self.assertEqual(metrics["token_usage"]["prompt_tokens"], 10)
        self.assertEqual(metrics["token_usage"]["completion_tokens"], 4)
        self.assertEqual(metrics["token_usage"]["total_tokens"], 14)
        self.assertEqual(metrics["token_usage"]["cached_tokens"], 3)
        self.assertEqual(metrics["token_usage"]["reasoning_tokens"], 2)
        self.assertGreaterEqual(metrics["model_elapsed_seconds"], 0.0)

    def test_complete_records_missing_usage_metrics(self):
        agent = OpenAICompatibleAgent(
            model="test-model",
            base_url="https://example.com/v1",
            api_key_env="TEST_KEY",
        )
        payload = {
            "choices": [{"message": {"content": '{"answer":"C"}'}}],
        }

        with patch.dict("os.environ", {"TEST_KEY": "test-key"}):
            with patch("urllib.request.urlopen", return_value=FakeHTTPResponse(payload)):
                agent.complete("Question?")

        metrics = agent.metrics_snapshot()
        self.assertEqual(metrics["llm_calls"], 1)
        self.assertEqual(metrics["usage_missing_calls"], 1)
        self.assertEqual(metrics["token_usage"]["total_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
