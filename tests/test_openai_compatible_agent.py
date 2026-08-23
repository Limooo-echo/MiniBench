import io
import json
import socket
from unittest.mock import patch
import unittest
import urllib.error

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


class FakeRawHTTPResponse(FakeHTTPResponse):
    def read(self):
        return self.payload


def make_http_error(status_code, *, retry_after=None):
    headers = {}
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return urllib.error.HTTPError(
        "https://example.com/v1/chat/completions",
        status_code,
        "request failed",
        headers,
        io.BytesIO(b'{"error":"temporary"}'),
    )


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

        self.assertEqual(model, "qwen3.8-max")
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

    def test_messages_payload_preserves_explicit_role_order(self):
        agent = OpenAICompatibleAgent(
            model="test-model",
            base_url="https://example.com/v1",
            api_key_env="TEST_KEY",
            default_system_prompt="Should not replace explicit system history.",
        )
        messages = [
            {"role": "system", "content": "Task rules."},
            {"role": "user", "content": "Clue 1"},
            {"role": "assistant", "content": '{"state":"A"}'},
            {"role": "user", "content": "Clue 2"},
        ]

        payload = agent.build_messages_payload(messages)

        self.assertEqual(payload["messages"], messages)

    def test_messages_payload_prepends_default_system_when_missing(self):
        agent = OpenAICompatibleAgent(
            model="test-model",
            base_url="https://example.com/v1",
            api_key_env="TEST_KEY",
            default_system_prompt="Shared task rules.",
        )

        payload = agent.build_messages_payload(
            [{"role": "user", "content": "Clue 1"}]
        )

        self.assertEqual(
            [message["role"] for message in payload["messages"]],
            ["system", "user"],
        )
        self.assertEqual(payload["messages"][0]["content"], "Shared task rules.")

    def test_complete_messages_sends_real_chat_history(self):
        agent = OpenAICompatibleAgent(
            model="test-model",
            base_url="https://example.com/v1",
            api_key_env="TEST_KEY",
        )
        response = {"choices": [{"message": {"content": '{"ok":true}'}}]}
        messages = [
            {"role": "system", "content": "Rules"},
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "State 1"},
            {"role": "user", "content": "Turn 2"},
        ]

        with patch.dict("os.environ", {"TEST_KEY": "test-key"}):
            with patch("urllib.request.urlopen", return_value=FakeHTTPResponse(response)) as urlopen:
                output = agent.complete_messages(messages)

        request = urlopen.call_args.args[0]
        sent_payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(output, '{"ok":true}')
        self.assertEqual(sent_payload["messages"], messages)

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

    def test_retries_transient_failures_with_exponential_backoff_and_metrics(self):
        agent = OpenAICompatibleAgent(
            model="test-model",
            base_url="https://example.com/v1",
            api_key_env="TEST_KEY",
            max_retries=3,
            retry_initial_backoff_seconds=1.0,
            retry_max_backoff_seconds=10.0,
        )
        payload = {
            "choices": [{"message": {"content": '{"answer":"C"}'}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
            },
        }
        effects = [
            make_http_error(503),
            urllib.error.URLError("connection reset"),
            socket.timeout("timed out"),
            FakeHTTPResponse(payload),
        ]

        with patch.dict("os.environ", {"TEST_KEY": "test-key"}):
            with patch("urllib.request.urlopen", side_effect=effects) as urlopen:
                with patch("minibench.factory.providers.sleep") as retry_sleep:
                    output = agent.complete("Question?")

        self.assertEqual(output, '{"answer":"C"}')
        self.assertEqual(urlopen.call_count, 4)
        self.assertEqual(
            [item.args[0] for item in retry_sleep.call_args_list],
            [1.0, 2.0, 4.0],
        )
        metrics = agent.metrics_snapshot()
        self.assertEqual(metrics["llm_calls"], 4)
        self.assertEqual(metrics["usage_missing_calls"], 3)
        self.assertEqual(metrics["token_usage"]["total_tokens"], 14)

    def test_retries_only_selected_http_statuses(self):
        response = {"choices": [{"message": {"content": '{"ok":true}'}}]}
        for status_code in (408, 429, 500, 599):
            with self.subTest(status_code=status_code):
                agent = OpenAICompatibleAgent(
                    model="test-model",
                    base_url="https://example.com/v1",
                    api_key_env="TEST_KEY",
                    max_retries=1,
                    retry_initial_backoff_seconds=0.0,
                    retry_max_backoff_seconds=0.0,
                )
                effects = [make_http_error(status_code), FakeHTTPResponse(response)]
                with patch.dict("os.environ", {"TEST_KEY": "test-key"}):
                    with patch("urllib.request.urlopen", side_effect=effects) as urlopen:
                        output = agent.complete("Question?")

                self.assertEqual(output, '{"ok":true}')
                self.assertEqual(urlopen.call_count, 2)

    def test_retry_after_is_respected_and_capped(self):
        agent = OpenAICompatibleAgent(
            model="test-model",
            base_url="https://example.com/v1",
            api_key_env="TEST_KEY",
            max_retries=1,
            retry_initial_backoff_seconds=1.0,
            retry_max_backoff_seconds=5.0,
        )
        response = {"choices": [{"message": {"content": '{"ok":true}'}}]}
        effects = [
            make_http_error(429, retry_after=120),
            FakeHTTPResponse(response),
        ]

        with patch.dict("os.environ", {"TEST_KEY": "test-key"}):
            with patch("urllib.request.urlopen", side_effect=effects):
                with patch("minibench.factory.providers.sleep") as retry_sleep:
                    output = agent.complete("Question?")

        self.assertEqual(output, '{"ok":true}')
        retry_sleep.assert_called_once_with(5.0)

    def test_max_retries_are_in_addition_to_the_initial_attempt(self):
        agent = OpenAICompatibleAgent(
            model="test-model",
            base_url="https://example.com/v1",
            api_key_env="TEST_KEY",
            max_retries=3,
            retry_initial_backoff_seconds=1.0,
            retry_max_backoff_seconds=10.0,
        )
        effects = [urllib.error.URLError("offline") for _ in range(4)]

        with patch.dict("os.environ", {"TEST_KEY": "test-key"}):
            with patch("urllib.request.urlopen", side_effect=effects) as urlopen:
                with patch("minibench.factory.providers.sleep") as retry_sleep:
                    with self.assertRaisesRegex(RuntimeError, "offline"):
                        agent.complete("Question?")

        self.assertEqual(urlopen.call_count, 4)
        self.assertEqual(
            [item.args[0] for item in retry_sleep.call_args_list],
            [1.0, 2.0, 4.0],
        )
        metrics = agent.metrics_snapshot()
        self.assertEqual(metrics["llm_calls"], 4)
        self.assertEqual(metrics["usage_missing_calls"], 4)

    def test_non_retryable_http_error_is_not_retried(self):
        agent = OpenAICompatibleAgent(
            model="test-model",
            base_url="https://example.com/v1",
            api_key_env="TEST_KEY",
            max_retries=3,
        )

        with patch.dict("os.environ", {"TEST_KEY": "test-key"}):
            with patch(
                "urllib.request.urlopen", side_effect=make_http_error(400)
            ) as urlopen:
                with patch("minibench.factory.providers.sleep") as retry_sleep:
                    with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
                        agent.complete("Question?")

        self.assertEqual(urlopen.call_count, 1)
        retry_sleep.assert_not_called()
        metrics = agent.metrics_snapshot()
        self.assertEqual(metrics["llm_calls"], 1)
        self.assertEqual(metrics["usage_missing_calls"], 1)

    def test_empty_content_is_not_retried(self):
        agent = OpenAICompatibleAgent(
            model="test-model",
            base_url="https://example.com/v1",
            api_key_env="TEST_KEY",
            max_retries=3,
        )
        response = {
            "choices": [
                {"finish_reason": "stop", "message": {"content": ""}}
            ]
        }

        with patch.dict("os.environ", {"TEST_KEY": "test-key"}):
            with patch(
                "urllib.request.urlopen", return_value=FakeHTTPResponse(response)
            ) as urlopen:
                with self.assertRaisesRegex(RuntimeError, "empty message content"):
                    agent.complete("Question?")

        self.assertEqual(urlopen.call_count, 1)
        self.assertEqual(agent.metrics_snapshot()["llm_calls"], 1)

    def test_invalid_json_is_not_retried_but_counts_the_attempt(self):
        agent = OpenAICompatibleAgent(
            model="test-model",
            base_url="https://example.com/v1",
            api_key_env="TEST_KEY",
            max_retries=3,
        )

        with patch.dict("os.environ", {"TEST_KEY": "test-key"}):
            with patch(
                "urllib.request.urlopen",
                return_value=FakeRawHTTPResponse(b"not-json"),
            ) as urlopen:
                with self.assertRaises(json.JSONDecodeError):
                    agent.complete("Question?")

        self.assertEqual(urlopen.call_count, 1)
        metrics = agent.metrics_snapshot()
        self.assertEqual(metrics["llm_calls"], 1)
        self.assertEqual(metrics["usage_missing_calls"], 1)

    def test_generate_messages_for_phase_delegates_to_message_generation(self):
        agent = OpenAICompatibleAgent(
            model="test-model",
            base_url="https://example.com/v1",
            api_key_env="TEST_KEY",
        )
        messages = [{"role": "user", "content": "Clue 1"}]

        with patch.object(
            agent, "generate_messages", return_value='{"ok":true}'
        ) as generate_messages:
            output = agent.generate_messages_for_phase(
                messages,
                None,
                phase="intermediate",
                max_tokens=32,
                json_mode=True,
            )

        self.assertEqual(output, '{"ok":true}')
        generate_messages.assert_called_once_with(
            messages,
            None,
            temperature=None,
            max_tokens=32,
            json_mode=True,
        )

    def test_generate_messages_for_phase_rejects_unknown_phase(self):
        agent = OpenAICompatibleAgent(
            model="test-model",
            base_url="https://example.com/v1",
            api_key_env="TEST_KEY",
        )

        with patch.object(agent, "generate_messages") as generate_messages:
            with self.assertRaisesRegex(ValueError, "Unsupported message phase"):
                agent.generate_messages_for_phase(
                    [{"role": "user", "content": "Clue 1"}],
                    None,
                    phase="unknown",
                )

        generate_messages.assert_not_called()


if __name__ == "__main__":
    unittest.main()
