import unittest

from minibench.agents import OpenAICompatibleAgent, resolve_provider


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


if __name__ == "__main__":
    unittest.main()
