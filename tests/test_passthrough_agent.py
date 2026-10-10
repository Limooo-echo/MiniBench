from copy import deepcopy
import unittest

from minibench.agents.passthrough import PassthroughAgent
from minibench.core.multimodal import ImageAttachment


class FakeClient:
    def __init__(self):
        self.calls = []

    def complete(
        self,
        prompt,
        *,
        system_prompt=None,
        temperature=None,
        max_tokens=None,
        json_mode=None,
        images=(),
    ):
        self.calls.append(
            {
                "kind": "prompt",
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_mode": json_mode,
                "images": images,
            }
        )
        return "response"

    def complete_messages(
        self,
        messages,
        *,
        system_prompt=None,
        temperature=None,
        max_tokens=None,
        json_mode=None,
        images=(),
    ):
        self.calls.append(
            {
                "kind": "messages",
                "messages": deepcopy(list(messages)),
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_mode": json_mode,
                "images": images,
            }
        )
        return "response"


class PassthroughAgentTests(unittest.TestCase):
    def test_generate_forwards_prompt_without_transforming_it(self):
        client = FakeClient()
        agent = PassthroughAgent(client)

        output = agent.generate("Original prompt", object())

        self.assertEqual(output, "response")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["prompt"], "Original prompt")
        self.assertIsNone(client.calls[0]["system_prompt"])

    def test_generate_multimodal_forwards_images(self):
        image = ImageAttachment(data=b"image", mime_type="image/png")
        client = FakeClient()
        agent = PassthroughAgent(client)

        agent.generate_multimodal("Prompt", object(), images=[image])

        self.assertEqual(client.calls[0]["images"], [image])

    def test_message_phases_preserve_history_and_visible_options(self):
        messages = [
            {"role": "system", "content": "Rules"},
            {"role": "user", "content": "Clue"},
        ]
        original = deepcopy(messages)
        client = FakeClient()
        agent = PassthroughAgent(client)

        for phase in ("intermediate", "final"):
            with self.subTest(phase=phase):
                output = agent.generate_messages_for_phase(
                    messages,
                    object(),
                    phase=phase,
                    temperature=0.25,
                    max_tokens=31,
                    json_mode=True,
                )
                self.assertEqual(output, "response")

        self.assertEqual(messages, original)
        self.assertEqual(len(client.calls), 2)
        for call in client.calls:
            self.assertEqual(call["messages"], original)
            self.assertEqual(call["temperature"], 0.25)
            self.assertEqual(call["max_tokens"], 31)
            self.assertTrue(call["json_mode"])

    def test_invalid_phase_fails_before_client_call(self):
        client = FakeClient()
        agent = PassthroughAgent(client)

        with self.assertRaisesRegex(ValueError, "Unsupported message phase"):
            agent.generate_messages_for_phase(
                [{"role": "user", "content": "Question"}],
                object(),
                phase="invalid",
            )

        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
