from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from minibench.core.multimodal import ImageAttachment, summarize_paired_modes
from minibench.factory.providers import OpenAICompatibleAgent


PNG_STUB = b"\x89PNG\r\n\x1a\nmultimodal-test"


class ImageAttachmentTests(unittest.TestCase):
    def test_requires_exactly_one_source(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            ImageAttachment()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            ImageAttachment(path=Path("x.png"), data=PNG_STUB)

    def test_path_and_bytes_produce_data_urls(self):
        byte_attachment = ImageAttachment(data=PNG_STUB)
        self.assertTrue(byte_attachment.data_url().startswith("data:image/png;base64,"))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.png"
            path.write_bytes(PNG_STUB)
            path_attachment = ImageAttachment(path=path, detail="low")
            part = path_attachment.content_part()
        self.assertEqual(part["image_url"]["detail"], "low")
        self.assertTrue(part["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_rejects_unknown_mime(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            ImageAttachment(data=b"not-an-image", mime_type="text/plain")
        with self.assertRaisesRegex(RuntimeError, "unsupported"):
            ImageAttachment(data=b"not-an-image").resolved()

    def test_rejects_mime_that_disagrees_with_bytes(self):
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            ImageAttachment(data=PNG_STUB, mime_type="image/jpeg").resolved()


class MultimodalPayloadTests(unittest.TestCase):
    def setUp(self):
        self.agent = OpenAICompatibleAgent(
            model="test-model",
            base_url="https://example.com/v1",
            api_key_env="TEST_KEY",
        )

    def test_text_payload_is_unchanged(self):
        payload = self.agent.build_payload("Question?")
        self.assertEqual(payload["messages"][1]["content"], "Question?")

    def test_multiple_images_are_attached_to_last_user_message(self):
        images = [
            ImageAttachment(data=PNG_STUB, mime_type="image/png"),
            ImageAttachment(data=PNG_STUB, mime_type="image/png", detail="auto"),
        ]
        payload = self.agent.build_messages_payload(
            [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Ack"},
                {"role": "user", "content": "Inspect these"},
            ],
            images=images,
        )
        messages = payload["messages"]
        self.assertEqual(messages[1]["content"], "First")
        parts = messages[-1]["content"]
        self.assertEqual(parts[0], {"type": "text", "text": "Inspect these"})
        self.assertEqual([part["type"] for part in parts], ["text", "image_url", "image_url"])


class PairedSummaryTests(unittest.TestCase):
    def test_visual_gap_is_paired_and_deterministic(self):
        results = [
            {"source_task_id": "a", "input_mode": "text", "success": True},
            {"source_task_id": "a", "input_mode": "image", "success": False},
            {"source_task_id": "b", "input_mode": "text", "success": False},
            {"source_task_id": "b", "input_mode": "image", "success": False},
        ]
        first = summarize_paired_modes(results, bootstrap_samples=100, seed=7)
        second = summarize_paired_modes(results, bootstrap_samples=100, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["visual_gap"]["image"]["visual_gap"], 0.5)
        self.assertEqual(first["visual_gap"]["image"]["paired_total"], 2)


if __name__ == "__main__":
    unittest.main()
