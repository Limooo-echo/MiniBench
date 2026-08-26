from __future__ import annotations

import unittest

from minibench.core.multimodal import ImageAttachment
from minibench.agents.passthrough import PassthroughAgent
from minibench.datasets.one_stroke.dataset import load_one_stroke_tasks
from minibench.datasets.one_stroke.prompting import (
    ONE_STROKE_SYSTEM_PROMPT,
    build_one_stroke_prompt,
    history_system_prompt,
)
from minibench.factory.agents import make_agent_from_config


class OneStrokePromptIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        agent = make_agent_from_config(
            {"name": "passthrough", "max_tokens": 1024},
            {"name": "qwen", "json_mode": True},
            system_prompt=ONE_STROKE_SYSTEM_PROMPT,
        )
        if not isinstance(agent, PassthroughAgent):
            raise AssertionError("expected a PassthroughAgent")
        cls.client = agent.client

    def test_a4_unsolvable_schema_survives_real_payload_construction(self):
        tasks = load_one_stroke_tasks("data/one_stroke/a4_multimodal.jsonl")
        task = next(item for item in tasks if not item.solution_exists)
        prompt = build_one_stroke_prompt(task, input_mode="challenge_image")
        image = ImageAttachment(path=task.image_variants["challenge"])

        payload = self.client.build_payload(prompt, images=(image,))

        messages = payload["messages"]
        self.assertIsInstance(messages, list)
        system_text = messages[0]["content"]
        self.assertIsInstance(system_text, str)
        self.assertIn(
            "schema requested in the task for both solvable and unsolvable cases",
            system_text,
        )
        self.assertNotIn('{"solvable":false}', system_text)

        user_content = messages[1]["content"]
        self.assertIsInstance(user_content, list)
        user_text = user_content[0]["text"]
        self.assertIn("all four fields", user_text)
        self.assertIn('"solvable":true,"path":["A","B"]', user_text)
        self.assertIn(
            "If no valid one-stroke path exists, set solvable to false and "
            "path to null.",
            user_text,
        )
        self.assertEqual(user_content[1]["type"], "image_url")
        self.assertEqual(user_content[1]["image_url"]["detail"], "high")
        self.assertTrue(
            user_content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        )

    def test_a1_and_a2_keep_minimal_unsolvable_object_in_task_prompt(self):
        cases = (
            ("data/one_stroke/a1_direct.jsonl", False),
            ("data/one_stroke/a2_rule_condition.jsonl", True),
        )
        for dataset_path, expects_edge_path in cases:
            with self.subTest(dataset=dataset_path):
                tasks = load_one_stroke_tasks(dataset_path)
                task = next(item for item in tasks if not item.solution_exists)
                prompt = build_one_stroke_prompt(task)

                payload = self.client.build_payload(prompt)

                messages = payload["messages"]
                system_text = messages[0]["content"]
                user_text = messages[1]["content"]
                self.assertNotIn('{"solvable":false}', system_text)
                self.assertIn(
                    'If no one-stroke path exists, return only JSON: '
                    '{"solvable":false}.',
                    user_text,
                )
                self.assertEqual('"edge_path"' in user_text, expects_edge_path)

    def test_a3_history_system_prompt_omits_task_metadata(self):
        tasks = load_one_stroke_tasks("data/one_stroke/a3_history.jsonl")
        task = next(item for item in tasks if item.difficulty == "hard")

        prompt = history_system_prompt(task, "incremental_state")

        self.assertNotIn("Task ID:", prompt)
        self.assertNotIn("Difficulty:", prompt)
        self.assertNotIn(task.id, prompt)
        self.assertIn(f"Initial vertex: {task.start}", prompt)
        self.assertIn("Static edge list:", prompt)


if __name__ == "__main__":
    unittest.main()
