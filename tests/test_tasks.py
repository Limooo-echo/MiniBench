import unittest

from minibench.datasets.multiple_choice.dataset import load_tasks


REQUIRED_TAG_PREFIXES = (
    "format:",
    "turn:",
    "source:",
    "domain:",
    "skill:",
    "difficulty:",
)


class TaskSetTests(unittest.TestCase):
    def test_seed_task_count(self):
        self.assertEqual(len(load_tasks()), 50)

    def test_task_ids_are_unique_and_sequential(self):
        tasks = load_tasks()
        ids = [task.id for task in tasks]

        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids[0], "mb-choice-001")
        self.assertEqual(ids[-1], "mb-choice-050")

    def test_tasks_have_required_tag_groups(self):
        for task in load_tasks():
            tags = set(task.tags)
            for prefix in REQUIRED_TAG_PREFIXES:
                matches = [tag for tag in tags if tag.startswith(prefix)]
                self.assertEqual(
                    len(matches),
                    1,
                    f"{task.id} must have exactly one {prefix} tag",
                )


if __name__ == "__main__":
    unittest.main()
