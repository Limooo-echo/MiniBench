import importlib.util
from pathlib import Path
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "import_zebra_tasks.py"
SPEC = importlib.util.spec_from_file_location("import_zebra_tasks", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ZebraImportTests(unittest.TestCase):
    def test_rejects_public_masked_solutions(self):
        public = [
            {
                "id": "masked",
                "size": "2*2",
                "puzzle": "Puzzle",
                "solution": {
                    "header": ["House", "Name", "Drink"],
                    "rows": [["___", "___", "___"], ["___", "___", "___"]],
                },
            }
        ]

        with self.assertRaisesRegex(ValueError, "masked solutions"):
            MODULE.convert_official_records(public)

    def test_converts_gold_and_selects_all_difficulties(self):
        official = []
        for index, size in enumerate(("2*2", "4*4", "6*6"), start=1):
            task_id = f"official-{index}"
            official.append(
                {
                    "id": task_id,
                    "size": size,
                    "puzzle": f"Puzzle {index}",
                    "solution": {
                        "header": ["House", "Name"],
                        "rows": [["1", "Alice"], ["2", "Bob"]],
                    },
                }
            )

        records = MODULE.convert_official_records(official)
        smoke = MODULE.select_smoke_records(records, per_difficulty=1, seed=7)

        self.assertEqual(len(smoke), 3)
        self.assertEqual({record["size"] for record in smoke}, {"2*2", "4*4", "6*6"})
        self.assertTrue(all(record["capability"] == "direct" for record in smoke))


if __name__ == "__main__":
    unittest.main()
