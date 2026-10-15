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

    def test_selects_balanced_reproducible_evaluation_set(self):
        official = []
        sizes = sorted(
            size
            for group in MODULE.ZEROEVAL_SIZE_GROUPS.values()
            for size in group
        )
        for size in sizes:
            for index in range(40):
                clues = "\n".join(
                    f"{clue}. Synthetic clue." for clue in range(1, index + 2)
                )
                official.append(
                    {
                        "id": f"task-{size}-{index:02d}",
                        "size": size,
                        "puzzle": f"## Clues:\n{clues}",
                        "solution": {
                            "header": ["House", "Name"],
                            "rows": [["1", "Alice"], ["2", "Bob"]],
                        },
                    }
                )
        records = MODULE.convert_official_records(official)
        excluded = {"task-2*2-00", "task-4*4-00", "task-6*6-00"}

        first = MODULE.select_evaluation_records(
            records,
            per_difficulty=15,
            seed=20260810,
            exclude_ids=excluded,
        )
        second = MODULE.select_evaluation_records(
            records,
            per_difficulty=15,
            seed=20260810,
            exclude_ids=excluded,
        )

        self.assertEqual([item["id"] for item in first], [item["id"] for item in second])
        self.assertEqual(len(first), 45)
        self.assertTrue(excluded.isdisjoint(item["id"] for item in first))
        for difficulty in MODULE.DIFFICULTIES:
            chosen = [
                item
                for item in first
                if MODULE.difficulty_for_size(item["size"]) == difficulty
            ]
            self.assertEqual(len(chosen), 15)
            if difficulty == "easy":
                expected_sizes = MODULE.ZEROEVAL_SIZE_GROUPS["small"]
            elif difficulty == "medium":
                expected_sizes = MODULE.ZEROEVAL_SIZE_GROUPS["medium"]
            else:
                expected_sizes = (
                    MODULE.ZEROEVAL_SIZE_GROUPS["large"]
                    | MODULE.ZEROEVAL_SIZE_GROUPS["x-large"]
                )
            size_counts = {
                size: sum(item["size"] == size for item in chosen)
                for size in expected_sizes
            }
            self.assertLessEqual(max(size_counts.values()) - min(size_counts.values()), 1)
            self.assertEqual(
                {
                    band: sum(f"clue-band:{band}" in item["tags"] for item in chosen)
                    for band in MODULE.CLUE_BANDS
                },
                {"low": 5, "middle": 5, "high": 5},
            )


    def test_target_total_and_clue_bands_are_frozen_before_exclusions(self):
        self.assertEqual(MODULE.difficulty_targets(100), {"easy": 34, "medium": 33, "hard": 33})
        sizes = sorted(size for group in MODULE.ZEROEVAL_SIZE_GROUPS.values() for size in group)
        pool = [{"id": f"{size}-{index:02d}", "size": size,
                 "puzzle": "\n".join(f"{i}. clue" for i in range(1, index + 2)), "tags": []}
                for size in sizes for index in range(40)]
        bands = MODULE.frozen_clue_bands(pool)
        excluded = {row["id"] for row in pool if row["id"].endswith("-00")}
        selected = MODULE.select_target_records(pool, target_total=100, seed=20261013,
                                                exclude_ids=excluded)
        self.assertEqual(len(selected), 100)
        self.assertTrue(excluded.isdisjoint(row["id"] for row in selected))
        self.assertTrue(all(f"clue-band:{bands[row['id']]}" in row["tags"] for row in selected))
        self.assertEqual(MODULE.build_parser().parse_args(["--target-total", "100"]).target_total, 100)


if __name__ == "__main__":
    unittest.main()
