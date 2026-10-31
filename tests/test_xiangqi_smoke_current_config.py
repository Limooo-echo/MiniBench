"""The smoke CLI must preserve the actual certified shortest release policy."""
import importlib.util
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("smoke_current_config",ROOT/"scripts/run_xiangqi_smoke.py")
smoke=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)

class CurrentSmokeConfigTests(unittest.TestCase):
    def test_real_frozen_configs_prepare_all_tasks_without_old_extension(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(smoke,"resolve_pikafish_executable",return_value=Path("/tmp/unused-pikafish")), patch.object(smoke,"pikafish_fingerprint",return_value={"test":"no-engine-process"}):
            plan, prepared=smoke.prepare_suite(Path(temporary)/"suite",["d3","h2","c2","m2"])
            self.assertEqual(set(prepared),{"d3","h2","c2","m2"})
            import json
            c2=json.loads((Path(temporary)/"suite/configs/c2.json").read_text())
            self.assertEqual(c2["evaluation"]["horizon_policy"],"certified_shortest_attacker_moves")
            self.assertNotIn("loss_unit",c2["evaluation"])
            self.assertTrue(c2["evaluation"]["solution_tree_dir"])
            self.assertEqual(plan["tasks"]["c2"]["selected_record_count"],6)

class SmokeFailureReportingTests(unittest.TestCase):
    def run_suite(self, first_result):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "suite"
            directory.mkdir()
            configs = {key: {"provider": {}} for key in ("d3", "h2", "c2")}
            with patch.object(smoke, "prepare_suite", return_value=({}, configs)), \
                 patch.object(smoke, "run_experiment", side_effect=[first_result, {"total": 1}, {"total": 1}]) as run, \
                 redirect_stdout(StringIO()):
                status = smoke.main(["--tasks", "d3,h2,c2", "--output-dir", str(directory)])
            report = json.loads((directory / "suite_results.json").read_text())
            self.assertEqual(run.call_count, 3)
            self.assertEqual(set(report["tasks"]), {"d3", "h2", "c2"})
            return status, report

    def test_saved_error_without_exception_fails_smoke_and_continues_other_tasks(self):
        for result in ({"total": 1, "missing": 1},
                       {"total": 1, "run_status": "completed_with_errors", "infrastructure_error_count": 1},
                       {"total": 1, "diagnostic_result_count": 1}):
            with self.subTest(result=result):
                status, report = self.run_suite(result)
                self.assertEqual(status, 1)
                self.assertEqual(report["status"], "completed_with_failures")
                self.assertEqual(set(report["failures"]), {"d3"})

    def test_validly_scored_model_failure_does_not_become_service_failure(self):
        status, report = self.run_suite({"total": 1, "missing": 0, "invalid": 1, "mate_rate": 0})
        self.assertEqual(status, 0)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["failures"], {})


if __name__=="__main__": unittest.main()
