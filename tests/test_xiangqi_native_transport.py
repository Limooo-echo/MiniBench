"""Native Xiangqi reuses durable SSE and rejects stale judge pins first."""
from pathlib import Path
import tempfile
import json
import unittest
from unittest.mock import patch, Mock
from minibench.experiments import xiangqi
from minibench.factory.agents import make_agent_from_config
from minibench.core.streaming import JournalClient

class NativeTransportTests(unittest.TestCase):
    def test_all_nine_agents_share_same_sse_runtime_reference(self):
        names=["passthrough","direct","cot","self-consistency","best-of-n","tot",
               "plan-then-solve","critic-refine","least-to-most"]
        provider={"name":"generic","model":"fixture","base_url":"http://unused",
                  "api_key_env":"UNUSED","omit_max_tokens":True,"extra_body":{"stream":True}}
        with tempfile.TemporaryDirectory() as d:
            for name in names:
                with self.subTest(agent=name):
                    config={"name":name}
                    if name in {"self-consistency","best-of-n"}:config["samples"]=3
                    agent=make_agent_from_config(config,provider)
                    result=xiangqi._xiangqi_transport(agent,provider,Path(d)/name)
                    self.assertIsInstance(result.client,JournalClient)
                    if hasattr(result,"runtime"):
                        self.assertIs(result.runtime.client,result.client)
                    payload=result.client.build_payload("fixture")
                    self.assertTrue(payload["stream"])
                    self.assertEqual(payload["stream_options"],{"include_usage":True})
                    self.assertNotIn("max_tokens",payload)
                    self.assertEqual(result.client.max_retries,0)

    def test_legacy_unconfigured_transport_unchanged(self):
        agent=object()
        self.assertIs(xiangqi._xiangqi_transport(agent,{},Path("/unused")),agent)

    def test_prediction_file_fixture_unchanged(self):
        agent=object()
        self.assertIs(xiangqi._xiangqi_transport(agent,{"extra_body":{"stream":True}},Path("/unused")),agent)

    def test_unpinned_fixture_never_resolves_an_engine(self):
        with patch("minibench.datasets.xiangqi.engines.pikafish.resolve_pikafish_executable") as resolve:
            xiangqi._preflight_xiangqi_engine("xiangqi-history",{})
            xiangqi._preflight_xiangqi_engine("xiangqi-rule-variants",{"pikafish_binary_sha256":"bad"})
            resolve.assert_not_called()

    def test_runner_rejects_engine_before_agent_construction_or_charge(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            spec=Mock()
            with patch.object(xiangqi,"_write_xiangqi_selected_tasks",return_value=root/"selected.jsonl"), \
                 patch.object(xiangqi,"_preflight_xiangqi_engine",side_effect=ValueError("stale judge")), \
                 patch.object(xiangqi,"make_agent_from_config") as make_agent, \
                 patch.object(xiangqi,"_evaluate") as evaluate:
                with self.assertRaisesRegex(ValueError,"stale judge"):
                    xiangqi._run_checkpointed_xiangqi_experiment(
                        spec,[{"id":"fixture"}],{"agent":{"name":"direct"}},
                        root/"tasks.jsonl","xiangqi-history",
                        {"pikafish_binary_sha256":"stale"},
                        {"output_dir":str(root),"run_name":"run"})
                make_agent.assert_not_called()
                evaluate.assert_not_called()
                state=json.loads((root/"run/run_state.json").read_text())
                self.assertEqual(state["status"],"error")
                self.assertEqual(state["completed_results"],0)
                self.assertEqual((root/"run/predictions.jsonl").read_text(),"")

    def test_pinned_engine_mismatch_propagates_before_agent_stage(self):
        with patch("minibench.datasets.xiangqi.engines.pikafish.resolve_pikafish_executable",return_value=Path("/fixture")) as resolve, patch("minibench.datasets.xiangqi.runtime.validate_engine_fingerprint",side_effect=ValueError("stale judge")) as validate:
            with self.assertRaisesRegex(ValueError,"stale judge"):
                xiangqi._preflight_xiangqi_engine("xiangqi-history",{"pikafish_binary_sha256":"expected"})
            resolve.assert_called_once()
            validate.assert_called_once_with(Path("/fixture"),binary_sha256="expected",nnue_sha256=None)

if __name__=="__main__":unittest.main()
