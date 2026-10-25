"""No paid APIs: hard decision deadlines, nine Agents, four tasks, cloud resume."""
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import io
import json
import os
from pathlib import Path
import signal
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import yaml

from minibench import cloud
from minibench.agents.passthrough import PassthroughAgent
from minibench.core.agent import ReasoningConfig
from minibench.core.streaming import JournalClient
from minibench.datasets.xiangqi import history, mate_in_one, multimodal, rule_variants
from minibench.datasets.xiangqi.deadline import (
    FINAL_MOVE_TIMEOUT_SECONDS, FinalMoveTimeout, call_agent_move,
    final_move_deadline, validate_move_timeout,
)
from minibench.datasets.xiangqi.runtime import validate_evaluation_config
from minibench.datasets.xiangqi.schema import FAMILY_PATHS, load_records, runtime_dict
from minibench.factory.agents import AGENT_NAMES, REASONING_AGENTS
from minibench.scoring.adapters import score_record
from tests.test_xiangqi_runtime_errors import FixedAgent, dict_fixture, no_engine, task_fixture


class SlowClient:
    def __init__(self): self.calls=0
    def complete(self,*args,**kwargs):
        self.calls+=1
        time.sleep(1)
        return '{"move":"a8d8"}'
    complete_messages=complete


class DeadlineTests(unittest.TestCase):
    def test_default_is_900_and_cannot_be_disabled_or_extended(self):
        self.assertEqual(FINAL_MOVE_TIMEOUT_SECONDS,900)
        for value in (None,True,0,-1,901,float("inf")):
            with self.subTest(value=value),self.assertRaises(ValueError): validate_move_timeout(value)
        for family in ("xiangqi-mate-in-one","xiangqi-history","xiangqi-rule-variants","xiangqi-multimodal"):
            validate_evaluation_config(family,{"final_move_timeout_seconds":900})
            with self.assertRaises(ValueError): validate_evaluation_config(family,{"final_move_timeout_seconds":901})

    def test_continuous_thinking_and_generic_retry_cannot_renew_deadline(self):
        chunks=[]
        def thinks_forever():
            try:
                while True:
                    chunks.append("reasoning, not a final move")
                    time.sleep(.005)
            except Exception:
                self.fail("A generic Agent retry must not swallow FinalMoveTimeout")
        previous=signal.getsignal(signal.SIGALRM)
        started=time.monotonic()
        with self.assertRaises(FinalMoveTimeout): call_agent_move(thinks_forever,timeout_seconds=.04)
        self.assertTrue(chunks)
        self.assertLess(time.monotonic()-started,.5)
        self.assertEqual(signal.getsignal(signal.SIGALRM),previous)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0],0)
        self.assertEqual(call_agent_move(lambda:'{"move":"a0a1"}',timeout_seconds=.04),'{"move":"a0a1"}')

    def test_internal_phases_share_one_deadline(self):
        phases=[]
        def two_phase_agent():
            time.sleep(.025)
            phases.append("intermediate completed")
            with final_move_deadline(900):
                time.sleep(.1)
            return '{"move":"a0a1"}'
        with self.assertRaises(FinalMoveTimeout): call_agent_move(two_phase_agent,timeout_seconds=.05)
        self.assertEqual(phases,["intermediate completed"])

    def test_every_agent_and_all_task_modes_score_a_timeout_as_failure(self):
        m2task=runtime_dict(load_records(FAMILY_PATHS["xiangqi-multimodal"])[0])
        for name in AGENT_NAMES:
            for case in ("d3","h2-full","h2-history","c2","m2-text","m2-image"):
                with self.subTest(agent=name,case=case):
                    client=SlowClient()
                    agent=(PassthroughAgent(client) if name=="passthrough" else
                           REASONING_AGENTS[name](client,ReasoningConfig(prompt_version="v2",trace="summary")))
                    if case=="d3":
                        with no_engine(mate_in_one):
                            result=asdict(mate_in_one.evaluate_mate_in_one_tasks(
                                [task_fixture()],agent,final_move_timeout_seconds=.02)[0])
                        family,dimension="xiangqi_mate_in_one","D"
                    elif case.startswith("h2"):
                        with no_engine(history):
                            result=asdict(history.evaluate_history_tasks([task_fixture()],agent,
                                history_mode="full-state" if case=="h2-full" else "move-history-only",
                                final_move_timeout_seconds=.02)[0])
                        family,dimension="xiangqi_history","H"
                    elif case=="c2":
                        result=asdict(rule_variants.evaluate_rule_variant_task(
                            dict_fixture(),agent,final_move_timeout_seconds=.02))
                        family,dimension="xiangqi_rule_variants","C"
                    else:
                        result=multimodal.evaluate_xiangqi_multimodal_tasks([m2task],agent,
                            modes=["text" if case=="m2-text" else "chinese-piece-image"],
                            verify_with_pikafish=False,final_move_timeout_seconds=.02)[0]
                        family,dimension="xiangqi_multimodal","M"
                    self.assertEqual(result["status"],"invalid",result)
                    self.assertFalse(result["success"])
                    self.assertEqual(result["termination_reason"],"final_move_timeout")
                    self.assertEqual(result["error"]["stage"],"timeout")
                    self.assertEqual(result["metrics"]["final_move_timeout_seconds"],.02)
                    scored=score_record(family,dimension,result,None,mode="single")
                    self.assertEqual(scored["y"],0)
                    self.assertEqual(scored["status"],"invalid")
                    if case=="m2-image":
                        self.assertTrue(result["steps"][0]["input_image_sha256"])

    def test_timeout_does_not_stop_following_native_task(self):
        class SlowThenFast(FixedAgent):
            def generate(self,prompt,task):
                self.calls+=1
                if self.calls==1:time.sleep(.2)
                return '{"move":"a8d8"}'
        with no_engine(mate_in_one):
            results=mate_in_one.evaluate_mate_in_one_tasks(
                [task_fixture(id="slow"),task_fixture(id="fast")],SlowThenFast(),final_move_timeout_seconds=.03)
        self.assertEqual(results[0].termination_reason,"final_move_timeout")
        self.assertFalse(results[0].success)
        self.assertTrue(results[1].success)

    def test_infrastructure_and_other_families_are_not_reclassified(self):
        record={"status":"invalid","success":False,"termination_reason":"final_move_timeout",
                "error":{"stage":"timeout","type":"FinalMoveTimeout","deadline_seconds":900}}
        self.assertEqual(score_record("zebra","D",record,None,mode="single")["status"],"missing")
        with no_engine(mate_in_one):
            result=mate_in_one.evaluate_mate_in_one_tasks(
                [task_fixture()],FixedAgent(error=RuntimeError("HTTP 401")),final_move_timeout_seconds=.02)[0]
        self.assertEqual(result.status,"error")
        self.assertIsNone(result.success)

    def test_stream_partial_saved_and_unknown_usage_on_deadline(self):
        class EndlessResponse:
            headers={"Content-Type":"text/event-stream"}
            def __init__(self):self.lines=0
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def readline(self):
                self.lines+=1
                if self.lines%2==0:return b"\n"
                time.sleep(.005)
                return ('data: '+json.dumps({"id":"unit","model":"unit","choices":[{
                    "index":0,"delta":{"reasoning_content":"thinking"},"finish_reason":None}]})+'\n').encode()
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{"UNIT_DEADLINE_KEY":"fixture"}):
            client=JournalClient(Path(tmp),model="unit",base_url="https://invalid.example/v1",
                                 api_key_env="UNIT_DEADLINE_KEY",max_retries=3,timeout=None)
            with no_engine(mate_in_one),patch("urllib.request.urlopen",return_value=EndlessResponse()) as network:
                result=mate_in_one.evaluate_mate_in_one_tasks(
                    [task_fixture()],PassthroughAgent(client),final_move_timeout_seconds=1.0)[0]
            self.assertEqual(network.call_count,1)
            self.assertEqual(result.termination_reason,"final_move_timeout")
            self.assertIsNone(result.metrics["token_usage"]["total_tokens"])
            partial=json.loads((Path(tmp)/"attempts/0000-0000/partial.json").read_text())
            self.assertTrue(partial["reasoning"])
            self.assertEqual(partial["content"],"")
            self.assertFalse((Path(tmp)/"0000.json").exists())

    def test_timeout_while_waiting_for_shared_quota_cannot_leak_a_slot(self):
        manager=cloud.GateManager()
        manager.start()
        self.addCleanup(manager.shutdown)
        gate=manager.RateGate({"default":{"max_in_flight":1,"reserve_tokens":10}})
        gate.acquire("default")  # Another job owns the only slot.
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{"UNIT_DEADLINE_KEY":"fixture"}):
            client=JournalClient(Path(tmp),model="unit",base_url="https://invalid.example/v1",
                                 api_key_env="UNIT_DEADLINE_KEY",gate=gate,timeout=None)
            with patch("urllib.request.urlopen",side_effect=AssertionError("queued call must not be sent")) as network:
                with self.assertRaises(FinalMoveTimeout):
                    call_agent_move(client.complete,"fixture",timeout_seconds=.3)
            self.assertEqual(network.call_count,0)
            self.assertFalse(list(Path(tmp).glob("attempts/*/started.json")))
        state=gate.snapshot()["default"]
        self.assertEqual(state["active_calls"],1)
        self.assertEqual(state["network_calls"],1)
        gate.release("default",{"total_tokens":1})
        self.assertTrue(gate.try_acquire("default"))
        gate.release("default",{"total_tokens":1})
        self.assertEqual(gate.snapshot()["default"]["active_calls"],0)

    def test_deadline_defers_only_the_tiny_reservation_commit_section(self):
        from minibench.core.streaming import _atomic_gate_rpc
        checkpoints=[]
        def reserve():
            with _atomic_gate_rpc():
                time.sleep(.04)
                checkpoints.append("grant acknowledged")
        with self.assertRaises(FinalMoveTimeout):
            call_agent_move(reserve,timeout_seconds=.01)
        self.assertEqual(checkpoints,["grant acknowledged"])


class CloudDeadlineTests(unittest.TestCase):
    def test_eight_worker_scheduler_continues_and_resume_does_not_rerun_timeout(self):
        from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
        calls=[]
        lock=threading.Lock()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with lock:
                    calls.append(body)
                    number=len(calls)
                self.send_response(200)
                self.send_header("Content-Type","text/event-stream")
                self.end_headers()
                try:
                    if number==1:
                        for _ in range(3000):
                            chunk={"id":"local","model":"fixture","choices":[{
                                "index":0,"delta":{"reasoning_content":"thinking"},"finish_reason":None}]}
                            self.wfile.write(('data: '+json.dumps(chunk)+'\n\n').encode())
                            self.wfile.flush()
                            time.sleep(.01)
                    else:
                        chunk={"id":"local","model":"fixture","choices":[{
                            "index":0,"delta":{"content":'{"move":"a0a0"}'},"finish_reason":"stop"}]}
                        end={"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}
                        self.wfile.write(('data: '+json.dumps(chunk)+'\n\ndata: '+json.dumps(end)+'\n\ndata: [DONE]\n\n').encode())
                        self.wfile.flush()
                except (BrokenPipeError,ConnectionResetError):pass
        server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{"UNIT_DEADLINE_KEY":"local-only"}):
            raw=yaml.safe_load((cloud.ROOT/"config/cloud/xiangqi-pilot.yaml").read_text())
            raw["experiments"]=[e for e in raw["experiments"] if e["id"]=="c2"]
            # Allow real fsync/process scheduling overhead; only the endless
            # reasoning response should exceed this scaled-down test deadline.
            raw["experiments"][0]["evaluation_overrides"]={"final_move_timeout_seconds":5.0}
            raw["engine_binding"]="reference"
            raw["workers"]=8
            raw["accounts"]["qwen"]["provider"].update(base_url=f"http://127.0.0.1:{server.server_port}/v1",
                model="fixture",api_key_env="UNIT_DEADLINE_KEY",timeout=None)
            raw["accounts"]["qwen"]["limits"]["max_in_flight"]=8
            root=Path(tmp)
            cfg=root/"cloud.yaml"
            cfg.write_text(yaml.safe_dump(raw))
            suite=root/"suite"
            plan=cloud.prepare(cfg,suite)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cloud.run(suite,plan),0)
                before=len(calls)
                self.assertEqual(cloud.run(suite,plan),0)
            self.assertEqual(before,6)
            self.assertEqual(len(calls),before)
            saved=cloud.completed_jobs(suite,plan)
            self.assertEqual(len(saved),6)
            rows=[row for batch in saved.values() for row in batch["records"]]
            timeouts=[row for row in rows if row["termination_reason"]=="final_move_timeout"]
            self.assertEqual(len(timeouts),1)
            self.assertFalse(timeouts[0]["success"])
            self.assertFalse(timeouts[0]["mate_within_shortest_steps"])
            self.assertNotIn("mate_within_shortest_plus_two_steps",timeouts[0])
            self.assertEqual(json.loads((suite/"state.json").read_text())["status"],"completed")
            summary=json.loads((suite/"summary.json").read_text())
            self.assertIsNone(summary["usage"]["total_tokens"])
            self.assertEqual(summary["usage"]["attempts_without_usage"],1)


if __name__=="__main__":unittest.main()
