"""Offline regression for shortest trees, strict horizons and future billing."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minibench.core.checkpoint import atomic_write_json
from minibench.datasets.xiangqi.dataset import xiangqi_task_from_dict
from minibench.datasets.xiangqi.history import _task_objective_hint
from minibench.datasets.xiangqi.mate_in_one import _build_mate_in_one_prompt, _extract_mate_in_one_uci
from minibench.datasets.xiangqi.multimodal import _build_multimodal_prompt
from minibench.datasets.xiangqi.resources import estimate_cost, validate_pricing
from minibench.datasets.xiangqi.schema import load_records
from minibench.datasets.xiangqi.solution_tree import (
    build_tree, TreeSession, canonical_board, summarize_tree_metrics, validate_tree_record,
)
from minibench.datasets.xiangqi.variants.board import VariantBoard

ONE = "5k3/9/7N1/9/7C1/9/2R1P3P/9/4K4/9 w - - 0 1"
TWO = "5ab2/3ka4/1NN6/p7p/2R3R2/2B1C4/2P3P2/9/4K4/3A1A3 w - - 0 1"
QUIET = "9/2ck5/2N6/4C4/2b3b2/6B2/9/3A5/9/4K4 w - - 0 1"
TWO_ROOTS = {"c7b9","c7d5","c7d9","c7e6","e4d4"}


class ShortestTreeTests(unittest.TestCase):
    def record(self, fen=TWO):
        return {"id":"fixture","fen":fen,"oracle":{"mate_in_plies":3}}

    def test_all_known_two_move_first_moves_including_quiet(self):
        tree=build_tree(self.record(),max_attacker_moves=2,seconds=10,include_all_defences=True)
        self.assertTrue(tree["complete"])
        self.assertTrue(tree["shortest_certified"])
        self.assertEqual(tree["shortest_attacker_moves"],2)
        self.assertEqual(set(tree["nodes"][tree["root"]]["edges"]),TWO_ROOTS)
        for node in tree["nodes"].values():
            if node["turn"]==-tree["attacker"] and not node["terminal"]:
                self.assertEqual(set(node["edges"]),set(node["legal_moves"]))

    def test_quiet_unique_and_one_move_not_inflated_to_two(self):
        for fen,n,roots in [(QUIET,2,{"e6e1"}),(ONE,1,{"c3c9"})]:
            tree=build_tree(self.record(fen),max_attacker_moves=2,seconds=10)
            self.assertEqual(tree["shortest_attacker_moves"],n)
            self.assertEqual(set(tree["nodes"][tree["root"]]["edges"]),roots)

    def test_budget_exhaustion_never_creates_a_certificate(self):
        tree=build_tree(self.record(),max_attacker_moves=2,node_budget=1)
        self.assertFalse(tree["complete"])
        self.assertFalse(tree["shortest_certified"])
        self.assertIsNone(tree["forced_mate"])

    def test_stronger_defender_allows_only_original_shortest_budget(self):
        record=self.record()
        tree=build_tree(record,max_attacker_moves=2,seconds=10)
        with tempfile.TemporaryDirectory() as directory:
            atomic_write_json(Path(directory)/"fixture.json",tree)
            session=TreeSession(record,directory)
            self.assertEqual(session.max_plies,3)
            self.assertTrue(session.scores(True,3)["mate_within_shortest_steps"])
            later=session.scores(True,7)
            self.assertFalse(later["mate_within_shortest_steps"])
            self.assertNotIn("mate_within_shortest_plus_two_steps",later)
            self.assertFalse(session.scores(True,5)["mate_within_shortest_steps"])
            self.assertEqual(later["solution_tree_metadata"]["fallback_max_attacker_moves"],2)
            self.assertIsNone(session.scores(None,None)["mate_within_shortest_steps"])

    def test_stale_tree_cannot_grant_more_than_shortest(self):
        record=self.record()
        tree=build_tree(record,max_attacker_moves=2,seconds=10)
        tree["max_attacker_moves"]=5
        with tempfile.TemporaryDirectory() as directory:
            atomic_write_json(Path(directory)/"fixture.json",tree)
            with self.assertRaisesRegex(ValueError,"cap.*must.*equal"):
                TreeSession(record,directory)

    def test_runtime_rejects_shortest_certified_but_incomplete_export(self):
        record=self.record()
        tree=build_tree(record,max_attacker_moves=2,seconds=10)
        tree['complete']=False
        with tempfile.TemporaryDirectory() as directory:
            atomic_write_json(Path(directory)/'fixture.json',tree)
            with self.assertRaisesRegex(ValueError,'complete certified shortest'):
                TreeSession(record,directory)

    def test_tree_header_is_checked_by_gate_and_before_model_turn(self):
        record=self.record()
        tree=build_tree(record,max_attacker_moves=2,seconds=10)
        for key,bad in [('forced_mate',False),('defender_coverage','typo'),
                        ('max_attacker_moves',10),('proof_defender_coverage','all_optimal'),
                        ('attacker',-1),('extension_attacker_moves',4)]:
            with self.subTest(field=key),tempfile.TemporaryDirectory() as directory:
                changed={**tree,key:bad}
                path=Path(directory)/'fixture.json'
                atomic_write_json(path,changed)
                self.assertFalse(validate_tree_record(record,path)['valid'])
                with self.assertRaises(ValueError):
                    TreeSession(record,directory)

    def test_legal_out_of_tree_move_is_not_automatically_illegal(self):
        from minibench.datasets.xiangqi.schema import fen_to_board
        record=self.record()
        tree=build_tree(record,max_attacker_moves=2,seconds=10)
        with tempfile.TemporaryDirectory() as directory:
            atomic_write_json(Path(directory)/"fixture.json",tree)
            session=TreeSession(record,directory)
            board,_=fen_to_board(TWO)
            root=tree["nodes"][tree["root"]]
            outside=next(m for m in root["legal_moves"] if m not in root["edges"])
            session.advance(board,1,0,outside)
            self.assertTrue(session.off_tree)
            self.assertEqual(session.off_tree_ply,1)

    def test_tree_cannot_be_reused_for_a_different_rule(self):
        record=self.record()
        tree=build_tree(record,max_attacker_moves=2,seconds=10)
        with tempfile.TemporaryDirectory() as directory:
            atomic_write_json(Path(directory)/"fixture.json",tree)
            with self.assertRaisesRegex(ValueError,"stale"):
                TreeSession({**record,"ruleset":"horse-no-leg-block"},directory)

    def test_partial_unknown_cannot_be_presented_as_shortest(self):
        record=self.record()
        tree=build_tree(record,max_attacker_moves=2,node_budget=1)
        with tempfile.TemporaryDirectory() as directory:
            atomic_write_json(Path(directory)/"fixture.json",tree)
            with self.assertRaisesRegex(ValueError,"not certified"):
                TreeSession(record,directory)

    def test_unmetered_or_missing_shortest_not_false_success(self):
        session=TreeSession(self.record(),None)
        self.assertIsNone(session.scores(True,3)["mate_within_shortest_steps"])
        self.assertNotIn("mate_within_shortest_plus_two_steps",session.scores(True,3))
        summary=summarize_tree_metrics([{"status":"error","mate_within_shortest_steps":True}])
        self.assertIsNone(summary["mate_within_shortest_rate"])

    def test_numbered_piece_ids_normalize_for_tree_lookup(self):
        self.assertEqual(canonical_board([[8,9,6,7,12,16]]),((8,8,6,6,12,12),))

    def test_export_replay_catches_tampered_legal_edge(self):
        record=self.record()
        tree=build_tree(record,max_attacker_moves=2,seconds=10)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"fixture.json"
            atomic_write_json(path,tree)
            self.assertTrue(validate_tree_record(record,path)["valid"])
            tree["nodes"][tree["root"]]["edges"]["a0a9"]="missing"
            atomic_write_json(path,tree)
            self.assertFalse(validate_tree_record(record,path)["valid"])

    def test_all_optimal_defences_share_the_parent_distance(self):
        tree=build_tree(self.record(),max_attacker_moves=2,seconds=10)
        self.assertEqual(tree["proof_defender_coverage"],"all_legal")
        self.assertEqual(tree["defender_coverage"],"all_optimal")
        for node in tree["nodes"].values():
            if node["turn"] == -tree["attacker"] and not node["terminal"]:
                self.assertTrue(node["edges"])
                self.assertTrue(all(tree["nodes"][t]["dtm_plies"] == node["dtm_plies"]-1
                                    for t in node["edges"].values()))


class CostTests(unittest.TestCase):
    PRICE={"currency":"CNY","input_per_million":2,"output_per_million":8}

    def test_no_school_price_does_not_mean_zero(self):
        self.assertIsNone(estimate_cost({"prompt_tokens":100,"completion_tokens":50},None)["amount"])

    def test_reasoning_not_double_counted(self):
        usage={"prompt_tokens":100,"completion_tokens":50,"reasoning_tokens":40}
        out=estimate_cost(usage,self.PRICE)
        self.assertAlmostEqual(out["amount"],0.0006)
        self.assertTrue(out["complete"])

    def test_missing_usage_preserves_known_lower_amount_not_total(self):
        out=estimate_cost({"prompt_tokens":100,"completion_tokens":50},self.PRICE,missing_calls=1)
        self.assertIsNone(out["amount"])
        self.assertAlmostEqual(out["known_amount"],0.0006)

    def test_cache_price_is_explicit_and_currency_not_assumed(self):
        usage={"prompt_tokens":100,"completion_tokens":50,"cached_tokens":20}
        self.assertIsNone(estimate_cost(usage,self.PRICE)["amount"])
        price={**self.PRICE,"cached_input_per_million":1}
        self.assertAlmostEqual(estimate_cost(usage,price)["amount"],0.00058)

    def test_invalid_prices_rejected_before_a_paid_call(self):
        for bad in [-1,float("inf"),True]:
            with self.assertRaises(ValueError):
                validate_pricing({**self.PRICE,"input_per_million":bad})

    def test_explicit_null_cache_price_is_ok_if_no_cache_hit(self):
        out=estimate_cost({"prompt_tokens":100,"completion_tokens":50},
                          {**self.PRICE,"cached_input_per_million":None})
        self.assertTrue(out["complete"])

    def test_physical_usage_includes_failed_partial_call_and_not_replay(self):
        from minibench.datasets.xiangqi.resources import (
            physical_bucket,add_physical_attempt,finish_physical_bucket)
        bucket=physical_bucket()
        add_physical_attempt(bucket,{"status":"success","elapsed_seconds":2,
            "usage":{"prompt_tokens":100,"completion_tokens":50,"total_tokens":150,
                     "completion_tokens_details":{"reasoning_tokens":40}}},self.PRICE)
        add_physical_attempt(bucket,{"status":"failed","elapsed_seconds":1,"usage":None},self.PRICE)
        report=finish_physical_bucket(bucket)
        self.assertEqual(report["physical_attempts"],2)
        self.assertIsNone(report["token_usage"]["total_tokens"])
        self.assertEqual(report["known_token_usage"]["total_tokens"],150)
        self.assertEqual(report["known_token_usage"]["reasoning_tokens"],40)
        self.assertIsNone(report["cost"]["amount"])
        self.assertAlmostEqual(report["cost"]["known_amount"],0.0006)
        self.assertEqual(report["attempts_with_unknown_usage"],1)


class SchedulerTests(unittest.TestCase):
    def test_composite_score_uses_shortest_not_extension_for_new_protocol(self):
        from minibench.scoring.adapters import score_record
        record={"protocol_version":"xiangqi-shortest-tree-v1","success":True,
                "goal_achieved":True,"mate_within_shortest_steps":False,
                "mate_within_shortest_plus_two_steps":True}
        for family,dimension in [("xiangqi_history","H"),("xiangqi_rule_variants","C")]:
            result=score_record(family,dimension,record,None,mode="single")
            self.assertEqual(result["y"],0)
            self.assertEqual(result["evidence"]["outcome_field"],"mate_within_shortest_steps")
            missing=score_record(family,dimension,{**record,"mate_within_shortest_steps":None},None,mode="single")
            self.assertIsNone(missing["y"])
            self.assertEqual(missing["status"],"missing")

    def test_failed_reexport_never_erases_an_existing_shortest_proof(self):
        from scripts.build_xiangqi_solution_trees import preserve_stronger_proof
        from minibench.datasets.xiangqi.solution_tree import VERSION
        prior={"version":VERSION,"record_key":"same","shortest_certified":True,
               "shortest_attacker_moves":4,"complete":False}
        failed={"version":VERSION,"record_key":"same","shortest_certified":False,
                "complete":False,"seconds":1,"visits":10,"stop_reason":"budget"}
        kept=preserve_stronger_proof(prior,failed)
        self.assertTrue(kept["shortest_certified"])
        self.assertEqual(kept["shortest_attacker_moves"],4)
        with self.assertRaisesRegex(ValueError,"Conflicting"):
            preserve_stronger_proof(prior,{**failed,"shortest_certified":True,"shortest_attacker_moves":3})

    def test_c2_variants_wait_for_all_own_standard_baselines(self):
        from minibench.cloud import c2_phase_dependencies
        def job(key,agent,phase,exp="c2"):
            return {"key":key,"agent":agent,"c2_phase":phase,"experiment":exp}
        jobs=[job("s1","direct","standard"),job("s2","direct","standard"),
              job("v","direct","variant"),job("cs","cot","standard"),
              job("cv","cot","variant"),job("h","direct",None,"h2")]
        self.assertEqual(c2_phase_dependencies({"jobs":jobs}),
                         {"v":{"s1","s2"},"cv":{"cs"}})


class TreeGameTests(unittest.TestCase):
    def test_h2_and_c2_stop_after_exactly_shortest_total_attacker_moves(self):
        from types import SimpleNamespace
        from minibench.datasets.xiangqi import history,rule_variants
        from minibench.datasets.xiangqi.dataset import XiangqiTask
        from minibench.datasets.xiangqi.schema import fen_to_board
        from tests.test_xiangqi_runtime_errors import FixedAgent,no_engine
        board,_=fen_to_board(TWO)
        record={"id":"fixture","fen":TWO,"oracle":{"mate_in_plies":3}}
        tree=build_tree(record,max_attacker_moves=2,seconds=10)
        # A deliberately non-terminal test double isolates the loop boundary.
        # Real chess correctness is covered by the real-board tests below.
        class NeverMateBoard:
            def __init__(self,state,rules): self.board=state
            def legal_moves(self,side):
                return [SimpleNamespace(to_uci=lambda: "a0a1" if side==1 else "a9a8")]
            def find_general(self,side): return (9,4) if side==1 else (0,4)
            def _is_in_check(self,side): return False
            def apply(self,move): pass
        def analysis(board,side,**kwargs):
            return [SimpleNamespace(move=board.legal_moves(side)[0],value=0)]
        def cp(engine,board,side,**kwargs):
            return board.legal_moves(side)[0].to_uci(),0
        with tempfile.TemporaryDirectory() as directory:
            for departure_ply in (0, 2):
                atomic_write_json(Path(directory)/"fixture.json",tree)
                def session(*args):
                    value=TreeSession(record,directory)
                    def advance(state, side, ply, uci):
                        if ply == departure_ply:
                            value.off_tree=True
                            value.off_tree_ply=ply+1
                    value.advance=advance
                    value.opponent_move=lambda *args: None
                    return value
                task=XiangqiTask(id="fixture",fen=TWO,board=board,side_to_move="ally",agent_side="ally",
                    max_steps=99,tags=(),schema_version=2,family="xiangqi-history",agent_color="red",
                    difficulty="short",oracle=record["oracle"])
                with no_engine(history),patch.object(history,"VariantBoard",NeverMateBoard),\
                        patch.object(history,"TreeSession",side_effect=session),\
                        patch.object(history,"_pikafish_analysis",side_effect=cp):
                    h=history.evaluate_history_tasks([task],FixedAgent('{"move":"a0a1"}'),
                                                     history_mode="full-state",solution_tree_dir=directory)[0]
                ctask={**record,"board":board,"rules":[],"ruleset":"standard","agent_color":"red",
                       "side_to_move":"ally","max_plies":99,"difficulty":"short"}
                with patch.object(rule_variants,"VariantBoard",NeverMateBoard),\
                        patch.object(rule_variants,"TreeSession",side_effect=session),\
                        patch.object(rule_variants,"score_moves_cpe",side_effect=analysis):
                    c=rule_variants.evaluate_rule_variant_task(ctask,FixedAgent('{"move":"a0a1"}'),
                                                              solution_tree_dir=directory)
                for result in (h,c):
                    self.assertEqual(result.status,"ok",result)
                    self.assertEqual(len(result.steps),3)
                    self.assertEqual(sum(s["actor"]=="agent" for s in result.steps),2)
                    self.assertFalse(result.success)
                    self.assertFalse(result.mate_within_shortest_steps)
                    self.assertNotIn("mate_within_shortest_plus_two_steps",vars(result))
                    self.assertIn("max_steps_reached",result.reasons)
                    self.assertEqual(result.solution_tree_metadata["first_off_tree_ply"],departure_ply+1)

    def test_h2_and_c2_actually_play_two_move_solution_and_score_shortest_window(self):
        from minibench.datasets.xiangqi import history,rule_variants
        from minibench.datasets.xiangqi.dataset import XiangqiTask
        from minibench.datasets.xiangqi.schema import fen_to_board
        from tests.test_xiangqi_runtime_errors import FixedAgent,no_engine
        board,_=fen_to_board(TWO)
        record={"id":"fixture","fen":TWO,"oracle":{"mate_in_plies":3}}
        tree=build_tree(record,max_attacker_moves=2,seconds=10)
        root=tree["nodes"][tree["root"]]
        first="c7d5"
        defender=tree["nodes"][root["edges"][first]]
        reply=min(defender["edges"])
        second=min(tree["nodes"][defender["edges"][reply]]["edges"])
        class SequenceAgent(FixedAgent):
            def generate(self,prompt,task):
                self.messages.append(prompt)
                self.calls+=1
                return json.dumps({"move":first if self.calls==1 else second})
        with tempfile.TemporaryDirectory() as directory:
            atomic_write_json(Path(directory)/"fixture.json",tree)
            task=XiangqiTask(id="fixture",fen=TWO,board=board,side_to_move="ally",agent_side="ally",
                max_steps=3,tags=(),schema_version=2,family="xiangqi-history",agent_color="red",
                difficulty="short",oracle=record["oracle"])
            agent=SequenceAgent()
            with no_engine(history):
                h=history.evaluate_history_tasks([task],agent,history_mode="full-state",
                                                 solution_tree_dir=directory)[0]
            self.assertEqual(h.status,"ok",h)
            self.assertTrue(h.mate_within_shortest_steps)
            self.assertEqual(h.checkmate_ply,3)
            self.assertIsNone(h.avg_cp_loss)
            self.assertTrue(any("certified shortest forced mate in 2" in p for p in agent.messages))
            ctask={**record,"board":board,"rules":[],"ruleset":"standard","agent_color":"red",
                   "side_to_move":"ally","max_plies":3,"difficulty":"short"}
            c=rule_variants.evaluate_rule_variant_task(ctask,SequenceAgent(),solution_tree_dir=directory)
            self.assertEqual(c.status,"ok",c)
            self.assertTrue(c.mate_within_shortest_steps)
            self.assertEqual(c.checkmate_ply,3)
            self.assertIsNotNone(c.avg_variant_loss)
            self.assertEqual([s["uci"] for s in c.steps],[first,reply,second])


    def test_real_off_tree_move_can_still_checkmate_on_last_allowed_move(self):
        from minibench.datasets.xiangqi import history, rule_variants
        from minibench.datasets.xiangqi.dataset import XiangqiTask
        from minibench.datasets.xiangqi.schema import fen_to_board
        from tests.test_xiangqi_runtime_errors import FixedAgent, no_engine
        from types import SimpleNamespace
        board, _ = fen_to_board(TWO)
        record = {"id": "fixture", "fen": TWO, "oracle": {"mate_in_plies": 3}}
        tree = build_tree(record, max_attacker_moves=2, seconds=10)
        root = tree["nodes"][tree["root"]]
        first = "c7d5"
        defence = tree["nodes"][root["edges"][first]]
        reply = min(defence["edges"])
        last = min(tree["nodes"][defence["edges"][reply]]["edges"])

        class SequenceAgent(FixedAgent):
            def generate(self, prompt, task):
                self.messages.append(prompt)
                self.calls += 1
                return json.dumps({"move": first if self.calls == 1 else last})

        # Simulate a missing book branch without changing the game or legality.
        def missing_branch_session(*args):
            session = TreeSession(record, directory)
            advance = session.advance
            def book_advance(board, side, ply, uci):
                if ply == 0:
                    session.off_tree = True
                    session.off_tree_ply = 1
                else:
                    advance(board, side, ply, uci)
            session.advance = book_advance
            return session

        def cp(engine, state, side, **kwargs):
            moves = state.legal_moves(side)
            preferred = next((m.to_uci() for m in moves if m.to_uci() == reply), None)
            return preferred or moves[0].to_uci(), 0.0

        def cpe(state, side, **kwargs):
            return [SimpleNamespace(move=m, value=0.0) for m in
                    sorted(state.legal_moves(side), key=lambda m: (m.to_uci() != reply, m.to_uci()))]

        with tempfile.TemporaryDirectory() as directory:
            atomic_write_json(Path(directory) / "fixture.json", tree)
            task = XiangqiTask(id="fixture", fen=TWO, board=board, side_to_move="ally",
                agent_side="ally", max_steps=99, tags=(), schema_version=2,
                family="xiangqi-history", agent_color="red", difficulty="short",
                oracle=record["oracle"])
            with no_engine(history), patch.object(history, "TreeSession", side_effect=missing_branch_session), \
                    patch.object(history, "_pikafish_analysis", side_effect=cp):
                h = history.evaluate_history_tasks([task], SequenceAgent(), history_mode="full-state",
                                                  solution_tree_dir=directory)[0]
            ctask = {**record, "board": board, "rules": [], "ruleset": "standard",
                     "agent_color": "red", "side_to_move": "ally", "max_plies": 99,
                     "difficulty": "short"}
            with patch.object(rule_variants, "TreeSession", side_effect=missing_branch_session), \
                    patch.object(rule_variants, "score_moves_cpe", side_effect=cpe):
                c = rule_variants.evaluate_rule_variant_task(ctask, SequenceAgent(),
                                                            solution_tree_dir=directory)
            for result, fallback in ((h, "pikafish-fallback"), (c, "depth3-fallback")):
                self.assertEqual(result.status, "ok", result)
                self.assertTrue(result.success)
                self.assertTrue(result.mate_within_shortest_steps)
                self.assertEqual(result.checkmate_ply, 3)
                self.assertEqual([s["uci"] for s in result.steps], [first, reply, last])
                self.assertEqual(result.steps[1]["opponent_source"], fallback)
                self.assertTrue(result.solution_tree_metadata["off_tree"])
                self.assertEqual(result.solution_tree_metadata["first_off_tree_ply"], 1)
                self.assertEqual(result.solution_tree_metadata["fallback_max_attacker_moves"], 2)


class PairedOutputTests(unittest.TestCase):
    def test_d3_and_m2_text_share_exact_prompt_and_output_schema(self):
        record=load_records("data/xiangqi/mate_in_one/tasks.jsonl")[0]
        task=xiangqi_task_from_dict(record)
        board=VariantBoard(task.board,[])
        self.assertEqual(_build_mate_in_one_prompt(task,board),
                         _build_multimodal_prompt(board,"text","",agent_side=1))
        raw=json.dumps({"move":"a0a1","pieces":{"a0":"R"}})
        self.assertEqual(_extract_mate_in_one_uci(raw),"a0a1")
        self.assertIsNone(_extract_mate_in_one_uci('{"move":"a0a1","reason":"x"}'))


if __name__=="__main__":
    unittest.main()
