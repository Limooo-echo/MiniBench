#!/usr/bin/env python3
"""Run real-Pikafish protocol acceptance with scripted agents, never an LLM.

This checks runtime behavior on one selected position per requested family. It
is not a model evaluation, a ranking, or a substitute for full corpus validation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minibench.datasets.xiangqi import mate_in_one as d3, history as h2, multimodal as m2, rule_variants as c2
from minibench.datasets.xiangqi.dataset import XiangqiTask, xiangqi_task_from_dict
from minibench.datasets.xiangqi.engines.pikafish import PikafishEngine, board_to_pikafish_fen, pikafish_fingerprint
from minibench.datasets.xiangqi.runtime import PROTOCOL_VERSION
from minibench.datasets.xiangqi.schema import FAMILY_PATHS, runtime_dict, validate_record
from minibench.datasets.xiangqi.variants.board import VariantBoard
from minibench.datasets.xiangqi.variants.rules import Rule
from minibench.datasets.xiangqi.variants.search import score_moves

FAMILIES = {"d3": "xiangqi-mate-in-one", "c2": "xiangqi-rule-variants", "h2": "xiangqi-history", "m2": "xiangqi-multimodal"}


def replay_history(board: list[list[int]], side: int, moves: list[str]) -> tuple[VariantBoard, int]:
    position = VariantBoard(board, [])
    for uci in moves:
        legal = {move.to_uci(): move for move in position.legal_moves(side)}
        if uci not in legal:
            raise ValueError(f"scripted-agent history contains illegal move {uci}")
        position.apply(legal[uci])
        side = -side
    return position, side


class CorrectAgent:
    """A transparent rules oracle; it deliberately receives task data for testing."""
    def __init__(self, engine: PikafishEngine | None = None):
        self.engine, self.calls, self.resets = engine, 0, 0

    def reset(self):
        self.resets += 1

    def _answer(self, task, moves=()):
        self.calls += 1
        if isinstance(task, XiangqiTask):
            side = 1 if task.side_to_move == "ally" else -1
            if task.family == "xiangqi-history":
                if self.engine is None:
                    raise ValueError("scripted H2 agent requires its independent reference engine")
                position, side = replay_history(task.board, side, list(moves))
                fen = board_to_pikafish_fen(position.board, side_to_move="ally" if side == 1 else "enemy")
                uci = self.engine.analyze_fen(fen, depth=20).bestmove
            else:
                uci = d3.enumerate_mate_in_one_moves(task.board, side)[0]
        elif task.get("family") == "xiangqi-rule-variants":
            position = VariantBoard(task["board"], [Rule.from_dict(rule) for rule in task.get("rules", [])])
            uci = score_moves(position, 1, 3, 1)[0][0].to_uci()
        else:
            uci = d3.enumerate_mate_in_one_moves(task["board"], m2._agent_side(task))[0]
        return json.dumps({"move": uci})

    def generate(self, prompt, task):
        moves = re.findall(r"^(?:agent|pikafish): ([a-i][0-9][a-i][0-9])$", prompt, re.MULTILINE)
        return self._answer(task, moves)

    def generate_messages(self, messages, task, **_kwargs):
        moves = []
        for message in messages:
            content = str(message["content"])
            if message["role"] == "assistant":
                moves.append(json.loads(content)["move"])
            elif message["role"] == "user":
                match = re.search(r"opponent (?:replied with|opened with) ([a-i][0-9][a-i][0-9])", content)
                if match:
                    moves.append(match.group(1))
        return self._answer(task, moves)

    def generate_multimodal(self, _prompt, task, *, images):
        return self._answer(task)


class FixedAgent(CorrectAgent):
    def __init__(self, output: str = "first invalid answer", *, timeout=False):
        super().__init__()
        self.output, self.timeout = output, timeout

    def _answer(self, task, moves=()):
        self.calls += 1
        if self.timeout:
            raise TimeoutError("intentional offline acceptance timeout")
        return self.output


def legal_non_mate(task: XiangqiTask, *, require_reply=False) -> str:
    side = 1 if task.side_to_move == "ally" else -1
    position = VariantBoard(task.board, [])
    mates = set(d3.enumerate_mate_in_one_moves(task.board, side))
    for move in position.legal_moves(side):
        trial = position.copy()
        trial.apply(move)
        if move.to_uci() not in mates and (not require_reply or trial.legal_moves(-side)):
            return move.to_uci()
    raise ValueError(f"{task.id} has no suitable legal non-mating move for this acceptance case")


def _read_first(family: str, *, source_task_id: str | None = None) -> tuple[dict, dict]:
    path = ROOT / FAMILY_PATHS[FAMILIES[family]]
    raw = path.read_bytes()
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if family == "h2":
        records.sort(key=lambda record: (record["max_plies"], record["id"]))
    if family == "c2":
        records.sort(key=lambda record: (record["ruleset"] == "standard", record["id"]))
    if source_task_id is not None:
        records = [record for record in records if record.get("source_task_id") == source_task_id]
    if not records:
        raise ValueError(f"no matching acceptance record in {path}")
    record = validate_record(records[0], expected_family=FAMILIES[family])
    return record, {"path": str(path), "sha256": sha256(raw).hexdigest(), "selected_id": record["id"]}


def run_acceptance(output_dir: Path, engine_path: Path, families: list[str]) -> dict[str, Any]:
    # An existing directory is never reused, even after an interrupted run.
    output_dir.mkdir(parents=True, exist_ok=False)
    selected, sources = {}, {}
    for family in families:
        source = selected.get("d3", {}).get("id") if family == "m2" else None
        selected[family], sources[family] = _read_first(family, source_task_id=source)
    config = {
        "protocol_version": PROTOCOL_VERSION, "kind": "scripted_offline_acceptance", "uses_language_model": False,
        "created_at": datetime.now(timezone.utc).isoformat(), "families": families,
        "scope": "one selected position per family; not full-dataset validation or model performance",
        "engine": pikafish_fingerprint(engine_path), "depths": {"d3_m2": 8, "h2_opponent": 16, "h2_scripted_agent": 20},
        "sources": sources,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "selected_tasks.jsonl").write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in selected.values()), encoding="utf-8")
    (output_dir / "plan.json").write_text(json.dumps({"cases": ["correct_scripted_agent", "returned_invalid_answer", "provider_timeout", "legal_wrong_d3", "diagnostic_engine_fault_d3_m2", "required_opponent_fault_h2"], "checkpoint": "append each individual task/mode result before continuing"}, indent=2) + "\n", encoding="utf-8")
    checks, recorded = {}, []
    checkpoint_path = output_dir / "predictions.jsonl"
    checkpoint_path.touch(exist_ok=False)

    def run_case(family, case, agent, *, path=engine_path):
        def checkpoint(result):
            payload = asdict(result) if is_dataclass(result) else result
            record = {"family": family, "case": case, "result": payload}
            with checkpoint_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            recorded.append(record)
        record = selected[family]
        if family == "d3":
            results = d3.evaluate_mate_in_one_tasks([xiangqi_task_from_dict(record)], agent, pikafish_path=path, pikafish_depth=8, pikafish_binary_sha256=config["engine"]["binary_sha256"], pikafish_nnue_sha256=config["engine"]["eval_file_sha256"], on_result=checkpoint)
        elif family == "h2":
            results = h2.evaluate_history_tasks([xiangqi_task_from_dict(record)], agent, history_mode="paired", pikafish_path=path, pikafish_depth=16, pikafish_binary_sha256=config["engine"]["binary_sha256"], pikafish_nnue_sha256=config["engine"]["eval_file_sha256"], on_result=checkpoint)
        elif family == "m2":
            results = m2.evaluate_xiangqi_multimodal_tasks([runtime_dict(record)], agent, pikafish_path=path, pikafish_depth=8, pikafish_binary_sha256=config["engine"]["binary_sha256"], pikafish_nnue_sha256=config["engine"]["eval_file_sha256"], on_result=checkpoint)
        else:
            results = c2.evaluate_rule_variant_tasks([runtime_dict(record)], agent, on_result=checkpoint)
        return [asdict(result) if is_dataclass(result) else result for result in results]

    try:
        with PikafishEngine(engine_path, timeout=60.0) as reference_engine:
            for family in families:
                results = run_case(family, "correct_scripted_agent", CorrectAgent(reference_engine))
                checks[family + ".correct"] = all(r["status"] == "ok" and r["success"] is True for r in results)
                bad = FixedAgent()
                results = run_case(family, "returned_invalid_answer", bad)
                checks[family + ".invalid_answer"] = all(r["status"] == "invalid" and r["success"] is False for r in results)
                checks[family + ".one_answer_per_condition"] = bad.calls == len(results)
                checks[family + ".state_isolation"] = bad.resets == len(results)
                timeout = FixedAgent(timeout=True)
                results = run_case(family, "provider_timeout", timeout)
                checks[family + ".timeout_missing"] = all(r["status"] == "error" and r["success"] is None for r in results)
                checks[family + ".no_evaluator_retry"] = timeout.calls == len(results)
            unavailable = output_dir / "intentionally-missing-pikafish"
            for family in ("d3", "m2"):
                if family in families:
                    results = run_case(family, "diagnostic_engine_fault", CorrectAgent(), path=unavailable)
                    checks[family + ".diagnostic_fault_keeps_success"] = all(r["status"] == "ok" and r["success"] is True for r in results)
                    checks[family + ".diagnostic_fault_unknown_cp"] = all(r.get("cp_loss", r.get("engine_cp_loss")) is None for r in results)
            if "d3" in families:
                wrong = legal_non_mate(xiangqi_task_from_dict(selected["d3"]))
                results = run_case("d3", "legal_wrong_answer", FixedAgent(json.dumps({"move": wrong})))
                checks["d3.legal_wrong_fails"] = all(r["is_legal"] and r["status"] == "ok" and r["success"] is False for r in results)
            if "h2" in families:
                move = legal_non_mate(xiangqi_task_from_dict(selected["h2"]), require_reply=True)
                results = run_case("h2", "required_opponent_fault", FixedAgent(json.dumps({"move": move})), path=unavailable)
                checks["h2.required_opponent_fault_missing"] = all(r["status"] == "error" and r["success"] is None and r["error"]["stage"] == "opponent_engine" for r in results)
    except BaseException as exc:
        report = {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", "error": {"type": type(exc).__name__, "reason": str(exc)}, "config": config, "checks": checks, "result_count": len(recorded)}
        (output_dir / "protocol_acceptance.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise
    report = {"status": "passed" if all(checks.values()) else "failed", "config": config, "checks": checks, "result_count": len(recorded), "failed_checks": [name for name, passed in checks.items() if not passed]}
    (output_dir / "protocol_acceptance.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pikafish-path", required=True, type=Path)
    parser.add_argument("--families", default="d3,h2,m2,c2")
    args = parser.parse_args()
    families = args.families.split(",")
    if len(set(families)) != len(families) or set(families) - set(FAMILIES):
        parser.error("families must be a unique comma-separated subset of d3,h2,m2,c2")
    report = run_acceptance(args.output_dir.resolve(), args.pikafish_path.resolve(), families)
    print(json.dumps({"status": report["status"], "result_count": report["result_count"], "failed_checks": report["failed_checks"], "report": str(args.output_dir / "protocol_acceptance.json")}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
