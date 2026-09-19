"""Interactive browser bridge for the exact D3/H2/C2/M2 evaluation code.

The script prints the prompt that should be sent to a web model, waits for the
user to paste its one-line JSON response, and then lets the normal MiniBench
evaluator validate/apply the move. H2 therefore receives real Pikafish replies
and M2 image inputs are written as uploadable PNG files.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
import hashlib
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from minibench.core.agent import ChatMessage
from minibench.core.multimodal import ImageAttachment
from minibench.core.prompts import FINAL_ANSWER_SYSTEM_PROMPT, direct_prompt
from minibench.datasets.xiangqi.engines.pikafish import resolve_pikafish_executable, pikafish_fingerprint
from minibench.datasets.xiangqi.runtime import PROTOCOL_VERSION
from minibench.factory.config import load_experiment_config
from minibench.factory.experiments import _evaluate, get_task_family_spec, _select_frozen_xiangqi_tasks


TASKS = {
    "d3": ("xiangqi-mate-in-one", "xiangqi_mate_in_one.yaml"),
    "h2": ("xiangqi-history", "xiangqi_history.yaml"),
    "c2": ("xiangqi-rule-variants", "xiangqi_rule_variants.yaml"),
    "m2": ("xiangqi-multimodal", "xiangqi_multimodal.yaml"),
}


class ManualWebAgent:
    """Agent adapter that obtains each answer from a browser chat manually."""

    name = "manual-web"

    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self.image_dir = session_dir / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_path = session_dir / "browser_transcript.jsonl"
        self.calls_by_session: dict[tuple[str, str], int] = {}
        self.call_index = 0
        self.task_system_prompt: str | None = None

    def reset(self) -> None:
        # Each task/mode starts a new chat; cumulative usage counters stay intact.
        self.calls_by_session.clear()

    @staticmethod
    def _task_id(task: Any) -> str:
        if isinstance(task, dict):
            return str(task.get("id", "unknown-task"))
        return str(getattr(task, "id", "unknown-task"))

    def _read_response(
        self,
        *,
        task: Any,
        transport: str,
        prompt: str | None = None,
        messages: Sequence[ChatMessage] | None = None,
        images: Sequence[ImageAttachment] = (),
        persistent: bool = False,
    ) -> str:
        task_id = self._task_id(task)
        session_key = (task_id, transport)
        task_call = self.calls_by_session.get(session_key, 0) + 1
        self.calls_by_session[session_key] = task_call
        self.call_index += 1
        image_paths: list[str] = []
        for image_index, attachment in enumerate(images, start=1):
            suffix = "png" if attachment.mime_type == "image/png" else "bin"
            path = self.image_dir / (
                f"{task_id}-call{self.call_index:04d}-image{image_index}.{suffix}"
            )
            path.write_bytes(attachment.data)
            image_paths.append(str(path.resolve()))

        print("\n" + "=" * 88)
        print(f"WEB CALL {self.call_index} | task={task_id} | transport={transport}")
        if messages is not None:
            if not persistent or task_call == 1:
                print("Open a NEW browser chat and send the following conversation:")
                for message in messages:
                    print(f"\n[{str(message['role']).upper()}]\n{message['content']}")
            else:
                print("Continue the SAME browser chat and send only this next user turn:")
                print(f"\n[USER]\n{messages[-1]['content']}")
        else:
            print("Open a NEW browser chat for this stateless call and send:")
            print(f"\n[USER]\n{prompt or ''}")
        for path in image_paths:
            print(f"\nAttach image: {path}")
        print("\nPaste the model's one-line JSON response below (`:quit` to stop):")
        response = input("> ").strip()
        if response == ":quit":
            raise KeyboardInterrupt("manual browser session stopped")

        record = {
            "call_index": self.call_index,
            "task_id": task_id,
            "task_call_index": task_call,
            "transport": transport,
            "prompt": prompt,
            "messages": list(messages) if messages is not None else None,
            "image_paths": image_paths,
            "response": response,
        }
        with self.transcript_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return response

    def generate(self, prompt: str, task: Any) -> str:
        system = "\n\n".join(
            item for item in (self.task_system_prompt, FINAL_ANSWER_SYSTEM_PROMPT)
            if item
        )
        messages: list[ChatMessage] = [
            {"role": "system", "content": system},
            {"role": "user", "content": direct_prompt(prompt)},
        ]
        return self._read_response(
            task=task, transport="stateless", messages=messages
        )

    def generate_multimodal(
        self,
        prompt: str,
        task: Any,
        *,
        images: Sequence[ImageAttachment],
    ) -> str:
        system = "\n\n".join(
            item for item in (self.task_system_prompt, FINAL_ANSWER_SYSTEM_PROMPT)
            if item
        )
        messages: list[ChatMessage] = [
            {"role": "system", "content": system},
            {"role": "user", "content": direct_prompt(prompt)},
        ]
        return self._read_response(
            task=task, transport="multimodal", messages=messages, images=images
        )

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        task: Any,
        **_options: Any,
    ) -> str:
        prepared = [dict(message) for message in messages]
        for message in reversed(prepared):
            if message["role"] == "user":
                message["content"] = direct_prompt(str(message["content"]))
                break
        for message in prepared:
            if message["role"] == "system":
                message["content"] = (
                    f"{message['content']}\n\n{FINAL_ANSWER_SYSTEM_PROMPT}"
                )
                break
        return self._read_response(
            task=task,
            transport="persistent-chat",
            messages=prepared,
            persistent=True,
        )

    def metrics_snapshot(self) -> dict[str, Any]:
        return {
            "model_elapsed_seconds": 0.0,
            "llm_calls": self.call_index,
            "usage_missing_calls": self.call_index,
            "token_usage": {},
        }


def _parse_tasks(value: str) -> list[str]:
    names = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(names) - set(TASKS))
    if unknown:
        raise ValueError(f"unknown task(s): {', '.join(unknown)}")
    return list(dict.fromkeys(names))


def _serializable(result: Any) -> Any:
    if is_dataclass(result):
        return asdict(result)
    return result


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_web_session(suite_dir: Path, session_dir: Path, names: list[str], *,
                        pikafish_path: Path | None = None, history_mode: str | None = None,
                        m2_modes: Sequence[str] | None = None):
    """Reuse saved configs and exact rosters; never silently resample or lower depth."""
    source_plan = json.loads((suite_dir / "smoke_plan.json").read_text(encoding="utf-8"))
    if source_plan.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("manual sessions require a prepared current-protocol smoke suite")
    session_dir.mkdir(parents=True, exist_ok=False)
    prepared = {}
    plan = {"protocol_version": PROTOCOL_VERSION, "source_suite_dir": str(suite_dir),
            "agent_adapter": "manual-web using direct prompts without automatic format repair",
            "status": "prepared", "tasks": {}}
    for key in names:
        entry = source_plan["tasks"][key]
        config_path, sample_path = suite_dir / entry["config_path"], suite_dir / entry["sample_path"]
        if _digest(config_path) != entry["config_sha256"] or _digest(sample_path) != entry["sample_sha256"]:
            raise ValueError(f"{key}: saved config/selection hash mismatch")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        family = config["task"]["family"]
        if family != TASKS[key][0]:
            raise ValueError(f"{key}: saved family mismatch")
        data_path = Path(config["task"]["path"])
        if _digest(data_path) != entry["source_sha256"]:
            raise ValueError(f"{key}: source dataset changed since suite preparation")
        config["task"]["selection"] = {"path": str(sample_path), "sha256": entry["sample_sha256"]}
        spec = get_task_family_spec(family)
        tasks = _select_frozen_xiangqi_tasks(spec.load_tasks(data_path), config["task"], data_path, family)
        run_dir = session_dir / key / "results"
        run_dir.mkdir(parents=True)
        selected = run_dir / "selected_tasks.jsonl"
        shutil.copyfile(sample_path, selected)
        config["task"]["selection"] = {"path": str(selected), "sha256": entry["sample_sha256"]}
        config["run"].update(output_dir=str(run_dir.parent), run_name="results", on_existing="error")
        evaluation = config.setdefault("evaluation", {})
        actual_engine = None
        if key in {"d3", "h2", "m2"}:
            engine = resolve_pikafish_executable(pikafish_path or evaluation.get("pikafish_path"), start_dir=ROOT)
            evaluation.update(pikafish_path=str(engine.resolve()), pikafish_depth=16 if key == "h2" else 8)
            actual_engine = pikafish_fingerprint(engine)
        if key == "h2" and history_mode is not None:
            evaluation["history_mode"] = history_mode
        if key == "m2":
            if m2_modes is not None:
                evaluation["input_modes"] = list(m2_modes)
            evaluation.update(max_plies=1, verify_with_pikafish=True, step_dir=str(run_dir / "rendered-inputs"))
        _write_json(run_dir / "resolved_config.json", config)
        _write_json(run_dir / "run_metadata.json", {
            "protocol_version": PROTOCOL_VERSION, "agent_adapter": plan["agent_adapter"],
            "source_sha256": entry["source_sha256"], "selection_sha256": entry["sample_sha256"],
            "engine": actual_engine, "evaluation": evaluation,
        })
        _write_json(run_dir / "run_state.json", {"status": "prepared", "completed_results": 0})
        (run_dir / "predictions.jsonl").touch(exist_ok=False)
        plan["tasks"][key] = {"family": family, "run_dir": str(run_dir), "selected_record_count": len(tasks),
                                "config_sha256": _digest(run_dir / "resolved_config.json"), "selection_sha256": entry["sample_sha256"]}
        prepared[key] = (spec, tasks, config, run_dir)
    _write_json(session_dir / "web_test_plan.json", plan)
    return prepared


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Use pasted web answers with the exact saved Xiangqi protocol and roster.")
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--tasks", default="d3,h2,c2,m2")
    parser.add_argument("--history-mode", choices=("paired", "full-state", "move-history-only"))
    parser.add_argument("--m2-modes", help="Optional explicit mode subset, saved in the actual configuration")
    parser.add_argument("--pikafish-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    suite_dir = args.suite_dir.resolve()
    session_dir = (args.output_dir or suite_dir / "web-tests" / datetime.now().strftime("%Y%m%d-%H%M%S")).resolve()
    names = _parse_tasks(args.tasks)
    if not names:
        raise ValueError("at least one task family is required")
    modes = [item.strip() for item in args.m2_modes.split(",") if item.strip()] if args.m2_modes else None
    prepared = prepare_web_session(suite_dir, session_dir, names, pikafish_path=args.pikafish_path,
                                   history_mode=args.history_mode, m2_modes=modes)
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "session_dir": str(session_dir), "model_calls": 0}))
        return 0
    agent = ManualWebAgent(session_dir)
    summaries = {}
    active_dir = None
    completed = []
    status = "completed"
    try:
        for key in names:
            spec, tasks, config, active_dir = prepared[key]
            agent.task_system_prompt = spec.system_prompt
            completed = []
            _write_json(active_dir / "run_state.json", {"status": "running", "completed_results": 0})
            def persist(result):
                record = _serializable(result)
                with (active_dir / "predictions.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                completed.append(record)
                _write_json(active_dir / "run_state.json", {"status": "running", "completed_results": len(completed)})
            results = _evaluate(spec, tasks, agent, config["task"]["family"], config["evaluation"], on_result=persist)
            spec.write_run(results, active_dir.parent, active_dir.name, write_predictions=False)
            summaries[key] = {"run_dir": str(active_dir), "summary": spec.summarize(results), "completed_results": len(completed)}
            _write_json(active_dir / "run_state.json", {"status": "completed", "completed_results": len(completed)})
    except KeyboardInterrupt:
        status = "stopped_by_user"
    except Exception as exc:
        status = "failed"
        summaries["error"] = f"{type(exc).__name__}: {exc}"
    if status != "completed" and active_dir is not None:
        _write_json(active_dir / "run_state.json", {"status": status, "completed_results": len(completed), "error": summaries.get("error")})
    report = {"status": status, "source_suite_dir": str(suite_dir), "session_dir": str(session_dir), "tasks": summaries}
    _write_json(session_dir / "web_test_results.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
