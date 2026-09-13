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
from minibench.datasets.xiangqi.engines.pikafish import resolve_pikafish_executable
from minibench.factory.config import load_experiment_config
from minibench.factory.experiments import _evaluate, get_task_family_spec


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
                f"{task_id}-call{task_call:02d}-image{image_index}.{suffix}"
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Drive MiniBench Xiangqi evaluation using responses pasted from a web UI."
    )
    parser.add_argument(
        "--suite-dir", type=Path, required=True,
        help="A directory produced by scripts/run_xiangqi_smoke.py.",
    )
    parser.add_argument("--tasks", default="d3,h2,c2,m2")
    parser.add_argument(
        "--history-mode",
        choices=("paired", "full-state", "move-history-only"),
        default="paired",
    )
    parser.add_argument(
        "--m2-modes",
        default="text,chinese-piece-image,latin-piece-image",
    )
    parser.add_argument("--pikafish-depth", type=int, default=8)
    args = parser.parse_args(argv)

    suite_dir = args.suite_dir.resolve()
    if not suite_dir.is_dir():
        raise SystemExit(f"suite directory does not exist: {suite_dir}")
    session_dir = suite_dir / "web-tests" / datetime.now().strftime("%Y%m%d-%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=False)
    agent = ManualWebAgent(session_dir)
    summaries: dict[str, Any] = {}
    pikafish_path = resolve_pikafish_executable(None, start_dir=ROOT)

    try:
        for short_name in _parse_tasks(args.tasks):
            family, config_name = TASKS[short_name]
            sample_path = suite_dir / "samples" / f"{short_name}.jsonl"
            if not sample_path.is_file():
                raise FileNotFoundError(f"missing saved sample: {sample_path}")
            task_dir = session_dir / short_name
            task_dir.mkdir()
            shutil.copy2(sample_path, task_dir / "selected_tasks.jsonl")
            spec = get_task_family_spec(family)
            agent.task_system_prompt = spec.system_prompt
            tasks = spec.load_tasks(sample_path)
            config = load_experiment_config(ROOT / "config/experiments" / config_name)
            evaluation = dict(config.get("evaluation") or {})
            evaluation["pikafish_depth"] = args.pikafish_depth
            if short_name in {"d3", "h2", "m2"}:
                evaluation["pikafish_path"] = str(pikafish_path)
            if short_name == "h2":
                evaluation["history_mode"] = args.history_mode
            if short_name == "m2":
                evaluation.update(
                    input_modes=tuple(
                        item.strip() for item in args.m2_modes.split(",") if item.strip()
                    ),
                    max_plies=1,
                    verify_with_pikafish=True,
                    step_dir=str(task_dir / "rendered-inputs"),
                )
            print(f"\n######## {short_name.upper()} ({len(tasks)} saved records) ########")
            results = _evaluate(spec, tasks, agent, family, evaluation)
            run_dir = spec.write_run(results, task_dir, "results")
            summaries[short_name] = {
                "run_dir": str(run_dir),
                "summary": spec.summarize(results),
                "results": [_serializable(result) for result in results],
            }
    except KeyboardInterrupt:
        status = "stopped_by_user"
    except Exception as exc:
        status = "failed"
        summaries["error"] = f"{type(exc).__name__}: {exc}"
    else:
        status = "completed"

    report = {
        "status": status,
        "source_suite_dir": str(suite_dir),
        "session_dir": str(session_dir),
        "tasks": summaries,
    }
    (session_dir / "web_test_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
