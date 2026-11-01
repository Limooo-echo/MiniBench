import json
from pathlib import Path
import unittest
from unittest.mock import patch

from minibench.factory.agents import make_agent_from_config
from minibench.factory.config import load_experiment_config, validate_experiment_config


class HTTPResponse:
    def __init__(self, content, *, reasoning=None, finish="stop", tokens=17):
        self.payload = {
            "choices": [{"message": {
                "role": "assistant", "content": content,
                "reasoning_content": reasoning,
            }, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": tokens,
                "total_tokens": 10 + tokens,
                "completion_tokens_details": {
                    "reasoning_tokens": tokens if reasoning else 0,
                },
            },
        }

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def thinking_config():
    path = Path("config/experiments/mahjong_solo.yaml")
    config = load_experiment_config(path)
    config["agent"].update({
        "native_thinking_max_tokens": 7168,
        "max_tokens": 1024,
        "trace": "full",
    })
    config["provider"]["api_key_env"] = "TEST_NATIVE_THINKING_KEY"
    config["provider"]["extra_body"]["thinking"]["type"] = "enabled"
    return validate_experiment_config(config)


class DirectNativeThinkingTests(unittest.TestCase):
    def test_native_thinking_carries_previous_reasoning_in_both_draw_discard_modes(self):
        from minibench.datasets.mahjong_solo.dataset import mahjong_solo_task_from_dict
        from minibench.datasets.mahjong_solo.evaluation import evaluate_mahjong_solo_task

        task = mahjong_solo_task_from_dict({"id": "native-memory",
            "initial_hand": "1m 2m 3m 4m 5m 6m 7p 8p 9p 2s 3s 4s E".split(),
            "wall": ["9s", "E"], "max_draws": 2})
        for mode in ("full-hand", "history-only"):
            with self.subTest(mode=mode):
                config = thinking_config()
                agent = make_agent_from_config(config["agent"], config["provider"])
                with patch.dict("os.environ", {"TEST_NATIVE_THINKING_KEY": "offline"}), patch(
                    "urllib.request.urlopen", side_effect=[
                        HTTPResponse('{"action":"discard","tile":"9s"}', reasoning="NATIVE_MEMORY_KEEP_E"),
                        HTTPResponse('{"action":"tsumo"}', reasoning="The draw completes the pair."),
                    ],
                ) as http:
                    result = evaluate_mahjong_solo_task(task, agent, observation_mode=mode)
                self.assertTrue(result.success)
                self.assertEqual(result.discards, ["9s"])
                second = json.loads(http.call_args_list[1].args[0].data)
                self.assertIn("NATIVE_MEMORY_KEEP_E", json.dumps(second["messages"]))
                observation = second["messages"][-1]["content"]
                self.assertIn("You draw E.", observation)
                self.assertEqual("Hand after previous accepted discard" in observation,
                                 mode == "full-hand")

    def test_mahjong_image_fallback_preserves_images_and_scores_a_truncated_answer(self):
        from minibench.datasets.mahjong.dataset import mahjong_task_from_dict
        from minibench.datasets.mahjong.evaluation import evaluate_mahjong_tasks

        source = Path("data/mahjong/visual_tasks.jsonl")
        raw = json.loads(source.read_text(encoding="utf-8").splitlines()[0])
        task = mahjong_task_from_dict(raw, source_path=source)
        for mode in ("text", "image"):
            with self.subTest(mode=mode):
                config = thinking_config()
                agent = make_agent_from_config(config["agent"], config["provider"])
                with patch.dict("os.environ", {"TEST_NATIVE_THINKING_KEY": "offline"}), patch(
                    "urllib.request.urlopen", side_effect=[
                        HTTPResponse("", reasoning="Partial tile analysis.", finish="length", tokens=7168),
                        HTTPResponse('{"winning_tiles":', finish="length", tokens=1024),
                    ],
                ) as http:
                    result = evaluate_mahjong_tasks([task], agent, input_modes=(mode,))[0]
                self.assertFalse(result.success)
                self.assertFalse(result.is_legal)
                self.assertEqual(result.metrics["llm_calls"], 2)
                bodies = [json.loads(call.args[0].data) for call in http.call_args_list]
                images = [[part["image_url"]["url"] for message in body["messages"]
                           if isinstance(message["content"], list)
                           for part in message["content"] if part.get("type") == "image_url"]
                          for body in bodies]
                self.assertEqual(images[0], images[1])
                self.assertEqual(len(images[0]), 1 if mode == "image" else 0)

    def test_mahjong_answer_failures_are_scored_but_transport_errors_stay_unknown(self):
        from urllib.error import URLError
        from minibench.datasets.mahjong_solo.dataset import mahjong_solo_task_from_dict
        from minibench.datasets.mahjong_solo.evaluation import (
            evaluate_mahjong_solo_task, summarize_mahjong_solo,
        )

        task = mahjong_solo_task_from_dict({"id": "native-answer-failure",
            "initial_hand": "1m 2m 3m 4m 5m 6m 7p 8p 9p 2s 3s 4s E".split(),
            "wall": ["E"], "max_draws": 1})
        for mode in ("full-hand", "history-only"):
            for final, finish in (('{"action":', "length"), ("bad JSON", "stop"), ("", "length")):
                with self.subTest(mode=mode, answer=final, finish=finish):
                    config = thinking_config()
                    agent = make_agent_from_config(config["agent"], config["provider"])
                    with patch.dict("os.environ", {"TEST_NATIVE_THINKING_KEY": "offline"}), patch(
                        "urllib.request.urlopen", side_effect=[
                            HTTPResponse("", reasoning="Partial reasoning.", finish="length", tokens=7168),
                            HTTPResponse(final, finish=finish, tokens=1024),
                        ],
                    ):
                        result = evaluate_mahjong_solo_task(task, agent, observation_mode=mode)
                    summary = summarize_mahjong_solo([result])
                    self.assertEqual((result.evaluation_status, result.success), ("completed", False))
                    self.assertEqual((summary["evaluated_total"], summary["api_error_total"],
                                      summary["success_rate"]), (1, 0, 0.0))
                    self.assertEqual(result.raw_outputs, [final])
                    self.assertEqual(result.metrics["llm_calls"], 2)
                    self.assertEqual(result.metrics["token_usage"]["completion_tokens"], 8192)
            for response, status in ((HTTPResponse("bad JSON", reasoning="Finished reasoning."), "completed"),
                                     (URLError("simulated transport failure"), "api_error")):
                config = thinking_config()
                config["provider"]["max_retries"] = 0
                agent = make_agent_from_config(config["agent"], config["provider"])
                with patch.dict("os.environ", {"TEST_NATIVE_THINKING_KEY": "offline"}), patch(
                    "urllib.request.urlopen", side_effect=[response],
                ):
                    result = evaluate_mahjong_solo_task(task, agent, observation_mode=mode)
                self.assertEqual(result.evaluation_status, status)
                self.assertEqual(result.success, False if status == "completed" else None)

    def test_mahjong_configs_use_native_thinking_with_a_reserved_final_budget(self):
        names = ["mahjong", "mahjong_multimodal", "mahjong_rule_variants",
                 "mahjong_solo", "mahjong_history_only"]
        for name in names:
            with self.subTest(config=name):
                config = load_experiment_config(Path("config/experiments") / f"{name}.yaml")
                self.assertEqual(config["agent"]["name"], "direct")
                self.assertEqual(config["agent"]["native_thinking_max_tokens"], 7168)
                self.assertEqual(config["agent"]["max_tokens"], 1024)
                self.assertEqual(config["provider"]["model"], "deepseek-flash")
                agent = make_agent_from_config(config["agent"], config["provider"])
                with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "offline"}), patch(
                    "urllib.request.urlopen", side_effect=[
                        HTTPResponse("", reasoning="Use the proposed answer.", finish="length", tokens=7168),
                        HTTPResponse('{"action":"tsumo"}'),
                    ],
                ) as http:
                    self.assertEqual(json.loads(agent.generate("Return a JSON answer.", {})),
                                     {"action": "tsumo"})
                requests = [json.loads(call.args[0].data) for call in http.call_args_list]
                self.assertEqual([r["max_tokens"] for r in requests], [7168, 1024])
                self.assertEqual([r["thinking"]["type"] for r in requests], ["enabled", "disabled"])

    def test_truncated_thinking_is_used_to_finish_with_reserved_answer_budget(self):
        config = thinking_config()
        agent = make_agent_from_config(config["agent"], config["provider"])
        partial = "The current hand is complete; the action should be tsumo."
        responses = [
            HTTPResponse("", reasoning=partial, finish="length", tokens=7168),
            HTTPResponse('{"action":"tsumo"}'),
        ]
        with patch.dict("os.environ", {"TEST_NATIVE_THINKING_KEY": "offline"}):
            with patch("urllib.request.urlopen", side_effect=responses) as http:
                output = agent.generate("Mahjong turn: return a JSON action.", {})

        self.assertEqual(json.loads(output), {"action": "tsumo"})
        requests = [json.loads(call.args[0].data) for call in http.call_args_list]
        self.assertEqual([r["max_tokens"] for r in requests], [7168, 1024])
        self.assertEqual([r["thinking"]["type"] for r in requests], ["enabled", "disabled"])
        self.assertIn(partial, json.dumps(requests[1]["messages"]))
        self.assertIn("Mahjong turn", json.dumps(requests[1]["messages"]))
        metrics = agent.runtime.metrics_snapshot()
        self.assertEqual(metrics["llm_calls"], 2)
        self.assertEqual(metrics["token_usage"]["completion_tokens"], 7185)
        self.assertEqual(metrics["token_usage"]["reasoning_tokens"], 7168)
        self.assertEqual(agent.last_run.stages[0].reasoning, partial)
        self.assertEqual(agent.last_run.stages[0].finish_reason, "length")

    def test_valid_answer_is_returned_without_a_finalization_request(self):
        config = thinking_config()
        agent = make_agent_from_config(config["agent"], config["provider"])
        response = HTTPResponse('{"action":"tsumo"}', reasoning="Complete hand.")
        with patch.dict("os.environ", {"TEST_NATIVE_THINKING_KEY": "offline"}):
            with patch("urllib.request.urlopen", return_value=response) as http:
                output = agent.generate("Return a JSON action.", {})

        self.assertEqual(json.loads(output), {"action": "tsumo"})
        self.assertEqual(http.call_count, 1)
        self.assertEqual(agent.runtime.metrics_snapshot()["llm_calls"], 1)

    def test_thinking_is_reenabled_for_the_next_decision(self):
        config = thinking_config()
        config["provider"]["extra_body"]["reasoning_effort"] = "high"
        agent = make_agent_from_config(config["agent"], config["provider"])
        responses = [
            HTTPResponse("", reasoning="Discard 9m.", finish="length", tokens=7168),
            HTTPResponse('{"action":"discard","tile":"9m"}'),
            HTTPResponse('{"action":"tsumo"}', reasoning="Complete hand."),
        ]
        with patch.dict("os.environ", {"TEST_NATIVE_THINKING_KEY": "offline"}):
            with patch("urllib.request.urlopen", side_effect=responses) as http:
                agent.generate("First turn.", {})
                output = agent.generate("Next turn.", {})

        requests = [json.loads(call.args[0].data) for call in http.call_args_list]
        self.assertEqual(json.loads(output), {"action": "tsumo"})
        self.assertEqual(
            [r["thinking"]["type"] for r in requests],
            ["enabled", "disabled", "enabled"],
        )
        self.assertEqual([r["max_tokens"] for r in requests], [7168, 1024, 7168])
        self.assertNotIn("reasoning_effort", requests[1])
        self.assertEqual(requests[2]["reasoning_effort"], "high")

    def test_plain_direct_keeps_one_request_and_its_original_budget(self):
        config = thinking_config()
        del config["agent"]["native_thinking_max_tokens"]
        config["agent"]["max_tokens"] = 8192
        config["provider"]["extra_body"]["thinking"]["type"] = "disabled"
        agent = make_agent_from_config(config["agent"], config["provider"])
        with patch.dict("os.environ", {"TEST_NATIVE_THINKING_KEY": "offline"}):
            with patch("urllib.request.urlopen", return_value=HTTPResponse('{"action":"tsumo"}')) as http:
                output = agent.generate("Return a JSON action.", {})

        self.assertEqual(json.loads(output), {"action": "tsumo"})
        self.assertEqual(http.call_count, 1)
        request = json.loads(http.call_args.args[0].data)
        self.assertEqual(request["max_tokens"], 8192)
        self.assertEqual(request["thinking"], {"type": "disabled"})

    def test_empty_nontruncated_response_is_not_mistaken_for_budget_exhaustion(self):
        config = thinking_config()
        agent = make_agent_from_config(config["agent"], config["provider"])
        with patch.dict("os.environ", {"TEST_NATIVE_THINKING_KEY": "offline"}):
            with patch("urllib.request.urlopen", return_value=HTTPResponse("", reasoning="Draft.")) as http:
                with self.assertRaisesRegex(RuntimeError, "empty message content"):
                    agent.generate("Return a JSON action.", {})
        self.assertEqual(http.call_count, 1)

    def test_answer_budget_exhaustion_does_not_trigger_a_third_request(self):
        config = thinking_config()
        agent = make_agent_from_config(config["agent"], config["provider"])
        responses = [
            HTTPResponse("", reasoning="Partial draft.", finish="length", tokens=7168),
            HTTPResponse('{"action":', finish="length", tokens=1024),
        ]
        with patch.dict("os.environ", {"TEST_NATIVE_THINKING_KEY": "offline"}):
            with patch("urllib.request.urlopen", side_effect=responses) as http:
                with self.assertRaisesRegex(RuntimeError, "truncated"):
                    agent.generate("Return a JSON action.", {})
        self.assertEqual(http.call_count, 2)

    def test_partial_json_is_finished_using_the_existing_reasoning(self):
        config = thinking_config()
        agent = make_agent_from_config(config["agent"], config["provider"])
        responses = [
            HTTPResponse('{"action":', reasoning="Declare tsumo.", finish="length", tokens=7168),
            HTTPResponse('{"action":"tsumo"}'),
        ]
        with patch.dict("os.environ", {"TEST_NATIVE_THINKING_KEY": "offline"}):
            with patch("urllib.request.urlopen", side_effect=responses) as http:
                output = agent.generate("Return a JSON action.", {})
        request = json.loads(http.call_args.args[0].data)
        self.assertEqual(json.loads(output), {"action": "tsumo"})
        self.assertIn('Partial final answer:\n{"action":', request["messages"][-1]["content"])

    def test_message_entrypoint_preserves_history_and_the_reserved_budget(self):
        config = thinking_config()
        agent = make_agent_from_config(config["agent"], config["provider"])
        messages = [
            {"role": "system", "content": "Mahjong rules."},
            {"role": "user", "content": "Initial hand."},
            {"role": "assistant", "content": '{"action":"discard","tile":"9m"}'},
            {"role": "user", "content": "Next draw."},
        ]
        original = json.dumps(messages)
        responses = [
            HTTPResponse("", reasoning="Complete hand.", finish="length", tokens=7168),
            HTTPResponse('{"action":"tsumo"}'),
        ]
        with patch.dict("os.environ", {"TEST_NATIVE_THINKING_KEY": "offline"}):
            with patch("urllib.request.urlopen", side_effect=responses) as http:
                output = agent.generate_messages(messages, {}, max_tokens=4096)
        request = json.loads(http.call_args.args[0].data)
        self.assertEqual(json.loads(output), {"action": "tsumo"})
        self.assertEqual(request["max_tokens"], 1024)
        self.assertIn("Initial hand.", json.dumps(request["messages"]))
        self.assertIn("Next draw.", json.dumps(request["messages"]))
        self.assertEqual(json.dumps(messages), original)
