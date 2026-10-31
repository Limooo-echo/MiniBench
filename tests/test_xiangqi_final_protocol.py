"""Regression acceptance for the FINAL Xiangqi-only protocol, without API calls."""
from copy import deepcopy
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
import warnings

from PIL import Image
import yaml

from minibench import cloud
from minibench.core.multimodal import ImageAttachment
from minibench.datasets.xiangqi.multimodal import (
    _extract_uci, evaluate_xiangqi_multimodal_tasks, render_board_manifest,
    render_board_png, score_board_recognition, summarize_xiangqi_multimodal,
)
from minibench.datasets.xiangqi.schema import FAMILY_PATHS, load_records, runtime_dict
from minibench.datasets.xiangqi.text_encoding import FILES, piece_symbol
from minibench.factory.providers import OpenAICompatibleClient
from minibench.datasets.xiangqi.solution_tree import validate_tree_record,TreeSession
from tests.test_reasoning_agents import FakeClient, one_level_tot_responses


def transcription(board):
    return {f'{FILES[c]}{9-r}': piece_symbol(p)
            for r, row in enumerate(board) for c, p in enumerate(row) if p}


class FinalProtocolTests(unittest.TestCase):
    def test_all_shipped_xiangqi_configs_share_thinking_and_deadline_policy(self):
        from minibench.factory.config import load_experiment_config
        paths=list(Path('config/experiments').glob('xiangqi_*.yaml'))
        paths+=list(Path('config/experiments/xiangqi_smoke').glob('*.yaml'))
        self.assertEqual(len(paths),8)
        for path in paths:
            with self.subTest(path=path):
                config=load_experiment_config(path)
                self.assertTrue(config['provider']['extra_body']['enable_thinking'])
                self.assertTrue(config['provider']['omit_max_tokens'])
                self.assertEqual(config['provider']['max_retries'],0)
                self.assertEqual(config['evaluation']['final_move_timeout_seconds'],900)
                self.assertNotIn('max_tokens',config['agent'])
                if config['task']['family']=='xiangqi-rule-variants':
                    self.assertEqual(config['evaluation']['horizon_policy'],
                                     'certified_shortest_attacker_moves')

    def test_candidate_tools_cannot_overwrite_release_even_via_alias(self):
        from minibench.datasets.xiangqi.provenance import require_candidate_outputs
        root=Path(__file__).resolve().parents[1]
        for target in ('data/xiangqi/history/tasks.jsonl',
                       'data/xiangqi/rule_variants/runtime_certificates.json',
                       'data/xiangqi/../xiangqi/provenance.json'):
            with self.subTest(target=target),self.assertRaisesRegex(ValueError,'Candidate-only'):
                require_candidate_outputs(root/target)
        with tempfile.TemporaryDirectory() as temporary:
            alias=Path(temporary)/'alias'
            alias.symlink_to(root/'data/xiangqi',target_is_directory=True)
            with self.assertRaises(ValueError):
                require_candidate_outputs(alias/'history/tasks.jsonl')
            require_candidate_outputs(Path(temporary)/'candidate.jsonl',None)

    def test_every_submitted_m2_png_and_serialized_request(self):
        client = OpenAICompatibleClient(model='offline-test', base_url='https://invalid.example',
                                        api_key_env='MINIBENCH_UNUSED_OFFLINE_KEY')
        records = load_records(FAMILY_PATHS['xiangqi-multimodal'])
        self.assertEqual(len(records), 100)
        for record in records:
            with self.subTest(task=record['id']):
                board = runtime_dict(record)['board']
                with warnings.catch_warnings():
                    warnings.simplefilter('error', UserWarning)
                    data = render_board_png(board, 'chinese-piece-image')
                attachment = ImageAttachment(data=data, mime_type='image/png')
                payload = client.build_payload('Current board: attached image.', images=[attachment])
                content = payload['messages'][-1]['content']
                part = next(p for p in content if p['type'] == 'image_url')
                self.assertEqual(base64.b64decode(part['image_url']['url'].split(',', 1)[1]), data)
                with Image.open(BytesIO(data)) as image:
                    image.load()
                    self.assertGreater(image.width, 500)
                    self.assertGreater(image.height, 500)
                    self.assertEqual(image.info['minibench_mode'], 'chinese-piece-image')
                    manifest = json.loads(image.info['minibench_piece_manifest'])
                    self.assertEqual(manifest, render_board_manifest(board, 'chinese-piece-image'))
                    self.assertEqual({p['square']: piece_symbol(p['piece_id']) for p in manifest}, transcription(board))

    def test_submitted100_four_corpora_and_one_m2_mode(self):
        raw = yaml.safe_load(Path('config/cloud/xiangqi-all-agents.yaml').read_text())
        raw['engine_binding'] = 'reference'
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / 'cloud.yaml'; cfg.write_text(yaml.safe_dump(raw))
            plan = cloud.prepare(cfg, Path(tmp) / 'suite')
        self.assertEqual(plan['planned_jobs'], 5400)
        for entry in plan['experiments'].values():
            rows = load_records(FAMILY_PATHS[entry['family']])
            self.assertEqual(len(rows), 200 if entry['family'] == 'xiangqi-rule-variants' else 100)
            self.assertEqual(entry['task_ids'], [r['id'] for r in rows])
            if entry['family'] in {'xiangqi-history', 'xiangqi-rule-variants'}:
                from minibench.datasets.xiangqi.schema import validate_record
                for row in rows:
                    self.assertEqual(row['max_plies'], 2*row['shortest_mate_moves']-1)
                    if entry['family'] == 'xiangqi-rule-variants':
                        self.assertEqual(row['c2_analysis']['max_plies_extension'], 0)
                    stale = deepcopy(row)
                    stale['max_plies'] += 4
                    with self.assertRaisesRegex(ValueError, 'horizon'):
                        validate_record(stale)
        self.assertEqual(plan['experiments']['m2']['condition_counts'],
                         {'chinese-piece-image': 100})
        self.assertEqual(plan['experiments']['h2']['condition_counts'],
                         {'full-state': 100, 'move-history-only': 100})

    def test_recognition_independent_of_move_quality_and_missing_transcription(self):
        task = runtime_dict(load_records(FAMILY_PATHS['xiangqi-multimodal'])[0])
        truth = transcription(task['board'])
        raw = json.dumps({'move': 'a0a9', 'pieces': truth})
        self.assertEqual(score_board_recognition(raw, task['board'])['recognition_accuracy'], 1)
        self.assertEqual(score_board_recognition('{"move":"a0a9"}', task['board'])['recognition_accuracy'], 0)
        predicted = dict(truth); key = next(iter(truth)); predicted[key] = truth[key].swapcase()
        expected = (len(truth) - 1) / len(truth)
        self.assertEqual(score_board_recognition(json.dumps({'pieces': predicted}), task['board'])['recognition_accuracy'], expected)
        extra = next(f'{f}{r}' for f in FILES for r in range(10) if f'{f}{r}' not in truth)
        predicted = {**truth, extra: 'P'}
        self.assertEqual(score_board_recognition(json.dumps({'pieces': predicted}), task['board'])['recognition_accuracy'], len(truth)/(len(truth)+1))
        self.assertEqual(score_board_recognition('{"pieces":{"z9":"K"}}', task['board'])['recognition_accuracy'], 0)
        self.assertEqual(_extract_uci(raw), 'a0a9')
        self.assertIsNone(_extract_uci('{"move":"a0a9","answer":1}'))

    def test_all_nine_agents_preserve_actual_png_in_every_internal_call(self):
        record = load_records(FAMILY_PATHS['xiangqi-multimodal'])[0]
        task = runtime_dict(record)
        answer = json.dumps({'move': record['oracle']['best_move_uci'], 'pieces': transcription(task['board'])})
        tot = one_level_tot_responses()
        tot[0] = json.dumps({**json.loads(tot[0]), 'proposed_answer': json.loads(answer)})
        tot[-1] = answer
        judge = json.dumps({'selected_id': 'candidate-1', 'scores': [
            {'candidate_id': 'candidate-1', 'score': .9, 'reason': 'ok'},
            {'candidate_id': 'candidate-2', 'score': .8, 'reason': 'ok'}]})
        cases = {
            'passthrough': [answer], 'direct': [answer], 'cot': ['reasoning', answer],
            'self-consistency': [answer, answer], 'best-of-n': [answer, answer, judge, answer],
            'tot': tot,
            'plan-then-solve': ['{"steps":["inspect"],"risks":[]}', '{"solution":"done"}', answer],
            'critic-refine': [answer, '{"is_valid":true,"issues":[],"corrections":[]}', answer],
            'least-to-most': ['{"subproblems":["one"]}', '{"subproblem":"one","result":"r","reason":"ok"}', answer],
        }
        for name, replies in cases.items():
            with self.subTest(agent=name):
                client = FakeClient(replies)
                options = {'samples': 2, 'branching_factor': 1, 'max_depth': 1, 'beam_width': 1,
                           'final_max_tokens': 1024, 'max_format_repairs': 0} if name != 'passthrough' else {}
                agent = cloud.build_agent(name, client, options)
                result = evaluate_xiangqi_multimodal_tasks([task], agent, modes=['chinese-piece-image'], verify_with_pikafish=False)[0]
                self.assertEqual(result['status'], 'ok', result)
                self.assertTrue(result['success'], result)
                self.assertEqual(result['recognition_accuracy'], 1)
                for call in client.calls:
                    images = call['options']['images']; self.assertEqual(len(images), 1)
                    data, mime = images[0].resolved(); self.assertEqual(mime, 'image/png')
                    self.assertEqual(hashlib.sha256(data).hexdigest(), result['steps'][0]['input_image_sha256'])
                    sent = base64.b64decode(images[0].content_part()['image_url']['url'].split(',', 1)[1])
                    self.assertEqual(data, sent)
                    with Image.open(BytesIO(data)) as im:
                        im.load()
                        self.assertEqual(json.loads(im.info['minibench_piece_manifest']), render_board_manifest(task['board'], 'chinese-piece-image'))
                summary = summarize_xiangqi_multimodal([result])['by_input_mode']['chinese-piece-image']
                self.assertEqual(summary['recognition_accuracy'], 1)

    def test_certificate_must_bind_code_dataset_and_exact_defender_choices(self):
        record = load_records(FAMILY_PATHS['xiangqi-rule-variants'])[0]
        path=Path(record['shortest_proof']['tree_path'])
        proof=validate_tree_record(record,path)
        self.assertTrue(proof['valid'],proof)
        self.assertTrue(proof['tree_complete'])
        with tempfile.TemporaryDirectory() as directory:
            from types import SimpleNamespace
            from unittest.mock import Mock
            from minibench.datasets.xiangqi.rule_variants import evaluate_rule_variant_task

            changed=json.loads(path.read_text())
            root=changed['nodes'][changed['root']]
            root['edges']['a0a9']='missing'
            data_root=Path(directory)/'data'/'xiangqi'
            tree_dir=data_root/'solution_trees'/'c2'
            tree_dir.mkdir(parents=True)
            source=data_root/'rule_variants'/'tasks.jsonl'
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps(record)+'\n')
            target=tree_dir/path.name
            target.write_text(json.dumps(changed))
            self.assertFalse(validate_tree_record(record,target)['valid'])
            # A real independent report is deliberately absent, so the stronger
            # gate fails first. Exact hash binding has dedicated proof-gate tests.
            with self.assertRaisesRegex(ValueError,'Independent shortest-tree verification required'):
                TreeSession(record,tree_dir)
            agent=SimpleNamespace(generate=Mock())
            with self.assertRaisesRegex(ValueError,'Independent shortest-tree verification required'):
                evaluate_rule_variant_task(runtime_dict(record),agent,solution_tree_dir=tree_dir)
            agent.generate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
