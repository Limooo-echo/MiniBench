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
from minibench.datasets.xiangqi.variants.certification import (
    CERTIFICATE_VERSION, POLICY, certificate_header_valid, policy_fingerprint,
    verify_winning_line,
)
from tests.test_reasoning_agents import FakeClient, one_level_tot_responses


def transcription(board):
    return {f'{FILES[c]}{9-r}': piece_symbol(p)
            for r, row in enumerate(board) for c, p in enumerate(row) if p}


class FinalProtocolTests(unittest.TestCase):
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

    def test_submitted100_four_corpora_and_two_m2_modes(self):
        raw = yaml.safe_load(Path('config/cloud/xiangqi-all-agents.yaml').read_text())
        raw['engine_binding'] = 'reference'
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / 'cloud.yaml'; cfg.write_text(yaml.safe_dump(raw))
            plan = cloud.prepare(cfg, Path(tmp) / 'suite')
        self.assertEqual(plan['planned_jobs'], 5400)
        for entry in plan['experiments'].values():
            rows = load_records(FAMILY_PATHS[entry['family']])
            self.assertEqual(len(rows), 100)
            self.assertEqual(entry['task_ids'], [r['id'] for r in rows])
        self.assertEqual(plan['experiments']['m2']['condition_counts'],
                         {'text': 100, 'chinese-piece-image': 100})
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
        certificates = json.loads(Path('data/xiangqi/rule_variants/runtime_certificates.json').read_text())
        line = certificates['tasks'][record['id']]['moves_uci']
        proof = verify_winning_line(record, line)
        self.assertTrue(proof['valid'], proof)
        wrong = list(line); wrong[1] = 'a0a9'
        self.assertFalse(verify_winning_line(record, wrong)['valid'])
        bundle = {'version': CERTIFICATE_VERSION, 'policy': POLICY, 'policy_fingerprint': policy_fingerprint(),
                  'dataset_sha256': 'sha', 'complete': True, 'tasks': {record['id']: {**proof, 'moves_uci': line}}}
        self.assertTrue(certificate_header_valid(bundle, [record], 'sha'))
        self.assertFalse(certificate_header_valid(bundle, [record], 'changed'))
        changed = deepcopy(bundle); changed['policy']['depth'] = 2
        self.assertFalse(certificate_header_valid(changed, [record], 'sha'))


if __name__ == '__main__':
    unittest.main()
