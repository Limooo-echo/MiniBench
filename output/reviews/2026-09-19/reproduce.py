"""Read-only, deterministic review reproductions. No external model/engine calls.
Run from an environment with the project's dependencies:
    python reproduce.py --repo /path/to/reviewed/MiniBench
"""
import argparse
import contextlib
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, required=True)
args = parser.parse_args()
repo = args.repo.resolve()
os.chdir(repo)
sys.path.insert(0, str(repo / 'src'))
from minibench.scoring.manifest import load_suite, read_document, file_hash
from minibench.scoring.adapters import score_record
from minibench.scoring.report import _costs
from minibench.factory.experiments import get_task_family_spec
from minibench.datasets.xiangqi.mate_in_one import evaluate_mate_in_one_tasks
from minibench.datasets.xiangqi.history import evaluate_history_tasks
from minibench.datasets.xiangqi.multimodal import _run_multimodal_game
from minibench.datasets.xiangqi.schema import load_records, runtime_dict, fen_to_board
from minibench.datasets.xiangqi.variants.board import VariantBoard, _crossed_river
from minibench.datasets.xiangqi.rule_variants import evaluate_rule_variant_task

out = {'reviewed_commit': 'eab452b069b40a1cfaec39fe2b3895d7503484f2'}
p = Path('config/scoring/manifest.example.yaml')
try:
    load_suite(p)
    out['manifest_error'] = None
except Exception as exc:
    out['manifest_error'] = f'{type(exc).__name__}: {exc}'
out['stale_manifest_entries'] = [
    {'experiment': e['id'], 'declared': e['dataset']['sha256'],
     'actual': file_hash(p.parent / e['dataset']['path'])}
    for e in read_document(p)['experiments']
    if file_hash(p.parent / e['dataset']['path']) != e['dataset']['sha256']
]
class TimeoutAgent:
    def generate(self, *args, **kwargs):
        raise TimeoutError('synthetic provider timeout')
    generate_messages = generate

task = get_task_family_spec('xiangqi-mate-in-one').load_tasks('data/xiangqi/mate_in_one/tasks.jsonl')[0]
engine = MagicMock()
engine.bestmove_for_fen.return_value = (task.oracle['best_move_uci'], ['info depth 15 score mate 1'])
with patch('minibench.datasets.xiangqi.mate_in_one.resolve_pikafish_executable', return_value=Path('/fake/engine')), patch('minibench.datasets.xiangqi.mate_in_one.PikafishEngine', return_value=engine), patch('minibench.datasets.xiangqi.mate_in_one._ensure_engine_alive'), contextlib.redirect_stdout(io.StringIO()):
    direct = asdict(evaluate_mate_in_one_tasks([task], TimeoutAgent())[0])
out['direct_timeout'] = {'raw_output': direct['raw_output'], 'goal_achieved': direct['goal_achieved'], 'scored': score_record('xiangqi_mate_in_one', 'D', direct, None, mode='single')}
htask = get_task_family_spec('xiangqi-history').load_tasks('data/xiangqi/history/tasks.jsonl')[0]
with patch('minibench.datasets.xiangqi.history.resolve_pikafish_executable', return_value=Path('/fake/engine')), patch('minibench.datasets.xiangqi.history.PikafishEngine', return_value=engine), patch('minibench.datasets.xiangqi.history._pikafish_analysis', return_value=(htask.oracle['best_move_uci'], 10000)), patch('minibench.datasets.xiangqi.history.time.sleep'), contextlib.redirect_stdout(io.StringIO()):
    history = asdict(evaluate_history_tasks([htask], TimeoutAgent())[0])
out['history_timeout'] = {'reasons': history['reasons'], 'goal_achieved': history['goal_achieved'], 'scored': score_record('xiangqi_history', 'H', history, None, mode='move-history-only')}
out['history_engine_error'] = score_record('xiangqi_history', 'H', {'goal_achieved': False, 'reasons': ['pikafish_error']}, None, mode='move-history-only')

m = runtime_dict(load_records('data/xiangqi/multimodal/tasks.jsonl')[0])
class RetryAgent:
    def __init__(self): self.calls = 0
    def generate(self, *args, **kwargs):
        self.calls += 1
        return 'invalid' if self.calls == 1 else json.dumps({'move': m['oracle']['best_move_uci']})
    generate_multimodal = generate
out['multimodal_retry'] = []
for mode in ['text', 'chinese-piece-image']:
    agent = RetryAgent()
    with patch('minibench.datasets.xiangqi.multimodal.render_board_png', return_value=b'fake'):
        _, success, reasons = _run_multimodal_game(m, agent, mode, opponent_depth=4, optimal_depth=3, max_steps=1, step_root=None, pikafish=None, pikafish_depth=15, initial_analysis=None)
    out['multimodal_retry'].append({'mode': mode, 'calls': agent.calls, 'success': success, 'reasons': reasons})

records = load_records('data/xiangqi/rule_variants/tasks.jsonl')
bad, retreats = [], []
for r in records:
    b, _ = fen_to_board(r['fen'])
    if any(v == -1 and not (row <= 2 and 3 <= col <= 5) for row, line in enumerate(b) for col, v in enumerate(line)):
        bad.append(r)
    u = r['oracle']['best_move_uci']
    fr, fc, tr = 9 - int(u[1]), ord(u[0]) - 97, 9 - int(u[3])
    if r['ruleset'] == 'soldier-free-retreat' and b[fr][fc] >= 12 and tr > fr and not _crossed_river(b[fr][fc], fr):
        retreats.append(r)
out['c2_outside_palace'] = {'records': len(bad), 'total': len(records), 'unique_fens': len({r['fen'] for r in bad}), 'ids': [r['id'] for r in bad]}
out['c2_uncrossed_retreat_oracles'] = [r['id'] for r in retreats]
target = next(r for r in records if r['id'] == 'xiangqi-rule-variants-c2-0164')
class FixedAgent:
    def generate(self, *args, **kwargs): return '{"move":"d2d1"}'
original = VariantBoard._gen_soldier
def corrected(self, r, c, pid, free):
    return original(self, r, c, pid, free and _crossed_river(pid, r))
native = evaluate_rule_variant_task(runtime_dict(target), FixedAgent())
with patch.object(VariantBoard, '_gen_soldier', corrected):
    fixed = evaluate_rule_variant_task(runtime_dict(target), FixedAgent())
out['c2_retreat_results'] = {name: {'legal': result.legality_rate, 'success': result.success, 'score': result.score} for name, result in [('current', native), ('in_memory_rule_fix', fixed)]}
out['h2_non_checkmate_reference_lines'] = []
for rec in load_records('data/xiangqi/history/tasks.jsonl'):
    raw, color = fen_to_board(rec['fen'])
    board, side = VariantBoard(raw, []), 1 if color == 'red' else -1
    for uci in rec['h2_analysis']['principal_variation_uci']:
        move = next((m for m in board.legal_moves(side) if m.to_uci() == uci), None)
        if move is None:
            raise RuntimeError(rec['id'] + ' illegal PV ' + uci)
        board.apply(move)
        side = -side
    if not board.is_checkmate(side):
        out['h2_non_checkmate_reference_lines'].append({'id': rec['id'], 'pv': rec['h2_analysis']['principal_variation_uci'], 'in_check': board._is_in_check(side), 'legal_replies': len(board.legal_moves(side))})
rows = [dict(profile='p', architecture='a', status='ok', metrics={'usage_available': True, 'llm_calls': 1, 'usage_missing_calls': 0, 'token_usage': {'total_tokens': 100}}), dict(profile='p', architecture='a', status='ok', metrics={'usage_available': False, 'llm_calls': 99})]
out['cost_coverage'] = _costs(rows)[0]
print(json.dumps(out, ensure_ascii=False, indent=2))
