"""Build the submitted stratified 100-row corpora from a frozen parent pool.

The parent pool stays outside the submitted repository. C2 counts complete
standard/focus pairs, uses only certified pairs, and never passes witnesses to
agents. M2 inherits the exact D3 positions, not a separate random draw.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from minibench.core.checkpoint import atomic_write_json, atomic_write_jsonl, sha256_file
from minibench.datasets.xiangqi.schema import load_records, sample_records
from minibench.datasets.xiangqi.variants.certification import (
    CERTIFICATE_VERSION, POLICY, policy_fingerprint, verify_winning_line,
)

FOLDERS = {'d3': 'mate_in_one', 'h2': 'history', 'c2': 'rule_variants', 'm2': 'multimodal'}


def select(parent: Path, destination: Path, seed=42):
    pools = {k: load_records(parent / f / 'tasks.jsonl') for k, f in FOLDERS.items()}
    if any(len(rows) != 250 for rows in pools.values()):
        raise ValueError('Expected immutable 250-row parent pools')
    old = json.loads((parent / 'rule_variants/runtime_certificates.json').read_text())
    if (old.get('version') != CERTIFICATE_VERSION or old.get('policy') != POLICY
            or old.get('policy_fingerprint') != policy_fingerprint()
            or old.get('dataset_sha256') != sha256_file(parent / 'rule_variants/tasks.jsonl')):
        raise ValueError('Parent C2 certificates do not bind current policy and pool')
    groups = defaultdict(list)
    for r in pools['c2']:
        groups[r['scenario_id']].append(r)
    eligible = [r for pair in groups.values() if len(pair) == 2
                and all(old.get('tasks', {}).get(p['id'], {}).get('valid') is True for p in pair)
                for r in pair]
    chosen = {
        'd3': sample_records(pools['d3'], count=100, seed=seed, strategy='stratified'),
        'h2': sample_records(pools['h2'], count=100, seed=seed, strategy='stratified'),
        'c2': sample_records(eligible, count=50, seed=seed, strategy='paired-stratified'),
    }
    modal = {r['source_task_id']: r for r in pools['m2']}
    chosen['m2'] = [modal[r['id']] for r in chosen['d3']]
    if any(len(rows) != 100 for rows in chosen.values()):
        raise ValueError('Every submitted corpus must have exactly 100 rows')
    if sum(r['c2_analysis']['mate_in_moves'] <= 5 for r in chosen['c2']) <= 50:
        raise ValueError('C2 majority must remain multi-move mates within five attacker moves')
    # Recompute, do not merely copy a stored valid flag.
    certified = {}
    for r in chosen['c2']:
        line = old['tasks'][r['id']]['moves_uci']
        proof = verify_winning_line(r, line)
        if not proof['valid']:
            raise ValueError(f'Runtime witness failed: {r["id"]}: {proof["reasons"]}')
        certified[r['id']] = {**proof, 'moves_uci': line}
    selected_h2 = {r['id'] for r in chosen['h2']}
    source_ids = {r['source_task_id'] for r in chosen['c2']}
    origins = [r for r in pools['h2'] if r['id'] in source_ids - selected_h2]
    if len(origins) != len(source_ids - selected_h2):
        raise ValueError('Missing independent C2 source evidence')
    manifest = {'version': 'xiangqi-submitted-stratified100-v1', 'seed': seed,
                'record_count_per_family': 100, 'source_pool_count_per_family': 250,
                'c2_eligible_certified_scenarios': len(eligible)//2,
                'c2_excluded_uncertified_scenarios': len(groups)-len(eligible)//2,
                'policy': {'d3': 'equal difficulty strata, 34/33/33',
                           'h2': 'short/medium/long 34/33/33',
                           'c2': '50 certified complete scenarios, equally stratified by focus rule; standard50, chariot17, horse17, soldier16',
                           'm2': 'exact D3 pairing; text and Chinese-piece image only'},
                'selection_limitations': 'C2 is sampled from certified pairs only; unknown-budget cases are excluded, not labelled unsolvable. Equal difficulty strata is a designed distribution, not parent population weighting.',
                'datasets': {}}
    for k, rows in chosen.items():
        path = destination / FOLDERS[k] / 'tasks.jsonl'
        atomic_write_jsonl(path, rows)
        manifest['datasets'][k] = {'path': path.relative_to(ROOT).as_posix(),
            'sha256': sha256_file(path), 'task_ids': [r['id'] for r in rows],
            'difficulty_counts': dict(Counter(r['difficulty'] for r in rows)),
            'parent_sha256': sha256_file(parent / FOLDERS[k] / 'tasks.jsonl')}
    origin_path = destination / 'sources/c2_h2_origins.jsonl'
    atomic_write_jsonl(origin_path, origins)
    manifest['c2_source_evidence'] = {'path': origin_path.relative_to(ROOT).as_posix(),
                                    'sha256': sha256_file(origin_path), 'record_count': len(origins)}
    bundle = {'version': CERTIFICATE_VERSION, 'policy': POLICY,
        'policy_fingerprint': policy_fingerprint(), 'dataset_sha256': manifest['datasets']['c2']['sha256'],
        'complete': True, 'tasks': certified,
        'claim': 'Existence of a bounded winning continuation against registered deterministic depth-3 policy only; not all-defences or shortest-mate proof.'}
    atomic_write_json(destination / 'rule_variants/runtime_certificates.json', bundle)
    manifest['c2_certificate_sha256'] = sha256_file(destination / 'rule_variants/runtime_certificates.json')
    atomic_write_json(destination / 'submission_manifest.json', manifest)
    atomic_write_json(destination / 'rule_variants/generation_report.json', {
        'version': manifest['version'], 'record_count': 100, 'scenario_count': 50,
        'ruleset_counts': dict(Counter(r['ruleset'] for r in chosen['c2'])),
        'difficulty_counts': manifest['datasets']['c2']['difficulty_counts'],
        'within_five_moves': sum(r['c2_analysis']['mate_in_moves'] <= 5 for r in chosen['c2']),
        'rule_sensitive_scenarios': dict(Counter(r['design_stratum'] for r in chosen['c2']
            if r['ruleset'] != 'standard' and r['c2_analysis']['rule_sensitive_along_reference'])),
        'runtime_certified_count': len(certified), 'selection_manifest': 'submission_manifest.json'})
    atomic_write_json(destination / 'history/validation_report.json', {
        'version': manifest['version'], 'selected_count': 100,
        'parent_dataset_sha256': manifest['datasets']['h2']['parent_sha256'],
        'selected_dataset_sha256': manifest['datasets']['h2']['sha256'],
        'note': 'Selected rows retain their two engine reference analyses; fresh independent_validation.json is the authoritative current gate.'})
    provenance_path = destination / 'provenance.json'
    provenance = json.loads(provenance_path.read_text())
    provenance['corpus_name'] = 'MiniBench Xiangqi submitted stratified 100 per task, 2026-10-10'
    provenance['submission'] = {'manifest': 'submission_manifest.json', 'sha256': sha256_file(destination / 'submission_manifest.json')}
    provenance['intentional_overlap']['D3_M2'] = 'All 100 submitted FENs shared to isolate modality.'
    provenance['intentional_overlap']['C2'] = '50 standard/focus pairs from H2 parent sources; sources not selected into H2 are retained as minimal source evidence, not benchmark tasks.'
    provenance['generation_command']['submitted_selection'] = 'python scripts/select_xiangqi_release.py --parent-pool <immutable-250-row-parent-pool> --replace --seed 42'
    provenance['generation_command']['c2_source_policy'] = '50 certified standard/focus pairs, all three focus rules; exclude active-rule mate-in-one and uncertified sources.'
    provenance['interpretation']['history_visual_gaps'] = 'Task gaps are not direct memory evidence; M2 separately scores explicit initial-board transcription.'
    provenance['validation']['variants'] = 'Independent legal/terminal replay plus exact depth-3 opponent choices on a winning witness for every submitted C2 row.'
    atomic_write_json(provenance_path, provenance)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-pool', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--replace', action='store_true', help='Explicitly replace the submitted canonical corpora')
    a = p.parse_args()
    if not a.replace:
        p.error('--replace is required after preserving an immutable parent-pool backup')
    manifest = select(a.parent_pool.resolve(), ROOT / 'data/xiangqi', a.seed)
    print(json.dumps({k: v['difficulty_counts'] for k, v in manifest['datasets'].items()}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
