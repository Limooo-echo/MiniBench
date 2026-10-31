"""Offline state-safety diagnostics for the frozen 1028 Zebra experiment.

Gold is used only in this post-run analysis. Runtime Pro used prefix entailment.
Keeping the gold position is necessary but does not prove prefix entailment.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

MODES = ('incremental_state', 'incremental_state_pro')
TREATMENTS = ('thinking_off', 'thinking_on')


def norm(value):
    return str(value).strip().lower()


def gold_positions(task):
    headers = task['solution']['header']
    truth = {}
    for number, row in enumerate(task['solution']['rows'], 1):
        for header, value in zip(headers[1:], row[1:]):
            key = (norm(header), norm(value))
            if key in truth:
                raise ValueError(f'duplicate gold value {task["id"]}: {key}')
            truth[key] = number
    return truth


def excluded_gold(state, truth):
    if state is None:
        return None
    domains = state['domains']
    found = set()
    excluded = []
    for header, values in domains.items():
        for value, houses in values.items():
            key = (norm(header), norm(value))
            if key not in truth or key in found:
                raise ValueError(f'unmatched or duplicate state value: {key}')
            found.add(key)
            if truth[key] not in houses:
                excluded.append({'attribute': header, 'value': value,
                                 'gold_house': truth[key], 'candidate_houses': houses})
    if found != set(truth):
        raise ValueError('state omitted declared values')
    return excluded


def final_failure(row):
    finish = row['history_trace'][-1]['completion_traces'][-1]['finish_reason']
    if row['success']:
        return 'correct'
    if row['no_answer']:
        return 'truncated_no_answer' if finish == 'length' else 'no_answer_other'
    return 'parsed_incorrect'


def subset_summary(rows):
    return {'tasks': len(rows), 'success': sum(r['success'] for r in rows),
            'no_answer': sum(r['no_answer'] for r in rows),
            'final_failures': dict(Counter(r['final_failure'] for r in rows))}


def analyze(batch: Path, output: Path, validate_inc: bool):
    tasks = {row['id']: row for row in map(json.loads, (batch / 'history_eval.jsonl').read_text(encoding='utf-8').splitlines())}
    if len(tasks) != 100:
        raise ValueError('expected 100 unique tasks')
    if validate_inc:
        sys.path.insert(0, str(batch / 'source'))
        from minibench.datasets.zebra.dataset import zebra_task_from_dict
        from minibench.datasets.zebra.state_validation import PrefixStateValidator
        task_objects = {key: zebra_task_from_dict(task) for key, task in tasks.items()}
    results, details, groups = {}, [], {}
    for treatment in TREATMENTS:
        with (batch / treatment / 'predictions.jsonl').open(encoding='utf-8') as handle:
            rows = [json.loads(line) for line in handle]
        for mode in MODES:
            selected = [row for row in rows if row['memory_mode'] == mode]
            if len(selected) != 100 or {row['task_id'] for row in selected} != set(tasks):
                raise ValueError(f'incomplete state-mode roster: {treatment}/{mode}')
            counter = Counter()
            per_task = []
            for row in selected:
                task = tasks[row['task_id']]
                truth = gold_positions(task)
                intermediate = [entry for entry in row['history_trace'] if entry['phase'] == 'intermediate']
                if len(intermediate) != len(task['clue_turns']):
                    raise ValueError('trace does not have one entry per clue')
                validator = (PrefixStateValidator(task_objects[row['task_id']], timeout_ms=10000)
                             if validate_inc and mode == 'incremental_state' else None)
                task_counts = Counter()
                steps, bad_seen, recovered_after_bad = [], False, False
                first_bad = None
                previous_bad = False
                for entry in intermediate:
                    index = entry['clue_index']
                    before = excluded_gold(entry['state_before'], truth)
                    after = excluded_gold(entry['state_after'], truth)
                    proposal = excluded_gold(entry['parsed_proposal'], truth)
                    accepted = entry['decision'] == 'accept'
                    schema = entry['schema_status']
                    status = entry['validation']['status']
                    if not accepted and entry['state_after'] != entry['state_before']:
                        raise ValueError('rejected state unexpectedly changed')
                    if accepted and entry['state_after'] != entry['parsed_proposal']:
                        raise ValueError('accepted state differs from proposal')
                    if mode.endswith('_pro') and accepted and status != 'entailed':
                        raise ValueError('Pro accepted non-entailed state')
                    if schema == 'invalid' and accepted:
                        raise ValueError('invalid schema accepted')
                    task_counts['clue_updates'] += 1
                    task_counts[f'schema_{schema}'] += 1
                    task_counts['accepted' if accepted else 'rejected'] += 1
                    task_counts[f'validation_{status}'] += 1
                    if not accepted:
                        task_counts['rejected_preserved_previous_state'] += 1
                        task_counts['rejected_schema' if schema == 'invalid' else f'rejected_{status}'] += 1
                    if proposal is not None:
                        task_counts['proposal_excludes_gold' if proposal else 'proposal_retains_gold'] += 1
                    if accepted:
                        task_counts['accepted_excludes_gold' if after else 'accepted_retains_gold'] += 1
                    elif proposal is not None:
                        task_counts['rejected_excludes_gold' if proposal else 'rejected_retains_gold'] += 1
                    if after:
                        task_counts['committed_steps_excluding_gold'] += 1
                        task_counts['excluded_gold_entries_across_committed_steps'] += len(after)
                        if first_bad is None:
                            first_bad = index
                        bad_seen = True
                    if previous_bad and not after:
                        task_counts['bad_to_safe_recoveries'] += 1
                        recovered_after_bad = True
                    if not previous_bad and after:
                        task_counts['safe_to_bad_transitions'] += 1
                    if before and after:
                        task_counts['bad_to_bad_transitions'] += 1
                        if len(after) > len(before):
                            task_counts['bad_to_bad_excluded_entries_increased'] += 1
                    previous_bad = bool(after)
                    finish = entry['completion_traces'][-1]['finish_reason']
                    task_counts[f'intermediate_finish_{finish}'] += 1
                    semantic_audit = None
                    if validator is not None and accepted:
                        semantic_audit = validator.validate(entry['state_after'], index)['status']
                        task_counts[f'offline_prefix_audit_{semantic_audit}'] += 1
                    steps.append({'clue_index': index, 'schema_status': schema,
                                  'schema_error': entry['schema_error'], 'decision': entry['decision'],
                                  'validation_status': status, 'finish_reason': finish,
                                  'proposal_excludes_gold': None if proposal is None else bool(proposal),
                                  'proposal_excluded_gold_entries': proposal,
                                  'committed_excludes_gold': bool(after),
                                  'committed_excluded_gold_entries': after,
                                  'offline_prefix_audit': semantic_audit})
                final_state = row['history_trace'][-1]['state_before']
                final_excluded = excluded_gold(final_state, truth)
                final_singletons = all(len(houses) == 1 for values in final_state['domains'].values() for houses in values.values())
                final_finish = row['history_trace'][-1]['completion_traces'][-1]['finish_reason']
                diagnostic = {'treatment': treatment, 'mode': mode, 'task_id': row['task_id'],
                              'source_id': row['source_id'], 'size': row['size'],
                              'success': row['success'], 'parsed': row['parsed'], 'no_answer': row['no_answer'],
                              'final_failure': final_failure(row), 'final_finish_reason': final_finish,
                              'ever_committed_excludes_gold': bad_seen, 'first_gold_exclusion_clue': first_bad,
                              'recovered_after_bad_at_least_once': recovered_after_bad,
                              'final_state_excludes_gold': bool(final_excluded),
                              'final_state_excluded_gold_entries': final_excluded,
                              'final_state_all_singletons': final_singletons,
                              'final_state_is_exact_gold': final_singletons and not final_excluded,
                              'counts': dict(task_counts), 'steps': steps}
                per_task.append(diagnostic)
                results[treatment, row['task_id'], mode] = diagnostic
                details.append(diagnostic)
                counter.update(task_counts)
            groups[f'{treatment}/{mode}'] = {
                'counts': dict(counter),
                'tasks': len(per_task), 'success': sum(r['success'] for r in per_task),
                'ever_committed_excludes_gold': subset_summary([r for r in per_task if r['ever_committed_excludes_gold']]),
                'never_committed_excludes_gold': subset_summary([r for r in per_task if not r['ever_committed_excludes_gold']]),
                'final_state_excludes_gold': subset_summary([r for r in per_task if r['final_state_excludes_gold']]),
                'final_state_retains_gold': subset_summary([r for r in per_task if not r['final_state_excludes_gold']]),
                'final_state_exact_gold': subset_summary([r for r in per_task if r['final_state_is_exact_gold']]),
                'had_bad_and_recovered_at_least_once': subset_summary([r for r in per_task if r['recovered_after_bad_at_least_once']]),
                'had_bad_and_finally_safe': subset_summary([r for r in per_task if r['ever_committed_excludes_gold'] and not r['final_state_excludes_gold']]),
                'had_any_reject': subset_summary([r for r in per_task if r['counts'].get('rejected', 0)]),
                'had_semantic_reject': subset_summary([r for r in per_task if r['counts'].get('rejected_not_entailed', 0)]),
            }
            print(f'Analyzed {treatment}/{mode}: {len(per_task)} tasks', flush=True)
    pairs = {}
    for treatment in TREATMENTS:
        paired = []
        counts = Counter()
        for task_id in tasks:
            inc = results[treatment, task_id, 'incremental_state']
            pro = results[treatment, task_id, 'incremental_state_pro']
            label = ('both_correct' if inc['success'] and pro['success'] else
                     'pro_only_correct' if pro['success'] else
                     'inc_only_correct' if inc['success'] else 'both_wrong')
            counts[label] += 1
            paired.append({'task_id': task_id, 'source_id': tasks[task_id]['source_id'], 'pair_result': label,
                           'inc_success': inc['success'], 'pro_success': pro['success'],
                           'inc_ever_excludes_gold': inc['ever_committed_excludes_gold'],
                           'inc_final_excludes_gold': inc['final_state_excludes_gold'],
                           'inc_final_exact_gold': inc['final_state_is_exact_gold'],
                           'inc_failure': inc['final_failure'], 'pro_failure': pro['final_failure'],
                           'pro_rejected': pro['counts'].get('rejected', 0),
                           'pro_rejected_unsafe_gold': pro['counts'].get('rejected_excludes_gold', 0),
                           'pro_final_exact_gold': pro['final_state_is_exact_gold']})
        pairs[treatment] = {'counts': dict(counts), 'all_tasks': paired,
                            'discordant': [r for r in paired if r['pair_result'] not in ('both_correct', 'both_wrong')]}
    document = {'definitions': {
        'gold_safety': 'Offline diagnostic: every retained candidate domain includes the final gold house for its declared value; necessary for correctness, not a proof of prefix entailment.',
        'prefix_entailment': 'Runtime Pro accepted only if every candidate restriction is entailed by background and clues received so far; safe broad states may still be incomplete.',
        'causal_limit': 'Paired independent model trajectories differ; correlations and discordant cases do not isolate a causal effect of state rejection.',
        'offline_inc_prefix_audit': 'All accepted ordinary states audited against the frozen prefix validator.' if validate_inc else 'Not performed; ordinary state diagnosis uses gold retention only.'},
        'groups': groups, 'pairs': pairs, 'task_diagnostics': details}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'pro_state_analysis.json').write_text(json.dumps(document, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    columns = ['treatment', 'mode', 'source_id', 'size', 'success', 'parsed', 'no_answer', 'final_failure',
               'final_finish_reason', 'ever_committed_excludes_gold', 'first_gold_exclusion_clue',
               'recovered_after_bad_at_least_once', 'final_state_excludes_gold', 'final_state_all_singletons',
               'final_state_is_exact_gold', 'clue_updates', 'accepted', 'rejected', 'schema_invalid',
               'rejected_not_entailed', 'proposal_excludes_gold', 'accepted_excludes_gold',
               'committed_steps_excluding_gold', 'bad_to_safe_recoveries', 'safe_to_bad_transitions']
    with (output / 'pro_task_diagnostics.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in details:
            writer.writerow({key: row.get(key, row['counts'].get(key, 0)) for key in columns})
    print(json.dumps({'groups': groups, 'pairs': {k: {'counts': v['counts'], 'discordant': v['discordant']} for k, v in pairs.items()}}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--validate-inc-prefixes', action='store_true')
    args = parser.parse_args()
    batch = args.batch.resolve()
    output = args.output or batch.with_name(batch.name + '_analysis')
    analyze(batch, output, args.validate_inc_prefixes)


if __name__ == '__main__':
    main()
