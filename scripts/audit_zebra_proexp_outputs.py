"""Offline audit of final Zebra scores and the actual 1028 HTTP payloads.

This independent implementation preserves the frozen scoring policy, including
its brace-based last-object extraction. Diagnostic key repair never changes
answer values and does not replace official scores.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import statistics

TREATMENTS = ('thinking_off', 'thinking_on')
MODES = ('deferred_reasoning', 'incremental_state', 'incremental_state_pro')
VALUE_LABEL_ALIASES = {
    'Birthday': {'january': 'jan', 'september': 'sept'},
    'CarModel': {'ford f-150': 'ford f150'},
    'Nationality': {'swedish': 'swede'},
    'Vacation': {'cruises': 'cruise'},
}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def last_object(text):
    depth, beginning, latest = 0, None, None
    for index, character in enumerate(text):
        if character == '{':
            if depth == 0:
                beginning = index
            depth += 1
        elif character == '}' and depth:
            depth -= 1
            if depth == 0:
                latest = text[beginning:index + 1]
                beginning = None
    if latest is None:
        return None
    try:
        value = json.loads(latest.replace('\n', ''))
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def normalize(value):
    return str(value).lower().strip()


def cell(value):
    if isinstance(value, list):
        value = value[0] if value else None
    return None if value is None or isinstance(value, (dict, list)) else value


def truth_table(task):
    return {f'House {index}': dict(zip(task['solution']['header'][1:], row[1:]))
            for index, row in enumerate(task['solution']['rows'], 1)}


def score(task, text):
    payload = last_object(text)
    solution = payload.get('solution') if payload is not None else None
    parsed = isinstance(solution, dict)
    truth = truth_table(task)
    total = sum(map(len, truth.values()))
    correct = 0
    for house, attributes in truth.items():
        predicted = solution.get(house) if parsed else None
        if not isinstance(predicted, dict):
            continue
        for name, expected in attributes.items():
            value = cell(predicted.get(name))
            correct += value is not None and normalize(value) == normalize(expected)
    return {'parsed': parsed, 'no_answer': not parsed,
            'success': parsed and correct == total,
            'correct_cells': correct, 'total_cells': total}, payload, solution


def compact(value):
    return re.sub('[^a-z0-9]', '', str(value).lower())


def diagnose(task, payload, solution, raw):
    flags, canonical = [], {}
    try:
        top = json.loads(raw)
        top_type = type(top).__name__
    except json.JSONDecodeError:
        top_type = 'invalid_json'
    if top_type == 'list':
        flags.append('top_array')
    truth = truth_table(task)
    if not isinstance(solution, dict):
        flags.append('missing_or_invalid_solution')
        if isinstance(payload, dict) and 'reasoning' in payload and 'solution' not in payload:
            flags.append('reasoning_only')
    else:
        if set(solution) != set(truth):
            flags.append('house_keys')
        if not solution:
            flags.append('empty_solution')
        for house, attributes in solution.items():
            match = re.fullmatch(r'(?:house|hause|haus|h)?([1-6])', compact(house))
            if not match or not isinstance(attributes, dict):
                continue
            normalized_house = 'House ' + match.group(1)
            if normalized_house in canonical:
                flags.append('duplicate_normalized_house')
                continue
            canonical[normalized_house] = {compact(k): v for k, v in attributes.items()}
            if not attributes:
                flags.append('empty_attributes')
        for house, attributes in truth.items():
            predicted = solution.get(house)
            if not isinstance(predicted, dict) or set(predicted) != set(attributes):
                flags.append('attribute_keys_or_missing_cells')
    correct, missing, wrong = 0, 0, []
    for house, attributes in truth.items():
        for name, expected in attributes.items():
            value = cell(canonical.get(house, {}).get(compact(name)))
            if value is None or value == '___':
                missing += 1
            elif normalize(value) == normalize(expected):
                correct += 1
            else:
                wrong.append({'house': house, 'attribute': name, 'predicted': value, 'expected': expected})
    return {'top_json_type': top_type, 'flags': sorted(set(flags)),
            'canonical_correct_cells': correct, 'canonical_missing_cells': missing,
            'canonical_wrong_cells': wrong, 'canonical_success': correct == sum(map(len, truth.values()))}


def summarize(values):
    return {'count': len(values), 'min': min(values) if values else None,
            'median': statistics.median(values) if values else None,
            'mean': statistics.mean(values) if values else None,
            'max': max(values) if values else None}


def label_review(task, diagnostic):
    """Bounded same-field alias check; never move assignments or rescore."""
    table = truth_table(task)
    domain = defaultdict(set)
    for attributes in table.values():
        for name, value in attributes.items():
            domain[name].add(normalize(value))
    details = []
    for wrong in diagnostic['canonical_wrong_cells']:
        name, predicted, expected = wrong['attribute'], normalize(wrong['predicted']), normalize(wrong['expected'])
        mapped = VALUE_LABEL_ALIASES.get(name, {}).get(predicted)
        if mapped is not None and mapped in domain[name]:
            verdict = 'same_house_label_alias' if mapped == expected else 'alias_maps_to_another_value'
        elif predicted in domain[name]:
            verdict = 'canonical_domain_value_at_wrong_house'
        else:
            verdict = 'unresolved_out_of_domain_value'
        details.append({**wrong, 'alias_target': mapped, 'verdict': verdict})
    counts = Counter(d['verdict'] for d in details)
    if not details:
        classification = 'no_wrong_filled_values'
    elif counts['same_house_label_alias'] == len(details) and not diagnostic['canonical_missing_cells']:
        classification = 'same_house_label_aliases_only'
    elif counts['unresolved_out_of_domain_value']:
        classification = 'unresolved_value_mismatch'
    elif counts['same_house_label_alias']:
        classification = 'mixed_alias_and_assignment_mismatch'
    else:
        classification = 'assignment_mismatch'
    return {'classification': classification, 'details': details}


def reasoning_tokens(usage):
    if not isinstance(usage, dict):
        return None
    return usage.get('completion_tokens_details', {}).get('reasoning_tokens')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch_dir', type=Path, nargs='?', default=Path(__file__).resolve().parents[1] / 'runs/1028_zebra_proexp_direct')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    batch = args.batch_dir.resolve()
    output = args.output_dir or batch.parent / (batch.name + '_analysis')
    output.mkdir(parents=True, exist_ok=True)
    tasks = {t['id']: t for t in map(json.loads, (batch / 'history_eval.jsonl').read_text(encoding='utf-8').splitlines())}
    rows = {(t, r['task_id'], r['memory_mode']): r for t in TREATMENTS
            for r in map(json.loads, (batch / t / 'predictions.jsonl').read_text(encoding='utf-8').splitlines())}
    assert len(tasks) == 100 and len(rows) == 600
    cases, mismatches = [], []
    group = defaultdict(lambda: {'total': 0, 'categories': Counter(), 'format_issue_counts': Counter(),
                                'final_finish_reasons': Counter(), 'final_length_parsed': 0,
                                'final_length_success': 0, 'canonical_successes': 0,
                                'final_content_characters': [],
                                'final_visible_reasoning_characters': [], 'final_native_reasoning_characters': [],
                                'final_completion_tokens': [], 'final_reasoning_tokens': []})
    request_groups = defaultdict(lambda: {'calls': 0, 'phases': Counter(), 'max_tokens': Counter(),
                                         'response_models': Counter(), 'finish_reasons': Counter(),
                                         'native_reasoning_nonempty': 0, 'native_reasoning_empty': 0,
                                         'reasoning_tokens_present': 0, 'reasoning_tokens_zero': 0,
                                         'reasoning_tokens_positive': 0, 'reasoning_tokens_sum': 0,
                                         'completion_tokens_sum': 0, 'token_usage_missing': 0,
                                         'http_error_records': 0, 'event_counts': Counter()})
    request_mismatches, finals, call_count = [], {}, 0
    for treatment in TREATMENTS:
        for folder in sorted((batch / 'raw_data' / treatment).iterdir()):
            if not folder.is_dir():
                continue
            metadata = read_json(folder / 'metadata.json')
            mode, task_id = metadata['mode'], metadata['task_id']
            task = tasks[task_id]
            n = len(task['clue_turns']) + 1
            assert metadata['planned_calls'] == n
            audit = request_groups[treatment + '/' + mode]
            audit['http_error_records'] += len(list((folder / 'calls').glob('*.error-*.json')))
            events = folder / 'events.jsonl'
            if events.exists():
                audit['event_counts'].update(json.loads(line)['event'] for line in events.read_text(encoding='utf-8').splitlines())
            for index in range(n):
                stem = folder / 'calls' / f'{index:04d}'
                request = read_json(stem.with_suffix('.request.json'))
                response = read_json(stem.with_suffix('.response.json'))
                phase = 'final' if index == n - 1 else 'clue'
                token_budget = 4096 if phase == 'final' else 32 if mode == 'deferred_reasoning' else 2048
                expected = {'model': 'qwen3.8-flash', 'enable_thinking': treatment == 'thinking_on',
                            'preserve_thinking': False, 'stream': False, 'temperature': 0.0,
                            'response_format': {'type': 'json_object'}, 'max_tokens': token_budget}
                if treatment == 'thinking_on':
                    expected['thinking_budget'] = 7168
                wrong = {key: {'expected': value, 'actual': request.get(key)}
                         for key, value in expected.items() if request.get(key) != value}
                if treatment == 'thinking_off' and 'thinking_budget' in request:
                    wrong['thinking_budget'] = {'expected': 'absent', 'actual': request['thinking_budget']}
                if wrong:
                    request_mismatches.append({'treatment': treatment, 'task_id': task_id, 'mode': mode, 'call': index, 'differences': wrong})
                choice = response['choices'][0]
                message, usage = choice['message'], response.get('usage')
                native = message.get('reasoning_content') or message.get('reasoning') or message.get('reasoning_details') or ''
                assert isinstance(native, str), type(native)
                tokens = reasoning_tokens(usage)
                audit['calls'] += 1
                audit['phases'][phase] += 1
                audit['max_tokens'][request.get('max_tokens')] += 1
                audit['response_models'][response.get('model')] += 1
                audit['finish_reasons'][choice.get('finish_reason')] += 1
                audit['native_reasoning_nonempty' if native.strip() else 'native_reasoning_empty'] += 1
                if tokens is not None:
                    audit['reasoning_tokens_present'] += 1
                    audit['reasoning_tokens_zero' if tokens == 0 else 'reasoning_tokens_positive'] += 1
                    audit['reasoning_tokens_sum'] += tokens
                if isinstance(usage, dict):
                    audit['completion_tokens_sum'] += usage.get('completion_tokens', 0)
                else:
                    audit['token_usage_missing'] += 1
                call_count += 1
                if phase == 'final':
                    finals[treatment, task_id, mode] = {'finish': choice.get('finish_reason'),
                        'native': native, 'usage': usage,
                        'path': str(stem.with_suffix('.response.json'))}
    assert call_count == 6468 and len(finals) == 600
    for key, row in sorted(rows.items()):
        treatment, task_id, mode = key
        task, final = tasks[task_id], finals[key]
        strict, payload, solution = score(task, row['raw_output'])
        differences = {name: {'recomputed': value, 'recorded': row[name]}
                       for name, value in strict.items() if value != row[name]}
        if differences:
            mismatches.append({'treatment': treatment, 'task_id': task_id, 'mode': mode, 'differences': differences})
        diagnostic = diagnose(task, payload, solution, row['raw_output'])
        value_label_review = label_review(task, diagnostic)
        category = ('correct' if strict['success'] else
                    'truncated_with_no_answer' if not strict['parsed'] and final['finish'] == 'length' else
                    'parse_or_schema_missing' if not strict['parsed'] else 'parsed_incorrect')
        visible = payload.get('reasoning', '') if isinstance(payload, dict) else ''
        visible = visible if isinstance(visible, str) else str(visible)
        usage = final['usage'] or {}
        case = {'treatment': treatment, 'task_id': task_id, 'source_id': row['source_id'],
                'mode': mode, 'size': row['size'], 'category': category, **strict,
                'final_finish_reason': final['finish'], **diagnostic,
                'value_label_review': value_label_review,
                'final_content_characters': len(row['raw_output']),
                'visible_reasoning_characters': len(visible), 'native_reasoning_characters': len(final['native']),
                'final_completion_tokens': usage.get('completion_tokens'),
                'final_reasoning_tokens': reasoning_tokens(usage), 'raw_final_response': final['path']}
        cases.append(case)
        summary = group[treatment + '/' + mode]
        summary['total'] += 1
        summary['categories'][category] += 1
        summary['format_issue_counts'].update(diagnostic['flags'])
        summary['final_finish_reasons'][final['finish']] += 1
        summary['final_length_parsed'] += final['finish'] == 'length' and strict['parsed']
        summary['final_length_success'] += final['finish'] == 'length' and strict['success']
        summary['canonical_successes'] += diagnostic['canonical_success']
        summary['final_content_characters'].append(len(row['raw_output']))
        summary['final_visible_reasoning_characters'].append(len(visible))
        summary['final_native_reasoning_characters'].append(len(final['native']))
        if usage.get('completion_tokens') is not None:
            summary['final_completion_tokens'].append(usage['completion_tokens'])
        if reasoning_tokens(usage) is not None:
            summary['final_reasoning_tokens'].append(reasoning_tokens(usage))
    for summary in group.values():
        for name in ('final_content_characters', 'final_visible_reasoning_characters', 'final_native_reasoning_characters',
                     'final_completion_tokens', 'final_reasoning_tokens'):
            summary[name] = summarize(summary[name])
    audit = {'scope': 'Independent implementation of the frozen scoring rules; diagnostic key normalization leaves values unchanged.',
             'instances_checked': len(rows), 'score_mismatches': mismatches,
             'bounded_value_label_review': {
                 'scope': 'All 20 parsed-but-incorrect outputs; only the declared same-field aliases are reviewed, with house positions unchanged. Official scores are not changed.',
                 'declared_aliases': VALUE_LABEL_ALIASES,
                 'group_counts': {key: dict(Counter(c['value_label_review']['classification'] for c in cases
                                       if c['treatment'] + '/' + c['mode'] == key and c['category'] == 'parsed_incorrect'))
                                  for key in group},
                 'cases': [{k: c[k] for k in ('treatment', 'source_id', 'mode', 'correct_cells', 'total_cells', 'value_label_review', 'raw_final_response')}
                           for c in cases if c['category'] == 'parsed_incorrect'],
             },
             'groups': dict(group), 'failure_cases': [c for c in cases if not c['success']],
             'all_cases': cases}
    (output / 'output_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    config_checks = {}
    for treatment in TREATMENTS:
        config_text = (batch / 'configs' / f'{treatment}.yaml').read_text(encoding='utf-8')
        config_checks[treatment] = {
            'max_retries_is_zero': bool(re.search(r'(?m)^\s*max_retries:\s*0\s*$', config_text)),
            'max_format_repairs_is_zero': bool(re.search(r'(?m)^\s*max_format_repairs:\s*0\s*$', config_text)),
        }
        assert all(config_checks[treatment].values()), config_checks
    raw_audit = {'calls_checked': call_count, 'request_parameter_mismatches': request_mismatches,
                 'config_checks': config_checks,
                 'groups': dict(request_groups),
                 'config_evidence': {t: str(batch / 'configs' / f'{t}.yaml') for t in TREATMENTS},
                 'limitations': ['Reasoning text is a returned output channel, not a direct observation of all internal computation.',
                                 'A format/length decomposition describes observed failures and does not establish why model behavior changed.',
                                 'Visible reasoning field lengths are measured only when a complete top-level payload could be extracted; final_content_characters covers the entire returned visible text even for truncated JSON.']}
    (output / 'raw_request_audit.json').write_text(json.dumps(raw_audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    fields = ['treatment', 'source_id', 'mode', 'size', 'category', 'success', 'parsed', 'no_answer',
              'correct_cells', 'total_cells', 'final_finish_reason', 'top_json_type', 'flags',
              'canonical_success', 'canonical_correct_cells', 'canonical_missing_cells', 'canonical_wrong_cells',
              'final_content_characters', 'visible_reasoning_characters', 'native_reasoning_characters', 'final_completion_tokens',
              'final_reasoning_tokens', 'value_label_review', 'raw_final_response']
    with (output / 'failure_cases.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for case in cases:
            if not case['success']:
                writer.writerow({**case, 'flags': ';'.join(case['flags']),
                                 'canonical_wrong_cells': json.dumps(case['canonical_wrong_cells'], ensure_ascii=False),
                                 'value_label_review': json.dumps(case['value_label_review'], ensure_ascii=False)})
    print(json.dumps({'instances': len(rows), 'calls': call_count, 'score_mismatches': len(mismatches),
                      'request_parameter_mismatches': len(request_mismatches),
                      'groups': {k: {'categories': v['categories'], 'canonical_successes': v['canonical_successes'],
                                     'format_issue_counts': v['format_issue_counts']} for k, v in group.items()}}, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
