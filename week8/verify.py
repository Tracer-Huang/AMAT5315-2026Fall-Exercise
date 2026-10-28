"""Audit the locally produced Week 8 records against the learning-sheet protocol."""
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
checks = []


def check(name, condition, detail):
    checks.append({'name': name, 'pass': bool(condition), 'detail': detail})


def read(path):
    return json.loads(path.read_text())


def before(a, b):
    return dt.datetime.fromisoformat(a) < dt.datetime.fromisoformat(b)


def main():
    c = read(ROOT / 'data/cpu-measurements.json')
    g = read(ROOT / 'data/grokking-measurements.json')
    p1, p2, p3 = c['part1'], c['part2'], c['part3b']
    expected_hash = '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'
    digest = hashlib.sha256((ROOT / 'data/input.txt').read_bytes()).hexdigest()
    check('corpus fingerprint', digest == expected_hash == p1['corpus_sha256'], digest)
    check('training, validation and scored counts',
          (p1['train_tokens'], p1['val_tokens'], p1['scored_tokens']) == (1003854, 111540, 111539),
          [p1['train_tokens'], p1['val_tokens'], p1['scored_tokens']])
    for name, reference in [('uniform', math.log(65)), ('unigram', 3.3473053354), ('bigram', 2.4818894321)]:
        value = p1[name]['loss']
        check(f'{name} loss and PPL', abs(value-reference) < 1e-5 and
              abs(p1[name]['ppl']-math.exp(value)) < 1e-8, [value, p1[name]['ppl']])
    check('two-target perplexity', abs(p1['two_target_example']['ppl']-math.sqrt(8)) < 1e-10,
          p1['two_target_example']['ppl'])
    check('part 2 held-out and independent score', p2['self_score']['token_count'] == 111539 and
          p2['independent_score']['token_count'] == 111539 and
          p2['self_vs_independent_abs_diff'] < 1e-4 and p2['self_score']['loss'] < 1.85,
          [p2['self_score']['loss'], p2['independent_score']['loss'], p2['self_vs_independent_abs_diff']])
    check('causal prefix and attention mask', p2['causality']['pairs'] == 8 and
          p2['causality']['max_prefix_logit_change'] == 0 and p2['attention_upper_triangle_max'] == 0 and
          p2['attention_row_sum_max_error'] < 1e-5, p2['causality'])
    check('two 500-character continuations', len(p2['samples']) == 2 and
          {s['temperature'] for s in p2['samples']} == {0.5, 1.0} and
          all(s['seed'] == 7 and s['text'].startswith(s['prompt']) and
              len(s['text'])-len(s['prompt']) == 500 for s in p2['samples']),
          [len(s['text'])-len(s['prompt']) for s in p2['samples']])
    pred, result = p3['prediction'], p3['result']
    check('scaling fit and excluded widths', [r['width'] for r in pred['fit_table']] ==
          [12, 16, 24, 32, 48, 64, 96, 128] and
          [r['width'] for r in pred['excluded_table']] == [8] and
          all(len(r['losses']) == 2 for r in pred['fit_table'] + pred['excluded_table']),
          [r['width'] for r in pred['fit_table']])
    check('held-out model twice as large', pred['heldout']['N'] > 2 * pred['fit_table'][-1]['N'],
          [pred['heldout']['N'], pred['fit_table'][-1]['N']])
    lo, hi = result['interval_95']
    check('scaling alpha and interval width', .05 <= pred['fit']['pure_power_law']['alpha'] <= .10 and
          hi-lo < .2, [pred['fit']['pure_power_law']['alpha'], lo, hi])
    frozen = read(ROOT / 'artifacts/scaling-prediction.json')
    check('prediction snapshot preserved', frozen == pred and not pred['heldout_runs_present_when_written'],
          pred['written'])
    all_widths = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192]
    all_runs = []
    for width in all_widths:
        for seed in (1, 2):
            path = ROOT / f'artifacts/scaling/w{width:03d}-s{seed}'
            r, meta = read(path / 'result.json'), read(path / 'model.json')
            sha = hashlib.sha256((path / 'model.npz').read_bytes()).hexdigest()
            all_runs.append(r)
            check(f'scaling width {width} seed {seed}', r['token_count'] == 111539 and
                  r['config']['max_steps'] == 1500 and abs(r['loss']-r['independent_loss']) < 1e-4 and
                  sha == r['checkpoint_sha256'] == meta['checkpoint_sha256'],
                  [r['loss'], r['independent_loss'], r['started'], r['finished']])
    held = [r for r in all_runs if r['width'] == 192]
    check('prediction predates both held-out starts', all(before(pred['written'], r['started']) for r in held),
          [pred['written']] + [r['started'] for r in held])
    check('held-out mean agrees with raw runs', abs(result['actual']-np.mean([r['loss'] for r in held])) < 1e-12,
          dict(predicted=result['predicted'], interval=[lo, hi], actual=result['actual'],
               within_interval=result['within_interval']))
    main_runs = g['groups']['main']['runs']
    check('five main grokking seeds', len(main_runs) == 5 and {r['seed'] for r in main_runs} == set(range(5)) and
          all(r['grokking_step'] is not None and r['grokking_step'] <= 10000 and r['delay_ratio'] > 10
              for r in main_runs), [(r['seed'], r['memorization_step'], r['grokking_step']) for r in main_runs])
    for group, expected in [('main', 5), ('wd0', 3), ('frac05', 3), ('random', 1)]:
        rows = g['groups'][group]['runs']
        check(f'grokking {group} run count', len(rows) == expected, len(rows))
        for r in rows:
            with np.load(ROOT / 'artifacts/grokking-runs' / f"{r['name']}.npz", allow_pickle=False) as z:
                meta = json.loads(str(z['meta']))
                steps, train_acc, test_acc = z['h_step'], z['h_train_acc'], z['h_test_acc']
                mem = steps[train_acc >= .99]
                grok = steps[test_acc >= .99]
                valid = (int(mem[0]) if len(mem) else None) == r['memorization_step'] and \
                        (int(grok[0]) if len(grok) else None) == r['grokking_step'] and \
                        meta['config']['seed'] == r['seed'] and \
                        np.intersect1d(z['train'], z['test']).size == 0
                if group == 'random':
                    a = np.repeat(np.arange(97), 97); b = np.tile(np.arange(97), 97)
                    valid = valid and np.array_equal(np.sort(z['labels']), np.sort((a+b) % 97))
                    valid = valid and not np.array_equal(z['labels'], (a+b) % 97)
                check(f"raw curve {r['name']}", valid, [r['memorization_step'], r['grokking_step']])
    mech = g['mechanism']
    check('five key-frequency tests', len(mech['per_seed']) == 5 and
          all(r['top_k_fraction_final'] > .8 and r['keep_key_test_acc'] > .95 and
              r['remove_key_test_acc'] < .05 for r in mech['per_seed'].values()), mech['per_seed'])
    ab = mech['ablation']
    check('non-key controls', len(ab['remove_random_nonkey']) == 10 and
          min(r['test_acc'] for r in ab['remove_random_nonkey']) > .95 and
          ab['keep_random_nonkey']['test_acc'] < .05,
          [ab['remove_random_nonkey_summary'], ab['keep_random_nonkey']['test_acc']])
    extras = p3.get('extra_heldout', {})
    frozen_extras = read(ROOT / 'artifacts/scaling-extra-predictions.json') if extras else {}
    extra_starts = []
    for width, entry in extras.items():
        if 'actual' not in entry:
            continue
        runs = [read(ROOT / f'artifacts/scaling/w{int(width):03d}-s{seed}/result.json') for seed in (1, 2)]
        extra_starts.extend(r['started'] for r in runs)
        check(f'optional width {width} prediction snapshot preserved',
              all(entry[k] == v for k, v in frozen_extras[width].items()), frozen_extras[width]['written'])
        check(f'optional width {width} prediction before starts',
              all(before(entry['written'], r['started']) for r in runs) and
              abs(entry['actual']-np.mean([r['loss'] for r in runs])) < 1e-12 and
              all(abs(r['loss']-r['independent_loss']) < 1e-4 for r in runs),
              dict(prediction=entry['prediction'], interval=entry['interval_95'], actual=entry['actual'],
                   within_interval=entry['within_interval']))
    if len(extra_starts) == 4:
        check('both optional predictions precede all optional runs',
              all(before(e['written'], s) for e in frozen_extras.values() for s in extra_starts),
              [e['written'] for e in frozen_extras.values()] + extra_starts)
    failed = [c['name'] for c in checks if not c['pass']]
    caveats = ['Course reference outcomes were already public; timing establishes procedural order only.',
               'The course package checker targets an additional speedrun contract; the independent NumPy evaluator was used.']
    if 'recovered' in p2.get('curve_precision', ''):
        caveats.insert(0, 'Part 2 curve recovered from the original four-decimal console log.')
    report = dict(passed=len(checks)-len(failed), total=len(checks), failed=failed, checks=checks, caveats=caveats)
    (ROOT / 'evidence/verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"{report['passed']}/{report['total']} checks passed; failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
