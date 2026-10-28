"""Recompute the Week 8 PDF requirements from local data and saved weights.

This audit is read-only except for its final evidence/strict-audit.json report.
It uses the course's independent NumPy decoder and an FFT-based grokking forward.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'cpu'), str(ROOT / 'checker'), str(ROOT / 'grokking')]
import evaluator
import lm
import numpy_check
import part2
from PIL import Image

CHECKS = []


def require(name, condition, detail=None):
    CHECKS.append(dict(name=name, passed=bool(condition), detail=detail))
    if not condition:
        raise AssertionError(f'{name}: {detail}')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def before(first, second):
    return dt.datetime.fromisoformat(first) < dt.datetime.fromisoformat(second)


def audit_baselines(c):
    raw = (ROOT / 'data/input.txt').read_bytes()
    fingerprint = '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'
    require('pinned corpus bytes', hashlib.sha256(raw).hexdigest() == fingerprint, fingerprint)
    text = raw.decode('utf-8')
    vocab = sorted(set(text))
    lookup = {ch: i for i, ch in enumerate(vocab)}
    ids = np.fromiter((lookup[ch] for ch in text), dtype=np.int64)
    split = int(.9 * len(ids))
    train, val = ids[:split], ids[split:]
    require('65 sorted characters and specified split',
            len(vocab) == 65 and len(train) == 1003854 and len(val) == 111540, [len(train), len(val)])
    uni = (np.bincount(train, minlength=65) + 1) / (len(train) + 65)
    counts = np.bincount(train[:-1] * 65 + train[1:], minlength=65 * 65).reshape(65, 65)
    bi = (counts + 1) / (counts.sum(axis=1, keepdims=True) + 65)
    loss = {'uniform': np.log(65), 'unigram': -np.log(uni[val[1:]]).mean(),
            'bigram': -np.log(bi[val[:-1], val[1:]]).mean()}
    for name, value in loss.items():
        require(f'{name}: independently recomputed loss and PPL',
                abs(value - c['part1'][name]['loss']) < 1e-10 and
                abs(np.exp(value) - c['part1'][name]['ppl']) < 1e-9, float(value))
    require('two-target example', abs(c['part1']['two_target_example']['loss'] - np.log(8)/2) < 1e-12 and
            abs(c['part1']['two_target_example']['ppl'] - np.sqrt(8)) < 1e-12)
    return vocab, val


def audit_transformer(c, vocab, val):
    p2 = c['part2']
    expected = dict(n_layer=2, n_head=4, n_embd=64, block_size=128, vocab_size=65,
                    batch_size=16, learning_rate=.003, min_lr=.0003, warmup_steps=100,
                    max_steps=3000, weight_decay=.1, seed=1337)
    require('transformer training settings', all(p2['config'].get(k) == v for k, v in expected.items()))
    path = ROOT / 'artifacts/part2'
    numeric, config, meta = evaluator.load_model(path, vocab)
    require('transformer saved checkpoint', meta['step'] == 3000 and meta['parameter_count'] == 116673 and
            meta['checkpoint_sha256'] == p2['checkpoint_sha256'] == sha(path/'model.npz'))
    full = evaluator.evaluate(numeric, config, val)
    require('fresh independent full-validation score', full['token_count'] == 111539 and
            abs(full['loss']-p2['independent_score']['loss']) < 1e-10 and full['loss'] < 1.85,
            dict(loss=full['loss'], ppl=full['ppl']))
    if 'recovered' in p2.get('curve_precision', ''):
        recovered = part2.rows_from_log((ROOT/'artifacts/logs/part2.log').read_text(), 16, 3000)
        require('training curve matches retained original log', recovered == p2['curve'])
    else:
        require('complete recorded training curve', [r['step'] for r in p2['curve']] == list(range(150, 3001, 150)) and
                all(np.isfinite(r['train_loss']) and np.isfinite(r['val_loss_subset']) for r in p2['curve']))
    params, _ = lm.load(path)
    rng = np.random.default_rng(0)
    numpy_prefix = jax_prefix = forward_diff = 0.0
    for _ in range(8):
        start = int(rng.integers(0, len(val) - 128))
        a = val[start:start + 128].astype(np.int32)[None]
        b = a.copy(); b[0, 64:] = rng.integers(0, 65, 64)
        an, _ = evaluator.forward(numeric, config, a)
        bn, _ = evaluator.forward(numeric, config, b)
        aj = np.asarray(lm.forward(params, a, p2['config']))
        bj = np.asarray(lm.forward(params, b, p2['config']))
        numpy_prefix = max(numpy_prefix, float(np.max(abs(an[:, :64] - bn[:, :64]))))
        jax_prefix = max(jax_prefix, float(np.max(abs(aj[:, :64] - bj[:, :64]))))
        forward_diff = max(forward_diff, float(np.max(abs(an-aj))), float(np.max(abs(bn-bj))))
    require('eight actual prefix-causality pairs', numpy_prefix == 0 and jax_prefix == 0,
            dict(numpy=numpy_prefix, jax=jax_prefix))
    require('trained JAX / independent NumPy logits', forward_diff < 2e-4, forward_diff)
    prompt = 'ROMEO:\nBut soft, what light'
    _, attention = evaluator.forward(numeric, config, np.array([[vocab.index(ch) for ch in prompt]]))
    require('four causal attention heads and normalized rows', attention.shape == (1, 4, len(prompt), len(prompt)) and
            np.max(abs(np.triu(attention, k=1))) == 0 and np.max(abs(attention.sum(-1)-1)) < 1e-12)
    for sample in p2['samples']:
        regenerated = lm.generate(params, p2['config'], vocab, 'ROMEO:\n', sample['temperature'], 500, 7)
        require(f"saved sample temperature {sample['temperature']}", regenerated == sample['text'] and
                len(regenerated)-len('ROMEO:\n') == 500)
    require('two requested sample temperatures', {s['temperature'] for s in p2['samples']} == {.5, 1.})


def audit_scaling(c, vocab, val):
    part = c['part3b']; pred = part['prediction']; res = part['result']
    extras = part.get('extra_heldout', {})
    if extras:
        require('additional experiments complete', set(extras) == {'256', '384'} and
                all('actual' in e for e in extras.values()))
    require('eight fit widths, width 8 excluded', [r['width'] for r in pred['fit_table']] ==
            [12, 16, 24, 32, 48, 64, 96, 128] and [r['width'] for r in pred['excluded_table']] == [8])
    runs = {}
    for width in [8, 12, 16, 24, 32, 48, 64, 96, 128, 192] + ([256, 384] if extras else []):
        for seed in [1, 2]:
            path = ROOT / f'artifacts/scaling/w{width:03d}-s{seed}'
            r = read(path/'result.json'); saved = read(path/'independent-eval.json'); cfg = r['config']
            expected = dict(n_layer=2, n_head=2, n_embd=width, block_size=128, vocab_size=65,
                            batch_size=16, learning_rate=.003, min_lr=.0003, warmup_steps=100,
                            max_steps=1500, weight_decay=.1, seed=seed)
            require(f'width {width} seed {seed} settings', all(cfg.get(k) == v for k, v in expected.items()) and
                    'cpu' in r['timing']['device'].lower() and read(path/'started.json')['started'] == r['started'])
            require(f'width {width} seed {seed} scoring and checkpoint',
                    saved['checkpoint_sha256'] == r['checkpoint_sha256'] == sha(path/'model.npz') and
                    saved['token_count'] == r['token_count'] == 111539 and
                    saved['block_tokens'] == [128]*871 + [51] and
                    abs(saved['loss'] - r['loss']) < 1e-4 and
                    abs(saved['loss'] - r['independent_loss']) < 1e-12)
            runs[(width, seed)] = r
    for width in [192] + ([384] if extras else []):
        path = ROOT / f'artifacts/scaling/w{width:03d}-s1'
        numeric, cfg, meta = evaluator.load_model(path, vocab)
        fresh = evaluator.evaluate(numeric, cfg, val)
        require(f'width {width} fresh independent full-validation score', fresh['token_count'] == 111539 and
                abs(fresh['loss']-runs[(width, 1)]['independent_loss']) < 1e-10, fresh['loss'])
    require('192 prediction snapshot before both starts', read(ROOT/'artifacts/scaling-prediction.json') == pred and
            all(before(pred['written'], runs[(192, seed)]['started']) for seed in [1, 2]))
    target = pred['heldout']['N']
    require('held-out parameter ratio', target > 2*pred['fit_table'][-1]['N'])
    require('192 held-out result and interval', abs(res['actual']-np.mean([runs[(192, s)]['loss'] for s in [1, 2]])) < 1e-12 and
            res['interval_95'][1]-res['interval_95'][0] < .2 and
            res['within_interval'] == (res['interval_95'][0] <= res['actual'] <= res['interval_95'][1]))
    if extras:
        extra = read(ROOT/'artifacts/scaling-extra-predictions.json')
        require('both additional predictions before either extra training',
                all(before(e['written'], runs[(w, s)]['started']) for e in extra.values() for w in [256, 384] for s in [1, 2]))
        for width in [256, 384]:
            item = extras[str(width)]
            require(f'width {width} extra result', abs(item['actual']-np.mean([runs[(width, s)]['loss'] for s in [1, 2]])) < 1e-12 and
                    item['within_interval'] == (item['interval_95'][0] <= item['actual'] <= item['interval_95'][1]))
    # The course reference uses weighted linear regression in log-loss space.
    # This retrospective fit minimizes the PDF's displayed raw-loss chi-square;
    # it is never used to replace the earlier, timestamped prediction.
    table = pred['fit_table']; fit = pred['fit']['pure_power_law']; n0 = pred['fit']['N0']
    n = np.array([r['N'] for r in table]); y = np.array([r['mean'] for r in table])
    sigma = np.array(pred['fit']['sigma_seed'])
    f = lambda z: (z[0] * (n/n0)**(-z[1]) - y) / sigma
    exact = least_squares(f, [fit['a'], fit['alpha']], bounds=([0, 0], [100, 5]),
                          xtol=1e-14, ftol=1e-14, gtol=1e-14)
    exact_predict = exact.x[0] * (target/n0)**(-exact.x[1])
    delta = float(exact_predict - res['predicted'])
    require('retrospective raw-loss WLS minimizes the displayed chi-square',
            np.isfinite(delta) and np.sum(exact.fun**2) <= np.sum(f([fit['a'], fit['alpha']])**2) + 1e-8,
            dict(prospective_log_space=float(res['predicted']), retrospective_raw_loss=float(exact_predict),
                 difference=delta, log_chi2=float(np.sum(f([fit['a'], fit['alpha']])**2)),
                 exact_chi2=float(np.sum(exact.fun**2))))


def audit_grokking(g):
    p = 97; a = np.repeat(np.arange(p), p); b = np.tile(np.arange(p), p); truth = (a+b) % p
    figure_data = np.load(ROOT/'data/grokking-evidence.npz', allow_pickle=False)
    settings = {'main': (5, .3, 1., 'true'), 'wd0': (3, .3, 0., 'true'),
                'frac05': (3, .5, 1., 'true'), 'random': (1, .3, 1., 'random')}
    for group, (number, frac, wd, labels_kind) in settings.items():
        rows = g['groups'][group]['runs']
        require(f'{group} seed set', len(rows) == number and {r['seed'] for r in rows} == set(range(number)))
        for row in rows:
            with np.load(ROOT/'artifacts/grokking-runs'/f"{row['name']}.npz", allow_pickle=False) as z:
                meta = json.loads(str(z['meta'])); cfg = meta['config']; tr, te, labels = z['train'], z['test'], z['labels']
                expected = dict(model='transformer', p=97, frac=frac, wd=wd, lr=.003, seed=row['seed'],
                                labels=labels_kind, log_every=100, betas=[.9, .98],
                                arch={'d_model': 64, 'n_heads': 4, 'd_mlp': 256})
                require(f"{row['name']} settings and split", all(cfg.get(k) == v for k, v in expected.items()) and
                        len(tr) == round(frac*p*p) and len(tr)+len(te) == p*p and
                        not np.intersect1d(tr, te).size and np.array_equal(np.sort(labels), np.sort(truth)))
                if labels_kind == 'true':
                    require(f"{row['name']} correct labels", np.array_equal(labels, truth))
                else:
                    require(f"{row['name']} labels are permuted", not np.array_equal(labels, truth))
                steps = z['h_step']; tracc = z['h_train_acc']; teacc = z['h_test_acc']
                for key in ['step', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'norm']:
                    require(f"{row['name']} {key} figure data", np.allclose(
                        figure_data[f"curve/{row['name']}/{key}"], z[f'h_{key}'], rtol=1e-6, atol=1e-7))
                mem = steps[tracc >= .99]; grok = steps[teacc >= .99]
                mem_step = int(mem[0]) if len(mem) else None
                grok_step = int(grok[0]) if len(grok) else None
                expected_end = 3000 if group == 'random' else (min(10000, grok_step+1000) if grok_step else 10000)
                require(f"{row['name']} logged steps and stop", np.array_equal(steps, np.arange(0, expected_end+1, 100)) and
                        mem_step == row['memorization_step'] and grok_step == row['grokking_step'])
                snap = lambda stage: {k.split('__', 2)[2]: z[k].astype(np.float64)
                                      for k in z.files if k.startswith(f'snap__{stage}__')}
                for stage, ix in [('memorized', int(np.where(steps == mem_step)[0][0])), ('final', -1)]:
                    weights = snap(stage)
                    norm = np.sqrt(sum(np.sum(w*w) for w in weights.values()))
                    require(f"{row['name']} {stage} parameter norm", abs(norm-z['h_norm'][ix]) < 1e-4)
                    for label, indices, recorded in [('train', tr, z['h_train_acc'][ix]), ('test', te, z['h_test_acc'][ix])]:
                        logits = numpy_check.transformer_logits(weights, a[indices], b[indices])
                        acc = np.mean(np.argmax(logits, axis=1) == labels[indices])
                        loss = np.mean(logsumexp(logits, axis=1)-logits[np.arange(len(indices)), labels[indices]])
                        record_loss = z['h_train_loss'][ix] if label == 'train' else z['h_test_loss'][ix]
                        require(f"{row['name']} {stage} {label} independent score",
                                abs(acc-recorded) < 1e-4 and abs(loss-record_loss) < 2e-4)
                if group == 'main':
                    require(f"{row['name']} delayed grokking", grok_step is not None and grok_step/mem_step > 10)
                    mechanism = g['mechanism']['per_seed'][row['name']]
                    for stage, key in [('init','top_k_fraction_init'), ('memorized','top_k_fraction_memorized'),
                                       ('final','top_k_fraction_final')]:
                        E = snap(stage)['W_E'][:p]
                        power = numpy_check.power(E)[1:]
                        share = np.sort(power)[-5:].sum()/power.sum()
                        require(f"{row['name']} {stage} FFT power", abs(share-mechanism[key]) < 1e-5)
                    weights = snap('final'); E = weights['W_E'][:p]
                    key = sorted((np.argsort(numpy_check.power(E)[1:])[-5:]+1).tolist())
                    require(f"{row['name']} key frequencies", key == mechanism['key_frequencies'])
                    for kind, keep, threshold, comparison in [('keep', True, .95, True), ('remove', False, .05, False)]:
                        filtered = numpy_check.fft_filter(E, key, keep)
                        logits = numpy_check.transformer_logits(weights, a[te], b[te], filtered)
                        acc = float(np.mean(np.argmax(logits, axis=1) == labels[te]))
                        recorded = mechanism[f'{kind}_key_test_acc']
                        require(f"{row['name']} independent {kind}-key ablation", abs(acc-recorded) < 1e-4 and
                                ((acc > threshold) if comparison else (acc < threshold)), acc)
    ab = g['mechanism']['ablation']
    require('ten random non-key removals and one non-key keep', len(ab['remove_random_nonkey']) == 10 and
            all(r['test_acc'] > .95 for r in ab['remove_random_nonkey']) and ab['keep_random_nonkey']['test_acc'] < .05)
    figure_data.close()


def audit_figures():
    names = ['cpu-part1-probabilities.png', 'cpu-part1-baselines.png', 'cpu-part2-loss.png',
             'cpu-part2-attention.png', 'cpu-part3-scaling.png', 'cpu-part3-scaling-extra.png',
             'grokking-curves.png', 'grokking-controls.png', 'grokking-fourier.png', 'grokking-ablation.png']
    for name in names:
        path = ROOT/'evidence'/name
        with Image.open(path) as image:
            size = image.size
            image.verify()
        require(f'figure {name}', min(size) >= 400, list(size))


def main():
    c = read(ROOT/'data/cpu-measurements.json'); g = read(ROOT/'data/grokking-measurements.json')
    published_c = read(ROOT/'evidence/cpu-measurements.json'); published_g = read(ROOT/'evidence/grokking-measurements.json')
    same_run = c['part3b']['prediction']['written'] == published_c['part3b']['prediction']['written']
    require('published measurements match the original run or identify a later reproduction',
            (c == published_c and g == published_g) if same_run else True,
            'original run' if same_run else 'new run with a distinct prediction timestamp')
    vocab, val = audit_baselines(c)
    audit_transformer(c, vocab, val)
    audit_scaling(c, vocab, val)
    audit_grokking(g)
    audit_figures()
    report = dict(checked_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                  passed=len(CHECKS), failed=[], checks=CHECKS,
                  caveats=['Reference outcomes were visible before this work.',
                           ('Part 2 training curve is recovered from a four-decimal log.' if
                            'recovered' in c['part2'].get('curve_precision', '') else 'Part 2 curve is full precision.'),
                           'The raw-loss WLS comparison is retrospective and does not replace the saved prediction.',
                           ('Scaling runs used four concurrent processes; macOS did not enforce a one-core affinity.'
                            if same_run else 'Verify CPU affinity independently for this reproduction.')])
    (ROOT/'evidence/strict-audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f"{len(CHECKS)} strict checks passed; failed=[]")


if __name__ == '__main__':
    main()
