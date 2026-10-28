"""Part 2: train one small causal transformer on CPU, rescore it independently,
probe causality, sample at two temperatures and record a first-layer attention map.

    python part2.py configs/part2.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lm  # noqa: E402

PROMPT = 'ROMEO:\nBut soft, what light'
SAMPLE_PROMPT = 'ROMEO:\n'


def rows_from_log(text, batch, expected_final):
    pattern = re.compile(r"^\s*step (\d+): train=([\d.]+) val\[96\]=([\d.]+) wall=(\d+)s$", re.M)
    rows = [dict(step=int(step), tokens=int(step) * batch * 128,
                 train_loss=float(train), val_loss_subset=float(val), wall_seconds=int(wall))
            for step, train, val, wall in pattern.findall(text)]
    if not rows or rows[-1]['step'] != expected_final:
        raise ValueError('training log does not contain the expected final step')
    if any(a['step'] >= b['step'] for a, b in zip(rows, rows[1:])):
        raise ValueError('training log steps are not strictly increasing')
    return rows


def main(config_path, out, recover_log=None):
    config = json.loads(Path(config_path).read_text())
    counts = lm.parameter_counts(config)
    print(f"config {config} params={counts}", flush=True)
    if recover_log:
        log = Path(recover_log).read_text()
        if f"config {config} params=" not in log:
            raise ValueError('training log configuration does not match')
        params, meta = lm.load(out)
        if meta['step'] != config['max_steps'] or any(
                meta['architecture'][key] != config[key]
                for key in ('n_layer', 'n_head', 'n_embd', 'block_size', 'vocab_size', 'dropout')):
            raise ValueError('saved model architecture or step does not match configuration')
        rows = rows_from_log(log, config['batch_size'], config['max_steps'])
        final = lm.score(params, config, lm.corpus()[1])
        timing = dict(wall_seconds_rounded_from_log=rows[-1]['wall_seconds'],
                      recovered_from_log=str(recover_log), jax_version=lm.jax.__version__,
                      device=str(lm.jax.devices()[0]))
    else:
        if (Path(out) / 'model.npz').exists() or (Path(out) / 'model.json').exists():
            raise FileExistsError(f'checkpoint already exists in {out}; choose a new output directory')
        params, rows, final, timing = lm.train(config, log_every=config.get('log_every', 100))
        meta = lm.export(out, params, config, config['max_steps'])
    independent = lm.independent_score(out)
    probe = lm.causality_probe(params, config)
    _, _, corpus = lm.corpus()
    vocabulary = corpus['vocabulary']
    samples = [dict(prompt=SAMPLE_PROMPT, temperature=t, seed=7, length=500,
                    checkpoint_sha256=meta['checkpoint_sha256'],
                    text=lm.generate(params, config, vocabulary, SAMPLE_PROMPT, t, 500, 7))
               for t in (0.5, 1.0)]
    ids = np.array([[vocabulary.index(c) for c in PROMPT]], np.int32)
    logits, attention = lm.forward(params, ids, config, return_attention=True)
    attention = np.asarray(attention)[0]          # heads x t x t, first layer
    upper = float(np.max(np.abs(np.triu(attention, k=1))))
    probabilities = np.asarray(lm.jax.nn.softmax(logits[0, -1]))
    top = np.argsort(probabilities)[::-1][:5]
    lm.save_arrays(part2_attention=attention, part2_prompt=np.frombuffer(PROMPT.encode(), np.uint8),
                   part2_next_probabilities=probabilities,
                   part2_steps=np.array([r['step'] for r in rows]),
                   part2_train=np.array([r['train_loss'] for r in rows]),
                   part2_val=np.array([r['val_loss_subset'] for r in rows]))
    result = dict(config=config, parameters=counts, timing=timing, curve=rows,
                  curve_precision=('four decimal places, recovered from the original training log'
                                   if recover_log else 'full precision'),
                  self_score={k: final[k] for k in ('loss', 'ppl', 'bpc', 'token_count', 'summed_nll')},
                  independent_score=independent,
                  self_vs_independent_abs_diff=abs(final['loss'] - independent['loss']),
                  causality=probe, attention_prompt=PROMPT,
                  attention_upper_triangle_max=upper,
                  attention_row_sum_max_error=float(np.max(np.abs(attention.sum(-1) - 1))),
                  next_char_top5=[(vocabulary[i], float(probabilities[i])) for i in top],
                  samples=samples, checkpoint_sha256=meta['checkpoint_sha256'])
    lm.record('part2', result)
    print(json.dumps({k: result[k] for k in ('self_score', 'independent_score', 'causality',
                                              'attention_upper_triangle_max')}, indent=2))
    if not recover_log:
        print(f"steps/s={timing['steps_per_second']:.2f} wall={timing['wall_seconds']:.0f}s")
    for s in samples:
        print(f"--- T={s['temperature']}\n{s['text']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--out', default=str(lm.HERE / 'work' / 'part2'))
    parser.add_argument('--recover-log', help='rescore an existing checkpoint and recover its logged curve')
    args = parser.parse_args()
    main(args.config, args.out, args.recover_log)
