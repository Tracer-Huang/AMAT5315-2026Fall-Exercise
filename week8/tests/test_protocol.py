"""Focused checks for divergences between the course code and the learning sheet."""
import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "grokking"))
import grok
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cpu"))
import scaling
import lm
import part2
import baselines


def test_random_labels_are_a_permutation_of_the_original_labels():
    (a, b, labels), train, test = grok.make_split(97, 0.3, 0, "random")
    truth = (a + b) % 97
    assert np.array_equal(np.sort(labels), np.sort(truth))
    assert not np.array_equal(labels, truth)
    assert len(np.intersect1d(train, test)) == 0
    assert len(train) + len(test) == 97 * 97


def test_baseline_summary_prints_counts_and_two_target_example(capsys):
    result, _, _ = baselines.baselines()
    baselines.print_summary(result)
    output = capsys.readouterr().out
    for value in ("1003854", "111540", "111539", "1.039721", "2.828427"):
        assert value in output


def test_blind_order_is_checked_against_start_not_finish():
    # A prediction written during a run cannot be called blind, even if it
    # predates that run's finish.
    prediction = "2026-10-28T10:01:00+00:00"
    assert not scaling.prediction_precedes_runs(prediction, [
        {"started": "2026-10-28T10:00:00+00:00", "finished": "2026-10-28T10:02:00+00:00"}
    ])
    assert scaling.prediction_precedes_runs(prediction, [
        {"started": "2026-10-28T10:02:00+00:00", "finished": "2026-10-28T10:03:00+00:00"}
    ])


def test_independent_evaluator_accepts_relative_checkpoint_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = lm.make_config(1, 8, 2)
    params = lm.initialize(config, lm.jax.random.PRNGKey(4))
    lm.export(Path("checkpoint"), params, config, 0)
    report = lm.independent_score(Path("checkpoint"))
    assert report["token_count"] == 111539
    assert np.isfinite(report["loss"])


def test_recovered_training_curve_requires_the_final_logged_step():
    import pytest

    log = "  step 150: train=2.9015 val[96]=2.4943 wall=3s\n"
    rows = part2.rows_from_log(log, batch=16, expected_final=150)
    assert rows[0]["step"] == 150
    assert rows[0]["tokens"] == 150 * 16 * 128
    assert rows[0]["train_loss"] == 2.9015
    assert rows[0]["val_loss_subset"] == 2.4943
    with pytest.raises(ValueError):
        part2.rows_from_log(log, batch=16, expected_final=3000)


def test_part2_refuses_to_overwrite_a_saved_model(tmp_path, monkeypatch):
    import pytest

    out = tmp_path / 'model'
    out.mkdir()
    (out / 'model.npz').write_bytes(b'previous run')
    monkeypatch.setattr(lm, 'train', lambda *args, **kwargs: pytest.fail('training must not start'))
    with pytest.raises(FileExistsError):
        part2.main(Path(__file__).resolve().parents[1] / 'cpu/configs/part2.json', out)
    assert (out / 'model.npz').read_bytes() == b'previous run'


def test_extra_prediction_uses_the_selected_protocol_and_pure_law(tmp_path, monkeypatch):
    monkeypatch.setattr(scaling, 'BOOTSTRAP', 12)
    monkeypatch.setattr(scaling, 'RUNS', tmp_path / 'runs')
    monkeypatch.setattr(lm, 'MEASUREMENTS', tmp_path / 'measurements.json')
    numbers = np.array([4000, 6500, 14000, 25000, 56000, 100000, 220000, 400000.])
    means = 2.3 * (numbers / scaling.N0) ** -0.07 + np.array([.01, -.01, .006, -.004, 0, .004, -.005, .002])
    table = [dict(N=float(n), N_total=float(n + 500), mean=float(y),
                  losses=[float(y - .01), float(y + .01)]) for n, y in zip(numbers, means)]
    base = scaling.analyse(table, 1e6, 2, .02, np.random.default_rng(2026))
    store = {'part3b': {'prediction': {'fit_table': table, 'pooled_seed_std': .02, 'fit': base}}}
    lm.MEASUREMENTS.write_text(json.dumps(store))
    spec = {'section': 'part3b', 'primary': 'pure', 'seeds': [1, 2],
            'n_layer': 2, 'n_head': 2, 'training': {}}
    scaling.stage_predict_extra(spec, 256)
    entry = json.loads(lm.MEASUREMENTS.read_text())['part3b']['extra_heldout']['256']
    assert entry['primary'] == 'pure'
    assert entry['prediction'] == entry['pure_power_law_prediction']
    assert entry['interval_95'][0] < entry['interval_95'][1]
