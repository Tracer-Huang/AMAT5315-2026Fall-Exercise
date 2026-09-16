import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import numpy as np
from stats import acf, block_se, bootstrap_chi, peak, susceptibility, tau_int


def test_chi_absolute_convention():
    assert susceptibility(np.array([-1.0, 1.0, -1.0, 1.0]), 4, 2) == 0
    assert susceptibility(np.array([0.0, 1.0, 0.0, 1.0]), 4, 2) == 2


def test_parabola_and_invalid_peak():
    t = np.linspace(2, 2.6, 13)
    assert abs(peak(t, 10 - (t - 2.31) ** 2) - 2.31) < 1e-10
    import pytest

    with pytest.raises(ValueError):
        peak(t, t)


def test_acf_matches_direct_and_block_error():
    x = np.array([1.0, 2.0, 4.0, -1.0, 3.0, 0.0])
    y = x - x.mean()
    direct = np.array([np.dot(y[: len(y) - k], y[k:]) / len(y) for k in range(len(y))])
    direct /= direct[0]
    np.testing.assert_allclose(acf(x), direct, atol=1e-12)
    assert np.isclose(block_se(x, 1), x.std(ddof=1) / np.sqrt(len(x)))
    blocks = x.reshape(3, 2).mean(1)
    assert np.isclose(block_se(x, 2), blocks.std(ddof=1) / np.sqrt(3))


def test_ar1_tau():
    rng = np.random.default_rng(987)
    x = np.zeros(100000)
    for i in range(1, len(x)):
        x[i] = 0.8 * x[i - 1] + rng.normal()
    assert abs(tau_int(x) - 4.5) < 0.5


def test_bootstrap_constant_and_nontrivial():
    rng = np.random.default_rng(9)
    np.testing.assert_allclose(bootstrap_chi(np.ones(100), 4, 2, 10, 50, rng), 0)
    x = np.tile([0.0, 1.0], 50)
    np.testing.assert_allclose(bootstrap_chi(x, 4, 2, 10, 50, rng), 2)


def test_basic_evidence_generation(tmp_path, monkeypatch):
    import analysis

    if not (analysis.ROOT / "artifacts/window-l64/series.jsonl").exists():
        import pytest

        pytest.skip("requires protocol rows; covered during reproduce")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    auxiliary = tmp_path / "auxiliary"
    auxiliary.mkdir()
    monkeypatch.setattr(analysis, "OUT", evidence)
    monkeypatch.setattr(analysis, "AUX", auxiliary)
    analysis.basic()
    import json

    assert (auxiliary / "boltzmann-check.json").exists()
    assert {p.name for p in evidence.iterdir()} == {
        "boltzmann.png",
        "magnetization.png",
        "susceptibility.png",
        "peaks.txt",
    }
    result = json.loads((auxiliary / "boltzmann-check.json").read_text())
    assert np.isfinite(result["unweighted_fitted_slope"])


def test_constant_chain_has_no_identifiable_tau():
    assert np.isnan(tau_int(np.full(10000, 0.1)))


def test_anticorrelation_not_floored_at_half():
    rng = np.random.default_rng(813)
    x = np.zeros(200000)
    for i in range(1, len(x)):
        x[i] = -0.2 * x[i - 1] + rng.normal()
    # AR(1) tau=(1+phi)/(2(1-phi))=1/3.
    assert 0.28 < tau_int(x) < 0.40


def test_nonfinite_fit_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        peak(np.arange(7.0), np.array([0, 1, 3, np.nan, 3, 1, 0]))


def test_zero_errors_not_evidence_of_stability():
    from stats import stable

    assert not stable([0.0, 0.0, 0.0])
