import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from check import validate_run


def fixture(tmp_path):
    meta = {
        "L": 2,
        "update": "metropolis",
        "t_grid": [2.0],
        "discard": 2,
        "measure": 2,
        "seed": 1,
        "sample_every": 1,
        "time_unit": "sweep",
    }
    (tmp_path / "run.json").write_text(json.dumps(meta))
    rows = [{"L": 2, "T": 2.0, "sweep": i, "M": 1.0, "E": -2.0} for i in [1, 2]]
    (tmp_path / "series.jsonl").write_text("\n".join(map(json.dumps, rows)))
    return rows


def test_valid_contract(tmp_path):
    fixture(tmp_path)
    assert validate_run(tmp_path).get("rows") == 2


def test_duplicate_or_missing_step_fails(tmp_path):
    rows = fixture(tmp_path)
    rows[1]["sweep"] = 1
    (tmp_path / "series.jsonl").write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError):
        validate_run(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [("T", float("nan")), ("T", float("inf")), ("sweep", True), ("L", 2.5)],
)
def test_reject_nonfinite_or_wrong_typed_row(tmp_path, field, value):
    rows = fixture(tmp_path)
    rows[0][field] = value
    (tmp_path / "series.jsonl").write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError):
        validate_run(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("update", "unknown"),
        ("seed", -1),
        ("measure", 2.0),
        ("discard", -1),
        ("t_grid", [float("nan")]),
    ],
)
def test_reject_invalid_metadata(tmp_path, field, value):
    fixture(tmp_path)
    meta = json.loads((tmp_path / "run.json").read_text())
    meta[field] = value
    if field == "update":
        meta["time_unit"] = "cluster_flip"
    (tmp_path / "run.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        validate_run(tmp_path)


def test_expected_protocol_seed_cannot_be_silently_changed(tmp_path):
    fixture(tmp_path)
    with pytest.raises(ValueError):
        validate_run(tmp_path, expected={"seed": 42})


def test_raw_moments_are_independently_accumulated(tmp_path):
    rows = fixture(tmp_path)
    rows[1]["M"] = 0.0
    (tmp_path / "series.jsonl").write_text("\n".join(map(json.dumps, rows)))
    result = validate_run(tmp_path)
    assert result.get("moments") == [{"T": 2.0, "mean_abs_M": 0.5, "mean_M2": 0.5}]


def test_numpy_boolean_checks_serialize_without_weakening_types():
    import numpy as np
    from check import normalize_checks

    result = normalize_checks({"ok": np.bool_(True), "no": False})
    assert type(result["ok"]) is bool
    assert json.loads(json.dumps(result)) == {"ok": True, "no": False}
    with pytest.raises(ValueError):
        normalize_checks({"bad": 1.2})
