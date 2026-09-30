"""Regression checks for the PDF's required evidence sources and printed fields."""
import json,sys,shutil
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import figures

@pytest.fixture
def evidence(tmp_path,monkeypatch):
    source=figures.ROOT/'artifacts'
    for name in ['forward','adjoint']+[f'checkpoint-{b}' for b in [1,3,5,10]]+['born']:
        (tmp_path/name).mkdir()
        for p in (source/name).iterdir():
            if p.suffix=='.json':shutil.copy2(p,tmp_path/name/p.name)
            elif p.suffix=='.npy':(tmp_path/name/p.name).symlink_to(p)
    monkeypatch.setattr(figures,'A',tmp_path)
    monkeypatch.setattr(figures,'save',lambda fig,path:figures.plt.close(fig))
    return tmp_path

def test_wave_check_rejects_inconsistent_recorded_times(evidence):
    p=evidence/'forward/run.json';r=json.loads(p.read_text());r['recording']['times'][35]+=1;p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='recording'):
        figures.small()

def test_full_history_baseline_is_loaded(evidence,monkeypatch):
    seen=[];original=figures.load
    def tracked(path):seen.append(Path(path));return original(path)
    monkeypatch.setattr(figures,'load',tracked)
    figures.checkpoints()
    assert evidence/'adjoint/result.json' in seen

def test_required_numerical_fields_are_printed(evidence,capsys):
    figures.small();figures.checkpoints();text=capsys.readouterr().out
    for key in ['depth_difference_km','relative_l2_error','peak_saved_states','missing_reverse_steps','duplicated_reverse_steps','reverse_steps_out_of_order','invalid_restores','budget_overruns']:
        assert key in text
