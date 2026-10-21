"""A deliberately failed measured target must block conditional submission."""
import json,subprocess,sys,shutil
from pathlib import Path

def test_failed_measured_target_blocks_submission(tmp_path):
    root=Path(__file__).resolve().parents[1]
    shutil.copy(root/'verify.py',tmp_path/'verify.py')
    shutil.copytree(root/'evidence',tmp_path/'evidence')
    for name in ['artifacts','inputs']:(tmp_path/name).symlink_to(root/name,target_is_directory=True)
    record=tmp_path/'evidence/toy-results.json'
    data=json.loads(record.read_text());data['velocity_relative_rms'][0]=.2;record.write_text(json.dumps(data))
    result=subprocess.run([sys.executable,'verify.py'],cwd=tmp_path,capture_output=True,text=True)
    audit=json.loads((tmp_path/'evidence/audit-summary.json').read_text())
    assert 'P3.velocity_full_rectangle' in audit['failed_checks']
    assert result.returncode==1, 'A failed PDF target must not return success to a submission workflow'
