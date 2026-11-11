import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class ReportingTests(unittest.TestCase):
    def test_evidence_audit_uses_documented_sample_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'artifacts').mkdir()
            for name, method in [('my-boundary64', 'boundary'), ('my-trg1024', 'trg')]:
                out = root / 'runs' / name
                out.mkdir(parents=True)
                data = {'method': method, 'L': 2, 'K': 0, 'g': 0}
                if method == 'boundary':
                    data.update(sites=[{'left': 1, 'right': 1, 'data': [1, 0]}], log_scales=[])
                else:
                    data.update(steps=[{'log_scale': 0, 'new_dim': 1}], final_tensor=[1])
                (out / 'evidence.json').write_text(json.dumps(data))
            p = subprocess.run([sys.executable, str(ROOT / 'audit_evidence.py')], cwd=root, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)

    def test_summary_deduplicates_equivalent_float_temperatures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'artifacts').mkdir()
            peak = {m: {str(l): {'16': {'T_peak': 2}, '32': {'T_peak': 2}, 'wide': {'T_peak': 2}} for l in sizes} for m, sizes in [('boundary', [16,32,64]), ('trg', [64,256,1024])]}
            (root/'artifacts/peak-results.json').write_text(json.dumps(peak))
            with (root/'artifacts/g-timings.csv').open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=['method','L','K','chi','seconds']);w.writeheader()
                for k in ['0.4775','0.47750000000000004']:
                    w.writerow({'method':'boundary','L':16,'K':k,'chi':16,'seconds':1})
            subprocess.run([sys.executable, str(ROOT/'summarize_results.py')], cwd=root, check=True)
            with (root/'artifacts/peak-summary.csv').open() as f:
                row = next(csv.DictReader(f))
            self.assertEqual(float(row['seconds_all_recorded']), 1.0)

    def test_small_checks_are_exported_as_a_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'artifacts').mkdir()
            checks = {'boundary_4_0.44': {'g': 0.85, 'exact_g': 0.85, 'abs_error': 0.0},
                      'boundary_4_cv': {'finite_difference': 0.439, 'energy_variance': 0.440, 'abs_error': 0.001}}
            (root / 'artifacts/small-checks.json').write_text(json.dumps(checks))
            p = subprocess.run([sys.executable, str(ROOT / 'summarize_checks.py')], cwd=root, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            with (root / 'artifacts/small-checks.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(float(rows[1]['reference']), 0.440)

if __name__ == '__main__':
    unittest.main()
