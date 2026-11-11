"""Convert the independent small-system checks into a readable CSV table."""
import csv
import json
from pathlib import Path

checks = json.loads(Path('artifacts/small-checks.json').read_text())
with Path('artifacts/small-checks.csv').open('w', newline='') as output:
    writer = csv.DictWriter(output, fieldnames=['check', 'measured', 'reference', 'absolute_error'])
    writer.writeheader()
    for name, values in checks.items():
        measured = values.get('g', values.get('finite_difference'))
        reference = values.get('exact_g', values.get('energy_variance'))
        writer.writerow({'check': name, 'measured': measured, 'reference': reference,
                         'absolute_error': values['abs_error']})
