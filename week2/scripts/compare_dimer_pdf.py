"""Optional diagnostic: compare Rust CSV with page 8's PDF vector curve.

Requires PyMuPDF. This extracts vector coordinates, not pixels. The axis
calibration is specific to the supplied 20-page Week 2 learning sheet.
No reference values enter the Rust solver or the displayed Rust curve.
"""
import argparse
import csv
import json
import math
import fitz

parser = argparse.ArgumentParser()
parser.add_argument("pdf")
parser.add_argument("csv")
args = parser.parse_args()
page = fitz.open(args.pdf)[7]
candidates = [d for d in page.get_drawings()
              if len(d["items"]) == 499 and d["rect"].x0 > page.rect.width/2]
if len(candidates) != 1:
    raise SystemExit("Expected one 500-point right-panel path on PDF page 8")
items = candidates[0]["items"]
points = [items[0][1]] + [item[2] for item in items]
with open(args.csv, newline="") as file:
    rows = list(csv.DictReader(file))
sampled = rows[10::10]
assert len(sampled) == len(points) == 500
# Read directly from the PDF axes: t=0 and 50; error*1000=0, +0.5, -0.5.
x_zero, x_fifty = 346.6777038574219, 502.58319091796875
y_zero, y_top, y_bottom = 444.0219421386719, 387.8959655761719, 500.14794921875
reference_t = [(p.x-x_zero)/(x_fifty-x_zero)*50 for p in points]
reference_e = [(y_zero-p.y)/(y_bottom-y_top)*1e-3 for p in points]
errors = [abs(float(row["verlet"])-e) for row,e in zip(sampled,reference_e)]
time_errors = [abs(float(row["time"])-t) for row,t in zip(sampled,reference_t)]
all_errors = [float(row["verlet"]) for row in rows]
result = {
    "pdf_page": 8,
    "pdf_right_panel_points": len(points),
    "integration_dt": 0.01,
    "display_stride_steps": 10,
    "max_pdf_time_reconstruction_error": max(time_errors),
    "max_relative_energy_difference_from_pdf": max(errors),
    "rms_relative_energy_difference_from_pdf": math.sqrt(sum(e*e for e in errors)/len(errors)),
    "full_series_min_relative_error": min(all_errors),
    "full_series_max_relative_error": max(all_errors),
    "full_series_max_absolute_relative_error": max(map(abs,all_errors)),
}
print(json.dumps(result,indent=2))
assert max(errors)<1e-8, "Rust curve differs from PDF beyond vector-coordinate precision"
assert max(time_errors)<1e-4
