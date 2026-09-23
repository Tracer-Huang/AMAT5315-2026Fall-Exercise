"""Downsample the actual 128-grid run, never run a substitute 64-grid flow."""
import argparse,json
from pathlib import Path
from numerics import frames,reduced

def main():
    p=argparse.ArgumentParser(); p.add_argument("--source",type=Path,default=Path("artifacts/random/fields.jsonl")); p.add_argument("--out",type=Path,default=Path("fields.jsonl")); a=p.parse_args()
    with a.out.open("w") as f:
        for frame in frames(a.source): f.write(json.dumps(reduced(frame),separators=(",",":"))+"\n")
    if a.out.stat().st_size>=5_000_000: raise RuntimeError("viewer recording exceeds 5 MB")
    print(f"{a.out}: {a.out.stat().st_size} bytes")
if __name__=="__main__": main()
