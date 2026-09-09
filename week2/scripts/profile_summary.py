"""Summarize inclusive force samples from actual samply data and symbol sidecar."""
import argparse
import gzip
import json
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument("profile",type=Path)
parser.add_argument("--symbols",type=Path,required=True)
parser.add_argument("--out",type=Path,required=True)
args=parser.parse_args()
opener=gzip.open if args.profile.suffix==".gz" else open
with opener(args.profile,"rt") as f: profile=json.load(f)
symbols=json.loads(args.symbols.read_text())
tables={entry["debug_name"]:entry for entry in symbols["data"]}
strings=symbols["string_table"]
result=[]
for thread in profile["threads"]:
    if not thread.get("isMainThread"): continue
    def names_for_frame(index):
        func=thread["frameTable"]["func"][index]
        resource=thread["funcTable"]["resource"][func]
        if resource<0: return []
        lib_index=thread["resourceTable"]["lib"][resource]
        if lib_index is None or lib_index<0: return []
        lib=profile["libs"][lib_index]
        table=tables.get(lib["debugName"])
        if table is None: return []
        address=thread["frameTable"]["address"][index]
        symbol_index=dict(table["known_addresses"]).get(address)
        if symbol_index is None: return []
        symbol=table["symbol_table"][symbol_index]
        return [strings[symbol["symbol"]]]+[strings[f["function"]] for f in symbol.get("frames",[])]
    force_frames=set()
    for index in range(thread["frameTable"]["length"]):
        if any(name.startswith("md::periodic::forces") or name.startswith("md::periodic::CellList::forces") for name in names_for_frame(index)):
            force_frames.add(index)
    samples=thread["samples"]
    weights=samples.get("weight") or [1]*samples["length"]
    total=0
    force=0
    for stack,weight in zip(samples["stack"],weights):
        if stack is None: continue
        total+=weight
        while stack is not None:
            if thread["stackTable"]["frame"][stack] in force_frames:
                force+=weight
                break
            stack=thread["stackTable"]["prefix"][stack]
    start=thread["processStartupTime"]
    end=thread["processShutdownTime"]
    result.append({"thread":thread["name"],"weighted_samples":total,"inclusive_force_samples":force,
                   "force_share_percent":100*force/total,
                   "process_start_ms":start,"process_end_ms":end,"elapsed_seconds":(end-start)/1000})
summary={"source":str(args.profile),"symbols":str(args.symbols),
         "method":"Count a sample once if any ancestor frame belongs to a force evaluator; elapsed is recorded process lifetime.",
         "threads":result}
args.out.write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
