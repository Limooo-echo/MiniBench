"""Verify the immutable legacy mapping; never zip old IDs onto changed boards.

For a new corpus revision, write an explicit replacement/deletion report. This
command intentionally no longer mutates the v1-to-v2 map or its answers.
"""
from hashlib import sha256
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def main():
    mapping = json.loads((ROOT/"data/xiangqi/migration_v1_to_v2.json").read_text(encoding="utf-8"))
    archive = ROOT/"data/xiangqi/legacy"/mapping["dataset_revision"]
    by_id = {}
    for path in archive.glob("*/tasks.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            row=json.loads(line); by_id[row["id"]]=row
    projections=[]
    for old_id,new_id in sorted(mapping["task_ids"].items()):
        record=by_id[new_id]
        projected={"old_id":old_id,"new_id":new_id,**{k:record[k] for k in
            ("fen","agent_color","goal","max_plies","difficulty","piece_count","oracle")}}
        for key in ("scenario_id","ruleset"):
            if key in record: projected[key]=record[key]
        projections.append(projected)
    digest=sha256(json.dumps(projections,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
    if digest != mapping["semantic_sha256"]: raise ValueError("legacy archive/map integrity mismatch")
    print(json.dumps({"verified":True,"revision":mapping["dataset_revision"],"sha256":digest,"mutated":False}))
    return 0
if __name__ == "__main__": raise SystemExit(main())
