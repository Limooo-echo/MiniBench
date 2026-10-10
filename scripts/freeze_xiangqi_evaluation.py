"""Freeze paired smoke/formal rosters and refresh reviewable evaluation manifests."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from minibench.datasets.xiangqi.schema import load_records, sample_records

TASKS = {"d3":("mate_in_one","xiangqi-direct"), "c2":("rule_variants","xiangqi-rule"),
         "h2":("history","xiangqi-history"), "m2":("multimodal","xiangqi-visual")}
RELEASE_ID = "xiangqi-2026-09-19-r1"
PROTOCOL_VERSION = "xiangqi-reasoning-v2"

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, content, replace=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != content and not replace:
        raise FileExistsError(f"frozen content differs: {path}; use a new release or --replace after review")
    path.write_text(content, encoding="utf-8", newline="\n")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--replace",action="store_true")
    args=parser.parse_args()
    datasets={key:load_records(ROOT/f"data/xiangqi/{folder}/tasks.jsonl") for key,(folder,_) in TASKS.items()}
    if len({len(rows) for rows in datasets.values()}) != 1 or len(datasets["d3"]) not in (100, 250):
        raise ValueError("complete 100-record submitted families or 250-record parent pools required")
    if any(row.get("release_id") != RELEASE_ID for rows in datasets.values() for row in rows):
        raise ValueError("all families must pass the new-release generators first")
    provenance=json.loads((ROOT/"data/xiangqi/provenance.json").read_text(encoding="utf-8"))
    engine=provenance["engine"]
    manifest={"release_id":RELEASE_ID,"protocol_version":PROTOCOL_VERSION,"samples":{}}
    for size,seed,count,c2count in (("smoke",20260909,6,3),("formal",42,30,10)):
        selected={"d3":sample_records(datasets["d3"],count=count,seed=seed,strategy="stratified"),
                  "h2":sample_records(datasets["h2"],count=count,seed=seed,strategy="stratified"),
                  "c2":sample_records(datasets["c2"],count=c2count,seed=seed,strategy="paired-stratified")}
        m2_by_source={r["source_task_id"]:r for r in datasets["m2"]}
        selected["m2"]=[m2_by_source[r["id"]] for r in selected["d3"]]
        sample={"seed":seed,"tasks":{}}
        for key,rows in selected.items():
            folder, _=TASKS[key]
            path=ROOT/f"data/xiangqi/evaluation_samples/{size}/{key}.jsonl"
            write(path,"".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),args.replace)
            sample["tasks"][key]={"path":path.relative_to(ROOT).as_posix(),"sha256":sha(path),
                "dataset_sha256":sha(ROOT/f"data/xiangqi/{folder}/tasks.jsonl"),
                "record_count":len(rows),"task_ids":[r["id"] for r in rows]}
            config_path=ROOT/f"config/experiments/xiangqi_{folder}.yaml"
            config=yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if key != "c2":
                config["evaluation"]["pikafish_binary_sha256"]=engine["binary_sha256"]
                config["evaluation"]["pikafish_nnue_sha256"]=engine["nnue_sha256"]
            config["task"].pop("sampling", None)
            config["task"]["selection"]={k:sample["tasks"][key][k] for k in ("path","sha256")}
            config["task"]["prompt_version"]="xiangqi-m2-recognition-v1" if key=="m2" else PROTOCOL_VERSION
            config["task"].pop("task_ids",None)
            config["task"].pop("limit",None)
            if size == "formal":
                # Canonical submitted corpora are already stratified 100-row
                # selections; full runs do not draw another sample.
                config["task"].pop("selection", None)
                config["task"]["limit"] = 100
            if key == "m2":
                config["evaluation"]["input_modes"] = ["text", "chinese-piece-image"]
            config["run"]["run_name"]=f"xiangqi_{key}_direct_{'submitted100' if size == 'formal' else size}_r1"
            config["run"]["on_existing"]="error"
            destination=config_path if size=="formal" else ROOT/f"config/experiments/xiangqi_smoke/{key}.yaml"
            write(destination,yaml.safe_dump(config,sort_keys=False,allow_unicode=True),True)
        manifest["samples"][size]=sample
    write(ROOT/"data/xiangqi/evaluation_samples/manifest.json",json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",args.replace)
    scoring_path=ROOT/"config/scoring/manifest.example.yaml"
    scoring=yaml.safe_load(scoring_path.read_text(encoding="utf-8"))
    for exp in scoring["experiments"]:
        match=next((key for key,(_,exp_id) in TASKS.items() if exp_id==exp["id"]),None)
        if match:
            dataset_path=(scoring_path.parent/exp["dataset"]["path"]).resolve()
            exp["dataset"]["sha256"]=sha(dataset_path)
            folder,_=TASKS[match]
            config=yaml.safe_load((ROOT/f"config/experiments/xiangqi_{folder}.yaml").read_text(encoding="utf-8"))
            roster=manifest["samples"]["formal"]["tasks"][match]
            exp["selection"]={"path":"../../"+roster["path"],"sha256":roster["sha256"]}
            exp["protocol"]={"version":config["task"]["prompt_version"],"release_id":RELEASE_ID,"evaluation":config["evaluation"]}
    write(scoring_path,yaml.safe_dump(scoring,sort_keys=False,allow_unicode=True),True)
    print(json.dumps({size:{k:v["record_count"] for k,v in data["tasks"].items()} for size,data in manifest["samples"].items()}))
    return 0
if __name__ == "__main__": raise SystemExit(main())
