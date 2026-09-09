from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from minibench.datasets.xiangqi.provenance import audit_xiangqi_release

parser = argparse.ArgumentParser(description="Audit Xiangqi release provenance and exact FEN reuse.")
parser.add_argument("--manifest", type=Path, default=Path("data/xiangqi/provenance.json"))
parser.add_argument("--output", type=Path)
parser.add_argument("--strict", action="store_true", help="Fail when provenance is unresolved or same-family FENs repeat.")
args = parser.parse_args()
report = audit_xiangqi_release(args.manifest)
serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
if args.output:
    args.output.write_text(serialized, encoding="utf-8")
print(serialized, end="")
if args.strict and not report["release_ready"]:
    raise SystemExit("release audit failed; fill provenance manifest or resolve duplicate positions")
