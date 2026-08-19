from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any, Sequence

from minibench.assets.fonts import font_path
from minibench.datasets.xiangqi.multimodal import render_board_png
from minibench.datasets.xiangqi.schema import (
    FAMILY_PATHS,
    LEGACY_FAMILY_NAMES,
    XIANGQI_FAMILIES,
    fen_to_board,
    load_records,
)


PIECE_TEXT = {
    1: "帅", -1: "将", 2: "仕", 3: "仕", -2: "士", -3: "士",
    4: "相", 5: "相", -4: "象", -5: "象", 6: "马", 7: "马",
    -6: "馬", -7: "馬", 8: "车", 9: "车", -8: "車", -9: "車",
    10: "炮", 11: "炮", -10: "砲", -11: "砲", 12: "兵", 13: "兵",
    14: "兵", 15: "兵", 16: "兵", -12: "卒", -13: "卒", -14: "卒",
    -15: "卒", -16: "卒",
}


def _require_family(family: str) -> str:
    normalized = family.strip()
    legacy_key = normalized.upper()
    if legacy_key in LEGACY_FAMILY_NAMES:
        replacement = LEGACY_FAMILY_NAMES[legacy_key]
        raise ValueError(
            f"legacy task name {family!r} is not supported; use {replacement!r}. "
            "To migrate old files, run: minibench migrate-xiangqi-v2 ..."
        )
    if normalized not in XIANGQI_FAMILIES:
        raise ValueError(
            f"unknown Xiangqi task {family!r}; choose one of "
            f"{', '.join(XIANGQI_FAMILIES)}"
        )
    return normalized


def find_record(family: str, task_id: str) -> dict[str, Any]:
    family = _require_family(family)
    for record in load_records(FAMILY_PATHS[family], expected_family=family):
        if record["id"] == task_id:
            return record
    raise ValueError(f"unknown task id {task_id!r} in {family}")


def terminal_board(board: Sequence[Sequence[int]]) -> str:
    lines = ["      a  b  c  d  e  f  g  h  i"]
    for row_index, row in enumerate(board):
        rank = 9 - row_index
        cells = " ".join(PIECE_TEXT.get(value, "·") if value else "·" for value in row)
        lines.append(f"  {rank}   {cells}   {rank}")
        if row_index == 4:
            lines.append("      ─────── 楚河  汉界 ───────")
    lines.append("      a  b  c  d  e  f  g  h  i")
    return "\n".join(lines)


def format_terminal(record: dict[str, Any]) -> str:
    board, _ = fen_to_board(record["fen"])
    oracle = record["oracle"]
    lines = [
        terminal_board(board),
        "",
        f"ID: {record['id']}",
        f"Family: {record['family']}",
        f"FEN: {record['fen']}",
        f"目标: {record['goal']}  难度: {record['difficulty']}  "
        f"最大半回合: {record['max_plies']}",
    ]
    if record.get("ruleset"):
        lines.append(f"规则组: {record['ruleset']}")
        for rule in record["rules"]:
            lines.append(
                f"  - {rule['kind']} / {rule['piece']} / {rule['effect']}"
            )
    lines.append(
        "Oracle: "
        f"best={oracle.get('best_move_uci')}, "
        f"mate_in_plies={oracle.get('mate_in_plies')}, "
        f"evaluation_cp={oracle.get('evaluation_cp')}"
    )
    lines.append(f"Tags: {', '.join(record['tags']) or '(none)'}")
    return "\n".join(lines)


def inspect_record(
    family: str,
    task_id: str,
    *,
    output_format: str,
    output: str | Path | None = None,
) -> str:
    record = find_record(family, task_id)
    board, side_to_move = fen_to_board(record["fen"])
    if output_format == "terminal":
        return format_terminal(record)
    if output_format == "json":
        return json.dumps(
            {"record": record, "decoded_board": board, "side_to_move": side_to_move},
            indent=2,
            ensure_ascii=False,
        )
    if output_format == "png":
        if output is None:
            raise ValueError("--output is required when --format png")
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(render_board_png(board, "chinese-piece-image"))
        return str(destination)
    raise ValueError("format must be terminal, json, or png")


def build_gallery(output: str | Path) -> Path:
    records: list[dict[str, Any]] = []
    for family in XIANGQI_FAMILIES:
        records.extend(load_records(FAMILY_PATHS[family], expected_family=family))
    regular = base64.b64encode(font_path().read_bytes()).decode("ascii")
    bold = base64.b64encode(font_path(bold=True).read_bytes()).decode("ascii")
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    document = _gallery_document(payload=payload, regular=regular, bold=bold)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8", newline="\n")
    return destination


def _gallery_document(*, payload: str, regular: str, bold: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MiniBench Xiangqi Gallery</title>
<style>
@font-face{{font-family:MiniBench;src:url(data:font/otf;base64,{regular})}}
@font-face{{font-family:MiniBench;src:url(data:font/otf;base64,{bold});font-weight:700}}
:root{{--paper:#f7f0df;--ink:#24201a;--red:#b52924;--line:#74552f}}
*{{box-sizing:border-box}} body{{margin:0;background:#f2eee6;color:var(--ink);font-family:MiniBench,sans-serif}}
header{{padding:20px 28px;background:#1f2833;color:white}} h1{{margin:0;font-size:25px}}
.controls{{display:grid;grid-template-columns:2fr repeat(3,1fr);gap:10px;padding:16px 28px;background:white;position:sticky;top:0;z-index:2;box-shadow:0 2px 8px #0002}}
input,select{{font:inherit;padding:9px;border:1px solid #bbb;border-radius:6px}}
main{{display:grid;grid-template-columns:minmax(350px,560px) minmax(300px,1fr);gap:24px;padding:24px 28px}}
.board{{display:grid;grid-template-columns:repeat(9,1fr);background:var(--paper);border:3px solid var(--line);aspect-ratio:9/10;max-height:72vh}}
.sq{{display:flex;align-items:center;justify-content:center;border:1px solid #b3905d;font-size:clamp(18px,3vw,35px);font-weight:700}}
.red{{color:var(--red)}} .black{{color:#191919}} .meta{{background:white;padding:22px;border-radius:10px;overflow-wrap:anywhere}}
.idlist{{max-height:180px;overflow:auto;border:1px solid #ddd;border-radius:6px;margin-top:14px}}
.idlist button{{display:block;width:100%;text-align:left;border:0;border-bottom:1px solid #eee;padding:7px 9px;background:white;font:inherit;cursor:pointer}}
.idlist button:hover{{background:#f4eee2}} dt{{font-weight:700;margin-top:10px}} dd{{margin-left:0}} code{{white-space:pre-wrap}}
@media(max-width:850px){{.controls{{grid-template-columns:1fr 1fr}}main{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>MiniBench Xiangqi Gallery <small id="count"></small></h1></header>
<div class="controls"><input id="query" placeholder="Search task ID">
<select id="family"><option value="">All families</option></select>
<select id="difficulty"><option value="">All difficulties</option></select>
<select id="ruleset"><option value="">All rulesets</option></select></div>
<main><section><div id="board" class="board"></div></section><section class="meta">
<h2 id="title"></h2><dl id="details"></dl><div id="ids" class="idlist"></div></section></main>
<script id="records" type="application/json">{payload}</script>
<script>
const all=JSON.parse(document.getElementById('records').textContent);let current=null;
const $=id=>document.getElementById(id), pieces={{K:'帅',k:'将',A:'仕',a:'士',B:'相',b:'象',N:'马',n:'馬',R:'车',r:'車',C:'炮',c:'砲',P:'兵',p:'卒'}};
function options(id,values){{for(const v of [...new Set(values.filter(Boolean))].sort()){{const o=document.createElement('option');o.value=o.textContent=v;$(id).append(o)}}}}
options('family',all.map(x=>x.family));options('difficulty',all.map(x=>x.difficulty));options('ruleset',all.map(x=>x.ruleset));
function decode(fen){{return fen.split(' ')[0].split('/').map(row=>{{const out=[];for(const c of row){{if(/\\d/.test(c))out.push(...Array(+c).fill(''));else out.push(c)}}return out}})}}
function show(x){{current=x;$('title').textContent=x.id;const b=decode(x.fen),root=$('board');root.textContent='';for(const row of b)for(const p of row){{const d=document.createElement('div');d.className='sq '+(p?(p===p.toUpperCase()?'red':'black'):'');d.textContent=pieces[p]||'';root.append(d)}}
const o=x.oracle,rules=(x.rules||[]).map(r=>`${{r.kind}} / ${{r.piece}} / ${{r.effect}}`).join('<br>')||'(standard)';
$('details').innerHTML=`<dt>Family</dt><dd>${{x.family}}</dd><dt>FEN</dt><dd><code>${{x.fen}}</code></dd><dt>Goal / difficulty</dt><dd>${{x.goal}} / ${{x.difficulty}}</dd><dt>Max plies / pieces</dt><dd>${{x.max_plies}} / ${{x.piece_count}}</dd><dt>Ruleset</dt><dd>${{x.ruleset||'(none)'}}</dd><dt>Rules</dt><dd>${{rules}}</dd><dt>Oracle</dt><dd>best=${{o.best_move_uci}}, mate=${{o.mate_in_plies}}, cp=${{o.evaluation_cp}}</dd><dt>Tags</dt><dd>${{x.tags.join(', ')||'(none)'}}</dd>`}}
function filter(){{const q=$('query').value.trim().toLowerCase(), f=$('family').value,d=$('difficulty').value,r=$('ruleset').value;const rows=all.filter(x=>(!q||x.id.toLowerCase().includes(q))&&(!f||x.family===f)&&(!d||x.difficulty===d)&&(!r||x.ruleset===r));$('count').textContent=`(${{rows.length}} / ${{all.length}})`;const box=$('ids');box.textContent='';for(const x of rows.slice(0,200)){{const b=document.createElement('button');b.textContent=x.id;b.onclick=()=>show(x);box.append(b)}}if(rows.length&&(!current||!rows.includes(current)))show(rows[0])}}
for(const id of ['query','family','difficulty','ruleset'])$(id).addEventListener('input',filter);filter();
</script></body></html>"""
