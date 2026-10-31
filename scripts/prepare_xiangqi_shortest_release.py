"""Prepare evidence-bearing candidates; publish only balanced certified pairs.

Intermediate files belong under output/, never in the submitted task corpus.
Derived H2 source records are NOT publishable until fresh engine reanalysis.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
from minibench.core.checkpoint import atomic_write_json, atomic_write_jsonl, sha256_file
from minibench.datasets.xiangqi import reference
from minibench.datasets.xiangqi.schema import load_records, fen_to_board, board_to_fen
from build_xiangqi_c2 import _candidate, _record

def applicable(record):
    board, color = fen_to_board(record["fen"])
    side = 1 if color == "red" else -1
    focus = record["design_stratum"]
    return (record["c2_analysis"]["rule_sensitive_along_reference"]
            or reference.legal_uci_moves(board, side, "standard")
            != reference.legal_uci_moves(board, side, focus))


def derived_sources(sources):
    seen = {r["fen"] for r in sources}
    for source in sources:
        board, color = fen_to_board(source["fen"])
        turn = 1 if color == "red" else -1
        pv = source["h2_analysis"]["principal_variation_uci"]
        for ply, uci in enumerate(pv, 1):
            moves = {m.to_uci():m for m in reference.legal_moves(board, turn, "standard")}
            if uci not in moves:
                raise ValueError("Parent PV failed replay")
            board = reference.apply_move(board, moves[uci])
            turn = -turn
            if ply % 2 or not 3 <= len(pv)-ply <= 9:
                continue
            fen = board_to_fen(board, active_color="red" if turn == 1 else "black")
            if fen in seen:
                continue
            seen.add(fen)
            item = deepcopy(source)
            tail = pv[ply:]
            n = (len(tail)+1)//2
            item.update(id="xiangqi-history-derived-"+hashlib.sha256(fen.encode()).hexdigest()[:16],
                        fen=fen, piece_count=sum(bool(p) for row in board for p in row),
                        difficulty="short" if n <= 3 else "medium",
                        max_plies=len(tail), derived_source_pending_reanalysis=True)
            item["oracle"].update(best_move_uci=tail[0], mate_in_plies=len(tail))
            item["h2_analysis"].update(mate_in_moves=n, reference_mate_plies=len(tail),
                                       principal_variation_uci=tail)
            item["source"]["derivation"] = {"parent_task_id":source["id"],
                "parent_fen":source["fen"],"moves_uci":pv[:ply],
                "offset_halfmoves":ply,"method":"H2 reference variation intermediate"}
            yield item


def tree_sources(sources, directory):
    """Trace alternate intermediate positions, not only the engine's one PV."""
    if directory is None:
        return
    for source in sources:
        path=directory/f"{source['id']}.json"
        if not path.exists():
            continue
        tree=json.loads(path.read_text())
        if not tree.get("complete") or not tree.get("forced_mate"):
            continue
        pending=[(tree["root"],[])]
        seen=set()
        while pending:
            key,prefix=pending.pop()
            if key in seen:
                continue
            seen.add(key)
            node=tree["nodes"][key]
            pending.extend((target,prefix+[move]) for move,target in node["edges"].items())
            if not prefix or node["turn"]!=tree["attacker"] or not 3<=node["dtm_plies"]<=9:
                continue
            tail=[]
            cursor=key
            while tree["nodes"][cursor]["edges"]:
                move,target=next(iter(tree["nodes"][cursor]["edges"].items()))
                tail.append(move)
                cursor=target
            if len(tail)!=node["dtm_plies"]:
                raise ValueError("Shortest tree reference path disagrees with node distance")
            item=deepcopy(source)
            fen=node["fen"]
            board,color=fen_to_board(fen)
            n=(len(tail)+1)//2
            item.update(id="xiangqi-history-derived-"+hashlib.sha256(fen.encode()).hexdigest()[:16],
                        fen=fen,agent_color=color,piece_count=sum(bool(p) for row in board for p in row),
                        difficulty="short" if n<=3 else "medium",max_plies=len(tail),
                        derived_source_pending_reanalysis=True)
            item["oracle"].update(best_move_uci=tail[0],mate_in_plies=len(tail))
            item["h2_analysis"].update(mate_in_moves=n,reference_mate_plies=len(tail),
                                       principal_variation_uci=tail)
            item["source"]["derivation"]={
                "parent_task_id":source["id"],"parent_fen":source["fen"],
                "moves_uci":prefix,"method":"certified standard solution DAG intermediate"}
            # Keep original PGN ply_index: this variation is explicitly NOT
            # mislabelled as a position appearing verbatim in that source PGN.
            yield item


def scan(parent, out, *, h2_tree_dir=None):
    h2 = load_records(parent/"history/tasks.jsonl")
    c2 = load_records(parent/"rule_variants/tasks.jsonl")
    sources = {r["id"]:r for r in h2}
    groups = defaultdict(list)
    for row in c2:
        groups[row["scenario_id"]].append(row)
    candidates, seen = [], set()
    for pair in groups.values():
        focus = next((r for r in pair if r["ruleset"] != "standard"), None)
        if len(pair) == 2 and focus and applicable(focus) and all(
                r["difficulty"] in {"short","medium"} for r in pair):
            candidates.extend(pair)
            seen.add((focus["fen"], focus["ruleset"]))
    expanded = [*h2, *derived_sources(h2), *tree_sources(h2,h2_tree_dir)]
    derived_used = set()
    for index, source in enumerate(expanded, 501):
        # The existing horse/soldier pools are sufficient. Expand the limiting
        # chariot group without adding mirrored/duplicate positions.
        focus = "chariot-no-center"
        if (source["fen"], focus) in seen or source["difficulty"] not in {"short","medium"}:
            continue
        item = _candidate(source, focus)
        if item is None or not item["sensitive"]:
            continue
        pair = [_record(item,scenario_number=index,ruleset=rule)
                for rule in ("standard",focus)]
        if not applicable(pair[1]):
            continue
        candidates.extend(pair)
        seen.add((source["fen"],focus))
        sources[source["id"]] = source
        if source.get("derived_source_pending_reanalysis"):
            derived_used.add(source["id"])
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(out/"c2-candidates.jsonl",candidates)
    atomic_write_jsonl(out/"candidate-origins.jsonl",list(sources.values()))
    counts=Counter((r["design_stratum"],r["difficulty"]) for r in candidates if r["ruleset"]!="standard")
    report={"counts":[{"rule":k[0],"band":k[1],"pairs":n} for k,n in sorted(counts.items())],
            "candidate_rows":len(candidates),"derived_sources_requiring_fresh_analysis":sorted(derived_used),
            "parent_sha256":sha256_file(parent/"history/tasks.jsonl"),
            "publication_status":"CANDIDATES_ONLY_NOT_A_RELEASE"}
    atomic_write_json(out/"scan.json",report)
    return report


def select_certified(records, tree_dir, *, count, seed, paired=False, require_complete=True):
    eligible=[]
    for r in records:
        path=tree_dir/f"{r['id']}.json"
        if not path.exists():
            continue
        tree=json.loads(path.read_text())
        from minibench.datasets.xiangqi.solution_tree import record_key
        if tree.get("record_key")!=record_key(r) or not tree.get("shortest_certified"):
            continue
        if require_complete and not tree.get("complete"):
            continue
        n=tree["shortest_attacker_moves"]
        band="short" if 2<=n<=3 else "medium" if 4<=n<=5 else None
        if band is not None:
            row=deepcopy(r)
            row["difficulty"]=band
            row["shortest_mate_moves"]=n
            eligible.append(row)
    rng=random.Random(seed)
    if not paired:
        output=[]
        for band in ("short","medium"):
            pool=sorted((r for r in eligible if r["difficulty"]==band),key=lambda r:r["id"])
            if len(pool)<count//2:
                raise ValueError(f"H2 {band}: only {len(pool)} certified candidates; need {count//2}")
            # Preserve source-game independence even with derived candidates.
            rng.shuffle(pool)
            used={r["source"]["file"] for r in output}
            used_fens={r["fen"] for r in output}
            picked=[]
            for row in pool:
                if row["source"]["file"] not in used and row["fen"] not in used_fens:
                    picked.append(row)
                    used.add(row["source"]["file"])
                    used_fens.add(row["fen"])
                if len(picked)==count//2:
                    break
            if len(picked)!=count//2:
                raise ValueError(f"H2 {band}: insufficient independent source games")
            output.extend(picked)
        return sorted(output,key=lambda r:r["id"])
    groups=defaultdict(list)
    for r in eligible:
        groups[r["scenario_id"]].append(r)
    def choices(rule,required,used):
        pool=[pair for pair in groups.values() if len(pair)==2
              and {r["ruleset"] for r in pair}=={"standard",rule}
              and pair[0]["fen"] not in used]
        pool.sort(key=lambda pair:pair[0]["scenario_id"])
        rng.shuffle(pool)
        states={(0,0):[]}
        for pair in pool:
            shorts=sum(r["difficulty"]=="short" for r in pair)
            for (size,s),prior in list(states.items())[::-1]:
                if size<required and s+shorts<=50:
                    states.setdefault((size+1,s+shorts),prior+[pair])
        return {s:selection for (size,s),selection in states.items() if size==required}
    # Rule changes may legitimately alter the shortest band within a pair.
    # Balance the 100 CONDITION ROWS, not a misleading shared reference label.
    for car_count, cars in sorted(choices("chariot-no-center",17,set()).items(),
                                  key=lambda item:abs(item[0]-17)):
        used={pair[0]["fen"] for pair in cars}
        for horse_count,horses in choices("horse-no-leg-block",17,used).items():
            used_more=used|{pair[0]["fen"] for pair in horses}
            soldiers=choices("soldier-free-retreat",16,used_more).get(50-car_count-horse_count)
            if soldiers is not None:
                output=[r for pair in [*cars,*horses,*soldiers] for r in pair]
                assert Counter(r["difficulty"] for r in output)=={"short":50,"medium":50}
                return sorted(output,key=lambda r:(r["ruleset"]!="standard",r["scenario_id"]))
    raise ValueError("Cannot form 17/17/16 certified pairs with exactly 50 short/50 medium rows")


def expand_defences(parent, work, *, engine_path, limit=24):
    """Add alternative legal defender replies from the existing H2 PVs.

    This expands candidate provenance, NOT the certified release. Each new
    standard position receives fresh depth16/20 evidence before becoming a pair.
    Both conditions still need the separate full-width proof stage.
    """
    from build_xiangqi_h2 import Verifier, make_record
    sources=load_records(parent/"history/tasks.jsonl")
    rows=[json.loads(l) for l in (work/"c2-candidates.jsonl").read_text().splitlines()]
    origins=[json.loads(l) for l in (work/"candidate-origins.jsonl").read_text().splitlines()]
    seen={r["fen"] for r in origins}
    candidates=[]
    for source in sources:
        board,color=fen_to_board(source["fen"])
        turn=1 if color=="red" else -1
        pv=source["h2_analysis"]["principal_variation_uci"]
        for ply,uci in enumerate(pv):
            legal={m.to_uci():m for m in reference.legal_moves(board,turn)}
            if ply%2 and 3<=len(pv)-ply-1<=5:
                for reply,move in legal.items():
                    if reply==uci:
                        continue
                    state=reference.apply_move(board,move)
                    fen=board_to_fen(state,active_color=source["agent_color"])
                    if fen in seen:
                        continue
                    seen.add(fen)
                    if reference.legal_uci_moves(state,-turn,"standard")==reference.legal_uci_moves(
                            state,-turn,"chariot-no-center"):
                        continue
                    candidates.append((sum(bool(p) for r in state for p in r),
                                       source,fen,pv[:ply]+[reply]))
            board=reference.apply_move(board,legal[uci])
            turn=-turn
    candidates.sort(key=lambda item:(item[0],len(item[3]),item[2]))
    verifier=Verifier(engine_path,(16,20),30)
    added=[]
    try:
        for _,source,fen,moves in candidates:
            metadata=deepcopy(source["source"])
            metadata["derivation"]={"parent_task_id":source["id"],"parent_fen":source["fen"],
                "moves_uci":moves,"method":"alternative legal reply to H2 reference variation"}
            item={"fen":fen,"source":metadata,"source_file":metadata["file"],
                  "ply_index":metadata["ply_index"]}
            checked=verifier.check(item,needed_bands={"short"})
            if not checked["valid"]:
                continue
            identifier="xiangqi-history-derived-"+hashlib.sha256(fen.encode()).hexdigest()[:16]
            origin=make_record(item,checked,identifier)
            candidate=_candidate(origin,"chariot-no-center")
            if candidate is None or not candidate["sensitive"]:
                continue
            pair=[_record(candidate,scenario_number=2001+len(added)//2,ruleset=rule)
                  for rule in ("standard","chariot-no-center")]
            origins.append(origin)
            rows.extend(pair)
            added.extend(pair)
            # Durable checkpoint: a later interrupted scan does not lose work.
            atomic_write_jsonl(work/"c2-candidates.jsonl",rows)
            atomic_write_jsonl(work/"candidate-origins.jsonl",origins)
            atomic_write_jsonl(work/"c2-alternative-reply-originals.jsonl",added)
            print(f"Added freshly checked pair {len(added)//2}: {identifier}",flush=True)
            if len(added)//2>=limit:
                break
    finally:
        verifier.close()
    return {"new_pairs":len(added)//2,"candidate_positions_considered":len(candidates)}


def expand_h2(parent,work,*,engine_path,limit=30):
    """Shorten existing long H2 variations; keep one candidate per source game."""
    from build_xiangqi_h2 import Verifier,make_record
    sources=load_records(parent/"history/tasks.jsonl")
    origins=[json.loads(l) for l in (work/"candidate-origins.jsonl").read_text().splitlines()]
    additions=[]
    verifier=Verifier(engine_path,(16,20),60)
    try:
        for source in sources:
            if source["difficulty"]!="long":
                continue
            candidates=[r for r in derived_sources([source]) if r["difficulty"]=="medium"]
            candidates.sort(key=lambda r:(r["h2_analysis"]["mate_in_moves"],r["id"]))
            for row in candidates:
                item={"fen":row["fen"],"source":row["source"],
                      "source_file":row["source"]["file"],"ply_index":row["source"]["ply_index"]}
                checked=verifier.check(item,needed_bands={"medium"})
                if not checked["valid"] or checked["band"]!="medium":
                    continue
                verified=make_record(item,checked,row["id"])
                additions.append(verified)
                origins=[r for r in origins if r["id"]!=verified["id"]]+[verified]
                atomic_write_jsonl(work/"candidate-origins.jsonl",origins)
                atomic_write_jsonl(work/"h2-expanded-originals.jsonl",sources+additions)
                atomic_write_jsonl(work/"h2-additional-originals.jsonl",additions)
                print(f"Added freshly checked H2 medium {len(additions)}: {row['id']}",flush=True)
                break
            if len(additions)>=limit:
                break
    finally:
        verifier.close()
    return {"new_h2_candidates":len(additions)}


def expand_foci(parent,work):
    """Use already complete medium standard proofs to fill variant strata."""
    from minibench.datasets.xiangqi.solution_tree import record_key
    source_path=work/"h2-expanded-originals.jsonl"
    sources=load_records(source_path if source_path.exists() else parent/"history/tasks.jsonl")
    rows=[json.loads(l) for l in (work/"c2-candidates.jsonl").read_text().splitlines()]
    origins={r["id"]:r for r in (json.loads(l) for l in
             (work/"candidate-origins.jsonl").read_text().splitlines())}
    seen={(r["fen"],r["ruleset"]) for r in rows if r["ruleset"]!="standard"}
    additions=[]
    for source in sources:
        proof_path=work/"h2-pool-v2"/f"{source['id']}.json"
        if not proof_path.exists():
            continue
        proof=json.loads(proof_path.read_text())
        if (not proof.get("complete") or not proof.get("shortest_certified")
                or not 4<=proof["shortest_attacker_moves"]<=5):
            continue
        for focus in ("horse-no-leg-block","soldier-free-retreat"):
            if (source["fen"],focus) in seen:
                continue
            item=_candidate(source,focus)
            if item is None or not item["sensitive"]:
                continue
            pair=[_record(item,scenario_number=3001+len(additions)//2,ruleset=rule)
                  for rule in ("standard",focus)]
            if record_key(pair[0])!=proof["record_key"]:
                raise ValueError("Standard proof identity changed")
            atomic_write_json(work/"c2-pool-v2"/f"{pair[0]['id']}.json",
                              {**proof,"task_id":pair[0]["id"]})
            additions.extend(pair)
            seen.add((source["fen"],focus))
            origins[source["id"]]=source
    atomic_write_jsonl(work/"c2-candidates.jsonl",rows+additions)
    atomic_write_jsonl(work/"candidate-origins.jsonl",list(origins.values()))
    atomic_write_jsonl(work/"c2-additional-foci-originals.jsonl",additions)
    return {"new_variant_pairs":len(additions)//2}


def publish(parent, work, *, seed=42, engine_path=None):
    """Publish selected, certified records; never infer a missing proof."""
    current_c2 = ROOT / "data/xiangqi/rule_variants/tasks.jsonl"
    if current_c2.exists() and len(load_records(current_c2)) == 200:
        raise ValueError(
            "The legacy combined publisher cannot replace the 100-pair C2 release; "
            "use expand_xiangqi_c2_pairs.py publish for the current C2 protocol"
        )
    expanded=work/"h2-expanded-originals.jsonl"
    h2=load_records(expanded if expanded.exists() else parent/"history/tasks.jsonl")
    c2=load_records(work/"c2-candidates.jsonl")
    selected={
        "h2":select_certified(h2,work/"h2-pool-v2",count=100,seed=seed),
        "c2":select_certified(c2,work/"c2-pool-v2",count=100,seed=seed,paired=True),
    }
    origin_rows=[json.loads(x) for x in (work/"candidate-origins.jsonl").read_text().splitlines()]
    origins={r["id"]:r for r in origin_rows}
    parents={r["id"]:r for r in load_records(parent/"history/tasks.jsonl")}
    for origin in origins.values():
        derivation=origin.get("source",{}).get("derivation")
        if derivation and "parent_fen" not in derivation:
            ancestor=parents[derivation["parent_task_id"]]
            offset=derivation["offset_halfmoves"]
            origin["source"]["ply_index"]=ancestor["source"]["ply_index"]
            derivation.update(parent_fen=ancestor["fen"],
                moves_uci=ancestor["h2_analysis"]["principal_variation_uci"][:offset],
                method="H2 reference variation intermediate")
    used_sources={r["source_task_id"] for r in selected["c2"]}
    # A derived source's old validation runs are never repurposed as new runs.
    derived=[origins[k] for k in sorted(used_sources)
             if origins[k].get("derived_source_pending_reanalysis")]
    if derived:
        if engine_path is None:
            raise ValueError("Fresh Pikafish 16/20 source reanalysis is required")
        from build_xiangqi_h2 import Verifier, make_record
        verifier=Verifier(engine_path,(16,20),60)
        try:
            for source in derived:
                item={"fen":source["fen"],"source":source["source"],
                      "source_file":source["source"]["file"],
                      "ply_index":source["source"]["ply_index"]}
                verified=verifier.check(item)
                if not verified["valid"]:
                    raise ValueError(f"Derived source failed fresh analysis: {source['id']}: {verified['reasons']}")
                origins[source["id"]]=make_record(item,verified,source["id"])
        finally:
            verifier.close()
    destination=ROOT/"data/xiangqi"
    backup=work/"before"
    for folder in ("history","rule_variants"):
        for path in (destination/folder).glob("*.json*"):
            target=backup/folder/path.name
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                shutil.copy2(path,target)
    for name in ("submission_manifest.json","provenance.json","independent_validation.json"):
        target=backup/name
        if not target.exists():
            shutil.copy2(destination/name,target)
    paths={"h2":"history","c2":"rule_variants"}
    for family,rows in selected.items():
        for row in rows:
            if family=="c2":
                # Fresh source evidence can have a different standard PV.
                # Preserve the C2 line, but never retain stale source metadata.
                row["source"]=deepcopy(origins[row["source_task_id"]]["source"])
            old_id=row["id"]
            n=row["shortest_mate_moves"]
            new_max=2*n-1
            if row["max_plies"]!=new_max:
                row["replaces_id"]=old_id
                row["id"]=row["family"]+"-shortest-"+hashlib.sha256(
                    f"{old_id}|{new_max}".encode()).hexdigest()[:16]
            row["max_plies"]=new_max
            if family == "c2":
                row["c2_analysis"]["max_plies_extension"] = 0
            row["tags"]=sorted(set(t for t in row["tags"] if not t.startswith("mate-band:"))
                               |{f"mate-band:{row['difficulty']}",f"certified-shortest:{n}"})
            tree=json.loads((work/(family+"-pool-v2")/f"{old_id}.json").read_text())
            if (not tree.get("shortest_certified") or not tree.get("complete")
                    or tree["shortest_attacker_moves"]!=n):
                raise ValueError("Missing shortest certificate")
            tree["task_id"]=row["id"]
            tree["extension_attacker_moves"]=0
            cursor=tree["root"]
            line=[]
            while tree["nodes"][cursor]["edges"]:
                move,target=min(tree["nodes"][cursor]["edges"].items())
                line.append(move)
                cursor=target
            if len(line)!=2*n-1:
                raise ValueError("Shortest principal strategy length mismatch")
            row["shortest_line_uci"]=line
            row["oracle"]={"best_move_uci":line[0],"mate_in_plies":2*n-1,"evaluation_cp":None}
            tree_path=destination/"solution_trees"/family/f"{row['id']}.json"
            atomic_write_json(tree_path,tree)
            row["shortest_proof"]={
                "version":tree["version"],"record_key":tree["record_key"],
                "tree_path":tree_path.relative_to(ROOT).as_posix(),
                "tree_sha256":sha256_file(tree_path),
                "shortest_attacker_moves":n,"tree_complete":tree["complete"],
            }
            from minibench.datasets.xiangqi.schema import validate_record
            validate_record(row)
            from minibench.datasets.xiangqi.solution_tree import validate_tree_record
            checked=validate_tree_record(row,tree_path)
            if not checked["valid"]:
                raise ValueError(f"Tree failed replay before publication: {row['id']}: {checked['reasons']}")
    # Validate ALL new rows before replacing either canonical task file.
    for family,rows in selected.items():
        atomic_write_jsonl(destination/paths[family]/"tasks.jsonl",rows)
    selected_h2_ids={r["id"] for r in selected["h2"]}
    evidence=[origins[k] for k in sorted(used_sources-selected_h2_ids)]
    atomic_write_jsonl(destination/"sources/c2_h2_origins.jsonl",evidence)
    manifest=json.loads((destination/"submission_manifest.json").read_text())
    manifest.pop("c2_eligible_certified_scenarios",None)
    manifest.pop("c2_excluded_uncertified_scenarios",None)
    manifest.update(version="xiangqi-shortest-paired100-strict-n-v3",seed=seed,
                    protocol="certified-shortest-total-attacker-moves-no-extension",
                    c2_rule_pair_counts={"standard":50,"chariot-no-center":17,
                                         "horse-no-leg-block":17,"soldier-free-retreat":16},
                    c2_derived_source_count=sum(bool(origins[k].get("source",{}).get("derivation"))
                                                for k in used_sources),
                    h2_derived_source_count=sum(bool(r.get("source",{}).get("derivation"))
                                               for r in selected["h2"]),
                    candidate_pool_rows={"h2":len(h2),"c2":len(c2)},
                    complete_tree_required=True)
    for family,rows in selected.items():
        path=destination/paths[family]/"tasks.jsonl"
        manifest["datasets"][family].update(
            path=path.relative_to(ROOT).as_posix(),sha256=sha256_file(path),
            task_ids=[r["id"] for r in rows],
            difficulty_counts=dict(Counter(r["difficulty"] for r in rows)),
            certified_shortest_count=len(rows),
            complete_tree_count=sum(r["shortest_proof"]["tree_complete"] for r in rows),
        )
    manifest["policy"].update(
        h2="50 certified shortest mate-in-2/3 and 50 mate-in-4/5",
        c2="50 complete standard/focus pairs, rule applies; condition rows 50 short/50 medium",
        m2="exact D3 positions and output protocol; Chinese images only; D3 is text baseline")
    manifest["c2_source_evidence"]={
        "path":"data/xiangqi/sources/c2_h2_origins.jsonl",
        "sha256":sha256_file(destination/"sources/c2_h2_origins.jsonl"),
        "record_count":len(evidence)}
    manifest.pop("c2_certificate_sha256",None)
    atomic_write_json(destination/"submission_manifest.json",manifest)
    atomic_write_json(destination/"history/validation_report.json",{
        "version":manifest["version"],"selected_count":100,
        "selected_dataset_sha256":sha256_file(destination/"history/tasks.jsonl"),
        "parent_dataset_sha256":sha256_file(parent/"history/tasks.jsonl"),
        "certified_shortest_count":100})
    atomic_write_json(destination/"rule_variants/generation_report.json",{
        "version":manifest["version"],"record_count":100,"scenario_count":50,
        "difficulty_counts":{"short":50,"medium":50},
        "ruleset_counts":manifest["c2_rule_pair_counts"],
        "certification":"shortest full-width AND/OR, all legal defences",
        "stored_defences":"optimal replies only, unless header says all_legal",
        "derived_sources":sum(bool(origins[k].get("source",{}).get("derivation"))
                              for k in used_sources)})
    provenance=json.loads((destination/"provenance.json").read_text())
    provenance.update(corpus_name="MiniBench shortest paired100 release 2026-10-19",
                      protocol_version="xiangqi-shortest-tree-strict-n-v2")
    provenance["generation_command"].update(
        h2="prepare_xiangqi_shortest_release.publish with reviewed complete trees; see submission_manifest.json",
        c2="prepare_xiangqi_shortest_release.publish with reviewed complete paired trees; see submission_manifest.json",
        submitted_selection="prepare_xiangqi_shortest_release.publish, seed=42; complete shortest trees required; see submission_manifest.json")
    provenance["validation"].pop("repair_manifest",None)
    provenance["validation"]["variants"]=(
        "Complete bounded shortest-mate AND/OR proofs cover all legal defensive replies; "
        "exported DAGs retain locally shortest attacks and optimal delaying defences. "
        "Independent rules and terminal replay are required. CPE depth-3 is a metric/fallback, not this proof.")
    provenance["submission"]={"manifest":"submission_manifest.json",
                              "sha256":sha256_file(destination/"submission_manifest.json")}
    provenance["interpretation"].update(
        h2="Certified shortest forced mate; 50 short / 50 medium",
        c2="Same-FEN standard/focus pairs, each certified under its own active rules",
        h2_horizon="2*certified_shortest_attacker_moves-1 half-moves; total attacker moves, no reset or extension",
        shortest_oracle="oracle and shortest_line_uci are certified shortest; h2/c2_analysis preserves historical reference evidence",
    )
    provenance["intentional_overlap"]["D3_M2"]="100 identical positions; D3 text versus M2 Chinese images; same output schema."
    atomic_write_json(destination/"provenance.json",provenance)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--publish",action="store_true")
    p.add_argument("--pikafish-path",type=Path)
    p.add_argument("--h2-tree-dir",type=Path)
    p.add_argument("--expand-defences",action="store_true")
    p.add_argument("--expand-h2",action="store_true")
    p.add_argument("--expand-foci",action="store_true")
    p.add_argument("--expansion-limit",type=int,default=24)
    args=p.parse_args()
    if args.expand_foci:
        result=expand_foci(args.parent,args.out)
    elif args.expand_h2:
        if not args.pikafish_path or args.expansion_limit<1:
            p.error("--expand-h2 requires --pikafish-path and positive --expansion-limit")
        result=expand_h2(args.parent,args.out,engine_path=args.pikafish_path,limit=args.expansion_limit)
    elif args.expand_defences:
        if not args.pikafish_path or args.expansion_limit<1:
            p.error("--expand-defences requires --pikafish-path and positive --expansion-limit")
        result=expand_defences(args.parent,args.out,engine_path=args.pikafish_path,
                               limit=args.expansion_limit)
    else:
        result=(publish(args.parent,args.out,engine_path=args.pikafish_path) if args.publish
                else scan(args.parent,args.out,h2_tree_dir=args.h2_tree_dir))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
