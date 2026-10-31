"""Prepare and prove C2 pairs without requiring a shared standard/variant PV.

All working files belong outside the submitted corpus. Cached trees are only
candidates: the separate VariantBoard checker must re-prove every published row.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from minibench.core.checkpoint import atomic_write_json, atomic_write_jsonl, sha256_file
from minibench.datasets.xiangqi import reference
from minibench.datasets.xiangqi.schema import fen_to_board, validate_record
from minibench.datasets.xiangqi.solution_tree import build_tree, record_key, VERSION
from minibench.datasets.xiangqi.validation import validate_position
from prepare_xiangqi_shortest_release import derived_sources
from minibench.datasets.xiangqi.rule_effects import graph_signature, paired_impact

FOCI = ("chariot-no-center", "horse-no-leg-block", "soldier-free-retreat")


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def identity(fen, rule):
    return hashlib.sha256(json.dumps([fen, rule], separators=(",", ":")).encode()).hexdigest()[:20]


def completed(tree):
    return (tree.get("version") == VERSION and tree.get("complete") is True
            and tree.get("forced_mate") is True and tree.get("shortest_certified") is True
            and 2 <= tree.get("shortest_attacker_moves", 0) <= 5)


def load_cache(directories):
    trees = {}
    for directory in directories:
        if not directory.exists():
            continue
        for path in directory.glob("*.json"):
            tree = json.loads(path.read_text())
            if completed(tree):
                trees[(tree["initial_fen"], tree["ruleset"])] = tree
    return trees


def subtree_candidate(tree, key):
    node = tree["nodes"][key]
    if node["remaining_plies"] != node["dtm_plies"]:
        return None
    pending, nodes = [key], {}
    while pending:
        cursor = pending.pop()
        if cursor in nodes:
            continue
        nodes[cursor] = deepcopy(tree["nodes"][cursor])
        pending.extend(nodes[cursor]["edges"].values())
    n = (node["dtm_plies"] + 1) // 2
    item = {**tree, "initial_fen": node["fen"], "root": key, "nodes": nodes,
            "shortest_attacker_moves": n, "max_attacker_moves": n,
            "deepest_mate_attacker_moves": n, "extension_attacker_moves": 0}
    item["record_key"] = record_key({"fen": node["fen"], "ruleset": "standard"})
    return item


def prepare(parent, work, legacy=None, *, max_derived=160):
    sources = {}
    for path in [ROOT / "data/xiangqi/history/tasks.jsonl", parent / "history/tasks.jsonl"]:
        for row in read_rows(path):
            sources.setdefault(row["fen"], row)
    if legacy:
        for row in read_rows(legacy / "candidate-origins.jsonl"):
            sources.setdefault(row["fen"], row)
    directories = [ROOT / "data/xiangqi/solution_trees/h2", ROOT / "data/xiangqi/solution_trees/c2"]
    if legacy:
        directories += [legacy / "h2-pool-v2", legacy / "c2-pool-v2"]
    cache = load_cache(directories)
    derived = []
    for fen, source in list(sources.items()):
        tree = cache.get((fen, "standard"))
        if not tree:
            continue
        queue, seen = deque([(tree["root"], [])]), set()
        while queue:
            key, prefix = queue.popleft()
            if key in seen:
                continue
            seen.add(key)
            node = tree["nodes"][key]
            queue.extend((target, prefix + [move]) for move, target in sorted(node["edges"].items()))
            if (not prefix or node["turn"] != tree["attacker"]
                    or not 3 <= node["dtm_plies"] <= 9):
                continue
            cropped = subtree_candidate(tree, key)
            if cropped:
                cache.setdefault((node["fen"], "standard"), cropped)
            if node["fen"] in sources:
                continue
            item = deepcopy(source)
            item.update(id="xiangqi-history-derived-" + identity(node["fen"], "standard"),
                        fen=node["fen"], derived_source_pending_reanalysis=True,
                        derived_shortest_candidate=(node["dtm_plies"] + 1)//2)
            item["source"]["derivation"] = {"parent_task_id": source["id"],
                "parent_fen": fen, "moves_uci": prefix,
                "method": "certified standard shortest-DAG intermediate"}
            sources[node["fen"]] = item
            derived.append(node["fen"])
            # Every selected intermediate receives a fresh standard proof too.
    for item in derived_sources(read_rows(parent / "history/tasks.jsonl")):
        if item["difficulty"] == "medium" and item["fen"] not in sources:
            item["derived_shortest_candidate"] = item["h2_analysis"]["mate_in_moves"]
            sources[item["fen"]] = item
            derived.append(item["fen"])
    if len(derived) > max_derived:
        order = lambda f: (sum(ch.isalpha() for ch in f.split()[0]), f)
        medium = sorted((f for f in derived if sources[f].get("derived_shortest_candidate", 2) >= 4), key=order)
        short = sorted((f for f in derived if f not in medium), key=order)
        selected = medium[:max_derived//2] + short[:max_derived-max_derived//2]
        selected += [f for f in medium+short if f not in selected][:max_derived-len(selected)]
        allowed = set(selected)
        sources = {f: r for f, r in sources.items() if f not in derived or f in allowed}
    work.mkdir(parents=True, exist_ok=True)
    proof_dir = work / "proofs"
    proof_dir.mkdir(exist_ok=True)
    origins, candidates = [], []
    for fen, source in sorted(sources.items()):
        board, color = fen_to_board(fen)
        side = 1 if color == "red" else -1
        if validate_position(board, side):
            continue
        known = cache.get((fen, "standard"))
        # Unproved long roots are expensive; their certified shorter intermediate
        # states remain eligible. All original 2..5-move roots are considered.
        if not known and source.get("h2_analysis", {}).get("mate_in_moves", 99) > 5 and not source.get("derived_source_pending_reanalysis"):
            continue
        origins.append(source)
        present = {reference.KINDS[abs(p)] for row in board for p in row if p}
        rules = []
        for focus, kind in zip(FOCI, ("chariot", "horse", "soldier")):
            if kind not in present or reference.in_check(board, -side, focus):
                continue
            # Different rules may yield a completely different mating line.
            # Never require the standard PV to remain legal under the variant.
            rules.append(focus)
        if not rules:
            continue
        for rule in ("standard", *rules):
            ident = "xiangqi-c2-p100-" + identity(fen, rule)
            row = {"id": ident, "fen": fen, "ruleset": rule, "origin_id": source["id"],
                   "difficulty": source.get("difficulty", "short"),
                   "h2_analysis": source.get("h2_analysis", {})}
            candidates.append(row)
            old = cache.get((fen, rule))
            path = proof_dir / f"{ident}.json"
            existing = json.loads(path.read_text()) if path.exists() else {}
            if old and not completed(existing):
                # Historical extension metadata cannot grant extra runtime turns.
                # The independent checker later recomputes all substantive claims.
                tree = {**old, "task_id": ident, "extension_attacker_moves": 0}
                atomic_write_json(path, tree)
    atomic_write_jsonl(work / "origins.jsonl", origins)
    atomic_write_jsonl(work / "candidates.jsonl", candidates)
    report = {"origins": len(origins), "candidates": len(candidates), "parent_data": str(parent.resolve()),
              "cached_candidate_proofs": sum((proof_dir / f"{r['id']}.json").exists() for r in candidates),
              "status": "CANDIDATES_NOT_CERTIFIED_RELEASE"}
    atomic_write_json(work / "prepare.json", report)
    return report


def prove_one(payload):
    row, seconds, budget, memory_mb = payload
    if memory_mb:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        cap = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (min(cap, hard) if hard != resource.RLIM_INFINITY else cap, hard))
    try:
        tree = build_tree(row, max_attacker_moves=5, seconds=seconds, node_budget=budget)
    except MemoryError:
        tree = {"version": VERSION, "task_id": row["id"], "record_key": record_key(row),
                "complete": False, "forced_mate": None, "shortest_certified": False,
                "stop_reason": "memory_budget_exhausted"}
    return row["id"], tree


def prove(work, *, workers=4, seconds=15, budget=250_000, memory_mb=1200, limit=0,
          rulesets=FOCI, phase_filter="all", band="all"):
    rows = read_rows(work / "candidates.jsonl")
    directory = work / "proofs"
    paired_path = work / "proved-pairs.jsonl"
    already_paired = {row["fen"] for row in read_rows(paired_path)} if paired_path.exists() else set()
    pending = []
    # Prove standard states before spending time on their variants.
    for phase in ("standard", "variants"):
        if phase_filter != "all" and phase != phase_filter:
            continue
        pending = []
        for row in rows:
            if (row["ruleset"] == "standard") != (phase == "standard"):
                continue
            if phase != "standard":
                if row["ruleset"] not in rulesets:
                    continue
                std = directory / f"xiangqi-c2-p100-{identity(row['fen'], 'standard')}.json"
                standard = json.loads(std.read_text()) if std.exists() else {}
                if not completed(standard):
                    continue
                if band != "all" and (standard["shortest_attacker_moves"] <= 3) != (band == "short"):
                    continue
            path = directory / f"{row['id']}.json"
            if path.exists() and json.loads(path.read_text()).get("complete"):
                continue
            pending.append(row)
        pending.sort(key=lambda r: (r["fen"] in already_paired if phase == "variants" else False,
                                    sum(ch.isalpha() for ch in r["fen"].split()[0]), r["ruleset"], r["id"]))
        if limit:
            pending = pending[:limit]
        print(json.dumps({"phase": phase, "pending": len(pending)}), flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(prove_one, (row, seconds, budget, memory_mb)) for row in pending]
            for future in as_completed(futures):
                ident, tree = future.result()
                atomic_write_json(directory / f"{ident}.json", tree)
                print(json.dumps({"id": ident, "complete": tree.get("complete"),
                                  "shortest": tree.get("shortest_attacker_moves"),
                                  "stop": tree.get("stop_reason")}), flush=True)
    return pool_summary(work)


def pool_summary(work):
    rows = read_rows(work / "candidates.jsonl")
    trees = {}
    for row in rows:
        p = work / "proofs" / f"{row['id']}.json"
        if p.exists():
            t = json.loads(p.read_text())
            if completed(t):
                trees[(row["fen"], row["ruleset"])] = t
    pairs = []
    for row in rows:
        if row["ruleset"] == "standard":
            continue
        std, var = trees.get((row["fen"], "standard")), trees.get((row["fen"], row["ruleset"]))
        if not std or not var:
            continue
        pairs.append({"fen": row["fen"], "ruleset": row["ruleset"], "origin_id": row["origin_id"],
                      "standard_id": std["task_id"], "variant_id": var["task_id"],
                      "standard_n": std["shortest_attacker_moves"], "variant_n": var["shortest_attacker_moves"],
                      **paired_impact(std, var)})
    atomic_write_jsonl(work / "proved-pairs.jsonl", pairs)
    report = {"proved_pairs": len(pairs), "unique_positions": len({r["fen"] for r in pairs}),
              "by_rule": {rule: {"pairs": sum(r["ruleset"] == rule for r in pairs),
                  "strategy_changed": sum(r["ruleset"] == rule and r["optimal_strategy_changed"] for r in pairs),
                  "first_moves_changed": sum(r["ruleset"] == rule and r["optimal_first_moves_changed"] for r in pairs),
                  "distance_changed": sum(r["ruleset"] == rule and r["shortest_distance_changed"] for r in pairs)} for rule in FOCI}}
    atomic_write_json(work / "pool-summary.json", report)
    return report


def select_pairs(work, *, count=100, rule_counts=None):
    import z3
    if isinstance(count, bool) or count < 2 or count % 2:
        raise ValueError("Pair count must be a positive even number")
    origins = {row["id"]: row for row in read_rows(work / "origins.jsonl")}
    rejected_path = work / "rejected-origins.json"
    rejected = set(json.loads(rejected_path.read_text())) if rejected_path.exists() else set()
    pairs = [row for row in read_rows(work / "proved-pairs.jsonl") if row["origin_id"] not in rejected]
    solver = z3.Optimize()
    solver.set(timeout=20_000)
    chosen = [z3.Bool(f"pair_{i}") for i in range(len(pairs))]
    solver.add(z3.PbEq([(x, 1) for x in chosen], count))
    solver.add(z3.PbEq([(x, int(row["standard_n"] <= 3)) for x, row in zip(chosen, pairs)], count // 2))
    for fen in {row["fen"] for row in pairs}:
        solver.add(z3.PbLe([(x, 1) for x, row in zip(chosen, pairs) if row["fen"] == fen], 1))
    for rule in FOCI:
        terms = [(x, 1) for x, row in zip(chosen, pairs) if row["ruleset"] == rule]
        if rule_counts:
            solver.add(z3.PbEq(terms, rule_counts[rule]))
        else:
            solver.add(z3.PbGe(terms, count // 5), z3.PbLe(terms, 2 * count // 5))
    for field in ("optimal_strategy_changed", "optimal_first_moves_changed", "shortest_distance_changed", "optimal_first_moves_disjoint"):
        solver.maximize(z3.Sum([z3.If(x, int(row[field]), 0) for x, row in zip(chosen, pairs)]))
    source_games = {row["source"]["file"] for row in origins.values()}
    solver.maximize(z3.Sum([z3.If(z3.Or([x for x, row in zip(chosen, pairs) if origins[row["origin_id"]]["source"]["file"] == game]), 1, 0) for game in source_games]))
    solver.minimize(z3.Sum([z3.If(x, i, 0) for i, x in enumerate(chosen)]))
    status = solver.check()
    if status == z3.unsat:
        raise ValueError(f"No proved {count}-pair selection with required baseline bands/rule coverage: {status}")
    model = solver.model()
    selected = [row for x, row in zip(chosen, pairs) if z3.is_true(model.eval(x))]
    # Budgeted optimization may leave global optimality unknown. A concrete
    # feasible roster is independently checked; this is not a mate certificate.
    rule_totals = Counter(row["ruleset"] for row in selected)
    if (len(selected) != count or len({r["fen"] for r in selected}) != count
            or sum(r["standard_n"] <= 3 for r in selected) != count//2
            or any(rule_totals[rule] != rule_counts[rule] if rule_counts else
                   not count//5 <= rule_totals[rule] <= 2*count//5 for rule in FOCI)):
        raise ValueError("Selection optimization has not produced a valid concrete roster")
    selected.sort(key=lambda r: (r["ruleset"], r["fen"]))
    atomic_write_jsonl(work / "selected-pairs.jsonl", selected)
    atomic_write_jsonl(work / "selected-origins.jsonl", [origins[row["origin_id"]] for row in selected])
    report = {"pairs": len(selected), "rules": dict(Counter(r["ruleset"] for r in selected)),
              "standard_bands": dict(Counter("short" if r["standard_n"] <= 3 else "medium" for r in selected)),
              "variant_bands": dict(Counter("short" if r["variant_n"] <= 3 else "medium" for r in selected)),
              "source_games": len({origins[r["origin_id"]]["source"]["file"] for r in selected}),
              "impact_counts": {field: sum(row[field] for row in selected) for field in
                  ("optimal_strategy_changed", "optimal_first_moves_changed", "shortest_distance_changed", "optimal_first_moves_disjoint")},
              "rule_coverage": "exact requested counts" if rule_counts else "each rule 20..40 percent; impact priority",
              "selection_optimality": "proved optimizer result" if status == z3.sat else "best feasible roster; global impact optimum not proved within selection budget",
              "variant_difficulty_policy": "recomputed; not forced to match baseline distribution"}
    atomic_write_json(work / "selection.json", report)
    return report


def repair_origin_trace(source, parents, visited=()):
    source = deepcopy(source)
    derivation = source.get("source", {}).get("derivation")
    if not derivation:
        return source
    parent_id = derivation.get("parent_task_id")
    if parent_id in visited:
        raise ValueError("Cyclic derived provenance")
    parent = parents.get(parent_id)
    if not parent:
        if not derivation.get("parent_fen") or not derivation.get("moves_uci"):
            raise ValueError("Missing parent for incomplete derived provenance")
        return source
    if not derivation.get("parent_fen"):
        derivation["parent_fen"] = parent["fen"]
        offset = derivation["offset_halfmoves"]
        derivation["moves_uci"] = parent["h2_analysis"]["principal_variation_uci"][:offset]
        source["source"]["ply_index"] = parent["source"]["ply_index"]
    parent = repair_origin_trace(parent, parents, (*visited, parent_id))
    parent_trace = parent.get("source", {}).get("derivation")
    if parent_trace and derivation["parent_fen"] == parent["fen"]:
        derivation.update(parent_fen=parent_trace["parent_fen"],
            moves_uci=parent_trace["moves_uci"] + derivation["moves_uci"],
            parent_task_id=parent_trace["parent_task_id"],
            method="flattened standard legal variation from original source position")
        source["source"]["ply_index"] = parent["source"]["ply_index"]
    return source


def refresh_origins(work, executable, *, workers=2):
    from concurrent.futures import ThreadPoolExecutor
    from build_xiangqi_h2 import Verifier, make_record
    verifier = Verifier(executable, (16, 20), 60)
    directory = work / "origin-evidence"
    directory.mkdir(exist_ok=True)
    preparation = json.loads((work / "prepare.json").read_text())
    parents = {row["id"]: row for path in (
        Path(preparation["parent_data"]) / "history/tasks.jsonl",
        ROOT / "data/xiangqi/history/tasks.jsonl", work / "origins.jsonl") for row in read_rows(path)}
    sources = [repair_origin_trace(row, parents) for row in read_rows(work / "selected-origins.jsonl")]
    atomic_write_jsonl(work / "selected-origins.jsonl", sources)
    rejected = []

    def refresh(source):
        target = directory / f"{identity(source['fen'], 'standard')}.json"
        if target.exists():
            cached = json.loads(target.read_text())
            if cached.get("fen") == source["fen"] and cached.get("source") == source["source"]:
                validate_record(cached, expected_family="xiangqi-history")
                return cached
        needs = bool(source.get("derived_source_pending_reanalysis"))
        try:
            validate_record(source, expected_family="xiangqi-history")
        except (ValueError, KeyError):
            needs = True
        if not needs:
            atomic_write_json(target, source)
            return source
        item = {"fen": source["fen"], "source": source["source"],
                "source_file": source["source"]["file"], "ply_index": source["source"]["ply_index"]}
        # The engine PV is source evidence, not the exact distance label. A
        # longer valid reference is allowed; the separate proof sets the bound.
        checked = verifier.check(item)
        if not checked["valid"]:
            atomic_write_json(directory / f"{identity(source['fen'], 'standard')}-failed.json", checked)
            return None
        result = make_record(item, checked, source["id"])
        atomic_write_json(target, result)
        return result

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(refresh, source): source for source in sources}
            for future in as_completed(futures):
                source = futures[future]
                result = future.result()
                if result is None:
                    rejected.append(source["id"])
                print(json.dumps({"source": source["id"], "valid": result is not None}), flush=True)
    finally:
        verifier.close()
    old = work / "rejected-origins.json"
    rejected = sorted(set(rejected) | (set(json.loads(old.read_text())) if old.exists() else set()))
    atomic_write_json(old, rejected)
    failed_selected = [source["id"] for source in sources if source["id"] in rejected]
    result = {"selected_origins": len(sources), "rejected_origins": rejected,
              "selected_rejected_origins": failed_selected, "ready": not failed_selected}
    atomic_write_json(work / "sources-ready.json", result)
    return result


def principal_line(tree):
    key, line = tree["root"], []
    while tree["nodes"][key]["edges"]:
        move, key = min(tree["nodes"][key]["edges"].items())
        line.append(move)
    if len(line) != 2 * tree["shortest_attacker_moves"] - 1:
        raise ValueError("Shortest strategy line length mismatch")
    return line


def line_sensitive(fen, line, focus, active):
    state, color = fen_to_board(fen)
    turn, changed = (1 if color == "red" else -1), False
    for uci in line:
        std = reference.legal_uci_moves(state, turn, "standard")
        var = reference.legal_uci_moves(state, turn, focus)
        changed |= std != var
        moves = {m.to_uci(): m for m in reference.legal_moves(state, turn, active)}
        if uci not in moves:
            raise ValueError("Active-rule reference line is illegal")
        state = reference.apply_move(state, moves[uci])
        turn = -turn
    return changed


def stage(work):
    import shutil
    from build_xiangqi_c2 import _public_rules, _production_replay, _independent_replay
    from minibench.datasets.xiangqi.variants.mate_search import LOSS_UNIT
    selected = read_rows(work / "selected-pairs.jsonl")
    if len(selected) != 100 or len({row["fen"] for row in selected}) != 100:
        raise ValueError("Official C2 staging requires 100 distinct paired positions")
    ready = json.loads((work / "sources-ready.json").read_text())
    if not ready.get("ready"):
        raise ValueError("Selected sources have not all passed fresh/reference validation")
    sources = {}
    for row in read_rows(work / "selected-origins.jsonl"):
        path = work / "origin-evidence" / f"{identity(row['fen'], 'standard')}.json"
        source = json.loads(path.read_text())
        validate_record(source, expected_family="xiangqi-history")
        if source["fen"] != row["fen"] or source["source"] != row["source"]:
            raise ValueError("Source reanalysis binding mismatch")
        sources[row["fen"]] = source
    destination = work / "stage/data/xiangqi"
    if not destination.exists():
        shutil.copytree(ROOT / "data/xiangqi", destination)
    rows, paths = [], set()
    for pair in selected:
        source = sources[pair["fen"]]
        focus = pair["ruleset"]
        scenario = "xiangqi-rule-scenario-p100-" + identity(pair["fen"], focus)
        for rule, key in (("standard", "standard_id"), (focus, "variant_id")):
            tree = json.loads((work / "proofs" / f"{pair[key]}.json").read_text())
            if not completed(tree) or tree["initial_fen"] != pair["fen"] or tree["ruleset"] != rule:
                raise ValueError("Candidate proof identity changed")
            ident = "xiangqi-rule-variants-c2-p100-" + identity(pair["fen"], rule)
            tree.update(task_id=ident, extension_attacker_moves=0)
            line, n = principal_line(tree), tree["shortest_attacker_moves"]
            tree_path = destination / "solution_trees/c2" / f"{ident}.json"
            atomic_write_json(tree_path, tree)
            paths.add(tree_path.resolve())
            board, color = fen_to_board(pair["fen"])
            side = 1 if color == "red" else -1
            row = {"schema_version": 2, "release_id": source["release_id"], "id": ident,
                "family": "xiangqi-rule-variants", "fen": pair["fen"], "agent_color": color,
                "goal": "checkmate", "max_plies": 2*n-1, "difficulty": "short" if n <= 3 else "medium",
                "piece_count": sum(bool(p) for rank in board for p in rank),
                "oracle": {"best_move_uci": line[0], "mate_in_plies": len(line), "evaluation_cp": None},
                "tags": sorted({"c2", "source:h2-verified-mate", f"focus:{focus}", f"ruleset:{rule}",
                                f"mate-band:{'short' if n <= 3 else 'medium'}", f"certified-shortest:{n}"}),
                "source": deepcopy(source["source"]), "source_id": source["source_id"],
                "source_task_id": source["id"], "ruleset": rule, "rules": _public_rules(rule),
                "scenario_id": scenario, "design_stratum": focus,
                "c2_analysis": {"version": "c2-multistep-rule-mate-v3", "source_task_id": source["id"],
                    "focus_ruleset": focus, "pair_rulesets": ["standard", focus], "mate_in_moves": n,
                    "reference_mate_plies": len(line), "principal_variation_uci": line,
                    "rule_sensitive_along_reference": line_sensitive(pair["fen"], line, focus, rule),
                    "search_depth": 3, "loss_unit": LOSS_UNIT, "max_plies_extension": 0,
                    "verification": {"production": _production_replay(board, side, line, rule),
                                     "independent": _independent_replay(board, side, line, rule)},
                    "interpretation": "Each active rule has its own certified shortest strategy; independent full-width reproof required."},
                "shortest_mate_moves": n, "shortest_line_uci": line,
                "shortest_proof": {"version": tree["version"], "record_key": tree["record_key"],
                    "tree_path": f"data/xiangqi/solution_trees/c2/{ident}.json", "tree_sha256": sha256_file(tree_path),
                    "shortest_attacker_moves": n, "tree_complete": True},
                "paired_rule_impact": {field: pair[field] for field in (
                    "shortest_distance_changed", "optimal_first_moves_changed", "optimal_first_moves_disjoint",
                    "optimal_strategy_changed", "standard_optimal_first_moves", "variant_optimal_first_moves")}}
            validate_record(row)
            rows.append(row)
    rows.sort(key=lambda r: (r["ruleset"] != "standard", r["scenario_id"]))
    for path in (destination / "solution_trees/c2").glob("*.json"):
        if path.resolve() not in paths:
            if not path.resolve().is_relative_to((work / "stage").resolve()):
                raise ValueError("Refuse cleanup outside staging directory")
            path.unlink()
    atomic_write_jsonl(destination / "rule_variants/tasks.jsonl", rows)
    canonical = {r["id"]: r for r in read_rows(destination / "history/tasks.jsonl")}
    evidence = []
    for source in sources.values():
        if source["id"] in canonical:
            if source != canonical[source["id"]]:
                raise ValueError("C2 source conflicts with unchanged H2 corpus")
        else:
            evidence.append(source)
    atomic_write_jsonl(destination / "sources/c2_h2_origins.jsonl", sorted(evidence, key=lambda r:r["id"]))
    selection = json.loads((work / "selection.json").read_text())
    atomic_write_json(destination / "rule_variants/generation_report.json", {
        "version": "c2-paired100-impact-priority-v1", "record_count": len(rows), "scenario_count": len(selected),
        "ruleset_counts": dict(Counter(r["ruleset"] for r in rows)), **selection,
        "source_policy": "old MiniBench pool and traced legal intermediates; derived sources freshly checked at depth16/20",
        "independent_verification_required": True})
    return {"staged_data_root": str(destination), "rows": len(rows), "pairs": len(selected)}


def publish(work):
    """Publish only complete independently re-proved staged pairs."""
    import subprocess
    from minibench.datasets.xiangqi.tree_certificate import require_saved_tree_certificate
    from validate_xiangqi_data import audit_collection, audit_c2_pair_impact, audit_c2_sources
    source = work / "stage/data/xiangqi"
    rows = read_rows(source / "rule_variants/tasks.jsonl")
    for row in rows:
        validate_record(row, expected_family="xiangqi-rule-variants")
    for folder, trees in (("history", "h2"), ("rule_variants", "c2")):
        data = source / folder / "tasks.jsonl"
        require_saved_tree_certificate(read_rows(data), data_path=data,
                                       tree_dir=source / "solution_trees" / trees)
    errors = audit_collection("xiangqi-rule-variants", rows)
    errors += audit_c2_pair_impact(rows, source)["reasons"]
    errors += audit_c2_sources(records=rows, data_root=source)["reasons"]
    if errors:
        raise ValueError("Staged C2 failed release checks: " + ";".join(errors))
    destination = ROOT / "data/xiangqi"
    # H2/D3/M2 are deliberately not copied from staging back into the release.
    for folder in ("history", "mate_in_one", "multimodal"):
        if sha256_file(source / folder / "tasks.jsonl") != sha256_file(destination / folder / "tasks.jsonl"):
            raise ValueError("An unrelated corpus changed since staging")
    trees = destination / "solution_trees/c2"
    trees.mkdir(exist_ok=True)
    expected = {f"{row['id']}.json" for row in rows}
    for name in sorted(expected):
        atomic_write_json(trees / name, json.loads((source / "solution_trees/c2" / name).read_text()))
    for relative in ("rule_variants/generation_report.json", "solution_tree_verification.json"):
        atomic_write_json(destination / relative, json.loads((source / relative).read_text()))
    atomic_write_jsonl(destination / "sources/c2_h2_origins.jsonl", read_rows(source / "sources/c2_h2_origins.jsonl"))
    atomic_write_jsonl(destination / "rule_variants/tasks.jsonl", rows)
    for path in trees.glob("*.json"):
        if path.name not in expected:
            if path.resolve().parent != trees.resolve():
                raise ValueError("Refuse obsolete-tree cleanup outside canonical C2 directory")
            path.unlink()
    selection = json.loads((work / "selection.json").read_text())
    evidence = destination / "sources/c2_h2_origins.jsonl"
    manifest_path = destination / "submission_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(version="xiangqi-c2-paired100-impact-priority-v4",
        record_count_per_family={"d3":100,"h2":100,"c2":200,"m2":100},
        c2_rule_pair_counts=dict(Counter(row["ruleset"] for row in rows)),
        c2_scenario_count=100,
        c2_derived_source_count=sum(bool(row.get("source", {}).get("derivation")) for row in rows if row["ruleset"]=="standard"),
        c2_source_evidence={"path":"data/xiangqi/sources/c2_h2_origins.jsonl",
                            "sha256":sha256_file(evidence),"record_count":len(read_rows(evidence))},
        c2_rule_impact=selection["impact_counts"],
        c2_baseline_difficulty_counts=selection["standard_bands"],
        c2_variant_difficulty_counts=selection["variant_bands"])
    manifest["datasets"]["c2"].update(record_count=200, scenario_count=100,
        sha256=sha256_file(destination / "rule_variants/tasks.jsonl"),
        task_ids=[row["id"] for row in rows],
        difficulty_counts=dict(Counter(row["difficulty"] for row in rows)),
        certified_shortest_count=200, complete_tree_count=200)
    manifest["policy"]["c2"] = (
        "100 standard/focus pairs; standard 50 mate-in-2/3 and 50 mate-in-4/5; "
        "variant distance independently certified and may change the band; impact-priority selection")
    manifest["c2_selection_optimality"] = selection["selection_optimality"]
    atomic_write_json(manifest_path, manifest)
    provenance_path = destination / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["corpus_name"] = "MiniBench shortest release with C2 100 paired positions 2026-10-28"
    provenance["generation_command"]["c2"] = (
        "python scripts/expand_xiangqi_c2_pairs.py prepare/prove/select/origins/stage; "
        "python scripts/verify_xiangqi_solution_trees.py --data-root <staging>; "
        "python scripts/expand_xiangqi_c2_pairs.py publish --work <reviewed-work>")
    provenance["generation_command"]["c2_source_policy"] = (
        "100 distinct standard/focus pairs; old MiniBench pools plus traced legal intermediates; "
        "new sources reanalysed depth16/20; each condition independently re-proved; no shared PV requirement")
    provenance["submission"] = {"manifest":"submission_manifest.json", "sha256":sha256_file(manifest_path)}
    atomic_write_json(provenance_path, provenance)
    subprocess.run([sys.executable, "-B", str(ROOT / "scripts/freeze_xiangqi_evaluation.py"), "--replace"],
                   cwd=ROOT, check=True)
    return {"published_rows":200,"paired_positions":100,"independent_tree_proof":"required and passed",
            "next_step":"regenerate independent_validation.json and technical release audit"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "prove", "summary", "select", "origins", "stage", "publish"))
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--legacy", type=Path)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--max-derived", type=int, default=160)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--node-budget", type=int, default=250_000)
    parser.add_argument("--memory-mb", type=int, default=1200)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rulesets", default=",".join(FOCI))
    parser.add_argument("--phase", choices=("all", "standard", "variants"), default="all")
    parser.add_argument("--band", choices=("all", "short", "medium"), default="all")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--rule-counts", help="Comma-separated car,horse,soldier counts; default 20..40 percent each")
    parser.add_argument("--pikafish", type=Path)
    args = parser.parse_args()
    if (args.work.resolve() == ROOT or args.work.resolve().is_relative_to(ROOT / "data")
            or args.workers < 1 or args.seconds <= 0 or not math.isfinite(args.seconds)
            or args.node_budget < 1 or args.memory_mb < 0 or args.limit < 0 or args.max_derived < 0):
        parser.error("Use an isolated work directory and valid positive search budgets")
    if args.command == "prepare":
        if args.parent is None:
            parser.error("prepare requires --parent")
        result = prepare(args.parent, args.work, args.legacy, max_derived=args.max_derived)
    elif args.command == "prove":
        result = prove(args.work, workers=args.workers, seconds=args.seconds,
                       budget=args.node_budget, memory_mb=args.memory_mb, limit=args.limit,
                       rulesets=tuple(args.rulesets.split(",")), phase_filter=args.phase, band=args.band)
    elif args.command == "select":
        counts = dict(zip(FOCI, map(int, args.rule_counts.split(",")))) if args.rule_counts else None
        result = select_pairs(args.work, count=args.count, rule_counts=counts)
    elif args.command == "origins":
        if args.pikafish is None:
            parser.error("origins requires --pikafish")
        result = refresh_origins(args.work, args.pikafish, workers=args.workers)
    elif args.command == "stage":
        result = stage(args.work)
    elif args.command == "publish":
        result = publish(args.work)
    else:
        result = pool_summary(args.work)
    print(json.dumps(result), flush=True)
    if args.command == "origins" and not result["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
