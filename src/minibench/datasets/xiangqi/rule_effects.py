"""Describe paired rule effects on independently checked shortest strategies."""
from __future__ import annotations


def graph_signature(tree):
    """Compare mating edges/distances, not merely unused legal candidates."""
    nodes = tree["nodes"]
    return sorted((node["fen"], node["turn"], node["dtm_plies"],
                   tuple(sorted((move, nodes[target]["fen"])
                                for move, target in node["edges"].items())))
                  for node in nodes.values())


def paired_impact(standard, variant):
    a = set(standard["nodes"][standard["root"]]["edges"])
    b = set(variant["nodes"][variant["root"]]["edges"])
    return {"shortest_distance_changed": standard["shortest_attacker_moves"] != variant["shortest_attacker_moves"],
            "optimal_first_moves_changed": a != b, "optimal_first_moves_disjoint": not bool(a & b),
            "optimal_strategy_changed": graph_signature(standard) != graph_signature(variant),
            "standard_optimal_first_moves": sorted(a), "variant_optimal_first_moves": sorted(b)}
