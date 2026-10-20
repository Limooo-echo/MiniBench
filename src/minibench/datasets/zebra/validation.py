"""Independent finite-grammar Zebra verifier.

Domains come exclusively from the puzzle's background. Gold grids are never
asserted into the solver. Every clue must have exactly one complete parse.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth")
BACKGROUND_HEADERS = {
    "Each person has a unique name": "Name",
    "The people are of nationalities": "Nationality",
    "People have unique favorite book genres": "BookGenre",
    "Everyone has something unique for lunch": "Food",
    "Each person has a favorite color": "Color",
    "The people keep unique animals": "Animal",
    "Each person has an occupation": "Occupation",
    "People use unique phone models": "PhoneModel",
    "Each mother is accompanied by their child": "Children",
    "People have unique favorite music genres": "MusicGenre",
    "People have unique heights": "Height",
    "The mothers' names in different houses are unique": "Mother",
    "Each person prefers a unique type of vacation": "Vacation",
    "Each person has a unique type of pet": "Pet",
    "Each person has a unique birthday month": "Birthday",
    "Each person lives in a unique style of house": "HouseStyle",
    "They all have a unique favorite flower": "Flower",
    "Each person has a unique level of education": "Education",
    "People have unique favorite sports": "FavoriteSport",
    "Everyone has a unique favorite cigar": "Cigar",
    "People own unique car models": "CarModel",
    "Each person has a unique hobby": "Hobby",
    "People have unique hair colors": "HairColor",
    "Everyone has a favorite smoothie": "Smoothie",
    "Each person has a unique favorite drink": "Drink",
}

# These are reviewed textual aliases, not inferred from answer positions.
SPECIAL_ALIASES = {
    "Nationality": {"brit": ["the British person"], "dane": ["the Dane"],
                    "swede": ["the Swedish person"], "chinese": ["the Chinese"],
                    "german": ["the German"], "norwegian": ["the Norwegian"]},
    "Food": {"pizza": ["the person who is a pizza lover"],
             "grilled cheese": ["the person who loves eating grilled cheese"],
             "spaghetti": ["the person who loves the spaghetti eater"],
             "stew": ["the person who loves the stew"],
             "soup": ["the person who loves the soup"],
             "stir fry": ["the person who loves stir fry"]},
    "Animal": {"bird": ["the bird keeper"], "cat": ["the cat lover"],
               "dog": ["the dog owner"], "fish": ["the fish enthusiast"],
               "horse": ["the person who keeps horses"], "rabbit": ["the rabbit owner"]},
    "Pet": {"bird": ["the person who keeps a pet bird"], "cat": ["the person who has a cat"],
            "dog": ["the person who owns a dog"], "fish": ["the person with an aquarium of fish"],
            "hamster": ["the person with a pet hamster"], "rabbit": ["the person who owns a rabbit"]},
    "Vacation": {"beach": ["the person who loves beach vacations"],
                 "camping": ["the person who enjoys camping trips"],
                 "city": ["the person who prefers city breaks"],
                 "cruise": ["the person who likes going on cruises"],
                 "cultural": ["the person who goes on cultural tours"],
                 "mountain": ["the person who enjoys mountain retreats"]},
    "HouseStyle": {"colonial": ["the person living in a colonial-style house"],
                   "craftsman": ["the person in a Craftsman-style house"],
                   "mediterranean": ["the person in a Mediterranean-style villa"],
                   "modern": ["the person in a modern-style house"],
                   "ranch": ["the person in a ranch-style home"],
                   "victorian": ["the person residing in a Victorian house"]},
    "Flower": {"carnations": ["the person who loves a carnations arrangement"],
               "daffodils": ["the person who loves a bouquet of daffodils"],
               "iris": ["the person who loves the boquet of iris"],
               "lilies": ["the person who loves the boquet of lilies"],
               "roses": ["the person who loves the rose bouquet"],
               "tulips": ["the person who loves the vase of tulips"]},
    "Education": {"associate": ["the person with an associate's degree"],
                  "bachelor": ["the person with a bachelor's degree"],
                  "doctorate": ["the person with a doctorate"],
                  "high school": ["the person with a high school diploma"],
                  "master": ["the person with a master's degree"],
                  "trade school": ["the person who attended trade school"]},
    "Cigar": {"blends": ["the person who smokes many unique blends"],
              "blue master": ["the person who smokes Blue Master"],
              "pall mall": ["the person partial to Pall Mall"],
              "dunhill": ["the Dunhill smoker"], "prince": ["the Prince smoker"],
              "yellow monster": ["the person who smokes Yellow Monster"]},
    "Hobby": {"cooking": ["the person who loves cooking"],
              "gardening": ["the person who enjoys gardening"],
              "knitting": ["the person who enjoys knitting"],
              "painting": ["the person who paints as a hobby"],
              "photography": ["the photography enthusiast"],
              "woodworking": ["the woodworking hobbyist"]},
    "Drink": {"boba tea": ["the boba tea drinker"], "coffee": ["the coffee drinker"],
              "milk": ["the person who likes milk"], "root beer": ["the root beer lover"],
              "tea": ["the tea drinker"], "water": ["the one who only drinks water"]},
}


class ValidationError(ValueError):
    pass


@dataclass(frozen=True)
class Clause:
    index: int
    text: str
    operator: str
    left: tuple[str, str]
    right: tuple[str, str] | int
    negated: bool = False


@dataclass(frozen=True)
class ParsedPuzzle:
    houses: int
    domains: tuple[tuple[str, tuple[str, ...]], ...]
    clauses: tuple[Clause, ...]


def entity_aliases(header: str, value: str) -> list[str]:
    if header in SPECIAL_ALIASES:
        if value not in SPECIAL_ALIASES[header]:
            raise ValidationError(f"unsupported declared value {header}/{value}")
        return SPECIAL_ALIASES[header][value]
    if header == "Name":
        return [value]
    if header == "Mother":
        return [f"the person whose mother's name is {value}"]
    if header == "Children":
        return (["the person who is the mother of Timothy"] if value == "Timothy"
                else [f"the person's child is named {value}"])
    if header == "BookGenre":
        return [f"the person who loves {value} books"]
    if header == "MusicGenre":
        return [f"the person who loves {value.replace('hip hop', 'hip-hop')} music"]
    if header == "Color":
        return [f"the person who loves {value}", f"the person whose favorite color is {value}"]
    if header == "HairColor":
        return [f"the person who has {value} hair"]
    if header == "FavoriteSport":
        return [f"the person who loves {value}"]
    if header == "Occupation":
        article = "an" if value[0] in "aeiou" else "a"
        return [f"the person who is {article} {value}"]
    if header == "Height":
        return (["the person who has an average height"] if value == "average"
                else [f"the person who is {value}"])
    if header == "Birthday":
        month = {"jan": "January", "feb": "February", "mar": "March",
                 "april": "April", "may": "May", "sept": "September"}[value]
        return [f"the person whose birthday is in {month}"]
    if header == "PhoneModel":
        article = "an" if value == "iphone 13" else "a"
        return [f"the person who uses {article} {value}"]
    if header == "CarModel":
        return [f"the person who owns a {value.replace('ford f150', 'ford f-150')}"]
    if header == "Smoothie":
        if value in ("blueberry", "lime"):
            return [f"the person who drinks {value} smoothies"]
        if value == "cherry":
            return ["the person who likes Cherry smoothies"]
        return [f"the {value} smoothie lover"]
    raise ValidationError(f"unsupported attribute {header}")


def parse_puzzle(puzzle: str, *, expected_size: str | None = None) -> ParsedPuzzle:
    marker = "\n\n## Clues:\n"
    if puzzle.count(marker) != 1:
        raise ValidationError("expected one complete clue section")
    background, clue_block = puzzle.split(marker)
    lines = background.splitlines()
    match = re.fullmatch(r"There are ([2-6]) houses, numbered 1 to ([2-6]) from left to right, as seen from across the street\. Each house is occupied by a different person\. Each house has a unique attribute for each of the following characteristics:", lines[0])
    if not match or match[1] != match[2]:
        raise ValidationError("unsupported or inconsistent background")
    houses = int(match[1])
    domains: list[tuple[str, tuple[str, ...]]] = []
    aliases: dict[str, set[tuple[str, str]]] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        m = re.fullmatch(r"\s*- (.+): (`[^`]+`(?:, `[^`]+`)*)", line)
        if not m or m[1] not in BACKGROUND_HEADERS:
            raise ValidationError(f"unsupported background line {line!r}")
        header = BACKGROUND_HEADERS[m[1]]
        values = tuple(re.findall(r"`([^`]+)`", m[2]))
        if len(values) != houses or len(set(values)) != houses:
            raise ValidationError(f"invalid declared domain {header}")
        if header in {h for h, _ in domains}:
            raise ValidationError(f"duplicate domain {header}")
        domains.append((header, values))
        for value in values:
            for alias in entity_aliases(header, value):
                aliases.setdefault(alias.casefold(), set()).add((header, value))
    if not 2 <= len(domains) <= 6:
        raise ValidationError("unsupported domain count")
    if expected_size and expected_size != f"{houses}*{len(domains)}":
        raise ValidationError("declared size disagrees with background")

    clauses: list[Clause] = []
    for index, line in enumerate(clue_block.splitlines(), 1):
        if not line.startswith(f"{index}. "):
            raise ValidationError(f"invalid or nonsequential clue {line!r}")
        text = line[len(f"{index}. "):]
        parses = _parse_clue(text, index, aliases)
        if len(parses) != 1:
            raise ValidationError(f"clue {index}: expected one complete parse, got {len(parses)}: {text}")
        clause = parses[0]
        if isinstance(clause.right, int) and clause.right > houses:
            raise ValidationError(f"clue {index}: ordinal out of range")
        clauses.append(clause)
    if not clauses:
        raise ValidationError("no clues")
    return ParsedPuzzle(houses, tuple(domains), tuple(clauses))


def _parse_clue(text: str, index: int, aliases: dict[str, set[tuple[str, str]]]) -> list[Clause]:
    if not text.endswith("."):
        return []
    body = text[:-1].casefold()
    found: set[Clause] = set()
    def entity(phrase: str) -> tuple[str, str] | None:
        values = aliases.get(phrase, set())
        return next(iter(values)) if len(values) == 1 else None
    def pair(a: str, b: str, operator: str, negated: bool = False) -> None:
        left, right = entity(a), entity(b)
        if left and right:
            found.add(Clause(index, text, operator, left, right, negated))
    # Enumerate boundary positions; internal 'is' in an entity is never guessed.
    for ordinal_index, ordinal in enumerate(ORDINALS, 1):
        for negation in (False, True):
            suffix = f" is {'not ' if negation else ''}in the {ordinal} house"
            if body.endswith(suffix) and (left := entity(body[:-len(suffix)])):
                found.add(Clause(index, text, "ordinal_house_reference", left, ordinal_index, negation))
    for prefix, operator in (("there is one house between ", "one_house_between"),
                             ("there are two houses between ", "two_houses_between")):
        if body.startswith(prefix):
            rest = body[len(prefix):]
            for m in re.finditer(" and ", rest):
                pair(rest[:m.start()], rest[m.end():], operator)
    for negation in (False, True):
        suffix = f" are {'not ' if negation else ''}next to each other"
        if body.endswith(suffix):
            rest = body[:-len(suffix)]
            for m in re.finditer(" and ", rest):
                pair(rest[:m.start()], rest[m.end():], "next_to", negation)
    relations = {"directly left of": "directly_left_of", "directly right of": "directly_right_of",
                 "somewhere to the left of": "somewhere_left_of", "somewhere to the right of": "somewhere_right_of",
                 "next to": "next_to", "": "equal"}
    for phrase, operator in relations.items():
        for negation in (False, True):
            boundary = " is " + ("not " if negation else "") + (phrase + " " if phrase else "")
            for m in re.finditer(re.escape(boundary), body):
                pair(body[:m.start()], body[m.end():], operator, negation)
    return sorted(found, key=lambda c: (c.operator, c.left, str(c.right), c.negated))


def applicable_operators(parsed: ParsedPuzzle) -> list[str]:
    return sorted({c.operator for c in parsed.clauses if c.operator != "equal"
                   and not (c.operator == "next_to" and parsed.houses < 3)})


def counterfactual_context(operator: str, houses: int) -> tuple[str, str, str]:
    """Canonical public rule contract, checked alongside the parsed override."""
    definitions = {
        "directly_left_of": ("house(B) - house(A) = 1",
            "house(B) - house(A) = 2" if houses >= 3 else "house(A) - house(B) = 1",
            '"A is directly left of B" means A is exactly two houses to the left of B.' if houses >= 3
            else '"A is directly left of B" means A is directly right of B.'),
        "directly_right_of": ("house(A) - house(B) = 1",
            "house(A) - house(B) = 2" if houses >= 3 else "house(B) - house(A) = 1",
            '"A is directly right of B" means A is exactly two houses to the right of B.' if houses >= 3
            else '"A is directly right of B" means A is directly left of B.'),
        "somewhere_left_of": ("house(A) < house(B)", "house(A) > house(B)",
            '"A is somewhere to the left of B" means A is somewhere to the right of B.'),
        "somewhere_right_of": ("house(A) > house(B)", "house(A) < house(B)",
            '"A is somewhere to the right of B" means A is somewhere to the left of B.'),
        "next_to": ("abs(house(A) - house(B)) = 1", "abs(house(A) - house(B)) = 2",
            '"A is next to B" means exactly one house lies between A and B. '
            '"A is not next to B" means their house-number distance is not 2.'),
        "one_house_between": ("abs(house(A) - house(B)) = 2", "abs(house(A) - house(B)) = 1",
            '"There is one house between A and B" means A and B are adjacent.'),
        "two_houses_between": ("abs(house(A) - house(B)) = 3", "abs(house(A) - house(B)) = 2",
            '"There are two houses between A and B" means exactly one house lies between A and B.'),
    }
    if operator == "ordinal_house_reference":
        original = ", ".join(f"{ORDINALS[i]}=House {i + 1}" for i in range(houses))
        changed = ", ".join(f"{ORDINALS[i]}=House {(i + 1) % houses + 1}" for i in range(houses))
        body = "; ".join(f'"{ORDINALS[i]} house" means House {(i + 1) % houses + 1}'
                         for i in range(houses)) + "."
    else:
        original, changed, body = definitions[operator]
    context = "\n".join(["Temporary counterfactual rule override:", body,
        "Apply this override to every matching clue in this puzzle.",
        "All other relation phrases retain their ordinary meanings.",
        "House labels in the requested answer always retain their ordinary numeric meanings."])
    return context, original, changed


def evaluate_clause(clause: Clause, positions: dict[tuple[str, str], int],
                    houses: int, override: str | None = None) -> bool:
    """Plain-integer cross-check independent of the SMT expression objects."""
    a = positions[clause.left]
    b = clause.right if isinstance(clause.right, int) else positions[clause.right]
    changed = clause.operator == override
    if clause.operator == "equal":
        result = a == b
    elif clause.operator == "ordinal_house_reference":
        expected = 1 if b == houses else b + 1
        result = a == (expected if changed else b)
    elif clause.operator in ("directly_left_of", "directly_right_of"):
        direction = 1 if clause.operator == "directly_left_of" else -1
        distance = (2 if houses >= 3 else -1) if changed else 1
        result = b == a + direction * distance
    elif clause.operator in ("somewhere_left_of", "somewhere_right_of"):
        leftward = clause.operator == "somewhere_left_of"
        if changed:
            leftward = not leftward
        result = a < b if leftward else a > b
    else:
        ordinary = {"next_to": 1, "one_house_between": 2, "two_houses_between": 3}
        altered = {"next_to": 2, "one_house_between": 1, "two_houses_between": 2}
        gap = (altered if changed else ordinary)[clause.operator]
        result = max(a, b) - min(a, b) == gap
    return not result if clause.negated else result


def solve_unique(parsed: ParsedPuzzle, *, override: str | None = None,
                 gold: dict[str, Any] | None = None, timeout_ms: int = 10000) -> dict[str, Any]:
    import z3
    solver = z3.Solver()
    solver.set(timeout=timeout_ms, random_seed=20261013)
    positions = {(header, value): z3.Int(f"p_{category}_{index}")
                 for category, (header, values) in enumerate(parsed.domains)
                 for index, value in enumerate(values)}
    for header, values in parsed.domains:
        variables = [positions[header, v] for v in values]
        solver.add(z3.Distinct(*variables))
        for variable in variables:
            solver.add(variable >= 1, variable <= parsed.houses)
    affected: list[int] = []
    for clause in parsed.clauses:
        left = positions[clause.left]
        right = clause.right if isinstance(clause.right, int) else positions[clause.right]
        op = clause.operator
        changed = op == override
        if changed:
            affected.append(clause.index)
        if op == "equal":
            predicate = left == right
        elif op == "ordinal_house_reference":
            predicate = left == (right % parsed.houses + 1 if changed else right)
        elif op == "directly_left_of":
            predicate = right - left == (2 if parsed.houses >= 3 else -1) if changed else right - left == 1
        elif op == "directly_right_of":
            predicate = left - right == (2 if parsed.houses >= 3 else -1) if changed else left - right == 1
        elif op == "somewhere_left_of":
            predicate = left > right if changed else left < right
        elif op == "somewhere_right_of":
            predicate = left < right if changed else left > right
        else:
            distance = {"next_to": 1, "one_house_between": 2, "two_houses_between": 3}[op]
            if changed:
                distance = {"next_to": 2, "one_house_between": 1, "two_houses_between": 2}[op]
            predicate = z3.Abs(left - right) == distance
        solver.add(z3.Not(predicate) if clause.negated else predicate)
    if override and not affected:
        raise ValidationError("override affects no parsed clauses")
    first = solver.check()
    evidence: dict[str, Any] = {"solver": f"z3 {z3.get_version_string()}", "timeout_ms": timeout_ms,
        "first_check": str(first), "override": override, "affected_clue_indices": affected,
        "clues_parsed": len(parsed.clauses), "domains_from_background": True,
        "gold_asserted": False, "variables": len(positions)}
    if first != z3.sat:
        evidence["status"] = "unsatisfiable" if first == z3.unsat else "solver_unknown"
        if first == z3.unknown:
            evidence["reason"] = solver.reason_unknown()
        return evidence
    model = solver.model()
    assignments = {key: model.eval(variable).as_long() for key, variable in positions.items()}
    solution = {"header": ["House", *(h for h, _ in parsed.domains)],
                "rows": [[str(house), *(next(v for v in values if assignments[header, v] == house)
                                        for header, values in parsed.domains)]
                         for house in range(1, parsed.houses + 1)]}
    if not all(evaluate_clause(c, assignments, parsed.houses, override) for c in parsed.clauses):
        raise ValidationError("SMT solution failed independent integer clause evaluation")
    solver.add(z3.Or(*(variable != assignments[key] for key, variable in positions.items())))
    second = solver.check()
    evidence.update(second_check=str(second), solution=solution, integer_clauses_satisfied=True)
    if second != z3.unsat:
        evidence["status"] = "multiple_solutions" if second == z3.sat else "solver_unknown"
        return evidence
    evidence["status"] = "verified_unique"
    evidence["solution_sha256"] = hashlib.sha256(json.dumps(solution, sort_keys=True).encode()).hexdigest()
    if gold is not None:
        evidence["gold_matches"] = solution == gold
        if solution != gold:
            evidence["status"] = "gold_mismatch"
    return evidence


def verify_record(record: dict[str, Any], *, timeout_ms: int = 10000) -> dict[str, Any]:
    try:
        if not isinstance(record.get("solution"), dict):
            raise ValidationError("scoreable record requires a gold solution object")
        puzzle = record["puzzle"]
        if record.get("rule_mode") == "temporary_codebook":
            mappings = record.get("rule_mapping")
            if not isinstance(mappings, list) or not mappings:
                raise ValidationError("missing codebook mapping")
            seen: set[str] = set()
            for mapping in mappings:
                token, meaning = mapping["token"], mapping["meaning"]
                if not isinstance(token, str) or not re.fullmatch(r"\[R[1-9]\d*\]", token) or token in seen:
                    raise ValidationError("invalid or repeated codebook token")
                if not isinstance(meaning, str) or puzzle.count(token) != mapping["occurrences"]:
                    raise ValidationError("invalid codebook meaning or count")
                seen.add(token)
                puzzle = puzzle.replace(token, meaning)
            expected_context = "\n".join([
                "The bracketed tokens below are temporary textual macros for this puzzle only.",
                "Expand each token to its quoted meaning before applying the clues.",
                "Do not reuse a token meaning from any other puzzle.", "",
                *(f'- {mapping["token"]} = "{mapping["meaning"]}"' for mapping in mappings)])
            if record.get("rule_context") != expected_context:
                raise ValidationError("codebook public context disagrees with the decoded mapping")
            if re.search(r"\[R\d+\]", puzzle):
                raise ValidationError("unresolved codebook token")
        if record.get("capability") == "history_memory":
            clues = record.get("clue_turns")
            if not isinstance(clues, list) or not all(isinstance(c, str) for c in clues):
                raise ValidationError("invalid history clues")
            puzzle += "\n\n## Clues:\n" + "\n".join(f"{i}. {c}" for i, c in enumerate(clues, 1))
        parsed = parse_puzzle(puzzle, expected_size=record["size"])
        override = None
        if record.get("rule_mode") == "counterfactual_semantics":
            override = record["counterfactual_rule"]["operator"]
            if override not in applicable_operators(parsed):
                raise ValidationError("unsupported or inapplicable counterfactual operator")
            context, ordinary, altered = counterfactual_context(override, parsed.houses)
            if (record.get("rule_context") != context or
                    record["counterfactual_rule"].get("original_semantics") != ordinary or
                    record["counterfactual_rule"].get("new_semantics") != altered):
                raise ValidationError("counterfactual textual rule contract mismatch")
        result = solve_unique(parsed, override=override, gold=record["solution"], timeout_ms=timeout_ms)
        if override:
            if result["affected_clue_indices"] != record["counterfactual_rule"]["affected_clue_indices"]:
                raise ValidationError("counterfactual affected-clue metadata mismatch")
            if result.get("solution") == record.get("original_solution"):
                raise ValidationError("counterfactual gold is unchanged")
        return result
    except (ValidationError, KeyError, TypeError) as exc:
        return {"status": "parse_rejected", "reason": str(exc), "gold_asserted": False}
