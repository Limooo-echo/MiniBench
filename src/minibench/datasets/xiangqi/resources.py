"""Xiangqi-only resource accounting; unknown usage/prices never mean free."""
from __future__ import annotations

import math
from typing import Any
from datetime import datetime

from minibench.core import metrics as core
from minibench.core.token_usage import assess_token_usage


def validate_pricing(pricing: dict | None) -> dict | None:
    if pricing is None:
        return None
    if not isinstance(pricing, dict):
        raise ValueError("pricing must be an object or null")
    allowed = {"currency", "input_per_million", "output_per_million",
               "cached_input_per_million", "source", "as_of", "model"}
    if set(pricing) - allowed:
        raise ValueError("unsupported pricing field")
    if not isinstance(pricing.get("currency"), str) or not pricing["currency"].strip():
        raise ValueError("pricing.currency is required")
    for key in ("input_per_million", "output_per_million", "cached_input_per_million"):
        value = pricing.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value < 0):
            raise ValueError(f"invalid pricing.{key}")
    return dict(pricing)


def estimate_cost(usage: dict, pricing: dict | None, *, missing_calls: int = 0) -> dict:
    pricing = validate_pricing(pricing)
    result = {"amount": None, "known_amount": None, "currency": None,
              "complete": False, "basis": "provider_reported_usage; not an invoice"}
    if pricing is None or any(pricing.get(k) is None for k in
                              ("input_per_million", "output_per_million")):
        return {**result, "reason": "pricing_not_configured"}
    result["currency"] = pricing["currency"]
    assessment = assess_token_usage(usage)
    if assessment["status"] == "invalid":
        return {**result, "reason": "invalid_provider_usage", "usage_issues": assessment["issues"]}
    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if any(isinstance(n, bool) or not isinstance(n, (int, float))
           or not math.isfinite(n) or n < 0
           for n in (prompt, completion)):
        return {**result, "reason": "provider_token_breakdown_missing"}
    total = usage.get("total_tokens")
    if total is not None and total != prompt+completion:
        return {**result, "reason": "token_totals_do_not_match_breakdown"}
    cached = usage.get("cached_tokens", usage.get("prompt_cache_hit_tokens", 0))
    if isinstance(cached, bool) or not isinstance(cached, (int, float)) or not 0 <= cached <= prompt:
        return {**result, "reason": "invalid_cached_token_count"}
    if cached and pricing.get("cached_input_per_million") is None:
        return {**result, "reason": "cached_input_price_not_configured"}
    # Completion tokens already include reasoning tokens. Never add them twice.
    cache_price = pricing.get("cached_input_per_million")
    if cache_price is None:
        cache_price = pricing["input_per_million"]  # Used only when cached == 0.
    amount = ((prompt-cached)*pricing["input_per_million"]
              + cached*cache_price
              + completion*pricing["output_per_million"])/1_000_000
    result.update(known_amount=round(amount, 10), pricing=pricing)
    if missing_calls:
        return {**result, "reason": "some_request_usage_unknown"}
    return {**result, "amount": round(amount, 10), "complete": True, "reason": None}


def start_task_metrics(agent: Any, *, pricing: dict | None = None) -> dict:
    start = core.start_task_metrics(agent)
    start["_xiangqi_pricing"] = validate_pricing(pricing)
    return start


def finish_task_metrics(agent: Any, start: dict) -> dict:
    result = core.finish_task_metrics(agent, start)
    if "_xiangqi_final_move_timeout_seconds" in start:
        result["final_move_timeout_seconds"] = start["_xiangqi_final_move_timeout_seconds"]
    missing = int(result["usage_missing_calls"])
    if result["llm_calls"] == 0:
        missing = max(missing, 1)  # Agents without metering are unknown, not free.
    if result["llm_calls"] and not result["token_usage"].get("total_tokens"):
        missing = max(missing, 1)  # Empty/missing provider usage must not become a free call.
    result["cost"] = estimate_cost(result.get("known_token_usage") or {}, start.get("_xiangqi_pricing"),
                                   missing_calls=missing)
    result["time_basis"] = "task wall duration; model duration includes all agent stages"
    return result


def summarize_metrics(results: list[Any]) -> dict:
    summary = core.summarize_metrics(results)
    costs = [((r.get("metrics") if isinstance(r, dict) else getattr(r, "metrics", {})) or {})
             .get("cost", {}) for r in results]
    currencies = {c.get("currency") for c in costs if c.get("currency")}
    complete = bool(costs) and len(currencies) == 1 and all(c.get("complete") for c in costs)
    known = sum(c.get("known_amount") or 0 for c in costs) if len(currencies) == 1 else None
    summary["total"]["cost"] = {
        "amount": round(known, 10) if complete else None,
        "known_amount": round(known, 10) if known is not None else None,
        "currency": next(iter(currencies)) if len(currencies) == 1 else None,
        "complete": complete, "unknown_conditions": sum(not c.get("complete") for c in costs),
        "note": "No provider price or unknown usage => unknown money, not zero.",
    }
    summary["time_semantics"] = (
        "Sum of episode/model durations, not concurrent batch elapsed time; "
        "cloud physical-attempt ledger is authoritative for replay/retry billing."
    )
    return summary


def physical_bucket():
    return {"physical_attempts":0, "attempts_with_unknown_usage":0,
            "token_usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0},
            "known_cost":0.0, "unknown_cost_attempts":0, "currencies":set(),
            "known_model_seconds":0.0, "attempts_with_unknown_time":0,
            "attempts_with_invalid_usage":0, "attempts_with_known_usage":0,
            "usage_issues":{}, "known_token_fields":{}}


def assess_attempt_usage(attempt):
    """Revalidate durable raw evidence, including old journals, without editing it."""
    assessment = assess_token_usage(
        attempt.get("usage"),
        reasoning_chars=attempt.get("reasoning_chars", len(attempt.get("reasoning") or "")),
        content_chars=attempt.get("content_chars", len(attempt.get("content") or "")),
    )
    recorded = attempt.get("usage_validation") or {}
    if recorded.get("status") == "invalid":
        assessment.update(status="invalid", complete=False, tokens=None, total_tokens=None,
                          issues=list(dict.fromkeys(assessment["issues"] + recorded.get("issues", []))))
    return assessment


def add_physical_attempt(bucket, attempt, pricing):
    """Called once per durable physical attempt, never on a model-call replay."""
    bucket["physical_attempts"] += 1
    assessment = assess_attempt_usage(attempt)
    usage = assessment["tokens"]
    if usage is not None:
        for key,value in usage.items():
            bucket["token_usage"][key] = bucket["token_usage"].get(key,0)+value
            bucket["known_token_fields"][key] = bucket["known_token_fields"].get(key,0)+1
    if not assessment["complete"]:
        bucket["attempts_with_unknown_usage"] += 1
        cost = {"complete":False}
    else:
        cost = estimate_cost(usage,pricing)
    if assessment["status"] == "invalid":
        bucket["attempts_with_invalid_usage"] += 1
    if assessment["total_tokens"] is not None:
        bucket["attempts_with_known_usage"] += 1
    for issue in assessment["issues"]:
        bucket["usage_issues"][issue] = bucket["usage_issues"].get(issue,0)+1
    if cost.get("currency"):
        bucket["currencies"].add(cost["currency"])
    if cost.get("known_amount") is not None:
        bucket["known_cost"] += cost["known_amount"]
    if not cost.get("complete"):
        bucket["unknown_cost_attempts"] += 1
    elapsed = attempt.get("elapsed_seconds")
    if elapsed is None and attempt.get("started_at") and attempt.get("at") and attempt["status"]=="success":
        elapsed = (datetime.fromisoformat(attempt["at"].replace("Z","+00:00"))
                   - datetime.fromisoformat(attempt["started_at"].replace("Z","+00:00"))).total_seconds()
    if isinstance(elapsed,(int,float)) and not isinstance(elapsed,bool) and elapsed>=0:
        bucket["known_model_seconds"] += elapsed
    else:
        bucket["attempts_with_unknown_time"] += 1


def finish_physical_bucket(bucket):
    result = dict(bucket)
    observed_fields = result.pop("known_token_fields")
    known_usage = {key:(value if observed_fields.get(key) else None)
                   for key,value in result["token_usage"].items()}
    result["usage_complete"] = (result["physical_attempts"] > 0
                                and result["attempts_with_unknown_usage"] == 0)
    result["known_token_usage"] = (known_usage if result["attempts_with_known_usage"]
                                   else {k:None for k in known_usage})
    if not result["usage_complete"]:
        result["token_usage"] = {k:None for k in known_usage}
    currencies = result.pop("currencies")
    known = round(result.pop("known_cost"),10)
    complete = result["unknown_cost_attempts"]==0 and len(currencies)==1
    result["cost"] = {"amount":known if complete else None,
                      "known_amount":known if len(currencies)==1 else None,
                      "currency":next(iter(currencies)) if len(currencies)==1 else None,
                      "complete":complete}
    result["basis"] = "physical journal attempts; replay excluded; unknown is not zero"
    return result


def resource_summary_lines(metrics):
    total=metrics["total"]
    cost=total.get("cost",{})
    amount=cost.get("amount")
    money="unknown" if amount is None else f"{amount:.8f} {cost['currency']}"
    tokens=total['token_usage'].get('total_tokens')
    known=total.get('known_token_usage',{}).get('total_tokens')
    shown='unknown' if tokens is None else str(tokens)
    subtotal='unknown' if known is None else str(known)
    return [
        f"Tokens: {shown}; validated subtotal: {subtotal}; "
        f"calls with unknown usage: {total['usage_missing_calls']}",
        f"Time: episode sum={total['task_elapsed_seconds']:.3f}s; "
        f"model sum={total['model_elapsed_seconds']:.3f}s (not concurrent wall time)",
        f"Estimated API cost: {money} (not an invoice; excludes server rental)",
    ]
