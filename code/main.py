"""Buy or Wait? - produce output.csv for every request in dataset/requests.csv.

    python code/main.py              # uses the disk cache, calls the API only for cache misses
    python code/main.py --dry-run    # cache only, zero API calls (evidence falls back to neutral)

Writes:
    <repo>/output.csv                          exactly 250 rows, 8 columns, validated
    code/evaluation/usage_report.md            model calls, tokens, cost for THIS run
    code/evaluation/failures.csv               one row per request that hit an exception
    <repo>/log.txt                             run record + requests flagged for review
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import logger  # noqa: E402
import schema  # noqa: E402
from extraction import PRICE_PER_MTOK, Extractor  # noqa: E402
from io_layer import load_dataset  # noqa: E402
from pipeline import Evidence, decide, gather_evidence  # noqa: E402

OUTPUT_PATH = os.path.join(ROOT, "output.csv")
FAILURES_PATH = os.path.join(HERE, "evaluation", "failures.csv")
USAGE_PATH = os.path.join(HERE, "evaluation", "usage_report.md")
PRICING_DATE = "2026-09-12"      # date the PRICE_PER_MTOK constants were checked against Anthropic's public pricing


def safe_row(request, profile, reason: str) -> dict:
    """Deterministic conservative row used when the pipeline fails for a request."""
    short = reason.strip().splitlines()[-1][:80] if reason.strip() else "unknown error"
    return {"request_id": request.request_id, "amount_safe_to_pay": "0", "affordability_status": "not_affordable",
            "recommended_payment_method": "not_recommended", "payment_plan": "none",
            "earliest_date_for_full_payment": "", "spending_changes_needed": "none",
            "decision_explanation": f"Do not make this payment. The evaluation could not be completed ({short})."}


def write_usage_report(stats: dict, n_requests: int, mode: str, wall: float, validation_failures: int,
                       pipeline_failures: int) -> None:
    calls, cache_hits = stats["calls"], stats["cache_hits"]
    in_tok, out_tok = stats["input_tokens"], stats["output_tokens"]
    cin, cout = stats["cached_input_tokens"], stats["cached_output_tokens"]
    total_live = in_tok + out_tok
    cost_live = Decimal(stats["estimated_cost_usd"])
    cost_cached = Decimal(0)
    for model, m in stats["by_model"].items():
        pin, pout = PRICE_PER_MTOK.get(model, (Decimal(0), Decimal(0)))
        cost_cached += (Decimal(m["cached_input_tokens"]) * pin + Decimal(m["cached_output_tokens"]) * pout) / Decimal(1_000_000)
    lines = [
        "# Token usage and cost report",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by `code/main.py` "
        f"for the full-dataset run that produced `output.csv` ({n_requests} requests, mode = **{mode}**).",
        "",
        "## Providers and models",
        "",
        "| provider | model | role |",
        "|---|---|---|",
    ]
    for model in stats["by_model"] or {stats["model"]: None}:
        lines.append(f"| Anthropic (Claude API) | `{model}` | message classification (call 1) and image reading (call 2) |")
    lines += [
        "",
        "Everything numeric in `output.csv` is computed deterministically in Python; the model only reads "
        "amounts, dates, currencies and intent off untrusted messages and images.",
        "",
        "## Pricing constants",
        "",
        f"USD per 1M tokens, as checked on {PRICING_DATE} against Anthropic's public price list:",
        "",
        "| model | input $/1M | output $/1M |",
        "|---|---|---|",
    ]
    for model, (pin, pout) in PRICE_PER_MTOK.items():
        lines.append(f"| `{model}` | {pin} | {pout} |")
    lines += [
        "",
        "## This run (live API calls)",
        "",
        "| metric | value |",
        "|---|---|",
        f"| model calls made | {calls} |",
        f"| cache hits (no call) | {cache_hits} |",
        f"| re-asks after schema rejection | {stats['reasks']} |",
        f"| fallbacks to the neutral default | {stats['fallbacks']} |",
        f"| retries (429/5xx backoff) | {stats['retries']} |",
        f"| refusals | {stats['refusals']} |",
        f"| injection flags | {stats['injections_flagged']} |",
        f"| input tokens | {in_tok:,} |",
        f"| output tokens | {out_tok:,} |",
        f"| total tokens | {total_live:,} |",
        f"| average tokens per request ({n_requests}) | {total_live / n_requests:,.1f} |",
        f"| average tokens per model call | {(total_live / calls):,.1f} |" if calls else "| average tokens per model call | n/a (no live calls) |",
        f"| estimated total cost (USD) | {cost_live:.4f} |",
        f"| estimated cost per request (USD) | {(cost_live / n_requests):.6f} |",
        f"| wall time (s) | {wall:.1f} |",
        f"| rows failing validation | {validation_failures} |",
        f"| rows produced by the failure fallback | {pipeline_failures} |",
        "",
        "## Per-model totals",
        "",
        "| model | live calls | cache hits | live input | live output | cached input | cached output | live cost (USD) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model, m in stats["by_model"].items():
        pin, pout = PRICE_PER_MTOK.get(model, (Decimal(0), Decimal(0)))
        c = (Decimal(m["input_tokens"]) * pin + Decimal(m["output_tokens"]) * pout) / Decimal(1_000_000)
        lines.append(f"| `{model}` | {m['calls']} | {m['cache_hits']} | {m['input_tokens']:,} | {m['output_tokens']:,} | "
                     f"{m['cached_input_tokens']:,} | {m['cached_output_tokens']:,} | {c:.4f} |")
    lines += [
        "",
        "## Cached responses replayed in this run",
        "",
        "Responses served from `.cache/` were produced by earlier live calls. Their recorded usage is "
        "reported here so the cost of the answers used by this run is visible even when the run itself made no calls.",
        "",
        "| metric | value |",
        "|---|---|",
        f"| cached input tokens | {cin:,} |",
        f"| cached output tokens | {cout:,} |",
        f"| cost of producing the cached answers (USD) | {cost_cached:.4f} |",
        f"| overall (live + cached) tokens | {total_live + cin + cout:,} |",
        f"| overall (live + cached) cost (USD) | {(cost_live + cost_cached):.4f} |",
        f"| overall cost per request (USD) | {((cost_live + cost_cached) / n_requests):.6f} |",
        "",
        "No API keys, credentials, or configuration values are included in this report.",
        "",
    ]
    os.makedirs(os.path.dirname(USAGE_PATH), exist_ok=True)
    with open(USAGE_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Buy or Wait? pipeline")
    ap.add_argument("--dry-run", action="store_true", help="use only the disk cache; make no API calls")
    ap.add_argument("--workers", type=int, default=8, help="bounded concurrency for extraction")
    ap.add_argument("--no-evidence", action="store_true", help="ignore messages and images entirely")
    args = ap.parse_args(argv)
    mode = "dry-run" if args.dry_run else ("no-evidence" if args.no_evidence else "live")

    t0 = time.perf_counter()
    logger.append_session_start(mode)
    ds = load_dataset()

    # ---- evidence, up front, concurrently
    extractor = None
    evidence = Evidence()
    if not args.no_evidence:
        extractor = Extractor(dry_run=args.dry_run or not os.environ.get("ANTHROPIC_API_KEY"))
        from extraction import run_all  # noqa: F401  (imported here so --no-evidence never touches the SDK)
        evidence = gather_evidence_concurrent(ds, extractor, args.workers)

    # ---- requests
    rows, failures, review = [], [], []
    validation_failures = 0
    for request in ds.requests:
        profile = ds.profile_by_user.get(request.user_id)
        try:
            if profile is None:
                raise KeyError(f"no profile for {request.user_id}")
            out = decide(ds, request, evidence)
            if out.error:
                raise RuntimeError(out.error)
            if out.validation_error:
                validation_failures += 1
                raise schema.ValidationError(out.validation_error)
            row = out.row
            if out.review:
                review.append((request.request_id, out.review))
        except Exception:  # noqa: BLE001 - one bad request must not kill the batch
            tb = traceback.format_exc()
            failures.append({"request_id": request.request_id, "user_id": request.user_id,
                             "error": tb.strip().splitlines()[-1], "traceback": tb})
            row = safe_row(request, profile, tb) if profile else {
                "request_id": request.request_id, "amount_safe_to_pay": "0", "affordability_status": "not_affordable",
                "recommended_payment_method": "not_recommended", "payment_plan": "none",
                "earliest_date_for_full_payment": "", "spending_changes_needed": "none",
                "decision_explanation": "Do not make this payment. The evaluation could not be completed (no profile)."}
            review.append((request.request_id, ["pipeline_failure: " + failures[-1]["error"][:120]]))
        rows.append(row)

    # ---- every row is re-validated as written (the fallback rows too)
    for row in rows:
        request = ds.request_by_id[row["request_id"]]
        profile = ds.profile_by_user.get(request.user_id)
        if profile is None:
            continue
        from pipeline import _event_dict, _option_dict, _profile_dict, _request_dict
        try:
            schema.validate_row(row, _request_dict(request), [_option_dict(o) for o in ds.request_options(request.request_id)],
                                _profile_dict(profile), [_event_dict(e) for e in ds.user_events(request.user_id)])
        except schema.ValidationError as err:
            validation_failures += 1
            failures.append({"request_id": row["request_id"], "user_id": request.user_id,
                             "error": f"final validation: {err}", "traceback": ""})
            rows[rows.index(row)] = safe_row(request, profile, str(err))

    # ---- output.csv at the repository root
    assert len(rows) == len(ds.requests), f"{len(rows)} rows for {len(ds.requests)} requests"
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=schema.OUTPUT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in schema.OUTPUT_COLUMNS})

    os.makedirs(os.path.dirname(FAILURES_PATH), exist_ok=True)
    with open(FAILURES_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["request_id", "user_id", "error", "traceback"])
        w.writeheader()
        w.writerows(failures)

    # ---- summary
    wall = time.perf_counter() - t0
    stats = extractor.stats.to_dict() if extractor else Extractor(dry_run=True).stats.to_dict()
    n = len(ds.requests)
    print(f"output.csv: {n} rows -> {OUTPUT_PATH}")
    print(f"mode={mode} wall_time={wall:.1f}s")
    print("model calls by model: " + (", ".join(f"{m}={v['calls']}" for m, v in stats["by_model"].items()) or "none"))
    print(f"input_tokens={stats['input_tokens']:,} output_tokens={stats['output_tokens']:,} "
          f"(cached replay: in={stats['cached_input_tokens']:,} out={stats['cached_output_tokens']:,})")
    print(f"cache_hits={stats['cache_hits']} reasks={stats['reasks']} fallbacks={stats['fallbacks']} "
          f"retries={stats['retries']} injection_flags={stats['injections_flagged']}")
    print(f"validation_failures={validation_failures} pipeline_failures={len(failures)} "
          f"requests_flagged_for_review={len(review)}")
    print(f"status counts: {dict(Counter(r['affordability_status'] for r in rows))}")
    print(f"estimated live cost USD {Decimal(stats['estimated_cost_usd']):.4f}")

    write_usage_report(stats, n, mode, wall, validation_failures, len(failures))
    logger.append_run_summary({"rows": n, "mode": mode, "calls": stats["calls"], "cache_hits": stats["cache_hits"],
                               "reasks": stats["reasks"], "fallbacks": stats["fallbacks"],
                               "injections": stats["injections_flagged"], "validation_failures": validation_failures,
                               "wall_time_s": round(wall, 1)}, review, [f["request_id"] for f in failures])
    return 0


def gather_evidence_concurrent(ds, extractor, workers: int) -> Evidence:
    """gather_evidence with the extraction pass run under a bounded thread pool."""
    import extraction
    original = extraction.run_all
    extraction.run_all = lambda ds_, x_: original(ds_, x_, workers=workers)
    try:
        return gather_evidence(ds, extractor)
    finally:
        extraction.run_all = original


if __name__ == "__main__":
    sys.exit(main())
