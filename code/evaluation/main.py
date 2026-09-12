"""Score the pipeline against dataset/sample_requests.csv.

    python code/evaluation/main.py --note "what changed"

Appends one row to code/evaluation/run_log.md, writes code/evaluation/errors_<run>.csv (one row per
mismatch with the solver's intermediate values) and code/evaluation/results_<run>.json (per-request
correctness, used to detect regressions on the next run).

Model evidence: when ANTHROPIC_API_KEY is set, or the .cache/ is warm, message and image evidence is
used; otherwise the run is evidence-free and says so in the note. Pass --no-evidence to force that.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)

import schema  # noqa: E402
from io_layer import load_dataset  # noqa: E402
from pipeline import candidate_table, decide, gather_evidence  # noqa: E402

RUN_LOG = os.path.join(HERE, "run_log.md")
STATUSES = ["affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"]
EXACT_FIELDS = ["affordability_status", "recommended_payment_method", "earliest_date_for_full_payment", "payment_plan"]
METRIC_COLUMNS = ["status_acc", "method_acc", "earliest_acc", "plan_acc", "amt_mae", "amt_mean_rel", "amt_within_1pct",
                  "changes_jaccard", "status_macro_f1", "all_fields_correct", "valid_rows"]


def jaccard(a: set, b: set) -> Decimal:
    if not a and not b:
        return Decimal(1)
    return Decimal(len(a & b)) / Decimal(len(a | b))


def relative_error(pred: Decimal, expected: Decimal, requested: Decimal) -> Decimal:
    """|pred - expected| over expected, or over requested_amount when expected is zero."""
    denom = expected if expected > 0 else requested
    return abs(pred - expected) / denom if denom > 0 else Decimal(0)


def macro_f1(pairs, labels):
    f1s = []
    for lab in labels:
        tp = sum(1 for p, e in pairs if p == lab and e == lab)
        fp = sum(1 for p, e in pairs if p == lab and e != lab)
        fn = sum(1 for p, e in pairs if p != lab and e == lab)
        if tp + fp + fn == 0:
            continue                               # label absent from both sides: skip, do not reward
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def previous_run():
    runs = sorted(int(m.group(1)) for f in os.listdir(HERE) for m in [re.match(r"results_(\d+)\.json$", f)] if m)
    if not runs:
        return 0, {}
    with open(os.path.join(HERE, f"results_{runs[-1]}.json"), encoding="utf-8") as f:
        return runs[-1], json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", required=True, help="one line: what changed since the last run")
    ap.add_argument("--no-evidence", action="store_true", help="ignore messages and images entirely")
    args = ap.parse_args(argv)

    ds = load_dataset()
    evidence_note = "no evidence"
    extractor = None
    if not args.no_evidence:
        from extraction import Extractor
        extractor = Extractor(dry_run=not os.environ.get("ANTHROPIC_API_KEY"))
    evidence = gather_evidence(ds, extractor)
    if extractor is not None:
        s = extractor.stats
        used = sum(len(v) for v in evidence.verdicts_by_user.values())
        evidence_note = (f"evidence: {used} verdicts, {len(evidence.facts_by_event)} image facts "
                         f"(api_calls={s.calls} cache_hits={s.cache_hits} fallbacks={s.fallbacks})")

    prev_run, prev_results = previous_run()
    run = prev_run + 1
    rows, pairs, per_request, errors = [], [], {}, []
    amt_abs, amt_rel, within = [], [], 0
    jac = []
    exact = Counter()
    valid = 0

    for r in ds.sample_requests:
        out = decide(ds, r, evidence)
        a = r.answer
        expected = {
            "amount_safe_to_pay": schema.format_amount_safe(a.amount_safe_to_pay),
            "affordability_status": a.affordability_status,
            "recommended_payment_method": a.recommended_payment_method,
            "payment_plan": a.payment_plan,
            "earliest_date_for_full_payment": a.earliest_date_for_full_payment.isoformat() if a.earliest_date_for_full_payment else "",
            "spending_changes_needed": a.spending_changes_needed,
        }
        got = out.row
        field_ok = {f: got[f] == expected[f] for f in EXACT_FIELDS}
        for f, ok in field_ok.items():
            exact[f] += ok
        pred_amt, exp_amt = Decimal(got["amount_safe_to_pay"]), a.amount_safe_to_pay
        abs_err = abs(pred_amt - exp_amt)
        rel = relative_error(pred_amt, exp_amt, r.requested_amount)
        amt_abs.append(abs_err / r.requested_amount)      # normalised so currencies are comparable
        amt_rel.append(rel)
        amt_ok = rel <= Decimal("0.01")
        within += amt_ok
        j = jaccard(set(schema.parse_spending_changes(got["spending_changes_needed"]) and got["spending_changes_needed"].split("|")),
                    set(schema.parse_spending_changes(expected["spending_changes_needed"]) and expected["spending_changes_needed"].split("|")))
        jac.append(j)
        pairs.append((got["affordability_status"], expected["affordability_status"]))
        valid += out.ok
        all_ok = all(field_ok.values()) and amt_ok and j == 1
        per_request[r.request_id] = {"all_ok": all_ok, **field_ok, "amount_ok": amt_ok, "changes_ok": j == 1}
        rows.append((r.request_id, got, expected, all_ok))
        mism = [f for f, ok in field_ok.items() if not ok] + (["amount_safe_to_pay"] if not amt_ok else []) + \
               (["spending_changes_needed"] if j < 1 else [])
        for f in mism:
            errors.append({
                "request_id": r.request_id, "field": f, "predicted": got[f], "expected": expected[f],
                "amount_safe_pred": got["amount_safe_to_pay"], "amount_safe_expected": expected["amount_safe_to_pay"],
                "earliest_pred": got["earliest_date_for_full_payment"], "earliest_expected": expected["earliest_date_for_full_payment"],
                "runner_up_rule": (out.facts.runner_up_rule if out.facts else ""),
                "candidates": candidate_table(out.decision),
                "validation_error": out.validation_error, "pipeline_error": out.error.strip().splitlines()[-1] if out.error else "",
                "explanation": got["decision_explanation"],
            })

    n = len(ds.sample_requests)
    conf = {(p, e): 0 for p in STATUSES for e in STATUSES}
    for p, e in pairs:
        conf[(p, e)] += 1
    metrics = {
        "status_acc": exact["affordability_status"] / n,
        "method_acc": exact["recommended_payment_method"] / n,
        "earliest_acc": exact["earliest_date_for_full_payment"] / n,
        "plan_acc": exact["payment_plan"] / n,
        "amt_mae": float(sum(amt_abs) / n),
        "amt_mean_rel": float(sum(amt_rel) / n),
        "amt_within_1pct": within / n,
        "changes_jaccard": float(sum(jac) / n),
        "status_macro_f1": macro_f1(pairs, STATUSES),
        "all_fields_correct": sum(1 for *_, ok in rows if ok) / n,
        "valid_rows": valid / n,
    }
    regressed = sorted(rid for rid, res in per_request.items()
                       if prev_results.get(rid, {}).get("all_ok") and not res["all_ok"])
    improved = sorted(rid for rid, res in per_request.items()
                      if prev_results and not prev_results.get(rid, {}).get("all_ok") and res["all_ok"])

    # ---- persist
    with open(os.path.join(HERE, f"results_{run}.json"), "w", encoding="utf-8") as f:
        json.dump(per_request, f, indent=1)
    err_path = os.path.join(HERE, f"errors_{run}.csv")
    with open(err_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(errors[0].keys()) if errors else ["request_id"])
        w.writeheader()
        w.writerows(errors)
    new_log = not os.path.exists(RUN_LOG)
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        if new_log:
            f.write("# Evaluation runs (dataset/sample_requests.csv, 25 rows)\n\n"
                    "amt_mae is mean |pred-expected| / requested_amount; amt_mean_rel is mean |pred-expected| / expected "
                    "(over requested_amount when expected is 0); within_1pct uses the same relative error.\n\n"
                    "| run | timestamp | note | " + " | ".join(METRIC_COLUMNS) + " | regressed | improved |\n"
                    "|---|---|---|" + "---|" * len(METRIC_COLUMNS) + "---|---|\n")
        f.write(f"| {run} | {datetime.now(timezone.utc).isoformat(timespec='seconds')} | {args.note} ({evidence_note}) | "
                + " | ".join(f"{metrics[c]:.3f}" for c in METRIC_COLUMNS)
                + f" | {', '.join(regressed) or '-'} | {', '.join(improved) or '-'} |\n")

    # ---- print
    print(f"run {run}: {args.note} ({evidence_note})")
    print(f"{'metric':22}{'value':>8}")
    for c in METRIC_COLUMNS:
        print(f"{c:22}{metrics[c]:8.3f}")
    print("\nconfusion (rows=predicted, cols=expected):")
    short = {s: s.replace("affordable_", "").replace("not_affordable", "not_aff") for s in STATUSES}
    print(f"{'':16}" + "".join(f"{short[e]:>10}" for e in STATUSES))
    for p in STATUSES:
        print(f"{short[p]:16}" + "".join(f"{conf[(p, e)]:>10}" for e in STATUSES))
    print(f"\nregressed: {regressed or '-'}   improved: {improved or '-'}")
    print(f"errors: {len(errors)} mismatches -> {os.path.relpath(err_path, CODE)}")
    if evidence.review_flags:
        flagged = Counter(reason.split(':')[0] for _, _, reason in evidence.review_flags)
        print(f"review flags: {dict(flagged)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
