"""Score the pipeline against dataset/sample_requests.csv.

    python code/evaluation/main.py --note "what changed"

Appends one row to code/evaluation/run_log.md, writes code/evaluation/errors_<run>.csv (one row per
mismatch with the solver's intermediate values) and code/evaluation/results_<run>.json (per-request
correctness, used to detect regressions on the next run).

Model evidence: when GEMINI_API_KEY is set, or the .cache/ is warm, message and image evidence is
used; otherwise the run is evidence-free and says so in the note. Pass --no-evidence to force that.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
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
                  "amt_within_10pct", "changes_jaccard", "status_macro_f1", "all_fields_correct", "valid_rows"]

# Stop criteria. within-1% was retired after the 2D sweep showed no cell in the parameter space
# exceeds 7/25, i.e. the target was unreachable by calibration; amt_mae (smooth) and within-10%
# (the error body) replace it. Counts are out of 25.
STOP_CRITERIA = [
    ("rows pass validate_row", lambda m, n: m["valid_rows"] * n, 25, "ge"),
    ("affordability_status", lambda m, n: m["status_acc"] * n, 22, "ge"),
    ("recommended_payment_method", lambda m, n: m["method_acc"] * n, 22, "ge"),
    ("earliest_date_for_full_payment", lambda m, n: m["earliest_acc"] * n, 22, "ge"),
    ("payment_plan where method matches", lambda m, n: m["plan_given_method"], 1.0, "ge"),
    ("amt_mae (<= run 18)", lambda m, n: m["amt_mae"], 0.103, "le"),
    ("amount within 10%", lambda m, n: m["amt_within_10pct"] * n, 16, "ge"),
    ("golden set, zero regressions", lambda m, n: m["golden_ok"], 1.0, "ge"),
]


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


BASELINE_PATH = os.path.join(HERE, "baseline.json")


def all_runs():
    return sorted(int(m.group(1)) for f in os.listdir(HERE) for m in [re.match(r"results_(\d+)\.json$", f)] if m)


def load_results(run: int) -> dict:
    with open(os.path.join(HERE, f"results_{run}.json"), encoding="utf-8") as f:
        return json.load(f)


def baseline_run() -> tuple[int, dict]:
    """The last run marked KEPT (via --keep). Regressions are measured against it, never against
    the previous run file - a reverted experiment must not become the yardstick."""
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH, encoding="utf-8") as f:
            run = int(json.load(f).get("baseline_run", 0))
        if run and os.path.exists(os.path.join(HERE, f"results_{run}.json")):
            return run, load_results(run)
    return 0, {}


def set_baseline(run: int) -> None:
    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump({"baseline_run": run,
                   "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}, f, indent=1)


GOLDEN_DIR = os.path.join(HERE, "golden")


def run_golden() -> list[dict]:
    """Run every fixture under golden/ through the real pipeline with stubbed model evidence.
    Returns one result dict per fixture with a `passed` flag and the failed checks."""
    from pipeline import evidence_from_stub
    results = []
    for name in sorted(os.listdir(GOLDEN_DIR)):
        d = os.path.join(GOLDEN_DIR, name)
        if not os.path.isdir(d):
            continue
        with open(os.path.join(d, "golden.json"), encoding="utf-8") as f:
            golden = json.load(f)
        failures = []
        try:
            ds = load_dataset(d)
            evidence = evidence_from_stub(ds, golden.get("evidence", {}))
            r = ds.sample_requests[0]
            out = decide(ds, r, evidence)
            a = r.answer
            expected = {"amount_safe_to_pay": schema.format_amount_safe(a.amount_safe_to_pay),
                        "affordability_status": a.affordability_status, "recommended_payment_method": a.recommended_payment_method,
                        "payment_plan": a.payment_plan,
                        "earliest_date_for_full_payment": a.earliest_date_for_full_payment.isoformat() if a.earliest_date_for_full_payment else "",
                        "spending_changes_needed": a.spending_changes_needed, "decision_explanation": a.decision_explanation}
            for fld, want in expected.items():
                if out.row[fld] != want:
                    failures.append(f"{fld}: got {out.row[fld]!r} want {want!r}")
            if out.error:
                failures.append("pipeline crashed: " + out.error.strip().splitlines()[-1])
            if out.validation_error:
                failures.append("validation: " + out.validation_error)
            if golden.get("review_expected") and not out.review:
                failures.append("expected a review flag, got none")
            if not golden.get("review_expected") and out.review:
                failures.append(f"unexpected review flags: {out.review}")
            needle = golden.get("review_contains")
            if needle and not any(needle in x for x in out.review):
                failures.append(f"review flags {out.review} do not mention {needle!r}")
            for eid, reason in golden.get("excluded", {}).items():
                x = out.ledger.exclusion_for(eid) if out.ledger else None
                if x is None or not x.reason.startswith(reason):
                    failures.append(f"{eid}: expected exclusion {reason!r}, got {x.reason if x else None!r}")
            needle = golden.get("conflict_contains")
            if needle and not any(needle in c.event_ids for c in (out.ledger.conflicts if out.ledger else [])):
                failures.append(f"no conflict recorded for {needle}")
            total = golden.get("candidate_total_paid")
            if total and (out.decision is None or out.decision.winner is None or str(out.decision.winner.total_paid) != total):
                failures.append(f"winner total_paid != {total}")
            from pipeline import _event_dict, _option_dict, _profile_dict, _request_dict
            for bad in golden.get("invalid_rows", []):
                row = {"request_id": r.request_id, **bad}
                try:
                    schema.validate_row(row, _request_dict(r), [_option_dict(o) for o in ds.request_options(r.request_id)],
                                        _profile_dict(ds.profile_by_user[r.user_id]),
                                        [_event_dict(e) for e in ds.user_events(r.user_id)])
                    failures.append(f"validate_row accepted an invalid row: {bad['spending_changes_needed']}")
                except schema.ValidationError:
                    pass
        except Exception as err:  # noqa: BLE001 - a fixture must never take the harness down
            failures.append(f"harness error: {type(err).__name__}: {err}")
        results.append({"fixture": name, "purpose": golden.get("purpose", ""), "passed": not failures, "failures": failures})
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", help="one line: what changed since the last run (required unless --rescore)")
    ap.add_argument("--no-evidence", action="store_true", help="ignore messages and images entirely")
    ap.add_argument("--skip-golden", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="replay the cache only; never call the API")
    ap.add_argument("--keep", action="store_true",
                    help="mark this run as the new baseline (use only when the change is kept)")
    ap.add_argument("--baseline", type=int, default=None, help="score against this run instead of the stored baseline")
    ap.add_argument("--rescore", nargs=2, type=int, metavar=("OLD", "NEW"),
                    help="print the regression diff between two stored runs and exit")
    ap.add_argument("--set-baseline", type=int, metavar="RUN",
                    help="advance the baseline pointer to RUN and exit (operator override)")
    args = ap.parse_args(argv)
    if not args.rescore and args.set_baseline is None and not args.note:
        ap.error("--note is required")

    ds = load_dataset()
    evidence_note = "no evidence"
    extractor = None
    if not args.no_evidence:
        from extraction import Extractor
        extractor = Extractor(dry_run=args.dry_run or not os.environ.get("GEMINI_API_KEY"))
    # evidence is only needed for the users behind the 25 samples: filter before extraction so a
    # free-tier run costs ~30 calls instead of 231 (the cache makes later runs free either way)
    sample_users = {r.user_id for r in ds.sample_requests}
    ds_evidence = dataclasses.replace(ds, messages=[m for m in ds.messages if m.user_id in sample_users],
                                      images=[i for i in ds.images if i.user_id in sample_users])
    evidence = gather_evidence(ds_evidence, extractor)
    if extractor is not None:
        s = extractor.stats
        used = sum(len(v) for v in evidence.verdicts_by_user.values())
        evidence_note = (f"evidence: {used} verdicts, {len(evidence.facts_by_event)} image facts "
                         f"(api_calls={s.calls} cache_hits={s.cache_hits} fallbacks={s.fallbacks})")

    if args.set_baseline is not None:
        set_baseline(args.set_baseline)
        print(f"baseline pointer -> run {args.set_baseline}")
        return 0

    if args.rescore:
        old_run, new_run = args.rescore
        a, b = load_results(old_run), load_results(new_run)
        regressed = sorted(r for r in b if a.get(r, {}).get("all_ok") and not b[r]["all_ok"])
        improved = sorted(r for r in b if r in a and not a[r]["all_ok"] and b[r]["all_ok"])
        print(f"re-score: run {new_run} against run {old_run}")
        print(f"  regressed: {', '.join(regressed) or '-'}")
        print(f"  improved:  {', '.join(improved) or '-'}")
        for f_ in ("affordability_status", "recommended_payment_method", "earliest_date_for_full_payment",
                   "payment_plan", "amount_ok"):
            up = sorted(r for r in b if r in a and not a[r][f_] and b[r][f_])
            down = sorted(r for r in b if r in a and a[r][f_] and not b[r][f_])
            print(f"  {f_:32} +{len(up)} {up}  -{len(down)} {down}")
        return 1 if regressed else 0

    golden = [] if args.skip_golden else run_golden()
    golden_failed = [g["fixture"] for g in golden if not g["passed"]]
    golden_note = "" if args.skip_golden else f"; golden {len(golden) - len(golden_failed)}/{len(golden)}" + (
        f" FAILED: {', '.join(golden_failed)}" if golden_failed else "")

    base_run, prev_results = baseline_run()
    if args.baseline is not None:
        base_run, prev_results = args.baseline, load_results(args.baseline)
    runs = all_runs()
    run = (runs[-1] if runs else 0) + 1
    rows, pairs, per_request, errors = [], [], {}, []
    amt_abs, amt_rel, within, within10 = [], [], 0, 0
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
        within10 += rel <= Decimal("0.10")
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
        "amt_within_10pct": within10 / n,
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
    kept = bool(args.keep) and not regressed and not golden_failed
    new_log = not os.path.exists(RUN_LOG)
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        if new_log:
            f.write("# Evaluation runs (dataset/sample_requests.csv, 25 rows)\n\n"
                    "amt_mae is mean |pred-expected| / requested_amount; amt_mean_rel is mean |pred-expected| / expected "
                    "(over requested_amount when expected is 0); within_1pct uses the same relative error.\n\n"
                    "| run | baseline_run | timestamp | note | " + " | ".join(METRIC_COLUMNS) + " | regressed | improved |\n"
                    "|---|---|---|---|" + "---|" * len(METRIC_COLUMNS) + "---|---|\n")
        f.write(f"| {run} | {base_run or '-'}{' KEPT' if kept else ''} | "
                f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} | {args.note} ({evidence_note}{golden_note}) | "
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
    if args.keep and not regressed and not golden_failed:
        set_baseline(run)
    print(f"\nbaseline: run {base_run or '-'}" + (f" -> run {run} (KEPT)" if args.keep and not regressed and not golden_failed else ""))
    print(f"regressed: {regressed or '-'}   improved: {improved or '-'}")
    print(f"errors: {len(errors)} mismatches -> {os.path.relpath(err_path, CODE)}")
    if evidence.review_flags:
        flagged = Counter(reason.split(':')[0] for _, _, reason in evidence.review_flags)
        print(f"review flags: {dict(flagged)}")

    if not args.skip_golden:
        print(f"\ngolden fixtures: {len(golden) - len(golden_failed)}/{len(golden)} passed")
        for g in golden:
            mark = "PASS" if g["passed"] else "FAIL"
            print(f"  {mark} {g['fixture']:34} {g['purpose']}")
            for fl in g["failures"]:
                print(f"       - {fl}")

    method_rows = [rid for rid, res in per_request.items() if res["recommended_payment_method"]]
    plan_bad = [rid for rid in method_rows if not per_request[rid]["payment_plan"]]
    extras = {"plan_given_method": (len(method_rows) - len(plan_bad)) / len(method_rows) if method_rows else 1.0,
              "golden_ok": 0.0 if golden_failed else 1.0}
    print("\nstop criteria:")
    failing = []
    for label, fn, target, sense in STOP_CRITERIA:
        value = fn({**metrics, **extras}, n)
        ok = value >= target if sense == "ge" else value <= target
        if not ok:
            failing.append(label)
        shown = f"{value:.3f}" if isinstance(value, float) and value < 2 else f"{value:.0f}/25"
        want = f"{'>=' if sense == 'ge' else '<='} {target}"
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:36} {shown:>9}  (need {want})")
    if plan_bad:
        print(f"         plan mismatches on method-correct rows: {plan_bad}")
    print(f"  {len(STOP_CRITERIA) - len(failing)}/{len(STOP_CRITERIA)} criteria pass"
          + (f"; failing: {', '.join(failing)}" if failing else " - ALL CRITERIA MET"))

    if regressed or golden_failed:
        print(f"\nREGRESSION: samples={regressed or '-'} golden={golden_failed or '-'}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
