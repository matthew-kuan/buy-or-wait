"""Joint 2D sweep of the two variable-spend parameters against the 25 samples and the golden set.

    python code/evaluation/sweep.py

Measurement only: it never writes run_log.md, never calls the API (evidence is replayed from
.cache/), and leaves reconcile.py's defaults untouched on exit. Every cell runs the real pipeline.

  rows    DISCRETIONARY_RATE  fraction of historical cadence projected for non-essential categories
  cols    ESSENTIAL_STAT      statistic over per-occurrence amounts for essential categories
"""
from __future__ import annotations

import os
import statistics
import sys
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)

import reconcile  # noqa: E402
import schema  # noqa: E402
from io_layer import load_dataset  # noqa: E402
from pipeline import decide, gather_evidence  # noqa: E402

import main as harness  # noqa: E402  (the scoring harness; run_golden + relative_error)

RATES = [Decimal("0"), Decimal("0.25"), Decimal("0.5"), Decimal("0.75"), Decimal("1.0")]
STATS = ["mean", "median", "p75", "p90", "max"]

# run-14 baseline that a cell must not fall below (selection rule 1)
BASE_STATUS, BASE_METHOD = 20, 22


def score(ds, evidence) -> dict:
    within = status = method = plan = earliest = 0
    rel_errors, abs_norm = [], []
    for r in ds.sample_requests:
        out = decide(ds, r, evidence)
        a, row = r.answer, out.row
        status += row["affordability_status"] == a.affordability_status
        method += row["recommended_payment_method"] == a.recommended_payment_method
        plan += row["payment_plan"] == a.payment_plan
        earliest += row["earliest_date_for_full_payment"] == (
            a.earliest_date_for_full_payment.isoformat() if a.earliest_date_for_full_payment else "")
        pred = Decimal(row["amount_safe_to_pay"])
        rel = harness.relative_error(pred, a.amount_safe_to_pay, r.requested_amount)
        rel_errors.append(rel)
        abs_norm.append(abs(pred - a.amount_safe_to_pay) / r.requested_amount)
        within += rel <= Decimal("0.01")
    n = len(ds.sample_requests)
    return {"within": within, "status": status, "method": method, "plan": plan, "earliest": earliest,
            "amt_mae": float(sum(abs_norm) / n), "amt_rel": float(sum(rel_errors) / n)}


def main() -> int:
    ds = load_dataset()
    from extraction import Extractor
    extractor = Extractor(dry_run=True)          # cache only: no API calls, no cost
    import dataclasses
    users = {r.user_id for r in ds.sample_requests}
    ds_ev = dataclasses.replace(ds, messages=[m for m in ds.messages if m.user_id in users],
                                images=[i for i in ds.images if i.user_id in users])
    evidence = gather_evidence(ds_ev, extractor)

    saved = (reconcile.ESSENTIAL_STAT, reconcile.DISCRETIONARY_RATE)
    cells = {}
    try:
        for rate in RATES:
            for stat in STATS:
                reconcile.DISCRETIONARY_RATE = rate
                reconcile.ESSENTIAL_STAT = stat
                m = score(ds, evidence)
                golden = harness.run_golden()
                m["golden"] = sum(g["passed"] for g in golden)
                m["golden_total"] = len(golden)
                cells[(rate, stat)] = m
                print(f"  ran rate={rate} stat={stat:6} within={m['within']:2}/25 "
                      f"status={m['status']:2} method={m['method']:2} golden={m['golden']}/{m['golden_total']}",
                      flush=True)
    finally:
        reconcile.ESSENTIAL_STAT, reconcile.DISCRETIONARY_RATE = saved

    # ---------------------------------------------------------------- grid
    print("\n" + "=" * 100)
    print("JOINT SWEEP: rows = DISCRETIONARY_RATE, cols = ESSENTIAL_STAT")
    print("cell = within-1% /25 | amt_mae | status /25 | method /25 | golden")
    print("=" * 100)
    print(f"{'rate':>6} " + "".join(f"{s:>18}" for s in STATS))
    for rate in RATES:
        line = f"{str(rate):>6} "
        for stat in STATS:
            m = cells[(rate, stat)]
            g = "ok" if m["golden"] == m["golden_total"] else "FAIL"
            line += f"{m['within']:>3}|{m['amt_mae']:.3f}|{m['status']:>2}|{m['method']:>2}|{g:>4}"
        print(line)

    for label, key in (("within-1% (of 25)", "within"), ("amt_mae (lower is better)", "amt_mae"),
                       ("status (of 25)", "status"), ("method (of 25)", "method")):
        print(f"\n{label}")
        print(f"{'rate':>6} " + "".join(f"{s:>9}" for s in STATS))
        for rate in RATES:
            vals = [cells[(rate, s)][key] for s in STATS]
            fmt = "{:>9.3f}" if key == "amt_mae" else "{:>9}"
            print(f"{str(rate):>6} " + "".join(fmt.format(v) for v in vals))

    # ---------------------------------------------------------------- selection
    print("\n" + "=" * 100)
    print("SELECTION")
    print("=" * 100)
    survivors = []
    for rate in RATES:
        for stat in STATS:
            m = cells[(rate, stat)]
            if m["status"] < BASE_STATUS or m["method"] < BASE_METHOD:
                continue                                        # rule 1
            if m["golden"] != m["golden_total"]:
                continue                                        # rule 2
            survivors.append((rate, stat, m))
    rejected_acc = 25 - len([1 for r in RATES for s in STATS
                             if cells[(r, s)]["status"] >= BASE_STATUS and cells[(r, s)]["method"] >= BASE_METHOD])
    rejected_golden = len([1 for r in RATES for s in STATS if cells[(r, s)]["golden"] != cells[(r, s)]["golden_total"]])
    print(f"rule 1 (status >= {BASE_STATUS} and method >= {BASE_METHOD}): rejected {rejected_acc} cells")
    print(f"rule 2 (golden must pass): rejected {rejected_golden} cells")
    print(f"survivors: {len(survivors)}")
    if not survivors:
        print("no cell survives; keeping the current defaults")
        return 1
    best_within = max(m["within"] for _, _, m in survivors)     # rule 3
    peak = [(r, s, m) for r, s, m in survivors if m["within"] == best_within]
    # rule 4: conservative tie-break = highest statistic, then highest discretionary rate
    order = {s: i for i, s in enumerate(STATS)}                 # mean < median < p75 < p90 < max
    peak.sort(key=lambda t: (order[t[1]], t[0]), reverse=True)
    print(f"rule 3: best within-1% among survivors = {best_within}/25, reached by "
          f"{[(str(r), s) for r, s, _ in peak]}")
    print(f"rule 4 (conservative tie-break): {str(peak[0][0])}, {peak[0][1]}")

    # ---------------------------------------------------------------- surface shape
    print("\n" + "=" * 100)
    print("SURFACE SHAPE")
    print("=" * 100)
    grid = [[cells[(r, s)]["within"] for s in STATS] for r in RATES]
    flat = [v for row in grid for v in row]
    print(f"within-1% over all 25 cells: min {min(flat)}, max {max(flat)}, mean {statistics.mean(flat):.2f}, "
          f"stdev {statistics.pstdev(flat):.2f}")
    jumps = []
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            if j + 1 < len(row):
                jumps.append(abs(row[j + 1] - v))
            if i + 1 < len(grid):
                jumps.append(abs(grid[i + 1][j] - v))
    print(f"adjacent-cell |delta| in within-1%: mean {statistics.mean(jumps):.2f}, max {max(jumps)}")
    at_peak = [(str(r), s) for r in RATES for s in STATS if cells[(r, s)]["within"] == max(flat)]
    print(f"cells at the global peak ({max(flat)}/25): {len(at_peak)} -> {at_peak}")
    principled = cells[(Decimal('0.5'), 'p75')]
    print(f"principled cell (rate 0.5, p75): within {principled['within']}/25, amt_mae {principled['amt_mae']:.3f}, "
          f"status {principled['status']}, method {principled['method']}, "
          f"golden {principled['golden']}/{principled['golden_total']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
