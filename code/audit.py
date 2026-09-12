"""Read-only audit of every file in dataset/. No API calls, no writes to dataset/.

Run from anywhere:  python code/audit.py
Writes the same report to stdout and code/evaluation/data_audit.txt.
"""
from __future__ import annotations

import csv
import os
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATASET = os.path.join(ROOT, "dataset")
OUT_PATH = os.path.join(HERE, "evaluation", "data_audit.txt")
CAP = 30

FILES = [
    "financial_profiles.csv",
    "financial_events.csv",
    "exchange_rates.csv",
    "requests.csv",
    "sample_requests.csv",
    "request_payment_options.csv",
    "messages.csv",
    "images.csv",
    "output.csv",
]

DISTINCT = [
    ("financial_events.csv", "event_type"),
    ("financial_events.csv", "status"),
    ("financial_events.csv", "direction"),
    ("financial_events.csv", "flexibility"),
    ("financial_events.csv", "category"),
    ("financial_events.csv", "currency"),
    ("financial_profiles.csv", "home_currency"),
    ("requests.csv", "request_type"),
    ("requests.csv", "allows_partial_payment"),
    ("financial_profiles.csv", "payment_methods_user_will_consider"),
    ("request_payment_options.csv", "payment_method"),
]

OUTPUT_COLS = [
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
]


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def load(name):
    with open(os.path.join(DATASET, name), encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def d(s):
    return date.fromisoformat(s)


def capped(items, fmt=str):
    items = list(items)
    lines = [f"  - {fmt(x)}" for x in items[:CAP]]
    if len(items) > CAP:
        lines.append(f"  ... ({len(items) - CAP} more, {len(items)} total)")
    return "\n".join(lines) if lines else "  (none)"


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out_file = open(OUT_PATH, "w", encoding="utf-8", newline="\n")
    console = sys.stdout
    if hasattr(console, "reconfigure"):
        console.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout = Tee(console, out_file)

    data = {name: load(name) for name in FILES}
    cols = {name: c for name, (c, _) in data.items()}
    rows = {name: r for name, (_, r) in data.items()}

    image_dir = os.path.join(DATASET, "media", "images")
    image_files = sorted(os.listdir(image_dir)) if os.path.isdir(image_dir) else []

    # ---------------------------------------------------------------- 1. shape
    section("1. Row counts, columns, null counts")
    for name in FILES:
        r, c = rows[name], cols[name]
        print(f"\n{name}: {len(r)} rows, {len(c)} columns")
        print(f"  columns: {', '.join(c)}")
        nulls = {col: sum(1 for x in r if (x.get(col) or "") == "") for col in c}
        nonzero = {k: v for k, v in nulls.items() if v}
        print(f"  null counts: {nonzero if nonzero else 'none'}")
    print(f"\nmedia/images: {len(image_files)} png files")

    # ------------------------------------------------------------- 2. distinct
    section("2. Distinct values with counts")
    for name, col in DISTINCT:
        cnt = Counter(x.get(col, "") for x in rows[name])
        print(f"\n{name}.{col} ({len(cnt)} distinct)")
        for val, n in cnt.most_common():
            print(f"  {val!r}: {n}")

    ev = rows["financial_events.csv"]
    ev_by_id = {e["event_id"]: e for e in ev}
    profiles = {p["user_id"]: p for p in rows["financial_profiles.csv"]}
    images = rows["images.csv"]
    img_by_event = defaultdict(list)
    for im in images:
        img_by_event[im["related_event_id"]].append(im["image_id"])

    # --------------------------------------------------------- 3. blank amount
    section("3. Events with blank amount, and image coverage")
    blank = [e for e in ev if e["amount"] == ""]
    covered = [e for e in blank if e["event_id"] in img_by_event]
    print(f"blank-amount events: {len(blank)}; with image row: {len(covered)}; without: {len(blank) - len(covered)}")
    print(capped(
        blank,
        lambda e: f"{e['event_id']} {e['user_id']} {e['event_type']}/{e['category']} {e['status']} "
                  f"settle={e['settlement_date']} images={img_by_event.get(e['event_id'], [])} "
                  f"file_present={[f'{i}.png' in image_files for i in img_by_event.get(e['event_id'], [])]}",
    ))
    orphan_images = [im for im in images if im["related_event_id"] and im["related_event_id"] not in ev_by_id]
    print(f"image rows whose related_event_id is not an event: {len(orphan_images)}")
    print(capped(orphan_images, lambda im: f"{im['image_id']} -> {im['related_event_id']}"))
    missing_files = [im["image_id"] for im in images if f"{im['image_id']}.png" not in image_files]
    print(f"image rows with no png on disk: {len(missing_files)} {missing_files[:CAP]}")

    # ---------------------------------------------------------- 4. linked ids
    section("4. linked_event_id integrity and status pairs")
    linked = [e for e in ev if e["linked_event_id"]]
    dangling = [e for e in linked if e["linked_event_id"] not in ev_by_id]
    print(f"events with linked_event_id: {len(linked)}; dangling: {len(dangling)}")
    print(capped(dangling, lambda e: f"{e['event_id']} -> {e['linked_event_id']}"))
    pairs = Counter()
    cross_user = []
    for e in linked:
        t = ev_by_id.get(e["linked_event_id"])
        if not t:
            continue
        pairs[(e["event_type"], e["status"], "->", t["event_type"], t["status"])] += 1
        if t["user_id"] != e["user_id"]:
            cross_user.append((e["event_id"], t["event_id"]))
    print("status pairs (child type/status -> parent type/status):")
    for k, n in pairs.most_common():
        print(f"  {k[0]}/{k[1]} -> {k[3]}/{k[4]}: {n}")
    print(f"links crossing users: {len(cross_user)} {cross_user[:CAP]}")

    # ------------------------------------------------------------ 5. duplicates
    section("5. Duplicate candidates (same user_id, event_date, amount, currency, description)")
    groups = defaultdict(list)
    for e in ev:
        groups[(e["user_id"], e["event_date"], e["amount"], e["currency"], e["description"])].append(e)
    dups = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"duplicate groups: {len(dups)}; rows involved: {sum(len(v) for v in dups.values())}")
    print(capped(
        dups.items(),
        lambda kv: f"{kv[0][0]} {kv[0][1]} {kv[0][2]} {kv[0][3]} {kv[0][4]!r}: "
                   + ", ".join(f"{e['event_id']}({e['status']})" for e in kv[1]),
    ))
    status_mix = Counter(tuple(sorted(e["status"] for e in v)) for v in dups.values())
    print(f"status combinations within groups: {dict(status_mix)}")

    # ------------------------------------------------------------------- 6. fx
    section("6. Foreign-currency events vs exchange_rates.csv")
    fx = rows["exchange_rates.csv"]
    fx_keys = {(f["rate_date"], f["from_currency"], f["to_currency"]) for f in fx}
    foreign = []
    misses = []
    no_profile = []
    for e in ev:
        p = profiles.get(e["user_id"])
        if not p:
            no_profile.append(e["event_id"])
            continue
        home = p["home_currency"]
        if e["currency"] == home:
            continue
        foreign.append(e)
        key = (e["settlement_date"], e["currency"], home)
        if key not in fx_keys:
            misses.append((e, key))
    print(f"foreign-currency events: {len(foreign)}; with exact rate row: {len(foreign) - len(misses)}; misses: {len(misses)}")
    print(f"  by type/status: {dict(Counter((e['event_type'], e['status']) for e in foreign))}")
    print(f"  by pair: {dict(Counter((e['currency'], profiles[e['user_id']]['home_currency']) for e in foreign))}")
    print("misses:")
    print(capped(misses, lambda m: f"{m[0]['event_id']} {m[0]['status']} needs {m[1]} (settlement_date={m[0]['settlement_date']!r})"))
    print(f"events whose user has no profile: {len(no_profile)} {no_profile[:CAP]}")
    unused = [f for f in fx if (f["rate_date"], f["from_currency"], f["to_currency"])
              not in {(e["settlement_date"], e["currency"], profiles[e["user_id"]]["home_currency"]) for e in foreign}]
    print(f"rate rows not matched by any event: {len(unused)} of {len(fx)}")
    print(f"  pairs: {dict(Counter((f['from_currency'], f['to_currency']) for f in fx))}")
    print(f"  date range: {min(f['rate_date'] for f in fx)} .. {max(f['rate_date'] for f in fx)}")

    # ------------------------------------------------------- 7. request joins
    section("7. Requests: profile and payment-option coverage")
    reqs = rows["requests.csv"]
    samples = rows["sample_requests.csv"]
    options = rows["request_payment_options.csv"]
    opt_by_req = defaultdict(list)
    for o in options:
        opt_by_req[o["request_id"]].append(o)
    for label, rs in (("requests.csv", reqs), ("sample_requests.csv", samples)):
        no_prof = [r["request_id"] for r in rs if r["user_id"] not in profiles]
        no_opt = [r["request_id"] for r in rs if not opt_by_req.get(r["request_id"])]
        no_full = [r["request_id"] for r in rs if not any(o["payment_method"] == "full_payment" for o in opt_by_req.get(r["request_id"], []))]
        print(f"{label}: {len(rs)} rows; no profile: {len(no_prof)} {no_prof[:CAP]}; "
              f"no options: {len(no_opt)} {no_opt[:CAP]}; no full_payment option: {len(no_full)} {no_full[:CAP]}")
    all_req_ids = {r["request_id"] for r in reqs} | {r["request_id"] for r in samples}
    orphan_opts = sorted({o["request_id"] for o in options} - all_req_ids)
    print(f"option rows for unknown requests: {len(orphan_opts)} {orphan_opts[:CAP]}")
    print(f"options per request: {dict(sorted(Counter(len(v) for v in opt_by_req.values()).items()))}")
    ids = Counter(r["request_id"] for r in reqs)
    print(f"duplicate request_ids in requests.csv: {[k for k, v in ids.items() if v > 1][:CAP]}")
    out_ids = [r["request_id"] for r in rows["output.csv"]]
    print(f"output.csv template ids == requests.csv ids (same order): {out_ids == [r['request_id'] for r in reqs]}")
    users_multi = [u for u, n in Counter(r["user_id"] for r in reqs + samples).items() if n > 1]
    print(f"users with more than one request (incl. samples): {len(users_multi)} {users_multi[:CAP]}")

    # ------------------------------------------------ 8. per-request evidence
    section("8. Per-request evidence (messages, images, scheduled events in the 90-day window)")
    msgs = rows["messages.csv"]
    msg_by_req = Counter(m["request_id"] for m in msgs if m["request_id"])
    msg_by_user = Counter(m["user_id"] for m in msgs)
    img_by_req = Counter(im["request_id"] for im in images if im["request_id"])
    img_by_user = Counter(im["user_id"] for im in images)
    ev_by_user = defaultdict(list)
    for e in ev:
        ev_by_user[e["user_id"]].append(e)
    summary = Counter()
    per_req = []
    for r in reqs:
        rd = d(r["request_date"])
        end = rd + timedelta(days=89)
        ues = ev_by_user[r["user_id"]]
        sched = [e for e in ues if e["status"] == "scheduled" and e["settlement_date"] and rd <= d(e["settlement_date"]) <= end]
        pend = [e for e in ues if e["status"] == "pending" and e["settlement_date"] and rd <= d(e["settlement_date"]) <= end]
        after = [e for e in ues if e["settlement_date"] and d(e["settlement_date"]) > rd]
        mc = msg_by_req[r["request_id"]] + sum(1 for m in msgs if m["user_id"] == r["user_id"] and not m["request_id"])
        ic = img_by_req[r["request_id"]] + sum(1 for im in images if im["user_id"] == r["user_id"] and not im["request_id"])
        summary["has_messages"] += bool(mc)
        summary["has_images"] += bool(ic)
        summary["has_scheduled_in_window"] += bool(sched)
        summary["has_pending_in_window"] += bool(pend)
        summary["has_any_event_after_request"] += bool(after)
        summary["has_scheduled_salary_in_window"] += any(e["category"] == "salary" for e in sched)
        per_req.append((r["request_id"], mc, ic, len(sched), len(pend)))
    print(f"of {len(reqs)} requests: {dict(summary)}")
    print(f"messages: total {len(msgs)}; with request_id {sum(1 for m in msgs if m['request_id'])}; "
          f"with related_event_id {sum(1 for m in msgs if m['related_event_id'])}; "
          f"user-level only {sum(1 for m in msgs if not m['request_id'] and not m['related_event_id'])}; "
          f"distinct users {len(msg_by_user)}; users with >1 message {sum(1 for n in msg_by_user.values() if n > 1)}")
    print(f"messages.source_type: {dict(Counter(m['source_type'] for m in msgs))}")
    print(f"messages with related_event_id not in events: "
          f"{[m['message_id'] for m in msgs if m['related_event_id'] and m['related_event_id'] not in ev_by_id][:CAP]}")
    print(f"messages dated after their request_date: "
          f"{sum(1 for m in msgs if m['request_id'] and m['request_id'] in {r['request_id'] for r in reqs + samples} and m['sent_at'][:10] > next(r['request_date'] for r in reqs + samples if r['request_id'] == m['request_id']))}")
    print(f"images: total {len(images)}; distinct users {len(img_by_user)}")
    print("message count distribution per request:", dict(sorted(Counter(x[1] for x in per_req).items())))
    print("image count distribution per request:", dict(sorted(Counter(x[2] for x in per_req).items())))
    print("scheduled-in-window count distribution:", dict(sorted(Counter(x[3] for x in per_req).items())))
    print("pending-in-window count distribution:", dict(sorted(Counter(x[4] for x in per_req).items())))
    print("requests with messages or images (request_id msgs imgs sched pend):")
    print(capped([x for x in per_req if x[1] or x[2]], lambda x: f"{x[0]} msgs={x[1]} imgs={x[2]} sched={x[3]} pend={x[4]}"))
    gaps = [(d(r["desired_completion_date"]) - d(r["request_date"])).days for r in reqs]
    print(f"desired_completion_date - request_date (days): min {min(gaps)}, max {max(gaps)}, "
          f"beyond 89 days: {sum(1 for g in gaps if g > 89)}")
    hist = []
    for r in reqs:
        dates = [d(e["settlement_date"]) for e in ev_by_user[r["user_id"]] if e["settlement_date"]]
        if dates:
            hist.append((d(r["request_date"]) - min(dates)).days)
    print(f"history length before request (days): min {min(hist)}, max {max(hist)}")

    # --------------------------------------------------------------- 9. samples
    section("9. sample_requests.csv: output distributions and per-row summary")
    for col in OUTPUT_COLS:
        vals = [s[col] for s in samples]
        if col in ("amount_safe_to_pay", "payment_plan", "earliest_date_for_full_payment"):
            derived = Counter()
            for s in samples:
                v = s[col]
                if col == "amount_safe_to_pay":
                    a, req = float(v), float(s["requested_amount"])
                    derived["== requested" if a == req else ("== 0" if a == 0 else "between")] += 1
                elif col == "payment_plan":
                    derived["none" if v == "none" else f"{v.count('|') + 1} payment(s)"] += 1
                else:
                    derived["empty" if not v else ("== request_date" if v == s["request_date"] else "later")] += 1
            print(f"\n{col}: {dict(derived)}")
        else:
            print(f"\n{col}: {dict(Counter(vals))}")
    sc = Counter()
    for s in samples:
        for part in s["spending_changes_needed"].split("|"):
            sc[part.split(":")[0]] += 1
    print(f"spending change kinds: {dict(sc)}")
    print(f"status x method: {dict(Counter((s['affordability_status'], s['recommended_payment_method']) for s in samples))}")
    print("\nper-row (id date | req amt | balance min | methods maxinst partial | safe status method plan earliest changes | expl chars):")
    for s in samples:
        p = profiles.get(s["user_id"], {})
        print(f"  {s['request_id']} {s['request_date']} | {s['requested_amount']} {p.get('home_currency', '?')} | "
              f"bal {p.get('current_available_balance', '?')} min {p.get('minimum_balance_to_keep', '?')} | "
              f"{p.get('payment_methods_user_will_consider', '?')} maxinst={p.get('max_installment_months') or '-'} "
              f"partial={s['allows_partial_payment']} | {s['amount_safe_to_pay']} {s['affordability_status']} "
              f"{s['recommended_payment_method']} {s['payment_plan']} earliest={s['earliest_date_for_full_payment'] or '-'} "
              f"{s['spending_changes_needed']} | {len(s['decision_explanation'])}")

    sys.stdout.flush()
    out_file.close()
    sys.stdout = sys.__stdout__
    print(f"\nreport written to {OUT_PATH}")


if __name__ == "__main__":
    main()
