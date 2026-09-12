"""Turn one user's raw events plus message/image evidence into a clean Ledger.

    ledger = build_ledger(profile, events, message_verdicts, image_facts, as_of_date, convert=ds.convert)

Ledger contents (all amounts in the user's home currency, Decimal):
    recurring_debits     detected recurring outflows, projected forward
    recurring_credits    detected salary-like recurring inflows, projected forward
    scheduled_one_offs   confirmed future cash events: pending debits, scheduled rows, verdict-added income
    history              settled cash events on or before as_of_date that survived exclusion
    excluded             (event_id, reason, source) for every row that does not enter the forecast
    conflicts            everything that needed precedence rules, with how it was resolved

Every exclusion carries the rule name or message_id that triggered it so explanations can cite it.

Verdicts and image facts are plain dataclasses defined here; extraction.py must produce these
shapes. Nothing in this module calls a model.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Callable, Optional

from io_layer import Event, FxMissing, Profile, event_sort_key

# --------------------------------------------------------------------------- tunables

MIN_OCCURRENCES = 3          # occurrences needed before a group is called recurring
GAP_TOLERANCE_DAYS = 3       # every gap must be within this many days of the median gap
AMOUNT_WINDOW = 3            # median of the last N amounts is projected forward
SAME_OCCURRENCE_DAYS = 3     # a projected occurrence this close to a confirmed row is dropped

NON_SALARY_WORDS = re.compile(r"\b(bonus|commission|refund|windfall|prize|lottery|reimburse|cashback|gift)\b", re.I)


# --------------------------------------------------------------------------- evidence shapes

@dataclass(frozen=True)
class MessageVerdict:
    """What one message establishes. `kind` is one of:
    confirm            no change (kept for the trace)
    cancel_event       event_id will not happen                    -> excluded
    not_cash           event_id is not withdrawable / not settled   -> excluded
    internal_transfer  event_ids are a transfer between own accounts -> excluded
    amend_amount       event_id's amount is `amount` (`currency` optional, default event's)
    delay_event        event_id now settles on `effective_date`
    confirmed_income   a confirmed inflow of `amount` `currency` on `effective_date` (next salary, invoice)
    retry_debit        the failed debit `event_id` will be attempted again on `effective_date`
    salary_change      recurring salary becomes `amount` (or scales by `percent`) from `effective_date`
    expense_change     recurring debit matching `event_id`'s series becomes `amount` / scales by `percent`
                       from `effective_date` (`temporary_cycles` > 0 limits it to that many occurrences)
    """
    message_id: str
    kind: str
    event_id: Optional[str] = None
    event_ids: tuple[str, ...] = ()
    amount: Optional[Decimal] = None
    currency: Optional[str] = None
    percent: Optional[Decimal] = None
    effective_date: Optional[date] = None
    temporary_cycles: int = 0
    note: str = ""
    sent_order: int = 0            # position by sent_at; later verdicts win within the same source


@dataclass(frozen=True)
class ImageFact:
    image_id: str
    event_id: str
    amount: Optional[Decimal]
    currency: Optional[str] = None
    settlement_date: Optional[date] = None
    note: str = ""


# --------------------------------------------------------------------------- ledger shapes

@dataclass(frozen=True)
class Exclusion:
    event_id: str
    reason: str
    source: str


@dataclass(frozen=True)
class Conflict:
    event_ids: tuple[str, ...]
    description: str
    resolution: str                # what precedence rule decided it, or "unresolved:<safer reading>"
    source: str


@dataclass(frozen=True)
class CashEvent:
    """A concrete dated cash movement in home currency."""
    on: date
    amount: Decimal                # always positive
    direction: str                 # debit | credit
    event_id: Optional[str]
    category: str
    description: str
    status: str
    source: str                    # rule / message_id / image_id that placed it here


@dataclass
class RecurringSeries:
    key: tuple[str, str]           # (category, normalized description)
    category: str
    description: str
    direction: str
    flexibility: str
    minimum_allowed_amount: Optional[Decimal]
    event_ids: list[str]
    dates: list[date]
    amounts: list[Decimal]         # home currency, aligned with dates
    gap_days: int
    amount: Decimal                # projected per-occurrence amount
    overrides: list[tuple[date, Optional[Decimal], Optional[Decimal], int, str]] = field(default_factory=list)
    # (effective_date, new_amount, percent, temporary_cycles, source)

    @property
    def last_event_id(self) -> str:
        return self.event_ids[-1]

    @property
    def last_date(self) -> date:
        return self.dates[-1]

    def occurrences(self, start: date, end: date) -> list[tuple[date, Decimal]]:
        """Projected (date, amount) strictly after last_date, within [start, end]."""
        out = []
        k = 1
        remaining_temp = {}
        while True:
            on = self.last_date + timedelta(days=self.gap_days * k)
            if on > end:
                break
            if on >= start:
                amount = self.amount
                for eff, new_amount, percent, cycles, _src in self.overrides:
                    if on < eff:
                        continue
                    if cycles:
                        used = remaining_temp.get((eff, _src), 0)
                        if used >= cycles:
                            continue
                        remaining_temp[(eff, _src)] = used + 1
                    if new_amount is not None:
                        amount = new_amount
                    elif percent is not None:
                        amount = (amount * (Decimal(100) + percent) / Decimal(100))
                out.append((on, amount))
            k += 1
        return out


@dataclass
class Ledger:
    user_id: str
    home_currency: str
    as_of: date
    recurring_debits: list[RecurringSeries] = field(default_factory=list)
    recurring_credits: list[RecurringSeries] = field(default_factory=list)
    scheduled_one_offs: list[CashEvent] = field(default_factory=list)
    history: list[CashEvent] = field(default_factory=list)
    excluded: list[Exclusion] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    non_recurring_groups: list[tuple[tuple[str, str], int, str]] = field(default_factory=list)  # (key, n, why)

    def flows(self, start: date, end: date, stopped: set[str] = frozenset(),
              reduced: dict[str, Decimal] | None = None) -> list[CashEvent]:
        """Every projected cash movement in [start, end], sorted by date (debits before credits
        on the same day - the conservative ordering). `stopped` / `reduced` are keyed by a
        series' last_event_id and model spending changes."""
        reduced = reduced or {}
        out = [c for c in self.scheduled_one_offs if start <= c.on <= end]
        for s in self.recurring_debits:
            if s.last_event_id in stopped:
                continue
            cap = reduced.get(s.last_event_id)
            confirmed = [(c.on, c.description) for c in self.scheduled_one_offs
                         if c.direction == "debit" and _norm(c.description) == s.key[1]]
            for on, amount in s.occurrences(start, end):
                if any(abs((on - cd).days) <= SAME_OCCURRENCE_DAYS for cd, _ in confirmed):
                    continue
                if cap is not None:
                    amount = min(amount, cap)
                out.append(CashEvent(on, amount, "debit", None, s.category, s.description, "projected",
                                     f"recurring:{s.last_event_id}"))
        for s in self.recurring_credits:
            confirmed = [c.on for c in self.scheduled_one_offs if c.direction == "credit"]
            for on, amount in s.occurrences(start, end):
                if any(abs((on - cd).days) <= SAME_OCCURRENCE_DAYS for cd in confirmed):
                    continue
                out.append(CashEvent(on, amount, "credit", None, s.category, s.description, "projected",
                                     f"recurring:{s.last_event_id}"))
        out.sort(key=lambda c: (c.on, 0 if c.direction == "debit" else 1, c.event_id or ""))
        return out

    def exclusion_for(self, event_id: str) -> Optional[Exclusion]:
        return next((x for x in self.excluded if x.event_id == event_id), None)


# --------------------------------------------------------------------------- helpers

def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _median_int(values: list[int]) -> int:
    return int(round(statistics.median(values)))


def _median_dec(values: list[Decimal]) -> Decimal:
    values = sorted(values)
    n = len(values)
    if n % 2:
        return values[n // 2]
    return (values[n // 2 - 1] + values[n // 2]) / 2


def _is_salary_like(e: Event) -> bool:
    if e.direction != "credit" or e.event_type != "income":
        return False
    if e.category in ("windfall", "investment"):
        return False
    return not NON_SALARY_WORDS.search(e.description or "")


# --------------------------------------------------------------------------- build

def build_ledger(user: Profile, events: list[Event], message_verdicts: list[MessageVerdict],
                 image_facts: list[ImageFact], as_of_date: date,
                 convert: Callable[[Decimal, str, str, date], Decimal] | None = None) -> Ledger:
    home = user.home_currency
    ledger = Ledger(user_id=user.user_id, home_currency=home, as_of=as_of_date)
    events = sorted((e for e in events if e.user_id == user.user_id),
                    key=lambda e: (e.cash_date, event_sort_key(e.event_id)))
    by_id = {e.event_id: e for e in events}

    # mutable working copy: event_id -> dict of overridable fields
    work: dict[str, dict] = {
        e.event_id: {"amount": e.amount, "currency": e.currency, "cash_date": e.cash_date,
                     "status": e.status, "excluded": False}
        for e in events
    }

    def exclude(event_id: str, reason: str, source: str):
        w = work.get(event_id)
        if w is None or w["excluded"]:
            return
        w["excluded"] = True
        ledger.excluded.append(Exclusion(event_id, reason, source))

    def conflict(ids, description, resolution, source):
        ledger.conflicts.append(Conflict(tuple(ids), description, resolution, source))

    # ---- 1. image facts fill blank amounts (or amend) ------------------------------------
    for fact in image_facts:
        w = work.get(fact.event_id)
        if w is None:
            conflict((fact.event_id,), f"image {fact.image_id} refers to an unknown event", "ignored", fact.image_id)
            continue
        if fact.amount is None:
            conflict((fact.event_id,), f"image {fact.image_id} yielded no amount", "unresolved:event kept without amount", fact.image_id)
            continue
        if w["amount"] is not None and w["amount"] != fact.amount:
            conflict((fact.event_id,), f"image {fact.image_id} amount {fact.amount} differs from row amount {w['amount']}",
                     "image (explicit amendment) wins", fact.image_id)
        w["amount"] = fact.amount
        if fact.currency:
            w["currency"] = fact.currency
        if fact.settlement_date:
            w["cash_date"] = fact.settlement_date

    # ---- 2. status / direction exclusions ------------------------------------------------
    for e in events:
        if e.status in ("cancelled", "failed"):
            exclude(e.event_id, f"status_{e.status}", "rule:status")
        elif e.direction == "non_cash":
            exclude(e.event_id, "direction_non_cash", "rule:non_cash")
        elif e.status == "unrealized":
            exclude(e.event_id, "status_unrealized", "rule:unrealized")
        elif e.event_type == "investment_valuation":
            exclude(e.event_id, "investment_valuation", "rule:non_cash")
        elif e.status == "pending" and e.direction == "credit":
            exclude(e.event_id, "pending_credit_not_counted", "rule:pending_credit")

    # ---- 3. duplicates --------------------------------------------------------------------
    groups: dict[tuple, list[Event]] = {}
    for e in events:
        if work[e.event_id]["excluded"]:
            continue
        key = (e.event_date, work[e.event_id]["amount"], work[e.event_id]["currency"], _norm(e.description))
        groups.setdefault(key, []).append(e)
    for key, members in groups.items():
        if len(members) < 2:
            continue
        members.sort(key=lambda e: event_sort_key(e.event_id))
        keep = members[0]
        for dup in members[1:]:
            exclude(dup.event_id, f"duplicate_of:{keep.event_id}", "rule:duplicate")
            conflict((keep.event_id, dup.event_id), "identical date/amount/currency/description", f"kept lowest id {keep.event_id}", "rule:duplicate")

    # ---- 4. linked lifecycles -------------------------------------------------------------
    for e in events:
        if not e.linked_event_id:
            continue
        t = by_id.get(e.linked_event_id)
        if t is None:
            conflict((e.event_id, e.linked_event_id), "linked_event_id points outside this user's events", "link ignored", "rule:link")
            continue
        we, wt = work[e.event_id], work[t.event_id]
        if e.direction == t.direction and {we["status"], wt["status"]} == {"settled", "pending"}:
            pending = e if we["status"] == "pending" else t
            settled = t if pending is e else e
            exclude(pending.event_id, f"superseded_by:{settled.event_id}", "rule:link_settled_supersedes_pending")
            conflict((pending.event_id, settled.event_id), "pending and settled rows of one transaction",
                     f"settled {settled.event_id} counted once", "rule:link")
        elif e.direction == t.direction and {we["status"], wt["status"]} == {"scheduled", "failed"}:
            conflict((e.event_id, t.event_id), "failed attempt with a scheduled retry",
                     "failed excluded, scheduled retry reserved", "rule:link")
        elif e.direction == t.direction and {we["status"], wt["status"]} == {"settled", "cancelled"}:
            conflict((e.event_id, t.event_id), "cancelled attempt re-issued and settled",
                     "cancelled excluded, settled counted once", "rule:link")

    # ---- 5. message verdicts: explicit cancellation / settlement / amendment first ----------
    for v in sorted(message_verdicts, key=lambda v: (v.sent_order, v.message_id)):
        _apply_verdict(v, user, work, by_id, ledger, exclude, conflict, as_of_date, convert)

    # ---- 6. currency conversion --------------------------------------------------------------
    for e in events:
        w = work[e.event_id]
        if w["excluded"]:
            continue
        if w["amount"] is None:
            exclude(e.event_id, "amount_unknown", "rule:blank_amount")
            conflict((e.event_id,), "blank amount and no image fact", "unresolved:excluded (cannot size it)", "rule:blank_amount")
            continue
        if w["currency"] != home:
            if convert is None:
                raise FxMissing(f"{e.event_id} is in {w['currency']} but no converter was supplied")
            try:
                w["amount"] = convert(w["amount"], w["currency"], home, w["cash_date"])
                w["currency"] = home
            except FxMissing as err:
                exclude(e.event_id, "fx_missing", "rule:fx")
                conflict((e.event_id,), str(err), "unresolved:excluded (debit would be safer but is unsizable)", "rule:fx")

    # ---- 7. split into history vs confirmed future ------------------------------------------
    for e in events:
        w = work[e.event_id]
        if w["excluded"]:
            continue
        ce = CashEvent(w["cash_date"], w["amount"], e.direction, e.event_id, e.category, e.description,
                       w["status"], "rule:row")
        if w["status"] == "settled" and w["cash_date"] <= as_of_date:
            ledger.history.append(ce)
        elif w["status"] == "pending" and e.direction == "debit":
            on = max(w["cash_date"], as_of_date)
            ledger.scheduled_one_offs.append(CashEvent(on, w["amount"], "debit", e.event_id, e.category, e.description, "pending", "rule:pending_debit_reserved"))
        elif w["status"] in ("scheduled", "settled"):
            if e.direction == "credit" and not _is_salary_like(e):
                exclude(e.event_id, "non_salary_credit_not_counted", "rule:income")
                continue
            on = max(w["cash_date"], as_of_date)
            ledger.scheduled_one_offs.append(CashEvent(on, w["amount"], e.direction, e.event_id, e.category, e.description, w["status"], "rule:scheduled_row"))
        else:
            exclude(e.event_id, f"status_{w['status']}_not_forecastable", "rule:status")

    # ---- 8. recurrence detection on settled history -----------------------------------------
    for direction, target in (("debit", ledger.recurring_debits), ("credit", ledger.recurring_credits)):
        groups: dict[tuple[str, str], list[CashEvent]] = {}
        for c in ledger.history:
            if c.direction != direction:
                continue
            if direction == "credit" and not _is_salary_like(by_id[c.event_id]):
                continue
            groups.setdefault((c.category, _norm(c.description)), []).append(c)
        for key, members in groups.items():
            members.sort(key=lambda c: (c.on, event_sort_key(c.event_id)))
            series, why = _detect(key, members, by_id)
            if series is None:
                ledger.non_recurring_groups.append((key, len(members), why))
            else:
                target.append(series)

    # ---- 9. series-level verdicts (salary_change / expense_change) --------------------------
    for v in sorted(message_verdicts, key=lambda v: (v.sent_order, v.message_id)):
        if v.kind == "salary_change":
            if not ledger.recurring_credits:
                conflict((), "salary_change but no recurring salary detected", "unresolved:ignored (no income invented)", v.message_id)
                continue
            for s in ledger.recurring_credits:
                s.overrides.append((v.effective_date or as_of_date, _to_home(v, s, convert, home), v.percent, v.temporary_cycles, v.message_id))
        elif v.kind == "expense_change":
            anchor = by_id.get(v.event_id or "")
            hit = None
            if anchor is not None:
                hit = next((s for s in ledger.recurring_debits if s.key == (anchor.category, _norm(anchor.description))), None)
            if hit is None:
                conflict((v.event_id or "",), "expense_change but no recurring series matches", "unresolved:ignored", v.message_id)
                continue
            hit.overrides.append((v.effective_date or as_of_date, _to_home(v, hit, convert, home), v.percent, v.temporary_cycles, v.message_id))

    ledger.scheduled_one_offs.sort(key=lambda c: (c.on, 0 if c.direction == "debit" else 1, c.event_id or ""))
    return ledger


def _to_home(v: MessageVerdict, series: RecurringSeries, convert, home: str) -> Optional[Decimal]:
    if v.amount is None:
        return None
    ccy = v.currency or home
    if ccy == home or convert is None:
        return v.amount
    try:
        return convert(v.amount, ccy, home, v.effective_date or series.last_date)
    except FxMissing:
        return v.amount


def _detect(key, members: list[CashEvent], by_id) -> tuple[Optional[RecurringSeries], str]:
    if len(members) < MIN_OCCURRENCES:
        return None, f"only {len(members)} occurrence(s)"
    dates = [m.on for m in members]
    gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
    if any(g <= 0 for g in gaps):
        return None, "two occurrences on one day"
    med = _median_int(gaps)
    if any(abs(g - med) > GAP_TOLERANCE_DAYS for g in gaps):
        return None, f"gaps {gaps} not within {GAP_TOLERANCE_DAYS}d of median {med}"
    last = by_id[members[-1].event_id]
    amounts = [m.amount for m in members]
    return RecurringSeries(
        key=key, category=last.category, description=last.description, direction=last.direction,
        flexibility=last.flexibility, minimum_allowed_amount=last.minimum_allowed_amount,
        event_ids=[m.event_id for m in members], dates=dates, amounts=amounts, gap_days=med,
        amount=_median_dec(amounts[-AMOUNT_WINDOW:]),
    ), ""


def _apply_verdict(v: MessageVerdict, user: Profile, work, by_id, ledger: Ledger, exclude, conflict,
                   as_of: date, convert):
    home = user.home_currency
    targets = [i for i in ((v.event_id,) + tuple(v.event_ids)) if i]
    known = [i for i in targets if i in work]
    for i in targets:
        if i not in work:
            conflict((i,), f"{v.kind} refers to an unknown event", "ignored", v.message_id)

    if v.kind == "confirm":
        # a message cannot reinstate a row the record explicitly cancelled or failed (precedence rule 1)
        for i in known:
            if work[i]["status"] in ("cancelled", "failed"):
                conflict((i,), f"message confirms a {work[i]['status']} event",
                         f"explicit {work[i]['status']} record wins; message from {v.message_id} not applied", v.message_id)
        return
    if v.kind in ("salary_change", "expense_change"):
        return   # series-level kinds are applied after recurrence detection

    if v.kind in ("cancel_event", "not_cash", "internal_transfer"):
        reason = {"cancel_event": "cancelled_by_message", "not_cash": "not_cash_per_message",
                  "internal_transfer": "internal_transfer_per_message"}[v.kind]
        for i in known:
            e = by_id[i]
            if v.kind == "cancel_event" and work[i]["status"] == "settled" and e.direction == "debit":
                conflict((i,), "message cancels a settled debit", "unresolved:settled debit kept (safer)", v.message_id)
                continue
            exclude(i, reason, v.message_id)
        return

    if v.kind == "amend_amount":
        for i in known:
            if v.amount is None:
                conflict((i,), "amend_amount without an amount", "ignored", v.message_id)
                continue
            old = work[i]["amount"]
            work[i]["amount"] = v.amount
            if v.currency:
                work[i]["currency"] = v.currency
            conflict((i,), f"amount {old} amended to {v.amount}", "message amendment wins", v.message_id)
        return

    if v.kind == "delay_event":
        for i in known:
            if v.effective_date is None:
                conflict((i,), "delay_event without a date", "ignored", v.message_id)
                continue
            work[i]["cash_date"] = v.effective_date
            if work[i]["status"] == "settled":
                work[i]["status"] = "scheduled"
        return

    if v.kind == "confirmed_income":
        if v.amount is None or v.effective_date is None:
            conflict((), "confirmed_income without amount/date", "ignored (no income invented)", v.message_id)
            return
        amount = v.amount
        ccy = v.currency or home
        if ccy != home:
            if convert is None:
                conflict((), f"confirmed_income in {ccy} without a converter", "ignored", v.message_id)
                return
            try:
                amount = convert(amount, ccy, home, v.effective_date)
            except FxMissing as err:
                conflict((), str(err), "ignored (no rate)", v.message_id)
                return
        # a scheduled row on (nearly) the same date is the same money: do not add twice
        for i, w in work.items():
            e = by_id[i]
            if (not w["excluded"] and e.direction == "credit" and w["status"] == "scheduled"
                    and abs((w["cash_date"] - v.effective_date).days) <= SAME_OCCURRENCE_DAYS):
                conflict((i,), "confirmed_income matches an existing scheduled credit", f"row {i} kept, message not added twice", v.message_id)
                return
        ledger.scheduled_one_offs.append(CashEvent(max(v.effective_date, as_of), amount, "credit", None, "salary",
                                                   v.note or "confirmed income", "confirmed", v.message_id))
        return

    if v.kind == "retry_debit":
        for i in known:
            e = by_id[i]
            already = any(o.linked_event_id == i and work[o.event_id]["status"] == "scheduled" for o in by_id.values())
            if already:
                conflict((i,), "retry already present as a scheduled row", "scheduled row kept", v.message_id)
                continue
            amount = work[i]["amount"]
            if amount is None:
                conflict((i,), "retry_debit on an event without an amount", "ignored", v.message_id)
                continue
            ccy = work[i]["currency"]
            if ccy != home and convert is not None:
                try:
                    amount = convert(amount, ccy, home, v.effective_date or as_of)
                except FxMissing:
                    pass
            ledger.scheduled_one_offs.append(CashEvent(max(v.effective_date or as_of, as_of), amount, "debit", None,
                                                       e.category, e.description, "retry", v.message_id))
        return

    conflict(tuple(known), f"unknown verdict kind {v.kind!r}", "ignored", v.message_id)
