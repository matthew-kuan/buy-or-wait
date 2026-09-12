"""Capacity questions over a Ledger. Pure functions, Decimal throughout, no I/O, no model calls.

    amount_safe_to_pay(ledger, request_date, requested_amount, balance, minimum_balance)
    earliest_full_payment_date(ledger, request_date, requested_amount, balance, minimum_balance)
    find_spending_changes(ledger, request_date, balance, minimum_balance, plan_payments, profile)

The first two ignore payment-method preferences and spending changes entirely: they measure what
the user's money can do, not what the user prefers. The third searches the smallest legal set of
spending changes (schema.py checks 12-14) that makes a concrete payment plan safe.
"""
from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal, ROUND_DOWN
from itertools import combinations
from typing import Optional, Sequence

from forecast import HORIZON_DAYS, build_series, is_safe
from io_layer import Profile
from reconcile import Ledger, RecurringSeries
from schema import REDUCIBLE, STOPPABLE, MAX_SPENDING_CHANGES

CENT = Decimal("0.01")
DAYS_PER_MONTH = Decimal(30)


def _safe_with(ledger: Ledger, request_date: date, balance: Decimal, minimum: Decimal,
               payments: Sequence[tuple[date, Decimal]], changes: Sequence[tuple] = (),
               horizon_days: int = HORIZON_DAYS) -> bool:
    series = build_series(ledger, request_date, balance, horizon_days, extra_payments=payments, changes=changes)
    return is_safe(series, minimum)


def amount_safe_to_pay(ledger: Ledger, request_date: date, requested_amount: Decimal, balance: Decimal,
                       minimum_balance: Decimal, horizon_days: int = HORIZON_DAYS) -> Decimal:
    """Largest payment on request_date (to the cent, rounded down) that keeps the 90-day series at or
    above the minimum, with no spending changes. Clamped to [0, requested_amount]."""
    requested_cents = int((requested_amount / CENT).to_integral_value(rounding=ROUND_DOWN))
    if requested_cents <= 0:
        return Decimal(0)

    def ok(cents: int) -> bool:
        return _safe_with(ledger, request_date, balance, minimum_balance,
                          [(request_date, Decimal(cents) * CENT)], horizon_days=horizon_days)

    if not ok(0):
        return Decimal(0)
    if ok(requested_cents):
        return requested_amount
    lo, hi = 0, requested_cents          # ok(lo) is True, ok(hi) is False
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return Decimal(lo) * CENT


def earliest_full_payment_date(ledger: Ledger, request_date: date, requested_amount: Decimal, balance: Decimal,
                               minimum_balance: Decimal, horizon_days: int = HORIZON_DAYS) -> Optional[date]:
    """First date within the horizon on which a single payment of requested_amount is safe, or None."""
    for offset in range(horizon_days):
        day = request_date + timedelta(days=offset)
        if _safe_with(ledger, request_date, balance, minimum_balance, [(day, requested_amount)],
                      horizon_days=horizon_days):
            return day
    return None


# --------------------------------------------------------------------------- spending changes

def legal_changes(ledger: Ledger, profile: Profile) -> list[tuple[tuple, Decimal]]:
    """Every single legal change as (change, monthly_reduction). A change is
    ("stop", last_event_id) or ("reduce_to", last_event_id, minimum_allowed_amount), keyed by the
    series' most recent occurrence - the id the sample answers cite. Legality mirrors schema.py
    checks 12-14: flexibility allows it, the category is in the matching willing list, and the
    category is not protected."""
    protect = set(profile.expense_categories_to_protect)
    may_stop = set(profile.expense_categories_user_is_willing_to_stop)
    may_reduce = set(profile.expense_categories_user_is_willing_to_reduce)
    out = []
    for s in ledger.recurring_debits:
        if s.category in protect:
            continue
        per_month = DAYS_PER_MONTH / Decimal(s.gap_days)
        if s.flexibility in STOPPABLE and s.category in may_stop:
            out.append((("stop", s.last_event_id), s.amount * per_month))
        if (s.flexibility in REDUCIBLE and s.category in may_reduce
                and s.minimum_allowed_amount is not None and s.minimum_allowed_amount < s.amount):
            out.append((("reduce_to", s.last_event_id, s.minimum_allowed_amount),
                        (s.amount - s.minimum_allowed_amount) * per_month))
    return out


def find_spending_changes(ledger: Ledger, request_date: date, balance: Decimal, minimum_balance: Decimal,
                          plan_payments: Sequence[tuple[date, Decimal]], profile: Profile,
                          max_changes: int = MAX_SPENDING_CHANGES,
                          horizon_days: int = HORIZON_DAYS) -> Optional[list[tuple]]:
    """Smallest legal combination (at most max_changes, one action per event) that makes
    plan_payments safe. Preference: fewer changes, then the smallest total monthly reduction, then
    the lowest event ids. None when no combination works (or the plan is already safe: returns [])."""
    if _safe_with(ledger, request_date, balance, minimum_balance, plan_payments, horizon_days=horizon_days):
        return []
    candidates = legal_changes(ledger, profile)
    if not candidates:
        return None
    # if even every change together is not enough, nothing smaller will be
    for size in range(1, min(max_changes, len(candidates)) + 1):
        best = None
        for combo in combinations(candidates, size):
            ids = [c[0][1] for c in combo]
            if len(set(ids)) != len(ids):
                continue                                   # never stop and reduce the same event
            changes = [c[0] for c in combo]
            if not _safe_with(ledger, request_date, balance, minimum_balance, plan_payments, changes,
                              horizon_days=horizon_days):
                continue
            total = sum((c[1] for c in combo), Decimal(0))
            rank = (total, sorted(_id_key(i) for i in ids))
            if best is None or rank < best[0]:
                best = (rank, changes)
        if best is not None:
            return best[1]
    return None


def _id_key(event_id: str):
    head, _, tail = event_id.rpartition("_")
    return (0, int(tail), head) if tail.isdigit() else (1, 0, event_id)
