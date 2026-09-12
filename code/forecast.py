"""Day-by-day balance projection over a Ledger. Pure functions, Decimal throughout.

    series = build_series(ledger, start_date, balance, horizon_days=90, extra_payments=(), changes=())
    min_balance(series) -> Decimal
    is_safe(series, minimum_balance) -> bool

`balance` is the opening balance at the start of `start_date`, before anything settles that day.
Each entry is (date, closing_balance). Within a day every movement - recurring, confirmed, and
`extra_payments` - is applied before the close, so a salary settling on day D is available for a
payment on day D. That matches the sample answers, whose "wait" and second-partial dates fall on
salary days themselves.

`changes` are spending changes in schema.py's parsed form: ("stop", event_id) or
("reduce_to", event_id, amount). An event_id may name any occurrence of a recurring series; it is
resolved to the series it belongs to.
"""
from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from typing import Iterable, Optional, Sequence

from reconcile import Ledger

HORIZON_DAYS = 90


def _series_key_map(ledger: Ledger) -> dict[str, str]:
    """Any event_id in a recurring series -> that series' last_event_id (the key flows() uses)."""
    out = {}
    for s in ledger.recurring_debits + ledger.recurring_credits:
        for event_id in s.event_ids:
            out[event_id] = s.last_event_id
    return out


def resolve_changes(ledger: Ledger, changes: Iterable[tuple]) -> tuple[set[str], dict[str, Decimal]]:
    """Split parsed spending changes into the (stopped, reduced) arguments Ledger.flows() takes."""
    keymap = _series_key_map(ledger)
    stopped: set[str] = set()
    reduced: dict[str, Decimal] = {}
    for change in changes:
        kind, event_id = change[0], change[1]
        key = keymap.get(event_id, event_id)
        if kind == "stop":
            stopped.add(key)
        elif kind == "reduce_to":
            reduced[key] = Decimal(change[2])
        else:
            raise ValueError(f"unknown spending change kind {kind!r}")
    return stopped, reduced


def build_series(ledger: Ledger, start_date: date, balance: Decimal, horizon_days: int = HORIZON_DAYS,
                 extra_payments: Sequence[tuple[date, Decimal]] = (),
                 changes: Sequence[tuple] = ()) -> list[tuple[date, Decimal]]:
    if not isinstance(balance, Decimal):
        raise TypeError(f"balance must be Decimal, got {type(balance).__name__}")
    end = start_date + timedelta(days=horizon_days - 1)
    stopped, reduced = resolve_changes(ledger, changes)

    daily: dict[date, Decimal] = {}
    for flow in ledger.flows(start_date, end, stopped=stopped, reduced=reduced):
        signed = flow.amount if flow.direction == "credit" else -flow.amount
        daily[flow.on] = daily.get(flow.on, Decimal(0)) + signed
    for on, amount in extra_payments:
        if not isinstance(amount, Decimal):
            raise TypeError(f"extra payment amount must be Decimal, got {type(amount).__name__}")
        if start_date <= on <= end:
            daily[on] = daily.get(on, Decimal(0)) - amount

    out = []
    running = balance
    for i in range(horizon_days):
        day = start_date + timedelta(days=i)
        running += daily.get(day, Decimal(0))
        out.append((day, running))
    return out


def min_balance(series: Sequence[tuple[date, Decimal]]) -> Optional[Decimal]:
    if not series:
        return None
    return min(b for _, b in series)


def is_safe(series: Sequence[tuple[date, Decimal]], minimum_balance: Decimal) -> bool:
    low = min_balance(series)
    if low is None:
        return True
    return low >= minimum_balance


def first_breach(series: Sequence[tuple[date, Decimal]], minimum_balance: Decimal) -> Optional[tuple[date, Decimal]]:
    """The first (date, balance) below the minimum, for explanations."""
    for day, balance in series:
        if balance < minimum_balance:
            return day, balance
    return None
