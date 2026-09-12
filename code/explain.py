"""decision_explanation from solver facts only. No model call, no free text from evidence.

    text = generate(facts)

`facts` is an ExplainFacts (or a dict with the same keys). Every number that appears in the
output must already be in the facts - after generation the text is scanned and any numeric token
not present in the facts triggers a deterministic fallback string, recorded in FALLBACK_LOG.
That is the hallucination containment for the only free-text output column.

Style follows dataset/sample_requests.csv: currency code + thousands separators, dates as
"15 November 2019", the minimum balance named, one or two sentences, under 160 characters.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Sequence

from schema import numbers_in_text

log = logging.getLogger("explain")
FALLBACK_LOG: list[dict] = []          # {"request_id", "reason", "attempted"}
MAX_CHARS = 160
FORECAST_DAYS = 90


@dataclass
class ExplainFacts:
    request_id: str
    currency: str
    method: str                                   # full_payment | partial_payment | installments | wait | not_recommended
    status: str
    requested_amount: Decimal
    amount_safe_to_pay: Decimal
    request_date: date
    desired_completion_date: date
    minimum_balance: Decimal
    payment_plan: list[tuple[date, Decimal]] = field(default_factory=list)
    earliest_date: Optional[date] = None
    forecast_min: Optional[Decimal] = None
    forecast_min_date: Optional[date] = None
    spending_changes: list[dict] = field(default_factory=list)   # {kind, event_id, description, new_amount}
    runner_up_rule: Optional[int] = None
    accepted_methods: Sequence[str] = ()
    allows_partial_payment: bool = False

    def numbers(self) -> set[Decimal]:
        out = {self.requested_amount, self.amount_safe_to_pay, self.minimum_balance,
               Decimal(FORECAST_DAYS), Decimal(len(self.payment_plan))}
        out.update(a for _, a in self.payment_plan)
        if self.forecast_min is not None:
            out.add(self.forecast_min)
        for c in self.spending_changes:
            if c.get("new_amount") is not None:
                out.add(Decimal(c["new_amount"]))
        if self.runner_up_rule is not None:
            out.add(Decimal(self.runner_up_rule))
        return out


# --------------------------------------------------------------------------- formatting

def money(ccy: str, amount: Decimal) -> str:
    amount = Decimal(amount)
    if amount == amount.to_integral_value():
        return f"{ccy} {int(amount):,}"
    return f"{ccy} {amount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):,}"


def day(d: date) -> str:
    return f"{d.day} {d:%B} {d.year}"


def _desc(change: dict) -> str:
    text = str(change.get("description") or change.get("event_id") or "expense").strip()
    return text[0].lower() + text[1:] if text else "expense"


def _change_phrase(ccy: str, changes: list[dict]) -> str:
    parts = []
    for c in changes:
        if c["kind"] == "stop":
            parts.append(f"stop the {_desc(c)}")
        else:
            parts.append(f"reduce the {_desc(c)} to {money(ccy, c['new_amount'])}")
    if len(parts) == 1:
        text = parts[0]
    else:
        text = ", ".join(parts[:-1]) + " and " + parts[-1]
    return text[0].upper() + text[1:]


# --------------------------------------------------------------------------- shapes

def _compose(f: ExplainFacts) -> list[str]:
    """Candidate texts in preference order; the first one that passes the guard and the length wins."""
    ccy, req, mn = f.currency, f.requested_amount, f.minimum_balance
    if f.method == "full_payment" and f.spending_changes:
        head = f"{_change_phrase(ccy, f.spending_changes)}, then pay {money(ccy, req)} today."
        return [f"{head} This leaves at least {money(ccy, mn)} available.", head]
    if f.method == "full_payment":
        return [f"Pay {money(ccy, req)} today. This leaves at least {money(ccy, mn)} available over the next {FORECAST_DAYS} days.",
                f"Pay {money(ccy, req)} today. This leaves at least {money(ccy, mn)} available."]
    if f.method == "installments":
        n = len(f.payment_plan)
        first_date, amount = f.payment_plan[0]
        return [f"Use {n} installments of {money(ccy, amount)}, starting {day(first_date)}. This leaves at least {money(ccy, mn)} available.",
                f"Use {n} installments of {money(ccy, amount)}, starting {day(first_date)}."]
    if f.method == "wait":
        when = f.payment_plan[0][0] if f.payment_plan else f.earliest_date
        return [f"Pay {money(ccy, req)} in full on {day(when)}. Paying earlier would take the balance below the {money(ccy, mn)} minimum.",
                f"Pay {money(ccy, req)} in full on {day(when)}."]
    if f.method == "partial_payment":
        (d1, a1), (d2, a2) = f.payment_plan
        return [f"Pay {money(ccy, a1)} today and the remaining {money(ccy, a2)} on {day(d2)}. This completes the full request and keeps the {money(ccy, mn)} minimum protected.",
                f"Pay {money(ccy, a1)} today and the remaining {money(ccy, a2)} on {day(d2)}. This keeps the {money(ccy, mn)} minimum protected."]
    # not_recommended
    partial_only = (set(f.accepted_methods) == {"partial_payment"} and f.allows_partial_payment
                    and f.amount_safe_to_pay > 0 and f.earliest_date is None)
    if partial_only:
        return [f"Do not proceed with the {money(ccy, req)} request. Although {money(ccy, f.amount_safe_to_pay)} is available today, the full amount cannot be completed safely within {FORECAST_DAYS} days.",
                f"Do not proceed with the {money(ccy, req)} request. The full amount cannot be completed safely within {FORECAST_DAYS} days."]
    return [f"Do not make this payment by {day(f.desired_completion_date)}. None of the available options keeps the {money(ccy, mn)} minimum protected.",
            f"Do not make this payment. None of the available options keeps the {money(ccy, mn)} minimum protected."]


def _fallback(f: ExplainFacts) -> str:
    mn = money(f.currency, f.minimum_balance)
    if f.method == "not_recommended":
        return f"Do not make this payment. The {mn} minimum cannot be protected."
    label = {"full_payment": "Pay in full today", "wait": "Wait, then pay in full",
             "partial_payment": "Pay part now and the rest later", "installments": "Use the installment plan"}[f.method]
    return f"{label}. This keeps the {mn} minimum protected."


# --------------------------------------------------------------------------- guard

def guard(text: str, facts: ExplainFacts) -> list[Decimal]:
    """Numeric tokens in `text` that are absent from the facts (dates excluded). Empty means clean."""
    allowed = facts.numbers()
    return [n for n in numbers_in_text(text) if n not in allowed]


def _coerce(facts) -> ExplainFacts:
    if isinstance(facts, ExplainFacts):
        return facts
    return ExplainFacts(**facts)


def generate(facts) -> str:
    f = _coerce(facts)
    attempted = []
    for text in _compose(f):
        stray = guard(text, f)
        if not stray and len(text) <= MAX_CHARS:
            return text
        attempted.append((text, [str(n) for n in stray], len(text)))
    fb = _fallback(f)
    reason = "; ".join(f"len={ln} stray={stray}" for _, stray, ln in attempted)
    FALLBACK_LOG.append({"request_id": f.request_id, "reason": reason, "attempted": [t for t, _, _ in attempted]})
    log.warning("explain fallback for %s: %s", f.request_id, reason)
    if guard(fb, f) or len(fb) > MAX_CHARS:            # the fallback itself must be clean by construction
        raise AssertionError(f"fallback explanation failed the guard for {f.request_id}: {fb!r}")
    return fb


# --------------------------------------------------------------------------- integration helper

def build_facts(decision, request, profile, ledger=None, series=None) -> ExplainFacts:
    """Assemble ExplainFacts from a ranker.Decision plus the objects that produced it."""
    from forecast import min_balance
    descriptions = {}
    if ledger is not None:
        for s in ledger.recurring_debits:
            for eid in s.event_ids:
                descriptions[eid] = s.description
    changes = []
    for c in decision.spending_changes_needed:
        changes.append({"kind": c[0], "event_id": c[1], "description": descriptions.get(c[1], c[1]),
                        "new_amount": c[2] if c[0] == "reduce_to" else None})
    fmin = fmin_date = None
    if series:
        fmin = min_balance(series)
        fmin_date = next(d for d, b in series if b == fmin)
    runner_up = None
    if decision.winner is not None:
        others = [c for c in decision.survivors if c is not decision.winner]
        if others:
            runner_up = others[0].eliminated_by
    return ExplainFacts(
        request_id=request.request_id, currency=profile.home_currency,
        method=decision.recommended_payment_method, status=decision.affordability_status,
        requested_amount=request.requested_amount, amount_safe_to_pay=decision.amount_safe_to_pay,
        request_date=request.request_date, desired_completion_date=request.desired_completion_date,
        minimum_balance=profile.minimum_balance_to_keep, payment_plan=list(decision.payment_plan),
        earliest_date=decision.earliest_date_for_full_payment, forecast_min=fmin, forecast_min_date=fmin_date,
        spending_changes=changes, runner_up_rule=runner_up,
        accepted_methods=tuple(profile.payment_methods_user_will_consider),
        allows_partial_payment=request.allows_partial_payment,
    )
