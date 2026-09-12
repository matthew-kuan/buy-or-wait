"""One request end to end: evidence -> ledger -> solver -> ranker -> explanation -> validated row.

    ev = gather_evidence(ds, extractor)          # once per run (cached model calls)
    out = decide(ds, request, ev)                # per request; never raises - see out.error

Shared by code/main.py (full dataset) and code/evaluation/main.py (scoring against the samples).
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Optional

import explain
import schema
from forecast import build_series
from io_layer import Dataset, Event, PaymentOption, Profile, Request
from ranker import Decision, SolverResults, compute_solver_results, rank
from reconcile import ImageFact, Ledger, MessageVerdict, build_ledger


@dataclass
class Evidence:
    verdicts_by_user: dict[str, list[MessageVerdict]] = field(default_factory=dict)
    facts_by_event: dict[str, ImageFact] = field(default_factory=dict)
    review_flags: list[tuple[str, str, str]] = field(default_factory=list)   # (user_id, source_id, reason)
    stats: Optional[dict] = None


def gather_evidence(ds: Dataset, extractor=None) -> Evidence:
    """Run (or replay from cache) the extraction passes and bridge them into reconcile shapes.
    With extractor=None no model output is used at all."""
    ev = Evidence()
    if extractor is None:
        return ev
    from extraction import run_all, to_image_fact, to_verdict
    classifications, extractions = run_all(ds, extractor)
    msg_by_id = {m.message_id: m for m in ds.messages}
    for order, (mid, c) in enumerate(sorted(classifications.items(), key=lambda kv: msg_by_id[kv[0]].sent_at)):
        m = msg_by_id[mid]
        if c.injection_suspected:
            ev.review_flags.append((m.user_id, mid, "injection_suspected"))
        if c.fallback:
            ev.review_flags.append((m.user_id, mid, f"extraction_fallback: {c.error[:80]}"))
        v = to_verdict(c, m, order)
        if v is not None:
            ev.verdicts_by_user.setdefault(m.user_id, []).append(v)
    for iid, e in extractions.items():
        im = next(i for i in ds.images if i.image_id == iid)
        if e.contains_instructions:
            ev.review_flags.append((im.user_id, iid, "image_contains_instructions"))
        if e.fallback:
            ev.review_flags.append((im.user_id, iid, f"extraction_fallback: {e.error[:80]}"))
        f = to_image_fact(e)
        if f.event_id:
            ev.facts_by_event[f.event_id] = f
    ev.stats = extractor.stats.to_dict()
    return ev


@dataclass
class Outcome:
    request: Request
    row: dict                                 # the 8 output columns as strings
    decision: Optional[Decision] = None
    results: Optional[SolverResults] = None
    ledger: Optional[Ledger] = None
    series: Optional[list] = None
    facts: Optional[explain.ExplainFacts] = None
    validation_error: str = ""
    error: str = ""                           # non-empty when the pipeline failed and a fallback row was used

    @property
    def ok(self) -> bool:
        return not self.error and not self.validation_error


# --------------------------------------------------------------------------- dict views for schema.validate_row

def _request_dict(r: Request) -> dict:
    return {"request_id": r.request_id, "user_id": r.user_id, "request_date": r.request_date.isoformat(),
            "request_type": r.request_type, "requested_amount": str(r.requested_amount),
            "desired_completion_date": r.desired_completion_date.isoformat(),
            "allows_partial_payment": "true" if r.allows_partial_payment else "false", "request_text": r.request_text}


def _profile_dict(p: Profile) -> dict:
    return {"user_id": p.user_id, "home_currency": p.home_currency,
            "current_available_balance": str(p.current_available_balance),
            "minimum_balance_to_keep": str(p.minimum_balance_to_keep),
            "expense_categories_to_protect": "|".join(p.expense_categories_to_protect),
            "expense_categories_user_is_willing_to_reduce": "|".join(p.expense_categories_user_is_willing_to_reduce),
            "expense_categories_user_is_willing_to_stop": "|".join(p.expense_categories_user_is_willing_to_stop),
            "payment_methods_user_will_consider": "|".join(p.payment_methods_user_will_consider),
            "max_installment_months": "" if p.max_installment_months is None else str(p.max_installment_months)}


def _option_dict(o: PaymentOption) -> dict:
    return {"payment_option_id": o.payment_option_id, "request_id": o.request_id, "payment_method": o.payment_method,
            "payment_amount": str(o.payment_amount), "number_of_payments": str(o.number_of_payments),
            "first_payment_date": o.first_payment_date.isoformat(),
            "payment_frequency_days": "" if o.payment_frequency_days is None else str(o.payment_frequency_days)}


def _event_dict(e: Event, override_amount: Optional[Decimal] = None) -> dict:
    amount = override_amount if override_amount is not None else e.amount
    return {"event_id": e.event_id, "user_id": e.user_id, "category": e.category, "flexibility": e.flexibility,
            "amount": "" if amount is None else str(amount),
            "minimum_allowed_amount": "" if e.minimum_allowed_amount is None else str(e.minimum_allowed_amount)}


# --------------------------------------------------------------------------- decide

def decide(ds: Dataset, request: Request, evidence: Evidence | None = None) -> Outcome:
    evidence = evidence or Evidence()
    profile = ds.profile_by_user[request.user_id]
    events = ds.user_events(request.user_id)
    options = ds.request_options(request.request_id)
    try:
        verdicts = evidence.verdicts_by_user.get(request.user_id, [])
        facts_img = [evidence.facts_by_event[e.event_id] for e in events if e.event_id in evidence.facts_by_event]
        ledger = build_ledger(profile, events, verdicts, facts_img, request.request_date, convert=ds.convert)
        results = compute_solver_results(ledger, request, profile)
        decision = rank(request, profile, options, ledger, results)
        series = build_series(ledger, request.request_date, profile.current_available_balance,
                              extra_payments=decision.payment_plan, changes=decision.spending_changes_needed)
        facts = explain.build_facts(decision, request, profile, ledger, series)
        row = {
            "request_id": request.request_id,
            "amount_safe_to_pay": schema.format_amount_safe(decision.amount_safe_to_pay),
            "affordability_status": decision.affordability_status,
            "recommended_payment_method": decision.recommended_payment_method,
            "payment_plan": schema.format_payment_plan(decision.payment_plan),
            "earliest_date_for_full_payment": (decision.earliest_date_for_full_payment.isoformat()
                                               if decision.earliest_date_for_full_payment else ""),
            "spending_changes_needed": schema.format_spending_changes(decision.spending_changes_needed),
            "decision_explanation": explain.generate(facts),
        }
        out = Outcome(request, row, decision, results, ledger, series, facts)
    except Exception:  # noqa: BLE001 - per-row isolation: any failure yields a valid conservative row
        out = Outcome(request, _fallback_row(request, profile), error=traceback.format_exc(limit=3))
        return out

    schema.normalize_row(out.row)
    image_amounts = {f.event_id: f.amount for f in facts_img if f.amount is not None}
    try:
        schema.validate_row(out.row, _request_dict(request), [_option_dict(o) for o in options], _profile_dict(profile),
                            [_event_dict(e, image_amounts.get(e.event_id)) for e in events],
                            facts={"full_payment_safe_within_90_days": results.earliest_full_payment_date is not None,
                                   "numbers": facts.numbers()})
    except schema.ValidationError as err:
        out.validation_error = str(err)
    return out


def _fallback_row(request: Request, profile: Profile) -> dict:
    mn = explain.money(profile.home_currency, profile.minimum_balance_to_keep)
    return {"request_id": request.request_id, "amount_safe_to_pay": "0", "affordability_status": "not_affordable",
            "recommended_payment_method": "not_recommended", "payment_plan": "none",
            "earliest_date_for_full_payment": "", "spending_changes_needed": "none",
            "decision_explanation": f"Do not make this payment. The {mn} minimum cannot be protected."}


def candidate_table(decision: Optional[Decision]) -> str:
    if decision is None:
        return ""
    parts = []
    for c in decision.candidates:
        parts.append(f"{c.method}[{c.payment_option_id or '-'}] n={c.payment_count} start={c.start_date} "
                     f"total={c.total_paid} deadline={'y' if c.completes_by_deadline else 'n'} "
                     f"changes={len(c.spending_changes)} -> {c.reason}")
    return " || ".join(parts)
