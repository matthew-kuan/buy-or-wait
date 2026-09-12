"""Candidate plans, eligibility, ranking, and the affordability_status mapping.

    results  = compute_solver_results(ledger, request, profile)
    decision = rank(request, profile, options, ledger, results)

Everything is deterministic and Decimal. The ranker never computes capacity itself - it asks
solver.py - and it never writes prose.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Callable, Optional, Sequence

import forecast
import solver
from io_layer import PaymentOption, Profile, Request
from reconcile import Ledger

RANKING_RULES = {
    1: "completes the full request by desired_completion_date",
    2: "requires no spending changes",
    3: "lowest total amount paid",
    4: "earliest start date",
    5: "fewest payments",
    6: "lowest payment_option_id",
}


@dataclass(frozen=True)
class SolverResults:
    amount_safe_to_pay: Decimal
    earliest_full_payment_date: Optional[date]
    full_safe_today: bool


def compute_solver_results(ledger: Ledger, request: Request, profile: Profile) -> SolverResults:
    safe = solver.amount_safe_to_pay(ledger, request.request_date, request.requested_amount,
                                     profile.current_available_balance, profile.minimum_balance_to_keep)
    earliest = solver.earliest_full_payment_date(ledger, request.request_date, request.requested_amount,
                                                 profile.current_available_balance, profile.minimum_balance_to_keep)
    return SolverResults(safe, earliest, safe == request.requested_amount)


@dataclass
class Candidate:
    method: str                                   # full_payment | partial_payment | installments | wait
    schedule: list[tuple[date, Decimal]]
    total_paid: Decimal
    completes_by_deadline: bool
    spending_changes: list[tuple]
    payment_count: int
    start_date: date
    payment_option_id: Optional[str]
    eligible: bool = True
    safe: Optional[bool] = None
    reason: str = ""                               # why it was filtered or which rule beat it
    eliminated_by: Optional[int] = None            # ranking rule index (1-6); None for winner / filtered

    @property
    def sort_key(self):
        return (0 if self.completes_by_deadline else 1,
                1 if self.spending_changes else 0,
                self.total_paid,
                self.start_date,
                self.payment_count,
                _option_key(self.payment_option_id))

    @property
    def survived(self) -> bool:
        return self.eligible and bool(self.safe)


@dataclass
class Decision:
    request_id: str
    amount_safe_to_pay: Decimal
    affordability_status: str
    recommended_payment_method: str
    payment_plan: list[tuple[date, Decimal]]
    earliest_date_for_full_payment: Optional[date]
    spending_changes_needed: list[tuple]
    winner: Optional[Candidate]
    candidates: list[Candidate] = field(default_factory=list)   # ranked: survivors first, then filtered

    @property
    def survivors(self) -> list[Candidate]:
        return [c for c in self.candidates if c.survived]


def _option_key(option_id: Optional[str]):
    if option_id is None:
        return (1, 0, "")
    head, _, tail = option_id.rpartition("_")
    return (0, int(tail), head) if tail.isdigit() else (0, 0, option_id)


# --------------------------------------------------------------------------- candidates

def enumerate_plans(request: Request, profile: Profile, options: Sequence[PaymentOption], ledger: Ledger,
                    results: SolverResults) -> list[Candidate]:
    rd, requested, desired = request.request_date, request.requested_amount, request.desired_completion_date
    safe, earliest = results.amount_safe_to_pay, results.earliest_full_payment_date
    full_option = next((o for o in options if o.payment_method == "full_payment"), None)
    full_id = full_option.payment_option_id if full_option else None
    out: list[Candidate] = []

    # full payment today, only if the whole amount is safe today
    if results.full_safe_today:
        out.append(Candidate("full_payment", [(rd, requested)], requested, rd <= desired, [], 1, rd, full_id))

    # full payment today with spending changes (only when it is not already safe)
    if not results.full_safe_today:
        changes = solver.find_spending_changes(ledger, rd, profile.current_available_balance,
                                               profile.minimum_balance_to_keep, [(rd, requested)], profile)
        if changes:
            out.append(Candidate("full_payment", [(rd, requested)], requested, rd <= desired, list(changes), 1, rd, full_id))

    # wait for the first safe full-payment date (identical to "full today" when that date is today)
    if earliest is not None and earliest > rd:
        out.append(Candidate("wait", [(earliest, requested)], requested, earliest <= desired, [], 1, earliest, None))

    # partial: exactly two payments per the spec
    if (request.allows_partial_payment and earliest is not None and Decimal(0) < safe < requested
            and earliest <= desired):
        out.append(Candidate("partial_payment", [(rd, safe), (earliest, requested - safe)], requested, True, [],
                             2, rd, None))

    # one candidate per installments option
    for opt in options:
        if opt.payment_method != "installments":
            continue
        schedule = opt.schedule()
        out.append(Candidate("installments", schedule, opt.total_payable_amount, schedule[-1][0] <= desired, [],
                             opt.number_of_payments, schedule[0][0], opt.payment_option_id))
    return out


# --------------------------------------------------------------------------- eligibility + safety

def default_safety(ledger: Ledger, profile: Profile, request: Request) -> Callable[[Candidate], bool]:
    def is_safe(c: Candidate) -> bool:
        series = forecast.build_series(ledger, request.request_date, profile.current_available_balance,
                                       extra_payments=c.schedule, changes=c.spending_changes)
        return forecast.is_safe(series, profile.minimum_balance_to_keep)
    return is_safe


def filter_candidates(candidates: list[Candidate], profile: Profile, options: Sequence[PaymentOption],
                      results: SolverResults, safety: Callable[[Candidate], bool]) -> None:
    accepted = set(profile.payment_methods_user_will_consider)
    by_id = {o.payment_option_id: o for o in options}
    for c in candidates:
        if c.method == "wait":
            if results.earliest_full_payment_date is None or "full_payment" not in accepted:
                c.eligible, c.reason = False, "ineligible: wait needs an earliest date and full_payment accepted"
        elif c.method not in accepted:
            c.eligible, c.reason = False, f"ineligible: {c.method} not in payment_methods_user_will_consider"
        if c.eligible and c.method == "installments":
            n = by_id[c.payment_option_id].number_of_payments
            if profile.max_installment_months is None:
                c.eligible, c.reason = False, "ineligible: max_installment_months is blank"
            elif n > profile.max_installment_months:
                c.eligible, c.reason = False, f"ineligible: {n} payments > max_installment_months {profile.max_installment_months}"
        if not c.eligible:
            continue
        c.safe = safety(c)
        if not c.safe:
            c.reason = "unsafe: balance would fall below minimum_balance_to_keep"


# --------------------------------------------------------------------------- ranking

def rank(request: Request, profile: Profile, options: Sequence[PaymentOption], ledger: Ledger,
         results: SolverResults, safety: Callable[[Candidate], bool] | None = None) -> Decision:
    candidates = enumerate_plans(request, profile, options, ledger, results)
    filter_candidates(candidates, profile, options, results, safety or default_safety(ledger, profile, request))

    survivors = sorted((c for c in candidates if c.survived), key=lambda c: c.sort_key)
    filtered = [c for c in candidates if not c.survived]
    winner = survivors[0] if survivors else None
    if winner:
        winner.reason = "winner"
        wk = winner.sort_key
        for c in survivors[1:]:
            ck = c.sort_key
            idx = next((i + 1 for i, (a, b) in enumerate(zip(ck, wk)) if a != b), None)
            c.eliminated_by = idx
            c.reason = f"lost on rule {idx}: {RANKING_RULES[idx]}" if idx else "tie with winner (identical key)"

    status, method = _status(winner, request, results)
    return Decision(
        request_id=request.request_id,
        amount_safe_to_pay=results.amount_safe_to_pay,
        affordability_status=status,
        recommended_payment_method=method,
        payment_plan=list(winner.schedule) if winner else [],
        earliest_date_for_full_payment=results.earliest_full_payment_date,
        spending_changes_needed=list(winner.spending_changes) if winner else [],
        winner=winner,
        candidates=survivors + filtered,
    )


def _status(winner: Optional[Candidate], request: Request, results: SolverResults) -> tuple[str, str]:
    if winner is None:
        return "not_affordable", "not_recommended"
    if winner.method == "full_payment":
        if (not winner.spending_changes and winner.start_date == request.request_date and results.full_safe_today):
            return "affordable_now", "full_payment"
        return "affordable_with_plan", "full_payment"
    if winner.method in ("partial_payment", "installments"):
        return "affordable_with_plan", winner.method
    if winner.method == "wait":
        return "affordable_later", "wait"
    raise ValueError(f"unknown winner method {winner.method!r}")


# --------------------------------------------------------------------------- sample validation

def _validate_against_samples() -> int:
    """Run the ranker over dataset/sample_requests.csv and print every disagreement.
    Pass 1 uses our own solver numbers; pass 2 injects the sample's amount_safe_to_pay and
    earliest date so the mapping and ranking are judged separately from the forecast."""
    import schema
    from io_layer import load_dataset
    from reconcile import build_ledger

    ds = load_dataset()
    disagreements = 0
    for label, inject in (("own solver", False), ("sample capacity injected", True)):
        print(f"\n=== {label} ===")
        agree = 0
        for r in ds.sample_requests:
            p = ds.profile_by_user[r.user_id]
            L = build_ledger(p, ds.user_events(r.user_id), [], [], r.request_date, convert=ds.convert)
            a = r.answer
            if inject:
                results = SolverResults(a.amount_safe_to_pay, a.earliest_date_for_full_payment,
                                        a.amount_safe_to_pay == r.requested_amount)
                base = default_safety(L, p, r)

                def safety(c, base=base, a=a, r=r):
                    if c.method == "full_payment" and not c.spending_changes:
                        return a.amount_safe_to_pay == r.requested_amount
                    if c.method in ("wait", "partial_payment"):
                        return True
                    return base(c)
                d = rank(r, p, ds.request_options(r.request_id), L, results, safety)
            else:
                d = rank(r, p, ds.request_options(r.request_id), L, compute_solver_results(L, r, p))
            got = (d.affordability_status, d.recommended_payment_method,
                   schema.format_payment_plan(d.payment_plan), schema.format_spending_changes(d.spending_changes_needed))
            want = (a.affordability_status, a.recommended_payment_method, a.payment_plan, a.spending_changes_needed)
            if got == want:
                agree += 1
            else:
                disagreements += 1
                print(f"{r.request_id}: want {want}")
                print(f"{' ' * len(r.request_id)}  got  {got}")
                for c in d.candidates:
                    print(f"{' ' * len(r.request_id)}    - {c.method:16} {c.payment_option_id or '-':18} "
                          f"n={c.payment_count} start={c.start_date} total={c.total_paid} "
                          f"deadline={'y' if c.completes_by_deadline else 'n'} {c.reason}")
        print(f"{agree}/{len(ds.sample_requests)} rows agree")
    return disagreements


if __name__ == "__main__":
    _validate_against_samples()
