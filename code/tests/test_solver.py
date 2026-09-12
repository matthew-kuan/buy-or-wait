"""Tests for code/forecast.py and code/solver.py. Run:  python -m pytest code/tests -q"""
import os
import sys
from datetime import date, timedelta
from decimal import Decimal

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import forecast  # noqa: E402
import solver  # noqa: E402
from io_layer import Profile  # noqa: E402
from reconcile import CashEvent, Ledger, RecurringSeries  # noqa: E402

D = Decimal
T0 = date(2026, 1, 1)


def day(n: int) -> date:
    return T0 + timedelta(days=n)


def series(key_id, category, amount, gap, last_offset, flexibility="fixed", minimum=None, direction="debit"):
    """A recurring series whose last occurrence was `last_offset` days before T0."""
    last = T0 - timedelta(days=last_offset)
    dates = [last - timedelta(days=gap * i) for i in (2, 1, 0)]
    return RecurringSeries(
        key=(category, category), category=category, description=category, direction=direction,
        flexibility=flexibility, minimum_allowed_amount=minimum,
        event_ids=[f"{key_id}_a", f"{key_id}_b", key_id], dates=dates, amounts=[amount] * 3,
        gap_days=gap, amount=amount,
    )


def ledger(debits=(), credits=(), one_offs=()):
    return Ledger(user_id="u", home_currency="ZAR", as_of=T0,
                  recurring_debits=list(debits), recurring_credits=list(credits),
                  scheduled_one_offs=list(one_offs))


def one_off(n, amount, direction, category="misc"):
    return CashEvent(day(n), amount, direction, f"ev_{n}", category, category, "scheduled", "test")


PROFILE = Profile(
    user_id="u", home_currency="ZAR", current_available_balance=D("1000"), minimum_balance_to_keep=D("500"),
    financial_priorities=[], expense_categories_to_protect=["rent"],
    expense_categories_user_is_willing_to_reduce=["dining"],
    expense_categories_user_is_willing_to_stop=["streaming", "gym"],
    payment_methods_user_will_consider=["full_payment"], max_installment_months=None,
)


# ----------------------------------------------------------------------------- forecast


def test_series_length_and_closing_balances():
    L = ledger(one_offs=[one_off(3, D("100"), "debit"), one_off(3, D("40"), "credit"), one_off(10, D("500"), "credit")])
    s = forecast.build_series(L, T0, D("1000"))
    assert len(s) == 90 and s[0][0] == T0 and s[-1][0] == day(89)
    assert s[2][1] == D("1000") and s[3][1] == D("940") and s[10][1] == D("1440")
    assert all(isinstance(b, Decimal) for _, b in s)


def test_recurring_projection_and_changes():
    rent = series("rent_1", "rent", D("300"), 30, 5)          # next on day 25
    stream = series("st_1", "streaming", D("20"), 30, 10, "stoppable")  # next on day 20
    dine = series("dn_1", "dining", D("100"), 30, 1, "reducible", D("40"))  # next on day 29
    L = ledger(debits=[rent, stream, dine])
    base = forecast.build_series(L, T0, D("1000"))
    assert base[25][1] == D("1000") - D("20") - D("300")
    changed = forecast.build_series(L, T0, D("1000"), changes=[("stop", "st_1_a"), ("reduce_to", "dn_1", D("40"))])
    assert changed[25][1] == D("700")
    assert changed[29][1] == D("660")


def test_balance_exactly_at_minimum_is_safe_one_cent_below_is_not():
    L = ledger(one_offs=[one_off(5, D("500"), "debit")])
    assert forecast.is_safe(forecast.build_series(L, T0, D("1000")), D("500"))
    assert not forecast.is_safe(forecast.build_series(L, T0, D("999.99")), D("500"))
    assert forecast.min_balance(forecast.build_series(L, T0, D("999.99"))) == D("499.99")


def test_float_rejected():
    with pytest.raises(TypeError):
        forecast.build_series(ledger(), T0, 1000.0)


# ----------------------------------------------------------------------------- solver: amount_safe_to_pay


def test_amount_safe_is_zero_when_balance_already_at_minimum():
    assert solver.amount_safe_to_pay(ledger(), T0, D("100"), D("500"), D("500")) == D("0")
    assert solver.amount_safe_to_pay(ledger(one_offs=[one_off(9, D("600"), "debit")]), T0, D("100"), D("1000"), D("500")) == D("0")


def test_amount_safe_equals_requested_when_comfortably_affordable():
    assert solver.amount_safe_to_pay(ledger(), T0, D("100"), D("1000"), D("500")) == D("100")


def test_amount_safe_converges_to_the_cent_and_is_monotone():
    L = ledger(one_offs=[one_off(9, D("123.45"), "debit"), one_off(20, D("50"), "credit")])
    balance, minimum, requested = D("1000"), D("500"), D("900")
    safe = solver.amount_safe_to_pay(L, T0, requested, balance, minimum)
    assert safe == D("376.55")
    # exactly safe at the result, unsafe one cent higher
    assert forecast.is_safe(forecast.build_series(L, T0, balance, extra_payments=[(T0, safe)]), minimum)
    assert not forecast.is_safe(forecast.build_series(L, T0, balance, extra_payments=[(T0, safe + D("0.01"))]), minimum)
    # monotone: every smaller payment is safe, every larger payment is not
    grid = [D(x) for x in ("0", "100", "376.54", "376.55", "376.56", "500", "900")]
    verdicts = [forecast.is_safe(forecast.build_series(L, T0, balance, extra_payments=[(T0, a)]), minimum) for a in grid]
    assert verdicts == [True, True, True, True, False, False, False]


# ----------------------------------------------------------------------------- solver: earliest date


def test_earliest_date_is_request_date():
    assert solver.earliest_full_payment_date(ledger(), T0, D("100"), D("1000"), D("500")) == T0


def test_payment_the_day_before_salary_is_unsafe_on_salary_day_it_is_safe():
    salary = one_off(10, D("1000"), "credit", "salary")
    rent = one_off(12, D("800"), "debit", "rent")
    L = ledger(one_offs=[salary, rent])
    balance, minimum, requested = D("1000"), D("500"), D("600")
    assert not forecast.is_safe(forecast.build_series(L, T0, balance, extra_payments=[(day(9), requested)]), minimum)
    assert forecast.is_safe(forecast.build_series(L, T0, balance, extra_payments=[(day(10), requested)]), minimum)
    assert solver.earliest_full_payment_date(L, T0, requested, balance, minimum) == day(10)
    assert solver.amount_safe_to_pay(L, T0, requested, balance, minimum) == D("500")


def test_earliest_date_is_none_when_never_safe():
    assert solver.earliest_full_payment_date(ledger(), T0, D("100000"), D("1000"), D("500")) is None
    # a payment on day 89 (last day of horizon) still counts; day 90 is outside
    L = ledger(one_offs=[one_off(89, D("5000"), "credit", "salary")])
    assert solver.earliest_full_payment_date(L, T0, D("5000"), D("1000"), D("500")) == day(89)
    L = ledger(one_offs=[one_off(90, D("5000"), "credit", "salary")])
    assert solver.earliest_full_payment_date(L, T0, D("5000"), D("1000"), D("500")) is None


# ----------------------------------------------------------------------------- solver: spending changes


def _flex_ledger():
    """Salary 2000 lands on days 30/60; recurring debits total 530 per cycle, so with balance 1000
    the trough is day 29 at 470 - P (P = payment on day 0) against a minimum of 500."""
    return ledger(
        credits=[series("sal_9", "salary", D("2000"), 30, 0, direction="credit")],
        debits=[
            series("rent_9", "rent", D("300"), 30, 5, "stoppable"),              # protected -> never legal
            series("st_9", "streaming", D("20"), 30, 10, "stoppable"),           # legal stop, 20/month
            series("gym_9", "gym", D("70"), 30, 12, "stoppable"),                # legal stop, 70/month
            series("dn_9", "dining", D("100"), 30, 1, "reducible", D("40")),     # legal reduce, 60/month
            series("fun_9", "entertainment", D("40"), 30, 3, "reducible_or_stoppable", D("10")),  # not permitted
        ])


def test_legal_changes_mirror_schema_rules():
    legal = {c for c, _ in solver.legal_changes(_flex_ledger(), PROFILE)}
    assert legal == {("stop", "st_9"), ("stop", "gym_9"), ("reduce_to", "dn_9", D("40"))}


def test_find_spending_changes_prefers_fewer_then_smallest_reduction():
    L = _flex_ledger()
    minimum = D("500")
    assert forecast.min_balance(forecast.build_series(L, T0, D("1000"))) == D("470")
    # already safe with a bigger opening balance -> no change needed
    assert solver.find_spending_changes(L, T0, D("2000"), minimum, [(T0, D("500"))], PROFILE) == []
    # 30 short: streaming (20) is not enough; dining (60) beats gym (70)
    assert solver.find_spending_changes(L, T0, D("1000"), minimum, [(T0, D("0"))], PROFILE) == [("reduce_to", "dn_9", D("40"))]
    # 120 short: no single change reaches it; gym + dining (130) is the only pair that does
    assert solver.find_spending_changes(L, T0, D("1000"), minimum, [(T0, D("90"))], PROFILE) == [("stop", "gym_9"), ("reduce_to", "dn_9", D("40"))]
    # hopeless -> None
    assert solver.find_spending_changes(L, T0, D("1000"), minimum, [(T0, D("5000"))], PROFILE) is None


def test_find_spending_changes_never_stops_and_reduces_same_event():
    L = ledger(credits=[series("sal_1", "salary", D("2000"), 30, 0, direction="credit")],
               debits=[series("x_1", "dining", D("100"), 30, 1, "reducible_or_stoppable", D("40"))])
    profile = Profile(**{**PROFILE.__dict__, "expense_categories_user_is_willing_to_stop": ["dining"]})
    legal = solver.legal_changes(L, profile)
    assert {c[0] for c, _ in legal} == {"stop", "reduce_to"}
    # trough day 29 = 1000 - 450 - 100 = 450: reduce (60/month) fixes it more cheaply than stop (100/month)
    result = solver.find_spending_changes(L, T0, D("1000"), D("500"), [(T0, D("450"))], profile)
    assert result == [("reduce_to", "x_1", D("40"))]
    # both actions on one event are never combined even when neither alone would do
    combos = [c for c, _ in solver.legal_changes(L, profile)]
    assert len({c[1] for c in combos}) == 1
