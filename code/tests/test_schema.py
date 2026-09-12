"""Tests for code/schema.py. Run:  python -m pytest code/tests -q"""
import os
import sys
from datetime import date
from decimal import Decimal

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import schema  # noqa: E402
from schema import ValidationError, validate_row  # noqa: E402

D = Decimal

# ----------------------------------------------------------------------------- fixtures

PROFILE = {
    "user_id": "user_01",
    "home_currency": "ZAR",
    "current_available_balance": "5000",
    "minimum_balance_to_keep": "1000",
    "financial_priorities": "education",
    "expense_categories_to_protect": "rent|groceries",
    "expense_categories_user_is_willing_to_reduce": "dining",
    "expense_categories_user_is_willing_to_stop": "streaming",
    "payment_methods_user_will_consider": "full_payment|partial_payment|installments",
    "max_installment_months": "3",
}

REQUEST = {
    "request_id": "request_01",
    "user_id": "user_01",
    "request_date": "2026-01-05",
    "request_type": "purchase",
    "requested_amount": "1000",
    "desired_completion_date": "2026-03-01",
    "allows_partial_payment": "true",
    "request_text": "Can I afford this?",
}

OPTIONS = [
    {"payment_option_id": "opt_1", "request_id": "request_01", "payment_method": "full_payment",
     "payment_amount": "1000", "number_of_payments": "1", "first_payment_date": "2026-01-05",
     "payment_frequency_days": "", "financing_fee": "0", "total_payable_amount": "1000"},
    {"payment_option_id": "opt_2", "request_id": "request_01", "payment_method": "installments",
     "payment_amount": "350", "number_of_payments": "3", "first_payment_date": "2026-01-10",
     "payment_frequency_days": "30", "financing_fee": "50", "total_payable_amount": "1050"},
    {"payment_option_id": "opt_3", "request_id": "request_01", "payment_method": "installments",
     "payment_amount": "220", "number_of_payments": "5", "first_payment_date": "2026-01-10",
     "payment_frequency_days": "30", "financing_fee": "100", "total_payable_amount": "1100"},
]


def _event(event_id, category, flexibility, amount, minimum="", user_id="user_01"):
    return {"event_id": event_id, "user_id": user_id, "event_type": "subscription", "description": category,
            "category": category, "direction": "debit", "amount": amount, "currency": "ZAR",
            "event_date": "2025-12-10", "settlement_date": "2025-12-10", "status": "settled",
            "linked_event_id": "", "flexibility": flexibility, "minimum_allowed_amount": minimum}


EVENTS = [
    _event("ev_stream", "streaming", "stoppable", "20"),
    _event("ev_dine", "dining", "reducible", "100", minimum="40"),
    _event("ev_fun", "entertainment", "reducible_or_stoppable", "60", minimum="10"),
    _event("ev_groc", "groceries", "reducible", "300", minimum="100"),
    _event("ev_rent", "rent", "fixed", "2000"),
    _event("ev_other", "streaming", "stoppable", "20", user_id="user_02"),
]


def row(**over):
    base = {
        "request_id": "request_01",
        "amount_safe_to_pay": "1000",
        "affordability_status": "affordable_now",
        "recommended_payment_method": "full_payment",
        "payment_plan": "2026-01-05:1000",
        "earliest_date_for_full_payment": "2026-01-05",
        "spending_changes_needed": "none",
        "decision_explanation": "Pay ZAR 1,000 today. This leaves at least ZAR 1,000 available over the next 90 days.",
    }
    base.update(over)
    return base


def _variant(defaults, over):
    merged = dict(defaults)
    merged.update(over)
    return row(**merged)


def partial_row(**over):
    return _variant(dict(
        amount_safe_to_pay="600", affordability_status="affordable_with_plan",
        recommended_payment_method="partial_payment", payment_plan="2026-01-05:600|2026-02-15:400",
        earliest_date_for_full_payment="2026-02-15",
        decision_explanation="Pay ZAR 600 today and ZAR 400 on 15 February 2026."), over)


def inst_row(**over):
    return _variant(dict(
        amount_safe_to_pay="600", affordability_status="affordable_with_plan",
        recommended_payment_method="installments",
        payment_plan="2026-01-10:350|2026-02-09:350|2026-03-11:350",
        earliest_date_for_full_payment="2026-02-15",
        decision_explanation="Use 3 installments of ZAR 350, starting 10 January 2026."), over)


def wait_row(**over):
    return _variant(dict(
        amount_safe_to_pay="600", affordability_status="affordable_later",
        recommended_payment_method="wait", payment_plan="2026-02-15:1000",
        earliest_date_for_full_payment="2026-02-15",
        decision_explanation="Pay ZAR 1,000 in full on 15 February 2026."), over)


def notrec_row(**over):
    return _variant(dict(
        amount_safe_to_pay="600", affordability_status="not_affordable",
        recommended_payment_method="not_recommended", payment_plan="none",
        earliest_date_for_full_payment="",
        decision_explanation="Do not make this payment. Only ZAR 600 is safe today."), over)


def change_row(**over):
    return _variant(dict(
        amount_safe_to_pay="980", affordability_status="affordable_with_plan",
        recommended_payment_method="full_payment", payment_plan="2026-01-05:1000",
        earliest_date_for_full_payment="2026-02-15", spending_changes_needed="stop:ev_stream",
        decision_explanation="Stop the streaming plan, then pay ZAR 1,000 today."), over)


def ok(r, **facts):
    validate_row(r, REQUEST, OPTIONS, PROFILE, EVENTS, facts or None)


def bad(r, check, **facts):
    with pytest.raises(ValidationError, match=rf"\[check {check}\]"):
        validate_row(r, REQUEST, OPTIONS, PROFILE, EVENTS, facts or None)


# ----------------------------------------------------------------------------- constants


def test_constants():
    assert schema.OUTPUT_COLUMNS[0] == "request_id" and len(schema.OUTPUT_COLUMNS) == 8
    assert schema.AFFORDABILITY_STATUS == {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
    assert schema.PAYMENT_METHOD == {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}


# ----------------------------------------------------------------------------- formatters and parsers


def test_format_amount_plan():
    assert schema.format_amount_plan(D("620.4")) == "620.40"
    assert schema.format_amount_plan(D("25256")) == "25256"
    assert schema.format_amount_plan(D("25256.00")) == "25256"
    assert schema.format_amount_plan(D("23.5")) == "23.50"
    assert schema.format_amount_plan(D("15952906.67")) == "15952906.67"


def test_format_amount_safe():
    assert schema.format_amount_safe(D("603.30")) == "603.3"
    assert schema.format_amount_safe(D("17229139.2")) == "17229139.2"
    assert schema.format_amount_safe(D("25256")) == "25256"
    assert schema.format_amount_safe(D("0")) == "0"
    assert schema.format_amount_safe(D("1E+4")) == "10000"


def test_float_rejected():
    with pytest.raises(ValidationError):
        schema.to_decimal(1.5)


def test_payment_plan_round_trip():
    text = "2025-08-08:15952906.67|2025-09-07:15952906.67|2025-10-07:15952906.67"
    parsed = schema.parse_payment_plan(text)
    assert parsed[0] == (date(2025, 8, 8), D("15952906.67"))
    assert schema.format_payment_plan(parsed) == text
    assert schema.parse_payment_plan("none") == []
    assert schema.format_payment_plan([]) == "none"
    assert schema.format_payment_plan([(date(2026, 1, 3), D("620.4"))]) == "2026-01-03:620.40"
    with pytest.raises(ValidationError):
        schema.parse_payment_plan("2026-01-03")


def test_spending_changes_round_trip():
    text = "stop:event_1815|reduce_to:event_1816:23.50"
    parsed = schema.parse_spending_changes(text)
    assert parsed == [("stop", "event_1815"), ("reduce_to", "event_1816", D("23.50"))]
    assert schema.format_spending_changes(parsed) == text
    assert schema.parse_spending_changes("none") == []
    assert schema.format_spending_changes([]) == "none"
    assert schema.format_spending_changes([("reduce_to", "e", D("665950"))]) == "reduce_to:e:665950"
    with pytest.raises(ValidationError, match="same event"):
        schema.parse_spending_changes("stop:e1|reduce_to:e1:5")
    with pytest.raises(ValidationError, match="maximum"):
        schema.parse_spending_changes("stop:a|stop:b|stop:c|stop:d")
    with pytest.raises(ValidationError, match="malformed"):
        schema.parse_spending_changes("pause:e1")


def test_normalize_row():
    r = row(affordability_status=" Affordable Now ", recommended_payment_method="Full-Payment",
            payment_plan="", spending_changes_needed=" None", amount_safe_to_pay="1,000.00")
    log = schema.normalize_row(r)
    assert r["affordability_status"] == "affordable_now"
    assert r["recommended_payment_method"] == "full_payment"
    assert r["payment_plan"] == "none"
    assert r["spending_changes_needed"] == "none"
    assert r["amount_safe_to_pay"] == "1000"
    assert len(log) == 5 and all("->" in line for line in log)
    assert schema.normalize_row(row()) == []


# ----------------------------------------------------------------------------- the 15 checks


def test_check_1_bounds():
    ok(row())
    bad(row(amount_safe_to_pay="1500"), 1)
    bad(notrec_row(amount_safe_to_pay="-1"), 1)


def test_check_2_vocabulary():
    ok(row())
    bad(row(affordability_status="maybe"), 2)
    bad(row(recommended_payment_method="cash"), 2)


def test_check_3_affordable_now_preconditions():
    ok(row())
    bad(row(amount_safe_to_pay="900"), 3)
    bad(row(spending_changes_needed="stop:ev_stream"), 3)
    no_full = dict(PROFILE, payment_methods_user_will_consider="installments")
    with pytest.raises(ValidationError, match=r"\[check 3\]"):
        validate_row(row(), REQUEST, OPTIONS, no_full, EVENTS)


def test_check_4_affordable_now_earliest_equals_request_date():
    ok(row())
    bad(row(earliest_date_for_full_payment="2026-01-06"), 4)


def test_check_5_earliest_empty_only_when_never_safe():
    ok(notrec_row(), full_payment_safe_within_90_days=False)
    ok(wait_row(), full_payment_safe_within_90_days=True)
    bad(notrec_row(), 5, full_payment_safe_within_90_days=True)
    bad(wait_row(), 5, full_payment_safe_within_90_days=False)
    bad(wait_row(earliest_date_for_full_payment=""), 5)          # status implies a date
    bad(wait_row(earliest_date_for_full_payment="2026-06-01", payment_plan="2026-06-01:1000"), 5)  # beyond day 89


def test_check_6_partial_payment():
    ok(partial_row())
    bad(partial_row(affordability_status="affordable_later"), 6)
    no_partial = dict(REQUEST, allows_partial_payment="false")
    with pytest.raises(ValidationError, match=r"\[check 6\]"):
        validate_row(partial_row(), no_partial, OPTIONS, PROFILE, EVENTS)
    bad(partial_row(amount_safe_to_pay="1000", payment_plan="2026-01-05:1000|2026-02-15:0"), 6)
    bad(partial_row(payment_plan="2026-01-05:600|2026-02-15:300"), 6)
    bad(partial_row(payment_plan="2026-01-05:600|2026-02-14:400"), 6)
    bad(partial_row(payment_plan="2026-01-05:600"), 6)
    bad(partial_row(earliest_date_for_full_payment="2026-03-15", payment_plan="2026-01-05:600|2026-03-15:400"), 6)


def test_check_7_installments_match_option():
    ok(inst_row())
    bad(inst_row(payment_plan="2026-01-10:351|2026-02-09:351|2026-03-11:351"), 7)
    bad(inst_row(payment_plan="2026-01-10:350|2026-02-10:350|2026-03-11:350"), 7)
    bad(inst_row(payment_plan="2026-01-10:350|2026-02-09:350"), 7)
    bad(inst_row(payment_plan="2026-01-10:220|2026-02-09:220|2026-03-11:220|2026-04-10:220|2026-05-10:220"), 7)
    bad(inst_row(affordability_status="affordable_later"), 7)


def test_check_8_wait():
    ok(wait_row())
    bad(wait_row(payment_plan="2026-02-15:999"), 8)
    bad(wait_row(payment_plan="2026-02-16:1000"), 8)
    bad(wait_row(affordability_status="affordable_with_plan"), 8)


def test_check_9_not_recommended_has_no_plan():
    ok(notrec_row())
    bad(notrec_row(payment_plan="2026-01-05:600"), 9)


def test_check_10_plan_ordering_and_positivity():
    ok(partial_row())
    bad(partial_row(earliest_date_for_full_payment="2026-01-05", payment_plan="2026-01-05:600|2026-01-05:400"), 10)
    bad(row(amount_safe_to_pay="0", affordability_status="affordable_later", recommended_payment_method="wait",
            payment_plan="2026-02-15:-5", earliest_date_for_full_payment="2026-02-15"), 8)  # negative caught upstream
    with pytest.raises(ValidationError):
        schema.parse_payment_plan("2026-01-05:abc")


def test_check_11_event_belongs_to_user():
    ok(change_row())
    bad(change_row(spending_changes_needed="stop:ev_missing"), 11)
    bad(change_row(spending_changes_needed="stop:ev_other"), 11)


def test_check_12_stop_requires_stoppable():
    ok(change_row())
    bad(change_row(spending_changes_needed="stop:ev_dine"), 12)


def test_check_13_reduce_bounds():
    ok(change_row(spending_changes_needed="reduce_to:ev_dine:40",
                  decision_explanation="Reduce dining to ZAR 40, then pay ZAR 1,000 today."))
    bad(change_row(spending_changes_needed="reduce_to:ev_dine:30"), 13)
    bad(change_row(spending_changes_needed="reduce_to:ev_dine:100"), 13)
    bad(change_row(spending_changes_needed="reduce_to:ev_stream:10"), 13)


def test_check_14_category_permissions():
    ok(change_row())
    bad(change_row(spending_changes_needed="stop:ev_fun"), 14)                 # not in willing-to-stop
    bad(change_row(spending_changes_needed="reduce_to:ev_groc:200"), 14)       # protected category


def test_check_15_explanation_grounded():
    ok(row())
    ok(row(decision_explanation="Pay 1000 today; the 90-day low is 2500."), numbers=[D("2500")])
    bad(row(decision_explanation=""), 15)
    bad(row(decision_explanation="Pay ZAR 1,000 today. This leaves ZAR 999 available."), 15)


def test_numbers_in_text_ignores_dates():
    nums = schema.numbers_in_text("Pay EUR 996.60 in full on 15 April 2025 (2025-04-15), keeping 1,300 over 90 days.")
    assert nums == [D("996.60"), D("1300"), D("90")]
