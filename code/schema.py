"""Output contract for Buy or Wait?: column order, closed vocabularies, parsers,
formatters, normalization and row validation.

All money is Decimal. Nothing in this module touches float.

validate_row(row, request, options, profile, events, facts=None)
    row      dict of the 8 output columns, values as strings (Decimal accepted for amounts)
    request  the requests.csv row for this request_id (dict of strings)
    options  the request_payment_options.csv rows for this request_id
    profile  the financial_profiles.csv row for the requesting user
    events   financial_events.csv rows (any user; filtered by user_id here) or a dict id -> row
    facts    optional dict of computed facts from the solver:
               full_payment_safe_within_90_days: bool   (needed for check 5)
               numbers: iterable of Decimal/str            (extra numbers allowed in the explanation)
             Without facts, check 5 falls back to what the status implies, and check 15 allows only
             numbers derivable from the row, request, profile and changed events.

Every check raises ValidationError whose message starts with "[check N]".
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

AFFORDABILITY_STATUS = frozenset({
    "affordable_now",
    "affordable_with_plan",
    "affordable_later",
    "not_affordable",
})

PAYMENT_METHOD = frozenset({
    "full_payment",
    "partial_payment",
    "installments",
    "wait",
    "not_recommended",
})

STOPPABLE = frozenset({"stoppable", "reducible_or_stoppable"})
REDUCIBLE = frozenset({"reducible", "reducible_or_stoppable"})

FORECAST_DAYS = 90
MAX_SPENDING_CHANGES = 3
NONE = "none"

_CENT = Decimal("0.01")


class ValidationError(ValueError):
    """A row violates the output contract. The message names the check."""


# --------------------------------------------------------------------------- money


def to_decimal(value, what="amount") -> Decimal:
    """Parse a CSV string (or Decimal/int) into Decimal. Floats are rejected on purpose."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool) or isinstance(value, float):
        raise ValidationError(f"{what} must not be a float: {value!r}")
    if isinstance(value, int):
        return Decimal(value)
    text = str(value).strip().replace(",", "")
    if text == "":
        raise ValidationError(f"{what} is blank")
    try:
        d = Decimal(text)
    except InvalidOperation:
        raise ValidationError(f"{what} is not a number: {value!r}") from None
    if not d.is_finite():
        raise ValidationError(f"{what} is not finite: {value!r}")
    return d


def format_amount_plan(d: Decimal) -> str:
    """2 decimals if the value has a fractional part, no decimals if integral.
    620.4 -> '620.40', 25256 -> '25256', 23.5 -> '23.50'."""
    d = to_decimal(d)
    if d == d.to_integral_value():
        return str(int(d))
    return str(d.quantize(_CENT, rounding=ROUND_HALF_UP))


def format_amount_safe(d: Decimal) -> str:
    """Plain representation, no zero padding, no exponent. 603.30 -> '603.3', 25256 -> '25256'."""
    d = to_decimal(d)
    if d == d.to_integral_value():
        return str(int(d))
    return format(d.normalize(), "f")


# --------------------------------------------------------------------------- dates


def parse_date(text, what="date") -> date:
    try:
        return date.fromisoformat(str(text).strip())
    except (TypeError, ValueError):
        raise ValidationError(f"{what} is not a YYYY-MM-DD date: {text!r}") from None


# --------------------------------------------------------------------------- payment_plan


def parse_payment_plan(text) -> list[tuple[date, Decimal]]:
    """'YYYY-MM-DD:amount|...' -> [(date, Decimal), ...]; 'none' -> []."""
    text = (text or "").strip()
    if text == "" or text == NONE:
        return []
    out = []
    for i, part in enumerate(text.split("|")):
        part = part.strip()
        if part.count(":") != 1:
            raise ValidationError(f"payment_plan entry {i} is not '<date>:<amount>': {part!r}")
        day, amount = part.split(":")
        out.append((parse_date(day, f"payment_plan entry {i} date"),
                    to_decimal(amount, f"payment_plan entry {i} amount")))
    return out


def format_payment_plan(payments) -> str:
    payments = list(payments)
    if not payments:
        return NONE
    return "|".join(f"{day.isoformat()}:{format_amount_plan(amount)}" for day, amount in payments)


# --------------------------------------------------------------------------- spending_changes


def parse_spending_changes(text) -> list[tuple]:
    """'stop:<id>|reduce_to:<id>:<amount>' -> [('stop', id), ('reduce_to', id, Decimal)]; 'none' -> [].
    Enforces the maximum of three changes and that no event appears in both forms."""
    text = (text or "").strip()
    if text == "" or text == NONE:
        return []
    out = []
    seen_stop, seen_reduce = set(), set()
    for i, part in enumerate(text.split("|")):
        bits = [b.strip() for b in part.strip().split(":")]
        if bits[0] == "stop" and len(bits) == 2 and bits[1]:
            out.append(("stop", bits[1]))
            seen_stop.add(bits[1])
        elif bits[0] == "reduce_to" and len(bits) == 3 and bits[1]:
            out.append(("reduce_to", bits[1], to_decimal(bits[2], f"spending change {i} amount")))
            seen_reduce.add(bits[1])
        else:
            raise ValidationError(f"spending change {i} is malformed: {part!r}")
    if len(out) > MAX_SPENDING_CHANGES:
        raise ValidationError(f"spending_changes_needed has {len(out)} changes; maximum is {MAX_SPENDING_CHANGES}")
    both = seen_stop & seen_reduce
    if both:
        raise ValidationError(f"spending_changes_needed stops and reduces the same event: {sorted(both)}")
    ids = [c[1] for c in out]
    if len(ids) != len(set(ids)):
        raise ValidationError("spending_changes_needed references the same event twice")
    return out


def format_spending_changes(changes) -> str:
    changes = list(changes)
    if not changes:
        return NONE
    if len(changes) > MAX_SPENDING_CHANGES:
        raise ValidationError(f"{len(changes)} spending changes; maximum is {MAX_SPENDING_CHANGES}")
    parts = []
    for c in changes:
        if c[0] == "stop":
            parts.append(f"stop:{c[1]}")
        elif c[0] == "reduce_to":
            parts.append(f"reduce_to:{c[1]}:{format_amount_plan(c[2])}")
        else:
            raise ValidationError(f"unknown spending change kind: {c[0]!r}")
    return "|".join(parts)


# --------------------------------------------------------------------------- normalize


def _vocab(text: str) -> str:
    return re.sub(r"[\s\-]+", "_", str(text).strip().lower())


def normalize_row(row: dict) -> list[str]:
    """Coerce near-miss values into the vocabulary in place. Returns the coercions applied
    as 'column: old -> new' strings (empty list when nothing changed)."""
    log = []

    def put(col, new):
        old = row.get(col)
        if old != new:
            row[col] = new
            log.append(f"{col}: {old!r} -> {new!r}")

    for col in OUTPUT_COLUMNS:
        if col not in row or row[col] is None:
            put(col, "")
    for col in ("affordability_status", "recommended_payment_method"):
        put(col, _vocab(row[col]))
    for col in ("payment_plan", "spending_changes_needed"):
        text = str(row[col]).strip()
        if text == "" or text.lower() == NONE:
            put(col, NONE)
        else:
            put(col, "|".join(":".join(b.strip() for b in p.split(":")) for p in text.split("|")))
    put("request_id", str(row["request_id"]).strip())
    put("earliest_date_for_full_payment", str(row["earliest_date_for_full_payment"]).strip())
    amount = row["amount_safe_to_pay"]
    if isinstance(amount, Decimal):
        put("amount_safe_to_pay", format_amount_safe(amount))
    else:
        text = str(amount).strip().replace(",", "")
        try:
            put("amount_safe_to_pay", format_amount_safe(to_decimal(text)))
        except ValidationError:
            put("amount_safe_to_pay", text)  # leave it for validate_row to reject
    put("decision_explanation", str(row["decision_explanation"]).strip())
    return log


# --------------------------------------------------------------------------- validate


def _split_list(text) -> set[str]:
    return {p.strip() for p in str(text or "").split("|") if p.strip()}


def _index_events(events) -> dict:
    if isinstance(events, dict):
        return events
    return {e["event_id"]: e for e in events}


_DATE_PATTERNS = [
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?,?\s+\d{4}\b",
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b",
]
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers_in_text(text: str) -> list[Decimal]:
    """Numbers mentioned in prose, ignoring dates. '1,000.50' -> Decimal('1000.50')."""
    for pat in _DATE_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    out = []
    for m in _NUMBER.finditer(text):
        try:
            out.append(Decimal(m.group(0).replace(",", "")))
        except InvalidOperation:
            continue
    return out


def validate_row(row: dict, request: dict, options, profile: dict, events, facts: dict | None = None) -> None:
    facts = facts or {}
    ev_index = _index_events(events)
    user_id = request["user_id"]

    # ---- parse the row and the request once
    missing = [c for c in OUTPUT_COLUMNS if c not in row]
    if missing:
        raise ValidationError(f"[check 0] row is missing columns {missing}")
    if str(row["request_id"]) != str(request["request_id"]):
        raise ValidationError(f"[check 0] row request_id {row['request_id']!r} != request {request['request_id']!r}")
    requested = to_decimal(request["requested_amount"], "requested_amount")
    request_date = parse_date(request["request_date"], "request_date")
    desired = parse_date(request["desired_completion_date"], "desired_completion_date")
    allows_partial = str(request.get("allows_partial_payment", "")).strip().lower() == "true"
    accepted = _split_list(profile.get("payment_methods_user_will_consider"))
    status = row["affordability_status"]
    method = row["recommended_payment_method"]
    safe = to_decimal(row["amount_safe_to_pay"], "amount_safe_to_pay")
    plan = parse_payment_plan(row["payment_plan"])
    changes = parse_spending_changes(row["spending_changes_needed"])
    earliest_text = str(row["earliest_date_for_full_payment"] or "").strip()
    earliest = parse_date(earliest_text, "earliest_date_for_full_payment") if earliest_text else None

    # 1
    if not (Decimal(0) <= safe <= requested):
        raise ValidationError(f"[check 1] amount_safe_to_pay {safe} is outside 0..{requested}")

    # 2
    if status not in AFFORDABILITY_STATUS:
        raise ValidationError(f"[check 2] affordability_status {status!r} not in {sorted(AFFORDABILITY_STATUS)}")
    if method not in PAYMENT_METHOD:
        raise ValidationError(f"[check 2] recommended_payment_method {method!r} not in {sorted(PAYMENT_METHOD)}")

    # 3
    if status == "affordable_now":
        if safe != requested:
            raise ValidationError(f"[check 3] affordable_now requires amount_safe_to_pay == requested_amount ({safe} != {requested})")
        if "full_payment" not in accepted:
            raise ValidationError("[check 3] affordable_now requires the user to accept full_payment")
        if changes:
            raise ValidationError("[check 3] affordable_now requires spending_changes_needed == none")
        if method != "full_payment":
            raise ValidationError(f"[check 3] affordable_now requires recommended_payment_method full_payment, got {method!r}")

    # 4
    if status == "affordable_now" and earliest != request_date:
        raise ValidationError(f"[check 4] affordable_now requires earliest_date_for_full_payment == request_date ({earliest_text!r} != {request_date})")

    # 5
    if "full_payment_safe_within_90_days" in facts:
        full_safe = bool(facts["full_payment_safe_within_90_days"])
        if earliest is None and full_safe:
            raise ValidationError("[check 5] earliest_date_for_full_payment is empty but the full amount is safe within 90 days")
        if earliest is not None and not full_safe:
            raise ValidationError("[check 5] earliest_date_for_full_payment is set but the full amount is never safe within 90 days")
    elif earliest is None and status in ("affordable_now", "affordable_later"):
        raise ValidationError(f"[check 5] earliest_date_for_full_payment is empty but status {status} implies a safe full-payment date")
    if earliest is not None and not (request_date <= earliest <= request_date + timedelta(days=FORECAST_DAYS - 1)):
        raise ValidationError(f"[check 5] earliest_date_for_full_payment {earliest} is outside request_date..request_date+{FORECAST_DAYS - 1}")

    # 6
    if method == "partial_payment":
        if status != "affordable_with_plan":
            raise ValidationError(f"[check 6] partial_payment requires status affordable_with_plan, got {status!r}")
        if not allows_partial:
            raise ValidationError("[check 6] partial_payment requires allows_partial_payment true")
        if "partial_payment" not in accepted:
            raise ValidationError("[check 6] partial_payment is not in the user's accepted methods")
        if not (Decimal(0) < safe < requested):
            raise ValidationError(f"[check 6] partial_payment requires 0 < amount_safe_to_pay < requested_amount, got {safe}")
        if len(plan) != 2:
            raise ValidationError(f"[check 6] partial_payment requires exactly two payments, got {len(plan)}")
        if earliest is None:
            raise ValidationError("[check 6] partial_payment requires earliest_date_for_full_payment")
        (d1, a1), (d2, a2) = plan
        if d1 != request_date or a1 != safe:
            raise ValidationError(f"[check 6] first partial payment must be {safe} on {request_date}, got {a1} on {d1}")
        if d2 != earliest or a2 != requested - safe:
            raise ValidationError(f"[check 6] second partial payment must be {requested - safe} on {earliest}, got {a2} on {d2}")
        if a1 + a2 != requested:
            raise ValidationError(f"[check 6] partial payments sum to {a1 + a2}, not {requested}")
        if earliest > desired:
            raise ValidationError(f"[check 6] earliest_date_for_full_payment {earliest} is after desired_completion_date {desired}")

    # 7
    if method == "installments":
        if status != "affordable_with_plan":
            raise ValidationError(f"[check 7] installments requires status affordable_with_plan, got {status!r}")
        if "installments" not in accepted:
            raise ValidationError("[check 7] installments is not in the user's accepted methods")
        if not plan:
            raise ValidationError("[check 7] installments requires a payment_plan")
        match = None
        for opt in options:
            if opt.get("payment_method") != "installments" or opt.get("request_id") != request["request_id"]:
                continue
            n = int(opt["number_of_payments"])
            first = parse_date(opt["first_payment_date"], "first_payment_date")
            freq = int(opt["payment_frequency_days"])
            amount = to_decimal(opt["payment_amount"], "payment_amount")
            expected = [(first + timedelta(days=freq * i), amount) for i in range(n)]
            if len(plan) == n and all(pd == ed and pa == ea for (pd, pa), (ed, ea) in zip(plan, expected)):
                match = opt
                break
        if match is None:
            raise ValidationError("[check 7] payment_plan does not exactly match any installments option for this request")
        max_months = str(profile.get("max_installment_months") or "").strip()
        if max_months and int(match["number_of_payments"]) > int(max_months):
            raise ValidationError(f"[check 7] option {match['payment_option_id']} has {match['number_of_payments']} payments; max_installment_months is {max_months}")

    # 8
    if method == "wait":
        if status != "affordable_later":
            raise ValidationError(f"[check 8] wait requires status affordable_later, got {status!r}")
        if earliest is None:
            raise ValidationError("[check 8] wait requires earliest_date_for_full_payment")
        if len(plan) != 1 or plan[0][0] != earliest or plan[0][1] != requested:
            raise ValidationError(f"[check 8] wait requires a single payment of {requested} on {earliest}, got {row['payment_plan']!r}")

    # 9
    if method == "not_recommended":
        if plan:
            raise ValidationError(f"[check 9] not_recommended requires payment_plan none, got {row['payment_plan']!r}")
        if status != "not_affordable":
            raise ValidationError(f"[check 9] not_recommended requires status not_affordable, got {status!r}")
    # full_payment: implied by the definitions of affordable_now / affordable_with_plan, not numbered in the spec
    if method == "full_payment":
        if len(plan) != 1 or plan[0][0] != request_date or plan[0][1] != requested:
            raise ValidationError(f"[check 9] full_payment requires a single payment of {requested} on {request_date}, got {row['payment_plan']!r}")
        if status not in ("affordable_now", "affordable_with_plan"):
            raise ValidationError(f"[check 9] full_payment requires status affordable_now or affordable_with_plan, got {status!r}")

    # 10
    for i in range(1, len(plan)):
        if plan[i][0] <= plan[i - 1][0]:
            raise ValidationError(f"[check 10] payment_plan dates are not strictly ascending at entry {i}")
    for i, (_, amount) in enumerate(plan):
        if amount <= 0:
            raise ValidationError(f"[check 10] payment_plan entry {i} amount {amount} is not > 0")

    # 11..14
    protect = _split_list(profile.get("expense_categories_to_protect"))
    may_stop = _split_list(profile.get("expense_categories_user_is_willing_to_stop"))
    may_reduce = _split_list(profile.get("expense_categories_user_is_willing_to_reduce"))
    changed_events = []
    for change in changes:
        kind, event_id = change[0], change[1]
        ev = ev_index.get(event_id)
        if ev is None:
            raise ValidationError(f"[check 11] spending change references unknown event {event_id!r}")
        if ev.get("user_id") != user_id:
            raise ValidationError(f"[check 11] event {event_id} belongs to {ev.get('user_id')!r}, not {user_id!r}")
        changed_events.append(ev)
        flex = str(ev.get("flexibility") or "").strip()
        category = str(ev.get("category") or "").strip()
        if kind == "stop":
            if flex not in STOPPABLE:
                raise ValidationError(f"[check 12] stop:{event_id} but flexibility is {flex!r}")
            if category not in may_stop:
                raise ValidationError(f"[check 14] stop:{event_id} category {category!r} is not in expense_categories_user_is_willing_to_stop")
        else:
            new_amount = change[2]
            if flex not in REDUCIBLE:
                raise ValidationError(f"[check 13] reduce_to:{event_id} but flexibility is {flex!r}")
            floor_text = str(ev.get("minimum_allowed_amount") or "").strip()
            if floor_text and new_amount < to_decimal(floor_text, "minimum_allowed_amount"):
                raise ValidationError(f"[check 13] reduce_to:{event_id}:{new_amount} is below minimum_allowed_amount {floor_text}")
            current_text = str(ev.get("amount") or "").strip()
            if not current_text:
                raise ValidationError(f"[check 13] reduce_to:{event_id} but the event has no amount to compare against")
            if new_amount >= to_decimal(current_text, "event amount"):
                raise ValidationError(f"[check 13] reduce_to:{event_id}:{new_amount} is not below the current amount {current_text}")
            if category not in may_reduce:
                raise ValidationError(f"[check 14] reduce_to:{event_id} category {category!r} is not in expense_categories_user_is_willing_to_reduce")
        if category in protect:
            raise ValidationError(f"[check 14] event {event_id} category {category!r} is in expense_categories_to_protect")

    # 15
    explanation = str(row["decision_explanation"] or "").strip()
    if not explanation:
        raise ValidationError("[check 15] decision_explanation is empty")
    allowed = {requested, safe, Decimal(FORECAST_DAYS), Decimal(len(plan))}
    allowed.update(a for _, a in plan)
    for key in ("current_available_balance", "minimum_balance_to_keep"):
        if str(profile.get(key) or "").strip():
            allowed.add(to_decimal(profile[key], key))
    for change in changes:
        if change[0] == "reduce_to":
            allowed.add(change[2])
    for ev in changed_events:
        for key in ("amount", "minimum_allowed_amount"):
            if str(ev.get(key) or "").strip():
                allowed.add(to_decimal(ev[key], key))
    for n in facts.get("numbers", ()):
        allowed.add(to_decimal(n, "fact"))
    stray = [n for n in numbers_in_text(explanation) if n not in allowed]
    if stray:
        raise ValidationError(f"[check 15] decision_explanation mentions numbers absent from the computed facts: {[str(n) for n in stray]}")
