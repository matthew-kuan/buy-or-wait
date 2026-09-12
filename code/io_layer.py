"""Load dataset/*.csv into typed records (dates parsed, money as Decimal) and build indexes.

    from io_layer import load_dataset
    ds = load_dataset()                      # resolves <repo>/dataset relative to this file
    ds.events_by_user["user_01"]             # list[Event]
    ds.convert(Decimal("100"), "USD", "INR", date(2026, 3, 15))

No float anywhere: blank numeric cells become None, everything else Decimal.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(HERE), "dataset")


class FxMissing(LookupError):
    """No exchange-rate row (direct or inverse) for the requested date and pair."""


class DataError(ValueError):
    """A CSV cell could not be parsed into its declared type."""


# --------------------------------------------------------------------------- cell parsers


def _dec(text, what) -> Optional[Decimal]:
    text = (text or "").strip().replace(",", "")
    if text == "":
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        raise DataError(f"{what}: not a number: {text!r}") from None


def _date(text, what) -> Optional[date]:
    text = (text or "").strip()
    if text == "":
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise DataError(f"{what}: not a YYYY-MM-DD date: {text!r}") from None


def _datetime(text, what) -> Optional[datetime]:
    text = (text or "").strip()
    if text == "":
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise DataError(f"{what}: not an ISO-8601 datetime: {text!r}") from None


def _int(text, what) -> Optional[int]:
    text = (text or "").strip()
    if text == "":
        return None
    try:
        return int(text)
    except ValueError:
        raise DataError(f"{what}: not an integer: {text!r}") from None


def _list(text) -> list[str]:
    return [p.strip() for p in (text or "").split("|") if p.strip()]


def _bool(text, what) -> bool:
    text = (text or "").strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no", ""):
        return False
    raise DataError(f"{what}: not a boolean: {text!r}")


def _opt(text) -> Optional[str]:
    text = (text or "").strip()
    return text or None


def event_sort_key(event_id: str):
    """'event_12' -> (0, 12) so ids sort numerically; non-numeric ids sort after, lexically."""
    head, _, tail = event_id.rpartition("_")
    return (0, int(tail), head) if tail.isdigit() else (1, 0, event_id)


# --------------------------------------------------------------------------- records


@dataclass(frozen=True)
class Profile:
    user_id: str
    home_currency: str
    current_available_balance: Decimal
    minimum_balance_to_keep: Decimal
    financial_priorities: list[str]
    expense_categories_to_protect: list[str]
    expense_categories_user_is_willing_to_reduce: list[str]
    expense_categories_user_is_willing_to_stop: list[str]
    payment_methods_user_will_consider: list[str]
    max_installment_months: Optional[int]


@dataclass(frozen=True)
class Event:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str                 # debit | credit | non_cash
    amount: Optional[Decimal]      # None when the CSV cell is blank (amount lives in an image)
    currency: str
    event_date: date
    settlement_date: Optional[date]
    status: str                    # settled | pending | scheduled | cancelled | failed | unrealized
    linked_event_id: Optional[str]
    flexibility: str               # fixed | reducible | stoppable | reducible_or_stoppable
    minimum_allowed_amount: Optional[Decimal]

    @property
    def cash_date(self) -> date:
        return self.settlement_date or self.event_date


@dataclass(frozen=True)
class SampleAnswer:
    amount_safe_to_pay: Decimal
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: Optional[date]
    spending_changes_needed: str
    decision_explanation: str


@dataclass(frozen=True)
class Request:
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: Decimal
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str
    answer: Optional[SampleAnswer] = None    # populated for sample_requests.csv only


@dataclass(frozen=True)
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str            # full_payment | installments
    payment_amount: Decimal
    number_of_payments: int
    first_payment_date: date
    payment_frequency_days: Optional[int]
    financing_fee: Decimal
    total_payable_amount: Decimal

    def schedule(self) -> list[tuple[date, Decimal]]:
        step = self.payment_frequency_days or 0
        return [(self.first_payment_date + timedelta(days=step * i), self.payment_amount)
                for i in range(self.number_of_payments)]

    @property
    def last_payment_date(self) -> date:
        return self.schedule()[-1][0]


@dataclass(frozen=True)
class Message:
    message_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]
    sent_at: datetime
    source_type: str
    message_text: str


@dataclass(frozen=True)
class Image:
    image_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]
    path: str


@dataclass(frozen=True)
class FxRate:
    rate_date: date
    from_currency: str
    to_currency: str
    rate: Decimal


# --------------------------------------------------------------------------- dataset


@dataclass
class Dataset:
    dataset_dir: str
    profiles: list[Profile]
    events: list[Event]
    requests: list[Request]
    sample_requests: list[Request]
    options: list[PaymentOption]
    messages: list[Message]
    images: list[Image]
    fx_rates: list[FxRate]

    profile_by_user: dict[str, Profile] = field(default_factory=dict)
    event_by_id: dict[str, Event] = field(default_factory=dict)
    events_by_user: dict[str, list[Event]] = field(default_factory=dict)
    request_by_id: dict[str, Request] = field(default_factory=dict)
    options_by_request: dict[str, list[PaymentOption]] = field(default_factory=dict)
    messages_by_user: dict[str, list[Message]] = field(default_factory=dict)
    messages_by_request: dict[str, list[Message]] = field(default_factory=dict)
    messages_by_event: dict[str, list[Message]] = field(default_factory=dict)
    images_by_user: dict[str, list[Image]] = field(default_factory=dict)
    images_by_request: dict[str, list[Image]] = field(default_factory=dict)
    images_by_event: dict[str, list[Image]] = field(default_factory=dict)
    fx: dict[tuple[date, str, str], Decimal] = field(default_factory=dict)

    def __post_init__(self):
        self.profile_by_user = {p.user_id: p for p in self.profiles}
        self.event_by_id = {e.event_id: e for e in self.events}
        self.events_by_user = _group(self.events, lambda e: e.user_id)
        for lst in self.events_by_user.values():
            lst.sort(key=lambda e: (e.cash_date, event_sort_key(e.event_id)))
        self.request_by_id = {r.request_id: r for r in self.requests + self.sample_requests}
        self.options_by_request = _group(self.options, lambda o: o.request_id)
        for lst in self.options_by_request.values():
            lst.sort(key=lambda o: event_sort_key(o.payment_option_id))
        self.messages_by_user = _group(self.messages, lambda m: m.user_id)
        self.messages_by_request = _group(self.messages, lambda m: m.request_id)
        self.messages_by_event = _group(self.messages, lambda m: m.related_event_id)
        self.images_by_user = _group(self.images, lambda i: i.user_id)
        self.images_by_request = _group(self.images, lambda i: i.request_id)
        self.images_by_event = _group(self.images, lambda i: i.related_event_id)
        self.fx = {(r.rate_date, r.from_currency, r.to_currency): r.rate for r in self.fx_rates}

    # -- lookups that never raise KeyError on a missing key
    def user_events(self, user_id: str) -> list[Event]:
        return self.events_by_user.get(user_id, [])

    def user_messages(self, user_id: str) -> list[Message]:
        return self.messages_by_user.get(user_id, [])

    def user_images(self, user_id: str) -> list[Image]:
        return self.images_by_user.get(user_id, [])

    def request_options(self, request_id: str) -> list[PaymentOption]:
        return self.options_by_request.get(request_id, [])

    # -- fx
    def convert(self, amount: Decimal, from_ccy: str, to_ccy: str, on_date: date) -> Decimal:
        """Convert using the rate row for `on_date` (an event's settlement_date) in the stated
        direction; fall back to the inverse pair on the same date; otherwise raise FxMissing."""
        if from_ccy == to_ccy:
            return amount
        rate = self.fx.get((on_date, from_ccy, to_ccy))
        if rate is not None:
            return amount * rate
        inverse = self.fx.get((on_date, to_ccy, from_ccy))
        if inverse is not None and inverse != 0:
            return amount / inverse
        raise FxMissing(f"no exchange rate for {from_ccy}->{to_ccy} on {on_date}")


def _group(items, key):
    out: dict = {}
    for item in items:
        k = key(item)
        if k is None:
            continue
        out.setdefault(k, []).append(item)
    return out


# --------------------------------------------------------------------------- loaders


def _read(path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _profile(r) -> Profile:
    u = r["user_id"]
    return Profile(
        user_id=u,
        home_currency=r["home_currency"].strip(),
        current_available_balance=_dec(r["current_available_balance"], f"{u} balance"),
        minimum_balance_to_keep=_dec(r["minimum_balance_to_keep"], f"{u} minimum"),
        financial_priorities=_list(r["financial_priorities"]),
        expense_categories_to_protect=_list(r["expense_categories_to_protect"]),
        expense_categories_user_is_willing_to_reduce=_list(r["expense_categories_user_is_willing_to_reduce"]),
        expense_categories_user_is_willing_to_stop=_list(r["expense_categories_user_is_willing_to_stop"]),
        payment_methods_user_will_consider=_list(r["payment_methods_user_will_consider"]),
        max_installment_months=_int(r["max_installment_months"], f"{u} max_installment_months"),
    )


def _event(r) -> Event:
    e = r["event_id"]
    return Event(
        event_id=e,
        user_id=r["user_id"],
        event_type=r["event_type"].strip(),
        description=r["description"].strip(),
        category=r["category"].strip(),
        direction=r["direction"].strip(),
        amount=_dec(r["amount"], f"{e} amount"),
        currency=r["currency"].strip(),
        event_date=_date(r["event_date"], f"{e} event_date"),
        settlement_date=_date(r["settlement_date"], f"{e} settlement_date"),
        status=r["status"].strip(),
        linked_event_id=_opt(r["linked_event_id"]),
        flexibility=r["flexibility"].strip(),
        minimum_allowed_amount=_dec(r["minimum_allowed_amount"], f"{e} minimum_allowed_amount"),
    )


def _request(r, with_answer: bool) -> Request:
    rid = r["request_id"]
    answer = None
    if with_answer:
        answer = SampleAnswer(
            amount_safe_to_pay=_dec(r["amount_safe_to_pay"], f"{rid} amount_safe_to_pay"),
            affordability_status=r["affordability_status"].strip(),
            recommended_payment_method=r["recommended_payment_method"].strip(),
            payment_plan=r["payment_plan"].strip(),
            earliest_date_for_full_payment=_date(r["earliest_date_for_full_payment"], f"{rid} earliest"),
            spending_changes_needed=r["spending_changes_needed"].strip(),
            decision_explanation=r["decision_explanation"].strip(),
        )
    return Request(
        request_id=rid,
        user_id=r["user_id"],
        request_date=_date(r["request_date"], f"{rid} request_date"),
        request_type=r["request_type"].strip(),
        requested_amount=_dec(r["requested_amount"], f"{rid} requested_amount"),
        desired_completion_date=_date(r["desired_completion_date"], f"{rid} desired_completion_date"),
        allows_partial_payment=_bool(r["allows_partial_payment"], f"{rid} allows_partial_payment"),
        request_text=r["request_text"],
        answer=answer,
    )


def _option(r) -> PaymentOption:
    o = r["payment_option_id"]
    return PaymentOption(
        payment_option_id=o,
        request_id=r["request_id"],
        payment_method=r["payment_method"].strip(),
        payment_amount=_dec(r["payment_amount"], f"{o} payment_amount"),
        number_of_payments=_int(r["number_of_payments"], f"{o} number_of_payments"),
        first_payment_date=_date(r["first_payment_date"], f"{o} first_payment_date"),
        payment_frequency_days=_int(r["payment_frequency_days"], f"{o} payment_frequency_days"),
        financing_fee=_dec(r["financing_fee"], f"{o} financing_fee") or Decimal(0),
        total_payable_amount=_dec(r["total_payable_amount"], f"{o} total_payable_amount"),
    )


def _message(r) -> Message:
    return Message(
        message_id=r["message_id"],
        user_id=r["user_id"],
        request_id=_opt(r["request_id"]),
        related_event_id=_opt(r["related_event_id"]),
        sent_at=_datetime(r["sent_at"], f"{r['message_id']} sent_at"),
        source_type=r["source_type"].strip(),
        message_text=r["message_text"],
    )


def _image(r, dataset_dir) -> Image:
    return Image(
        image_id=r["image_id"],
        user_id=r["user_id"],
        request_id=_opt(r["request_id"]),
        related_event_id=_opt(r["related_event_id"]),
        path=os.path.join(dataset_dir, "media", "images", f"{r['image_id']}.png"),
    )


def _fx(r) -> FxRate:
    return FxRate(
        rate_date=_date(r["rate_date"], "rate_date"),
        from_currency=r["from_currency"].strip(),
        to_currency=r["to_currency"].strip(),
        rate=_dec(r["rate"], "rate"),
    )


def load_dataset(dataset_dir: str | None = None) -> Dataset:
    d = dataset_dir or DEFAULT_DATASET_DIR
    p = lambda name: os.path.join(d, name)  # noqa: E731
    return Dataset(
        dataset_dir=d,
        profiles=[_profile(r) for r in _read(p("financial_profiles.csv"))],
        events=[_event(r) for r in _read(p("financial_events.csv"))],
        requests=[_request(r, False) for r in _read(p("requests.csv"))],
        sample_requests=[_request(r, True) for r in _read(p("sample_requests.csv"))],
        options=[_option(r) for r in _read(p("request_payment_options.csv"))],
        messages=[_message(r) for r in _read(p("messages.csv"))],
        images=[_image(r, d) for r in _read(p("images.csv"))],
        fx_rates=[_fx(r) for r in _read(p("exchange_rates.csv"))],
    )


if __name__ == "__main__":
    ds = load_dataset()
    print(f"profiles={len(ds.profiles)} events={len(ds.events)} requests={len(ds.requests)} "
          f"samples={len(ds.sample_requests)} options={len(ds.options)} messages={len(ds.messages)} "
          f"images={len(ds.images)} fx={len(ds.fx)}")
