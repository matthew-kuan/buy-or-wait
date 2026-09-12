# Buy or Wait? — deterministic affordability agent

Decides, for every request in `dataset/requests.csv`, whether the user should pay in full, pay
partially, use a supplied installment option, wait, or not proceed — and writes `output.csv`.

## Run

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # or copy .env.example to .env and export it; see below
python code/main.py                 # full run: cache first, API only for cache misses
python code/main.py --dry-run       # zero API calls: replays .cache/ (shipped in code.zip)
```

`output.csv` lands in the **repository root** with exactly the 8 required columns and one row per
request. `code/evaluation/usage_report.md` is rewritten on every run with that run's token usage
and cost. Run from any working directory — every path resolves relative to `code/`.

Other commands:

```bash
python -m pytest code/tests -q                       # unit tests (schema, forecast, solver)
python code/evaluation/main.py --note "what changed" # score against the 25 samples + 13 golden fixtures
python code/audit.py                                 # data audit of dataset/
python code/package.py                               # build code.zip for submission
```

Secrets are read from environment variables only (`ANTHROPIC_API_KEY`, optional
`BUY_OR_WAIT_MODEL`). Nothing reads `.env` directly; export it with your shell or a dotenv runner.

## Architecture

```text
dataset/*.csv, dataset/media/images/*.png
        │
        ▼
  io_layer.py      typed records (Decimal money, parsed dates), indexes, dated FX conversion
        │
        ├─────────────► extraction.py   (the ONLY model calls)
        │                 call 1: one classification per message  → verdict JSON
        │                 call 2: one reading per image           → amount/currency/date JSON
        │                 validate_json → one re-ask → neutral default; sha256 disk cache
        │                 <<<UNTRUSTED>>> delimiters + injection flag; nothing here computes
        ▼
  reconcile.py     per-user Ledger: exclusions (cancelled/failed/non-cash/unrealized/pending
                   credits), duplicates, linked lifecycles, message verdicts, image amounts,
                   FX to home currency, recurrence detection → recurring series + one-offs,
                   every exclusion carries (event_id, reason, source), conflicts recorded
        ▼
  forecast.py      90-day daily balance series; is_safe = min balance ≥ minimum_balance_to_keep
        ▼
  solver.py        amount_safe_to_pay (cent-level binary search), earliest safe full-payment
                   date (90-day scan), smallest legal spending-change set (checks 12–14)
        ▼
  ranker.py        candidates: full / full+changes / wait / partial / each installment option
                   eligibility (accepted methods, max_installment_months) → forecast safety →
                   six ranking rules → affordability_status mapping
        ▼
  explain.py       templated decision_explanation; numeric guard: every number in the text
                   must exist in the solver facts, else a deterministic fallback
        ▼
  schema.py        normalize_row + validate_row (15 checks) on every row before it is written
        ▼
  main.py          per-request try/except, safe fallback row, output.csv, usage_report.md,
                   failures.csv, run record in log.txt
```

Support: `pipeline.py` (the per-request path shared by `main.py` and the evaluator),
`logger.py` (run records), `audit.py` (data audit), `evaluation/` (scoring harness, golden
fixtures, run log), `tests/`.

## Why the numbers are deterministic — and the model never produces one

Every value that reaches `output.csv` is computed by Python from the CSVs with `Decimal`
arithmetic; no float is used anywhere on the money path. The model is called exactly twice
per piece of evidence — once per message, once per image — and returns a fixed-shape JSON that
only *describes* the evidence: a verdict (cancel / amend / delay / confirm / irrelevant), an
amount as printed on a document, a currency code, a date. Those values are validated against
a closed schema, converted to `Decimal`, and handed to `reconcile.py`, which decides what to do
with them under the challenge's precedence rules. The model cannot add an event the rules
would reject, cannot change a rule, and never sees a balance, a minimum, or a request amount.

Three containment layers make this enforceable rather than aspirational:

1. **Closed vocabulary in** — `validate_json` rejects unknown keys, missing keys and
   out-of-vocabulary values; one re-ask with the exact error, then a neutral default.
2. **Untrusted content stays data** — evidence is wrapped in `<<<UNTRUSTED>>>` delimiters, the
   system prompt states that embedded instructions are to be reported, and a local heuristic
   flags injection attempts independently of the model; flagged requests are listed in
   `log.txt` for review. Golden fixtures 01 and 02 assert the decision is unchanged.
3. **Numeric guard out** — `explain.py` scans the only free-text column and rejects any
   number not present in the solver's facts; `schema.validate_row` runs on every row.

Reproducibility: every model response is cached under `.cache/` keyed on
`sha256(model, prompt_version, system, prompt, input bytes)`. Claude Opus 5 does not accept a
temperature parameter, so run-to-run stability comes from the cache; `--dry-run` replays it
with zero calls. `code.zip` ships the cache.

## Decision rules, briefly

* Pending debits are reserved; pending credits, refunds, bonuses, windfalls, unrealized
  valuations, cancelled and failed rows are never counted.
* Recurrence needs ≥ 3 occurrences with a consistent gap; projection uses the median gap and
  the median of the last three amounts. Confirmed scheduled rows supersede a projected
  occurrence within 3 days.
* A plan is safe only if the 90-day closing balance never drops below
  `minimum_balance_to_keep` with the plan's payments and spending changes applied.
* `amount_safe_to_pay` and `earliest_date_for_full_payment` measure capacity with no spending
  changes and no preferences. Preferences enter only in the ranker.
* Ranking: completes by deadline → no spending changes → lowest total paid → earliest start →
  fewest payments → lowest `payment_option_id`.
* Conflicts: explicit cancellation/settlement/amendment, then newer record from the same
  source, then settled over estimate, then the financially safer reading. Unresolved cases are
  recorded and flagged.

## Files in code.zip

```text
code/            all modules above, tests/, evaluation/ (usage_report.md, run_log.md, golden/)
.cache/          cached model responses (no secrets; enables --dry-run)
README.md        this file
requirements.txt
.env.example     variable names only, no values
```

## Known limitations

_To be completed by the author._

*
*
*
