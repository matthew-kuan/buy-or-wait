# Token usage and cost report

Generated 2026-09-13T03:58:08+00:00 by `code/main.py` for the full-dataset run that produced `output.csv` (250 requests, mode = **live**).

**Quota note.** Runs use free-tier AI Studio quota (no billing); the list prices are shown for reference only. Gemini output pricing includes thinking tokens, which are counted separately below.

**Model choice: gemini-3.5-flash-lite was used because the free-tier requests-per-day allowance for gemini-3.5-flash was exhausted (27 of 20 used); flash-lite has a separate free-tier quota of 15 RPM / 500 RPD, which covers all 231 extraction calls in one run.**

## Providers and models

| provider | model | role |
|---|---|---|
| Google (Gemini API, AI Studio key) | `gemini-3.5-flash-lite` | message classification (call 1) and image reading (call 2); native structured output (response_mime_type=application/json + response_schema) |

Everything numeric in `output.csv` is computed deterministically in Python; the model only reads amounts, dates, currencies and intent off untrusted messages and images.

## Pricing constants

List price, USD per 1M tokens, pulled 2026-09-12 from https://ai.google.dev/gemini-api/docs/pricing (paid tier; the free tier is charged nothing). Gemini bills thinking tokens at the output rate.

| model | input $/1M | output $/1M (incl. thinking) |
|---|---|---|
| `gemini-3.5-flash` | 1.50 | 9.00 |
| `gemini-3.6-flash` | 0.75 | 3.75 |
| `gemini-3.7-flash` | 0.75 | 3.75 |
| `gemini-3.8-flash` | 0.75 | 3.75 |
| `gemini-3.5-flash-lite` | 0.30 | 2.50 |

Token field mapping: input = `usage_metadata.prompt_token_count`, output = `usage_metadata.candidates_token_count`, reasoning = `usage_metadata.thoughts_token_count` (reported separately, priced as output).

## This run (live API calls)

| metric | value |
|---|---|
| model calls made | 227 |
| cache hits (no call) | 4 |
| re-asks after schema rejection (second line of defense) | 0 |
| fallbacks to the neutral default | 0 |
| retries (429/5xx backoff) | 8 |
| seconds spent waiting on the rate limiter | 0 |
| empty/blocked responses | 0 |
| injection flags | 0 |
| input tokens (prompt_token_count) | 147,849 |
| output tokens (candidates_token_count) | 21,968 |
| reasoning tokens (thoughts_token_count, billed as output) | 0 |
| total tokens | 169,817 |
| average tokens per request (250) | 679.3 |
| average tokens per model call | 748.1 |
| list-price cost of this run (USD) | 0.0993 |
| list-price cost per request (USD) | 0.000397 |
| wall time (s) | 899.8 |
| rows failing validation | 0 |
| rows produced by the failure fallback | 0 |

## Per-model totals

| model | live calls | cache hits | live input | live output | live reasoning | cached input | cached output | cached reasoning | list cost live (USD) |
|---|---|---|---|---|---|---|---|---|---|
| `gemini-3.5-flash-lite` | 227 | 4 | 147,849 | 21,968 | 0 | 3,129 | 402 | 0 | 0.0993 |

## Cached responses replayed in this run

Responses served from `.cache/` were produced by earlier live calls. Their recorded usage is reported here so the cost of the answers used by this run is visible even when the run itself made no calls.

| metric | value |
|---|---|
| cached input tokens | 3,129 |
| cached output tokens | 402 |
| cached reasoning tokens | 0 |
| list-price cost of producing the cached answers (USD) | 0.0019 |
| overall (live + cached) tokens | 173,348 |
| overall (live + cached) list-price cost (USD) | 0.1012 |
| overall list-price cost per request (USD) | 0.000405 |

No API keys, credentials, or configuration values are included in this report.
