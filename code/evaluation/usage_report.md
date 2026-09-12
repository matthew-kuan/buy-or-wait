# Token usage and cost report

Generated 2026-09-12T22:21:35+00:00 by `code/main.py` for the full-dataset run that produced `output.csv` (250 requests, mode = **dry-run**).

## Providers and models

| provider | model | role |
|---|---|---|
| Anthropic (Claude API) | `claude-opus-5` | message classification (call 1) and image reading (call 2) |

Everything numeric in `output.csv` is computed deterministically in Python; the model only reads amounts, dates, currencies and intent off untrusted messages and images.

## Pricing constants

USD per 1M tokens, as checked on 2026-09-12 against Anthropic's public price list:

| model | input $/1M | output $/1M |
|---|---|---|
| `claude-opus-5` | 5.00 | 25.00 |
| `claude-sonnet-5` | 2.00 | 10.00 |
| `claude-haiku-4-5` | 1.00 | 5.00 |

## This run (live API calls)

| metric | value |
|---|---|
| model calls made | 0 |
| cache hits (no call) | 0 |
| re-asks after schema rejection | 0 |
| fallbacks to the neutral default | 231 |
| retries (429/5xx backoff) | 0 |
| refusals | 0 |
| injection flags | 0 |
| input tokens | 0 |
| output tokens | 0 |
| total tokens | 0 |
| average tokens per request (250) | 0.0 |
| average tokens per model call | n/a (no live calls) |
| estimated total cost (USD) | 0.0000 |
| estimated cost per request (USD) | 0.000000 |
| wall time (s) | 8.9 |
| rows failing validation | 0 |
| rows produced by the failure fallback | 0 |

## Per-model totals

| model | live calls | cache hits | live input | live output | cached input | cached output | live cost (USD) |
|---|---|---|---|---|---|---|---|

## Cached responses replayed in this run

Responses served from `.cache/` were produced by earlier live calls. Their recorded usage is reported here so the cost of the answers used by this run is visible even when the run itself made no calls.

| metric | value |
|---|---|
| cached input tokens | 0 |
| cached output tokens | 0 |
| cost of producing the cached answers (USD) | 0.0000 |
| overall (live + cached) tokens | 0 |
| overall (live + cached) cost (USD) | 0.0000 |
| overall cost per request (USD) | 0.000000 |

No API keys, credentials, or configuration values are included in this report.
