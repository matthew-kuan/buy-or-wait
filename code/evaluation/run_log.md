# Evaluation runs (dataset/sample_requests.csv, 25 rows)

amt_mae is mean |pred-expected| / requested_amount; amt_mean_rel is mean |pred-expected| / expected (over requested_amount when expected is 0); within_1pct uses the same relative error.

| run | timestamp | note | status_acc | method_acc | earliest_acc | plan_acc | amt_mae | amt_mean_rel | amt_within_1pct | changes_jaccard | status_macro_f1 | all_fields_correct | valid_rows | regressed | improved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-09-12T21:31:56+00:00 | baseline (evidence: 0 verdicts, 16 image facts (api_calls=0 cache_hits=0 fallbacks=231)) | 0.720 | 0.800 | 0.320 | 0.640 | 0.183 | 2.151 | 0.160 | 0.880 | 0.718 | 0.160 | 1.000 | - | - |
