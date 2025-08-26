
# Pilot 20-symbol Summary

Date: 2025-08-26

Executive summary
- Symbols processed: 20
- Historical window: 90 days
- Anomalies detected: 20 (approximately 1 anomaly per symbol in this run)

Sample anomalies by symbol

| Symbol | Timestamp (ISO) | Close |
|---|---:|---:|
| AAPL | 2025-08-08T00:00:00 | 229.09 |
| MSFT | 2025-07-31T00:00:00 | 532.62 |
| GOOG | 2025-08-25T00:00:00 | 209.16 |
| AMZN | 2025-07-31T00:00:00 | 234.11 |
| TSLA | 2025-06-05T00:00:00 | 284.70 |

Short technical notes
- The detector used: IsolationForest with `contamination=0.01`, applied to numeric columns only.
- `yfinance` may return columns with ticker suffixes (e.g. `Close_AAPL`); the pipeline flattens and extracts the Close value automatically.
- Output CSV: `results_pilot_20.csv`. Each row includes `symbol`, `timestamp`, `close`, and a `raw` field containing the original downloaded data for auditing.

Conclusion and next steps (senior team decision executed)
- Status: PoC pilot completed successfully on a 20-symbol sample. The detector flags suspicious observations; manual review is required to confirm false positives.
- Next step (scheduled): run the same pilot in batches for 100 symbols and produce an aggregated report with per-symbol anomaly counts and percentiles.

Relevant artifacts
- `tools/run_pilot.py` — pilot runner (download, analyze, save).
- `results_pilot_20.csv` — pilot results.

Audit note
- Keep `results_pilot_20.csv` and `reports/pilot_20_summary.md` under version control to preserve the experiment record.

