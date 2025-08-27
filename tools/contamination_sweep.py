"""Run a sweep over different IsolationForest contamination values and report counts.

Usage: PYTHONPATH=. python tools/contamination_sweep.py --symbols SYMBOLS --days 90 --contaminations 0.005,0.01,0.02 --out reports/contamination_sweep_20.md
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None

from src.cerverus.models.tier2 import IsolationForestDetector


def fetch_symbol_history(symbol, days=90):
    if yf is None:
        return None
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    df = yf.download(
        symbol,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
    )
    if df.empty:
        return None
    # Flatten multiindex columns if present
    if hasattr(df.columns, "nlevels") and getattr(df.columns, "nlevels", 1) > 1:
        df.columns = [
            "_".join([str(c) for c in col]).strip() for col in df.columns.values
        ]
    df = df.reset_index()
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["symbol"] = symbol
    return df


def analyze_with_contamination(df, contamination):
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty or len(numeric) < 10:
        return 0
    detector = IsolationForestDetector(contamination=contamination)
    detector.fit(numeric)
    preds = detector.predict(numeric)
    return int((preds == -1).sum())


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--symbols", default="AAPL,MSFT,GOOG,AMZN,TSLA", help="Comma separated symbols"
    )
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--contaminations", default="0.005,0.01,0.02")
    parser.add_argument("--out", default="reports/contamination_sweep.md")
    args = parser.parse_args(argv)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    conts = [float(x) for x in args.contaminations.split(",")]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Collect counts
    counts = {c: {} for c in conts}

    for sym in symbols:
        print("Processing", sym)
        df = None
        if yf is not None:
            df = fetch_symbol_history(sym, days=args.days)
        if df is None:
            # synth fallback
            now = datetime.utcnow()
            rows = []
            for i in range(200):
                rows.append(
                    {
                        "Date": now - pd.Timedelta(minutes=30 * i),
                        "Close": 100 + np.random.randn() * 2,
                        "Volume": abs(int(1000 + np.random.randn() * 100)),
                        "symbol": sym,
                    }
                )
            df = pd.DataFrame(rows)

        for c in conts:
            n = analyze_with_contamination(df, c)
            counts[c][sym] = n

    # Write report
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("# Contamination sweep report\n")
        fh.write(f"Date: {datetime.utcnow().isoformat()}\n\n")
        fh.write("## Summary table (anomaly counts)\n\n")
        # header
        fh.write("| Symbol | " + " | ".join([str(c) for c in conts]) + " |\n")
        fh.write("|---|" + "---:|" * len(conts) + "\n")
        for sym in symbols:
            fh.write(f"| {sym} ")
            for c in conts:
                fh.write(f"| {counts[c].get(sym,0)} ")
            fh.write("|\n")

    print("Wrote report to", args.out)


if __name__ == "__main__":
    main(sys.argv[1:])
