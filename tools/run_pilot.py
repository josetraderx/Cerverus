"""Pilot runner: download price data for a list of symbols (or generate synthetic), run IsolationForest per symbol, and save results to CSV.

Usage: python tools/run_pilot.py --symbols AAPL,MSFT --days 90 --out results.csv
"""
import argparse
import sys
import time
from datetime import datetime, timedelta
import csv
import os

try:
    import yfinance as yf
except Exception:
    yf = None

import numpy as np
import pandas as pd
from src.cerverus.models.tier2 import IsolationForestDetector


def fetch_symbol_history(symbol, days=90):
    if yf is None:
        return None
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    df = yf.download(symbol, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), progress=False)
    if df.empty:
        return None
    # yfinance may return MultiIndex columns when querying multiple tickers;
    # flatten them to single level like 'Close' or 'Close_AAPL' for robustness.
    if hasattr(df.columns, 'levels') and getattr(df.columns, 'nlevels', 1) > 1:
        df.columns = ["_".join([str(c) for c in col]).strip() for col in df.columns.values]
    df = df.reset_index()
    # Ensure a 'Date' column exists
    if 'Date' not in df.columns:
        # sometimes index name is 'Datetime' or similar
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
    df['symbol'] = symbol
    return df


def synth_symbol(symbol, n=200):
    now = datetime.utcnow()
    rows = []
    for i in range(n):
        rows.append({
            'Date': now - pd.Timedelta(minutes=30*i),
            'Open': 100 + np.random.randn()*2,
            'High': 100 + np.random.randn()*2,
            'Low': 100 + np.random.randn()*2,
            'Close': 100 + np.random.randn()*2,
            'Volume': abs(int(1000 + np.random.randn()*100)),
            'symbol': symbol
        })
    return pd.DataFrame(rows)


def analyze_symbol(df):
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return []
    detector = IsolationForestDetector(contamination=0.01)
    detector.fit(numeric)
    preds = detector.predict(numeric)
    anomalies = df[preds == -1]
    return anomalies


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', default='AAPL,MSFT', help='Comma separated symbols')
    parser.add_argument('--days', type=int, default=90)
    parser.add_argument('--out', default='pilot_results.csv')
    parser.add_argument('--use-synthetic', action='store_true')
    args = parser.parse_args(argv)

    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
    results = []

    for sym in symbols:
        print('Processing', sym)
        if args.use_synthetic or yf is None:
            df = synth_symbol(sym, n=200)
        else:
            df = fetch_symbol_history(sym, days=args.days)
            if df is None:
                print('No data for', sym, '-> using synthetic')
                df = synth_symbol(sym, n=200)

        anomalies = analyze_symbol(df)
        for _, row in anomalies.iterrows():
            r = row.to_dict()
            ts = r.get('Date') or r.get('Datetime') or r.get('date')
            close = r.get('Close') or r.get('close') or r.get('Adj Close') or None
            # Normalize timestamp to ISO
            try:
                if ts is not None and not pd.isna(ts):
                    ts = pd.to_datetime(ts).isoformat()
            except Exception:
                ts = str(ts) if ts is not None else ''
            # Normalize close to float if possible
            try:
                close_val = float(close) if close is not None and not pd.isna(close) else ''
            except Exception:
                close_val = ''
            # Include raw row for debugging
            raw = {k: (v if not pd.isna(v) else None) for k, v in r.items()}
            results.append({'symbol': sym, 'timestamp': ts or '', 'close': close_val, 'raw': raw})

    # Post-process results: if 'close' is empty, try to extract from raw keys like 'Close' or 'Close_<SYMBOL>'
    for r in results:
        if r.get('close') in (None, ''):
            raw = r.get('raw') or {}
            found = None
            for k, v in raw.items():
                if isinstance(k, str) and 'Close' in k:
                    found = v
                    break
            try:
                if found is not None:
                    r['close'] = float(found)
            except Exception:
                r['close'] = ''

    # save results
    out = args.out
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['symbol', 'timestamp', 'close', 'raw'])
        writer.writeheader()
        for r in results:
            # Serialize raw as JSON-like string
            row = {k: (v if k != 'raw' else str(v)) for k, v in r.items()}
            writer.writerow(row)

    print('Wrote', len(results), 'anomalies to', out)


if __name__ == '__main__':
    main(sys.argv[1:])
