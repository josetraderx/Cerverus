"""Collector script for PoC: fetch historical OHLCV and write CSVs for ingestion.

Usage (developer):
    python -m src.cerverus.infrastructure.etl.collector_write_csv

This script writes a single CSV with columns matching the daily_prices table:
stock_id,symbol,trade_date,open_price,high_price,low_price,close_price,volume,adjusted_close

It uses the `yahoo_finance_client.fetch_history` function.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from datetime import date, timedelta

import pandas as pd

from ..external_apis.yahoo_finance_client import fetch_history
from .tickers_50 import TICKERS_50

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tickers_to_csv(tickers, start_date: str, end_date: str, out_path: Path):
    logger.info("Fetching data for %s tickers", len(tickers))
    df = fetch_history(tickers, start_date, end_date)

    if df.empty:
        logger.warning("No data fetched; exiting")
        return False

    # Map DataFrame columns to daily_prices schema
    df = df.rename(columns={
        "date": "trade_date",
        "open": "open_price",
        "high": "high_price",
        "low": "low_price",
        "close": "close_price",
        "adj_close": "adjusted_close",
        "volume": "volume",
    })

    # We don't have stock_id mapping here; leave stock_id empty for loader to resolve or populate with client logic
    df["stock_id"] = ""

    # Order columns as required
    out_cols = ["stock_id", "symbol", "trade_date", "open_price", "high_price", "low_price", "close_price", "volume", "adjusted_close"]
    df = df[out_cols]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Wrote CSV to %s (%d rows)", out_path, len(df))
    return True


def default_dates(days: int = 365):
    end = date.today()
    start = end - timedelta(days=days)
    return start.isoformat(), end.isoformat()


def main():
    start, end = default_dates(365)
    out = Path("data/ingestion/daily_prices_50.csv")
    ok = tickers_to_csv(TICKERS_50, start, end, out)
    if not ok:
        logger.error("Collector failed")


if __name__ == "__main__":
    main()
