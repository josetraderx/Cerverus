"""Simple Yahoo Finance client used by the PoC collector.

This module provides a minimal wrapper around `yfinance` to fetch historical
OHLCV data for a list of tickers and return a merged pandas DataFrame with
uniform columns.

All code and filenames are in English; CLI and comments are English.
"""
from typing import List, Optional
import time
import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_history(tickers: List[str], start: str, end: str, interval: str = "1d", retry: int = 3, pause: float = 1.0) -> pd.DataFrame:
	"""Fetch historical OHLCV data for a list of tickers using yfinance.

	Returns a DataFrame with columns: symbol, date, open, high, low, close, adj_close, volume
	Date is returned as a date (not datetime).
	start/end are ISO date strings (YYYY-MM-DD).
	"""
	all_frames = []

	for symbol in tickers:
		attempts = 0
		while attempts < retry:
			try:
				ticker = yf.Ticker(symbol)
				df = ticker.history(start=start, end=end, interval=interval, actions=True)
				if df is None or df.empty:
					logger.warning("No data for %s (%s to %s)", symbol, start, end)
					break

				# keep only necessary columns and normalize names
				df = df.rename(columns={
					"Open": "open",
					"High": "high",
					"Low": "low",
					"Close": "close",
					"Volume": "volume",
					"Dividends": "dividends",
					"Stock Splits": "stock_splits",
					"Adj Close": "adj_close",
				})

				df = df.reset_index()
				# ensure date column is date only
				df["date"] = pd.to_datetime(df["Date"]).dt.date

				out = pd.DataFrame({
					"symbol": symbol,
					"date": df["date"],
					"open": df.get("open"),
					"high": df.get("high"),
					"low": df.get("low"),
					"close": df.get("close"),
					"adj_close": df.get("adj_close"),
					"volume": df.get("volume"),
				})

				all_frames.append(out)
				break
			except Exception as exc:
				attempts += 1
				logger.exception("Error fetching %s (attempt %s/%s): %s", symbol, attempts, retry, exc)
				time.sleep(pause * attempts)

	if not all_frames:
		return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"])

	result = pd.concat(all_frames, ignore_index=True)
	# Reorder columns explicitly
	result = result[["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]]
	return result


def fetch_single(symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
	return fetch_history([symbol], start, end, interval)


if __name__ == "__main__":
	# quick manual test (won't run in CI); user can run this file directly
	logging.basicConfig(level=logging.INFO)
	df = fetch_history(["AAPL", "MSFT"], "2024-08-01", "2025-08-01")
	print(df.head())
