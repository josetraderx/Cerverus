"""
Cerverus System - Stage 1: Data Collection
Main extraction module implementing polymorphic adapters and fault tolerance
Following documentation requirements strictly
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import time
import json
from pathlib import Path
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import asyncio
from enum import Enum

# Configure structured logging as per documentation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class RateLimit:
    """Rate limiting configuration per data source as specified in documentation."""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_limit: int

@dataclass
class DataExtractionResult:
    """Result of data extraction operation with metadata tracking."""
    success: bool
    data: Optional[Any]
    records_count: int
    source: str
    timestamp: datetime
    s3_path: Optional[str]
    error_message: Optional[str] = None
    data_quality_score: Optional[float] = None

class DataSourceAdapter(ABC):
    """
    Abstract interface for data source adapters.
    Implements Polymorphic Adapter Pattern as per Stage 1 documentation.
    """
    
    def __init__(self, source_name: str, rate_limit: RateLimit):
        self.source_name = source_name
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.request_count = {"minute": 0, "hour": 0, "day": 0}
        self.logger = logging.getLogger(f"{__name__}.{source_name}")
        self._setup_session()
    
    def _setup_session(self):
        """Setup HTTP session with retry strategy."""
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    @abstractmethod
    def extract_data(self, symbols: List[str], start_date: datetime, 
                    end_date: datetime) -> DataExtractionResult:
        """Extract data from external source with metadata."""
        pass
    
    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate connectivity with data source."""
        pass
    
    @abstractmethod
    def get_rate_limits(self) -> RateLimit:
        """Return current rate limits for this source."""
        pass
    
    def _check_rate_limit(self) -> bool:
        """
        Check if new request can be made according to rate limits.
        Implements Strategy pattern for rate limiting.
        """
        current_time = time.time()
        
        # Reset counters if time has passed
        if current_time - self.last_request_time > 60:  # 1 minute
            self.request_count["minute"] = 0
        if current_time - self.last_request_time > 3600:  # 1 hour
            self.request_count["hour"] = 0
        if current_time - self.last_request_time > 86400:  # 1 day
            self.request_count["day"] = 0
        
        # Check limits
        if (self.request_count["minute"] >= self.rate_limit.requests_per_minute or
            self.request_count["hour"] >= self.rate_limit.requests_per_hour or
            self.request_count["day"] >= self.rate_limit.requests_per_day):
            return False
        
        return True
    
    def _update_rate_limit_counters(self):
        """Update rate limiting counters."""
        self.request_count["minute"] += 1
        self.request_count["hour"] += 1
        self.request_count["day"] += 1
        self.last_request_time = time.time()
    
    def _calculate_data_quality_score(self, data: Any) -> float:
        """Calculate data quality score based on completeness and consistency."""
        if not data:
            return 0.0
        
        total_fields = 0
        complete_fields = 0
        
        if isinstance(data, dict):
            for symbol, symbol_data in data.items():
                if isinstance(symbol_data, dict) and 'ohlc' in symbol_data:
                    ohlc_data = symbol_data['ohlc']
                    for record in ohlc_data:
                        total_fields += 4  # OHLC
                        complete_fields += sum(1 for v in record.values() if v is not None)
        
        return complete_fields / total_fields if total_fields > 0 else 0.0
    
    def _log_extraction_start(self, symbols: List[str], start_date: datetime, 
                             end_date: datetime):
        """Structured logging for extraction start."""
        self.logger.info(
            "Starting data extraction",
            extra={
                "source": self.source_name,
                "symbols": symbols,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "symbol_count": len(symbols)
            }
        )
    
    def _log_extraction_result(self, result: DataExtractionResult):
        """Structured logging for extraction result."""
        if result.success:
            self.logger.info(
                "Extraction completed successfully",
                extra={
                    "source": result.source,
                    "records_count": result.records_count,
                    "s3_path": result.s3_path,
                    "data_quality_score": result.data_quality_score,
                    "timestamp": result.timestamp.isoformat()
                }
            )
        else:
            self.logger.error(
                "Data extraction failed",
                extra={
                    "source": result.source,
                    "error_message": result.error_message,
                    "timestamp": result.timestamp.isoformat()
                }
            )


class YahooFinanceAdapter(DataSourceAdapter):
    """
    Yahoo Finance adapter for OHLC prices, volumes, and intraday data.
    Configured for ~500MB/day capacity for top 500 stocks as per documentation.
    """
    
    def __init__(self):
        rate_limit = RateLimit(
            requests_per_minute=60,
            requests_per_hour=2000,
            requests_per_day=100000,
            burst_limit=10
        )
        super().__init__("yahoo_finance", rate_limit)
        self.daily_capacity_mb = 500
        self.max_symbols_per_request = 50
    
    def extract_data(self, symbols: List[str], start_date: datetime, 
                    end_date: datetime) -> DataExtractionResult:
        """Extract data from Yahoo Finance with validation and rate limiting."""
        self._log_extraction_start(symbols, start_date, end_date)
        
        if not self._check_rate_limit():
            return DataExtractionResult(
                success=False,
                data=None,
                records_count=0,
                source=self.source_name,
                timestamp=datetime.now(),
                s3_path=None,
                error_message="Rate limit exceeded"
            )
        
        try:
            all_data = {}
            total_records = 0
            
            # Process symbols in batches for efficiency
            for i in range(0, len(symbols), self.max_symbols_per_request):
                batch = symbols[i:i + self.max_symbols_per_request]
                
                for symbol in batch:
                    try:
                        ticker = yf.Ticker(symbol)
                        
                        # Extract historical data with different intervals
                        hist_daily = ticker.history(start=start_date, end=end_date, interval="1d")
                        
                        if not hist_daily.empty:
                            # Get company information
                            info = ticker.info
                            
                            all_data[symbol] = {
                                "ohlc": hist_daily[['Open', 'High', 'Low', 'Close']].to_dict('records'),
                                "volume": hist_daily['Volume'].to_list(),
                                "adjusted_close": hist_daily.get('Adj Close', hist_daily['Close']).to_list(),
                                "dividends": ticker.dividends.to_dict() if hasattr(ticker, 'dividends') else {},
                                "splits": ticker.splits.to_dict() if hasattr(ticker, 'splits') else {},
                                "company_info": {
                                    "market_cap": info.get('marketCap'),
                                    "pe_ratio": info.get('trailingPE'),
                                    "sector": info.get('sector'),
                                    "industry": info.get('industry'),
                                    "full_time_employees": info.get('fullTimeEmployees')
                                },
                                "metadata": {
                                    "extraction_timestamp": datetime.now().isoformat(),
                                    "interval": "1d",
                                    "source": "yahoo_finance"
                                }
                            }
                            total_records += len(hist_daily)
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to extract data for {symbol}: {str(e)}")
                        continue
            
            self._update_rate_limit_counters()
            
            # Generate S3 path following Bronze layer structure from documentation
            s3_path = f"s3://cerverus-bronze/market_data/yahoo_finance/{datetime.now().strftime('%Y/%m/%d')}/{int(time.time())}.parquet"
            
            result = DataExtractionResult(
                success=True,
                data=all_data,
                records_count=total_records,
                source=self.source_name,
                timestamp=datetime.now(),
                s3_path=s3_path,
                data_quality_score=self._calculate_data_quality_score(all_data)
            )
            
            self._log_extraction_result(result)
            return result
            
        except Exception as e:
            error_result = DataExtractionResult(
                success=False,
                data=None,
                records_count=0,
                source=self.source_name,
                timestamp=datetime.now(),
                s3_path=None,
                error_message=f"Yahoo Finance extraction error: {str(e)}"
            )
            self._log_extraction_result(error_result)
            return error_result
    
    def validate_connection(self) -> bool:
        """Validate connection with Yahoo Finance."""
        try:
            test_ticker = yf.Ticker("AAPL")
            test_data = test_ticker.history(period="1d", interval="1d")
            return not test_data.empty
        except Exception as e:
            self.logger.error(f"Yahoo Finance connection validation failed: {str(e)}")
            return False
    
    def get_rate_limits(self) -> RateLimit:
        """Return current rate limits for Yahoo Finance."""
        return self.rate_limit


class SECEdgarAdapter(DataSourceAdapter):
    """
    SEC EDGAR adapter for quarterly (10-Q), annual (10-K), insider trading (Form 4), and 8-K reports.
    Configured for ~2GB/day capacity on peak reporting days as per documentation.
    """
    
    def __init__(self):
        rate_limit = RateLimit(
            requests_per_minute=10,  # SEC has strict rate limiting
            requests_per_hour=100,
            requests_per_day=1000,
            burst_limit=5
        )
        super().__init__("sec_edgar", rate_limit)
        self.daily_capacity_gb = 2
        self.base_url = "https://www.sec.gov/Archives/edgar"
        
        # Set required User-Agent header for SEC compliance
        self.session.headers.update({
            'User-Agent': 'Cerverus Financial Fraud Detection System admin@cerverus.com'
        })
    
    def extract_data(self, symbols: List[str], start_date: datetime, 
                    end_date: datetime) -> DataExtractionResult:
        """Extract SEC filings data with event-driven capability."""
        self._log_extraction_start(symbols, start_date, end_date)
        
        if not self._check_rate_limit():
            return DataExtractionResult(
                success=False,
                data=None,
                records_count=0,
                source=self.source_name,
                timestamp=datetime.now(),
                s3_path=None,
                error_message="Rate limit exceeded"
            )
        
        try:
            all_filings = {}
            total_records = 0
            
            for symbol in symbols:
                try:
                    # In production, implement actual SEC EDGAR API calls
                    # For now, simulate the structure that would be returned
                    filings_data = {
                        "10-K": self._extract_10k_filings(symbol, start_date, end_date),
                        "10-Q": self._extract_10q_filings(symbol, start_date, end_date),
                        "8-K": self._extract_8k_filings(symbol, start_date, end_date),
                        "Form-4": self._extract_form4_filings(symbol, start_date, end_date),
                        "metadata": {
                            "extraction_timestamp": datetime.now().isoformat(),
                            "cik": self._get_cik_for_symbol(symbol),
                            "source": "sec_edgar"
                        }
                    }
                    
                    all_filings[symbol] = filings_data
                    total_records += sum(len(v) if isinstance(v, list) else 1 
                                       for k, v in filings_data.items() if k != "metadata")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to extract SEC data for {symbol}: {str(e)}")
                    continue
            
            self._update_rate_limit_counters()
            
            s3_path = f"s3://cerverus-bronze/regulatory_data/sec_edgar/{datetime.now().strftime('%Y/%m/%d')}/{int(time.time())}.json"
            
            result = DataExtractionResult(
                success=True,
                data=all_filings,
                records_count=total_records,
                source=self.source_name,
                timestamp=datetime.now(),
                s3_path=s3_path,
                data_quality_score=0.9  # High quality for regulatory data
            )
            
            self._log_extraction_result(result)
            return result
            
        except Exception as e:
            error_result = DataExtractionResult(
                success=False,
                data=None,
                records_count=0,
                source=self.source_name,
                timestamp=datetime.now(),
                s3_path=None,
                error_message=f"SEC EDGAR extraction error: {str(e)}"
            )
            self._log_extraction_result(error_result)
            return error_result
    
    def _extract_10k_filings(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Extract 10-K annual filings."""
        # Placeholder for actual implementation
        return []
    
    def _extract_10q_filings(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Extract 10-Q quarterly filings."""
        # Placeholder for actual implementation
        return []
    
    def _extract_8k_filings(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Extract 8-K material events filings."""
        # Placeholder for actual implementation
        return []
    
    def _extract_form4_filings(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Extract Form 4 insider trading filings."""
        # Placeholder for actual implementation
        return []
    
    def _get_cik_for_symbol(self, symbol: str) -> Optional[str]:
        """Get CIK (Central Index Key) for symbol."""
        # Placeholder for actual implementation
        return None
    
    def validate_connection(self) -> bool:
        """Validate connection with SEC EDGAR."""
        try:
            response = self.session.get("https://www.sec.gov/edgar/search/", timeout=10)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"SEC EDGAR connection validation failed: {str(e)}")
            return False
    
    def get_rate_limits(self) -> RateLimit:
        """Return current rate limits for SEC EDGAR."""
        return self.rate_limit


class FINRAAdapter(DataSourceAdapter):
    """
    FINRA adapter for dark pool data, suspensions, regulations, and short interest.
    Configured for ~100MB/day capacity as per documentation.
    """
    
    def __init__(self):
        rate_limit = RateLimit(
            requests_per_minute=30,
            requests_per_hour=500,
            requests_per_day=5000,
            burst_limit=5
        )
        super().__init__("finra", rate_limit)
        self.daily_capacity_mb = 100
    
    def extract_data(self, symbols: List[str], start_date: datetime, 
                    end_date: datetime) -> DataExtractionResult:
        """Extract FINRA regulatory data with daily + event-driven frequency."""
        self._log_extraction_start(symbols, start_date, end_date)
        
        if not self._check_rate_limit():
            return DataExtractionResult(
                success=False,
                data=None,
                records_count=0,
                source=self.source_name,
                timestamp=datetime.now(),
                s3_path=None,
                error_message="Rate limit exceeded"
            )
        
        try:
            all_data = {}
            total_records = 0
            
            for symbol in symbols:
                try:
                    finra_data = {
                        "dark_pool_data": self._extract_dark_pool_data(symbol, start_date, end_date),
                        "short_interest": self._extract_short_interest_data(symbol, start_date, end_date),
                        "regulatory_alerts": self._extract_regulatory_alerts(symbol, start_date, end_date),
                        "suspensions": self._extract_suspensions_data(symbol, start_date, end_date),
                        "metadata": {
                            "extraction_timestamp": datetime.now().isoformat(),
                            "source": "finra",
                            "reporting_period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                        }
                    }
                    
                    all_data[symbol] = finra_data
                    total_records += sum(len(v) if isinstance(v, list) else 1 
                                       for k, v in finra_data.items() if k != "metadata")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to extract FINRA data for {symbol}: {str(e)}")
                    continue
            
            self._update_rate_limit_counters()
            
            s3_path = f"s3://cerverus-bronze/regulatory_data/finra/{datetime.now().strftime('%Y/%m/%d')}/{int(time.time())}.json"
            
            result = DataExtractionResult(
                success=True,
                data=all_data,
                records_count=total_records,
                source=self.source_name,
                timestamp=datetime.now(),
                s3_path=s3_path,
                data_quality_score=0.85
            )
            
            self._log_extraction_result(result)
            return result
            
        except Exception as e:
            error_result = DataExtractionResult(
                success=False,
                data=None,
                records_count=0,
                source=self.source_name,
                timestamp=datetime.now(),
                s3_path=None,
                error_message=f"FINRA extraction error: {str(e)}"
            )
            self._log_extraction_result(error_result)
            return error_result
    
    def _extract_dark_pool_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Extract dark pool trading data."""
        # Placeholder for actual implementation
        return []
    
    def _extract_short_interest_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict:
        """Extract short interest data."""
        # Placeholder for actual implementation
        return {}
    
    def _extract_regulatory_alerts(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Extract regulatory alerts and notifications."""
        # Placeholder for actual implementation
        return []
    
    def _extract_suspensions_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Extract trading suspensions data."""
        # Placeholder for actual implementation
        return []
    
    def validate_connection(self) -> bool:
        """Validate connection with FINRA."""
        try:
            response = self.session.get("https://www.finra.org", timeout=10)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"FINRA connection validation failed: {str(e)}")
            return False
    
    def get_rate_limits(self) -> RateLimit:
        """Return current rate limits for FINRA."""
        return self.rate_limit


class AlphaVantageAdapter(DataSourceAdapter):
    """
    Alpha Vantage adapter for technical indicators, forex, commodities, and sentiment.
    Configured for ~50MB/day capacity as per documentation.
    """
    
    def __init__(self, api_key: str):
        rate_limit = RateLimit(
            requests_per_minute=5,  # Alpha Vantage free tier limit
            requests_per_hour=25,
            requests_per_day=500,
            burst_limit=2
        )
        super().__init__("alpha_vantage", rate_limit)
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.daily_capacity_mb = 50
    
    def extract_data(self, symbols: List[str], start_date: datetime, 
                    end_date: datetime) -> DataExtractionResult:
        """Extract Alpha Vantage technical indicators and sentiment data."""
        self._log_extraction_start(symbols, start_date, end_date)
        
        if not self._check_rate_limit():
            return DataExtractionResult(
                success=False,
                data=None,
                records_count=0,
                source=self.source_name,
                timestamp=datetime.now(),
                s3_path=None,
                error_message="Rate limit exceeded"
            )
        
        try:
            all_data = {}
            total_records = 0
            
            for symbol in symbols:
                try:
                    symbol_data = {
                        "technical_indicators": {
                            "rsi": self._get_rsi(symbol),
                            "macd": self._get_macd(symbol),
                            "bollinger_bands": self._get_bollinger_bands(symbol)
                        },
                        "forex_data": self._get_forex_data(symbol) if self._is_forex_symbol(symbol) else {},
                        "commodities_data": self._get_commodities_data(symbol) if self._is_commodity_symbol(symbol) else {},
                        "sentiment": self._get_sentiment_data(symbol),
                        "news": self._get_news_data(symbol),
                        "metadata": {
                            "extraction_timestamp": datetime.now().isoformat(),
                            "source": "alpha_vantage",
                            "api_key_used": self.api_key[:8] + "..."  # Partial key for tracking
                        }
                    }
                    
                    all_data[symbol] = symbol_data
                    total_records += sum(len(v) if isinstance(v, (list, dict)) else 1 
                                       for category in symbol_data.values() 
                                       for v in (category.values() if isinstance(category, dict) else [category])
                                       if category != symbol_data["metadata"])
                    
                except Exception as e:
                    self.logger.warning(f"Failed to extract Alpha Vantage data for {symbol}: {str(e)}")
                    continue
            
            self._update_rate_limit_counters()
            
            s3_path = f"s3://cerverus-bronze/technical_data/alpha_vantage/{datetime.now().strftime('%Y/%m/%d')}/{int(time.time())}.json"
            
            result = DataExtractionResult(
                success=True,
                data=all_data,
                records_count=total_records,
                source=self.source_name,
                timestamp=datetime.now(),
                s3_path=s3_path,
                data_quality_score=0.8
            )
            
            self._log_extraction_result(result)
            return result
            
        except Exception as e:
            error_result = DataExtractionResult(
                success=False,
                data=None,
                records_count=0,
                source=self.source_name,
                timestamp=datetime.now(),
                s3_path=None,
                error_message=f"Alpha Vantage extraction error: {str(e)}"
            )
            self._log_extraction_result(error_result)
            return error_result
    
    def _get_rsi(self, symbol: str) -> Dict:
        """Get RSI technical indicator."""
        # Placeholder for actual Alpha Vantage API call
        return {"rsi_values": [], "period": 14, "overbought": 70, "oversold": 30}
    
    def _get_macd(self, symbol: str) -> Dict:
        """Get MACD technical indicator."""
        # Placeholder for actual Alpha Vantage API call
        return {"macd_line": [], "signal_line": [], "histogram": []}
    
    def _get_bollinger_bands(self, symbol: str) -> Dict:
        """Get Bollinger Bands technical indicator."""
        # Placeholder for actual Alpha Vantage API call
        return {"upper_band": [], "middle_band": [], "lower_band": [], "period": 20, "std_dev": 2}
    
    def _get_forex_data(self, symbol: str) -> Dict:
        """Get forex exchange rate data."""
        # Placeholder for actual Alpha Vantage API call
        return {"exchange_rate": [], "bid": [], "ask": []}
    
    def _get_commodities_data(self, symbol: str) -> Dict:
        """Get commodities price data."""
        # Placeholder for actual Alpha Vantage API call
        return {"commodity_prices": [], "unit": "", "currency": "USD"}
    
    def _get_sentiment_data(self, symbol: str) -> Dict:
        """Get market sentiment analysis."""
        # Placeholder for actual Alpha Vantage API call
        return {"sentiment_score": 0.0, "label": "neutral", "relevance_score": 0.0}
    
    def _get_news_data(self, symbol: str) -> List[Dict]:
        """Get financial news articles."""
        # Placeholder for actual Alpha Vantage API call
        return []
    
    def _is_forex_symbol(self, symbol: str) -> bool:
        """Check if symbol is a forex pair."""
        return len(symbol) == 6 and symbol.isupper()
    
    def _is_commodity_symbol(self, symbol: str) -> bool:
        """Check if symbol is a commodity."""
        commodities = ["WTI", "BRENT", "NATURAL_GAS", "COPPER", "ALUMINUM", "WHEAT", "CORN", "COTTON", "SUGAR", "COFFEE"]
        return symbol.upper() in commodities
    
    def validate_connection(self) -> bool:
        """Validate connection with Alpha Vantage."""
        try:
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": "AAPL",
                "apikey": self.api_key
            }
            response = self.session.get(self.base_url, params=params, timeout=10)
            return response.status_code == 200 and "Error Message" not in response.text
        except Exception as e:
            self.logger.error(f"Alpha Vantage connection validation failed: {str(e)}")
            return False
    
    def get_rate_limits(self) -> RateLimit:
        """Return current rate limits for Alpha Vantage."""
        return self.rate_limit