-- PostgreSQL Schema Design for Cerverus PoC
-- Daily OHLCV data for top 100 S&P 500 stocks

-- ============================================================================
-- 1. STOCKS MASTER TABLE
-- ============================================================================
CREATE TABLE stocks (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL UNIQUE,
    company_name VARCHAR(255) NOT NULL,
    market_cap DECIMAL(15,2),
    sector VARCHAR(100),
    industry VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for fast symbol lookups
CREATE INDEX idx_stocks_symbol ON stocks(symbol);
CREATE INDEX idx_stocks_market_cap ON stocks(market_cap DESC);

-- ============================================================================
-- 2. DAILY PRICE DATA (OHLCV) - PARTITIONED BY DATE
-- ============================================================================
CREATE TABLE daily_prices (
    id BIGSERIAL,
    stock_id INTEGER NOT NULL REFERENCES stocks(id),
    symbol VARCHAR(10) NOT NULL, -- Denormalized for query performance
    trade_date DATE NOT NULL,
    open_price DECIMAL(12,4) NOT NULL,
    high_price DECIMAL(12,4) NOT NULL,
    low_price DECIMAL(12,4) NOT NULL,
    close_price DECIMAL(12,4) NOT NULL,
    volume BIGINT NOT NULL,
    adjusted_close DECIMAL(12,4), -- For dividend/split adjustments
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT daily_prices_pkey PRIMARY KEY (id, trade_date),
    CONSTRAINT daily_prices_unique UNIQUE (stock_id, trade_date),
    CONSTRAINT valid_ohlc CHECK (
        open_price > 0 AND high_price > 0 AND 
        low_price > 0 AND close_price > 0 AND
        high_price >= low_price AND
        high_price >= open_price AND high_price >= close_price AND
        low_price <= open_price AND low_price <= close_price
    ),
    CONSTRAINT valid_volume CHECK (volume >= 0)
) PARTITION BY RANGE (trade_date);

-- Create monthly partitions for last 2 years
-- August 2024 - August 2025 (14 partitions)
CREATE TABLE daily_prices_2024_08 PARTITION OF daily_prices
    FOR VALUES FROM ('2024-08-01') TO ('2024-09-01');
CREATE TABLE daily_prices_2024_09 PARTITION OF daily_prices
    FOR VALUES FROM ('2024-09-01') TO ('2024-10-01');
CREATE TABLE daily_prices_2024_10 PARTITION OF daily_prices
    FOR VALUES FROM ('2024-10-01') TO ('2024-11-01');
CREATE TABLE daily_prices_2024_11 PARTITION OF daily_prices
    FOR VALUES FROM ('2024-11-01') TO ('2024-12-01');
CREATE TABLE daily_prices_2024_12 PARTITION OF daily_prices
    FOR VALUES FROM ('2024-12-01') TO ('2025-01-01');
CREATE TABLE daily_prices_2025_01 PARTITION OF daily_prices
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE daily_prices_2025_02 PARTITION OF daily_prices
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
CREATE TABLE daily_prices_2025_03 PARTITION OF daily_prices
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');
CREATE TABLE daily_prices_2025_04 PARTITION OF daily_prices
    FOR VALUES FROM ('2025-04-01') TO ('2025-05-01');
CREATE TABLE daily_prices_2025_05 PARTITION OF daily_prices
    FOR VALUES FROM ('2025-05-01') TO ('2025-06-01');
CREATE TABLE daily_prices_2025_06 PARTITION OF daily_prices
    FOR VALUES FROM ('2025-06-01') TO ('2025-07-01');
CREATE TABLE daily_prices_2025_07 PARTITION OF daily_prices
    FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');
CREATE TABLE daily_prices_2025_08 PARTITION OF daily_prices
    FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE daily_prices_2025_09 PARTITION OF daily_prices
    FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');

-- Indexes optimized for time series analysis
CREATE INDEX idx_daily_prices_symbol_date ON daily_prices(symbol, trade_date DESC);
CREATE INDEX idx_daily_prices_date_symbol ON daily_prices(trade_date DESC, symbol);
CREATE INDEX idx_daily_prices_stock_id_date ON daily_prices(stock_id, trade_date DESC);
CREATE INDEX idx_daily_prices_volume ON daily_prices(volume DESC);
CREATE INDEX idx_daily_prices_close_price ON daily_prices(close_price);

-- ============================================================================
-- 3. DATA QUALITY TRACKING
-- ============================================================================
CREATE TABLE data_ingestion_log (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    ingestion_date DATE NOT NULL,
    records_processed INTEGER NOT NULL DEFAULT 0,
    records_inserted INTEGER NOT NULL DEFAULT 0,
    records_updated INTEGER NOT NULL DEFAULT 0,
    records_failed INTEGER NOT NULL DEFAULT 0,
    status VARCHAR(20) NOT NULL DEFAULT 'pending', -- pending, success, failed, partial
    error_message TEXT,
    processing_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_ingestion_log_symbol_date ON data_ingestion_log(symbol, ingestion_date DESC);
CREATE INDEX idx_ingestion_log_status ON data_ingestion_log(status);

-- ============================================================================
-- 4. FEATURE ENGINEERING CACHE (FOR ML)
-- ============================================================================
CREATE TABLE daily_features (
    id BIGSERIAL PRIMARY KEY,
    stock_id INTEGER NOT NULL REFERENCES stocks(id),
    symbol VARCHAR(10) NOT NULL,
    trade_date DATE NOT NULL,
    
    -- Price-based features
    price_change DECIMAL(12,4), -- close - previous_close
    price_change_pct DECIMAL(8,4), -- (close - previous_close) / previous_close
    daily_return DECIMAL(8,4), -- log(close / previous_close)
    
    -- Volume features
    volume_ratio DECIMAL(8,4), -- volume / avg_volume_20d
    volume_change_pct DECIMAL(8,4), -- (volume - previous_volume) / previous_volume
    
    -- Volatility features
    daily_range DECIMAL(8,4), -- (high - low) / close
    gap_up DECIMAL(8,4), -- (open - previous_close) / previous_close
    
    -- Moving averages (for context)
    sma_5 DECIMAL(12,4), -- 5-day simple moving average
    sma_20 DECIMAL(12,4), -- 20-day simple moving average
    
    -- Technical indicators
    rsi_14 DECIMAL(5,2), -- 14-day RSI
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT daily_features_unique UNIQUE (stock_id, trade_date)
);

CREATE INDEX idx_daily_features_symbol_date ON daily_features(symbol, trade_date DESC);
CREATE INDEX idx_daily_features_price_change ON daily_features(ABS(price_change_pct) DESC);
CREATE INDEX idx_daily_features_volume_ratio ON daily_features(volume_ratio DESC);

-- ============================================================================
-- 5. ANOMALY DETECTION RESULTS
-- ============================================================================
CREATE TABLE anomaly_detections (
    id BIGSERIAL PRIMARY KEY,
    stock_id INTEGER NOT NULL REFERENCES stocks(id),
    symbol VARCHAR(10) NOT NULL,
    trade_date DATE NOT NULL,
    
    -- Detection metadata
    detection_run_id UUID NOT NULL, -- To group batch detections
    algorithm VARCHAR(50) NOT NULL DEFAULT 'isolation_forest',
    model_version VARCHAR(20) NOT NULL DEFAULT 'v1.0',
    
    -- Scores and classification
    anomaly_score DECIMAL(8,4) NOT NULL, -- -1 to 1, higher = more anomalous
    is_anomaly BOOLEAN NOT NULL DEFAULT FALSE,
    confidence_score DECIMAL(5,4), -- 0 to 1
    
    -- Feature contributions (for explainability)
    primary_feature VARCHAR(50), -- Which feature contributed most
    feature_values JSONB, -- Store all feature values used
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT anomaly_detections_unique UNIQUE (stock_id, trade_date, algorithm, model_version)
);

CREATE INDEX idx_anomaly_detections_symbol_date ON anomaly_detections(symbol, trade_date DESC);
CREATE INDEX idx_anomaly_detections_is_anomaly ON anomaly_detections(is_anomaly, trade_date DESC);
CREATE INDEX idx_anomaly_detections_score ON anomaly_detections(anomaly_score DESC);
CREATE INDEX idx_anomaly_detections_run_id ON anomaly_detections(detection_run_id);

-- ============================================================================
-- 6. INITIAL SEED DATA - TOP 100 STOCKS
-- ============================================================================
INSERT INTO stocks (symbol, company_name, market_cap, sector) VALUES
('MSFT', 'Microsoft', 3510000000000, 'Technology'),
('AAPL', 'Apple', 3350000000000, 'Technology'),
('NVDA', 'NVIDIA', 2980000000000, 'Technology'),
('GOOGL', 'Alphabet (Google)', 2250000000000, 'Technology'),
('AMZN', 'Amazon', 1950000000000, 'Consumer Discretionary'),
('META', 'Meta Platforms', 1290000000000, 'Technology'),
('BRK-B', 'Berkshire Hathaway', 915800000000, 'Financial Services'),
('LLY', 'Eli Lilly', 850100000000, 'Healthcare'),
('AVGO', 'Broadcom', 790500000000, 'Technology'),
('JPM', 'JPMorgan Chase', 601200000000, 'Financial Services'),
('TSLA', 'Tesla', 595700000000, 'Consumer Discretionary'),
('V', 'Visa', 575300000000, 'Financial Services'),
('XOM', 'Exxon Mobil', 520400000000, 'Energy'),
('JNJ', 'Johnson & Johnson', 515100000000, 'Healthcare'),
('UNH', 'UnitedHealth Group', 505600000000, 'Healthcare'),
('MA', 'Mastercard', 450700000000, 'Financial Services'),
('PG', 'Procter & Gamble', 401900000000, 'Consumer Staples'),
('COST', 'Costco', 380100000000, 'Consumer Staples'),
('HD', 'Home Depot', 355800000000, 'Consumer Discretionary'),
('ORCL', 'Oracle', 350200000000, 'Technology'),
('MRK', 'Merck & Co.', 335400000000, 'Healthcare'),
('CVX', 'Chevron', 301600000000, 'Energy'),
('ABBV', 'AbbVie', 298500000000, 'Healthcare'),
('CRM', 'Salesforce', 295300000000, 'Technology'),
('NFLX', 'Netflix', 290100000000, 'Communication Services'),
('KO', 'Coca-Cola', 275900000000, 'Consumer Staples'),
('AMD', 'AMD', 270800000000, 'Technology'),
('BAC', 'Bank of America', 268400000000, 'Financial Services'),
('ADBE', 'Adobe', 265700000000, 'Technology'),
('PEP', 'PepsiCo', 230100000000, 'Consumer Staples'),
('WMT', 'Walmart', 228600000000, 'Consumer Staples'),
('LIN', 'Linde', 220500000000, 'Materials'),
('MCD', 'McDonald''s', 215900000000, 'Consumer Discretionary'),
('TMO', 'Thermo Fisher', 214300000000, 'Healthcare'),
('CSCO', 'Cisco Systems', 210700000000, 'Technology'),
('CAT', 'Caterpillar', 205100000000, 'Industrials'),
('DHR', 'Danaher', 201800000000, 'Healthcare'),
('ABT', 'Abbott Labs', 198600000000, 'Healthcare'),
('INTU', 'Intuit', 195400000000, 'Technology'),
('GE', 'GE', 190200000000, 'Industrials'),
('WFC', 'Wells Fargo', 188700000000, 'Financial Services'),
('NOW', 'ServiceNow', 185300000000, 'Technology'),
('ISRG', 'Intuitive Surgical', 180100000000, 'Healthcare'),
('VZ', 'Verizon', 178900000000, 'Communication Services'),
('QCOM', 'QUALCOMM', 175500000000, 'Technology'),
('AMGN', 'Amgen', 170600000000, 'Healthcare'),
('AMAT', 'Applied Materials', 168400000000, 'Technology'),
('PM', 'Philip Morris', 165700000000, 'Consumer Staples'),
('TXN', 'Texas Instruments', 163200000000, 'Technology'),
('PFE', 'Pfizer', 160100000000, 'Healthcare');

-- Continue with remaining 50 stocks...
INSERT INTO stocks (symbol, company_name, market_cap, sector) VALUES
('TMUS', 'T-Mobile US', 158900000000, 'Communication Services'),
('HON', 'Honeywell', 155800000000, 'Industrials'),
('IBM', 'IBM', 154300000000, 'Technology'),
('UBER', 'Uber', 152700000000, 'Consumer Discretionary'),
('GS', 'Goldman Sachs', 150100000000, 'Financial Services'),
('CMCSA', 'Comcast', 148900000000, 'Communication Services'),
('UNP', 'Union Pacific', 147500000000, 'Industrials'),
('PLD', 'Prologis', 145300000000, 'Real Estate'),
('LRCX', 'Lam Research', 143200000000, 'Technology'),
('C', 'Citigroup', 140800000000, 'Financial Services'),
('MU', 'Micron Technology', 138600000000, 'Technology'),
('T', 'AT&T', 135400000000, 'Communication Services'),
('DE', 'Deere & Company', 133200000000, 'Industrials'),
('SPGI', 'S&P Global', 131100000000, 'Financial Services'),
('PGR', 'Progressive', 129800000000, 'Financial Services'),
('BSX', 'Boston Scientific', 128500000000, 'Healthcare'),
('SYK', 'Stryker', 127300000000, 'Healthcare'),
('AXP', 'American Express', 126100000000, 'Financial Services'),
('REGN', 'Regeneron', 125000000000, 'Healthcare'),
('BLK', 'BlackRock', 124800000000, 'Financial Services'),
('COP', 'ConocoPhillips', 123600000000, 'Energy'),
('CI', 'Cigna', 122500000000, 'Healthcare'),
('ADI', 'Analog Devices', 121300000000, 'Technology'),
('GILD', 'Gilead Sciences', 120100000000, 'Healthcare'),
('BKNG', 'Booking Holdings', 119000000000, 'Consumer Discretionary'),
('CB', 'Chubb', 118800000000, 'Financial Services'),
('MMC', 'Marsh & McLennan', 117600000000, 'Financial Services'),
('LMT', 'Lockheed Martin', 116500000000, 'Industrials'),
('VRTX', 'Vertex', 115300000000, 'Healthcare'),
('KLAC', 'KLA Corporation', 114200000000, 'Technology'),
('SO', 'Southern Company', 113100000000, 'Utilities'),
('DUK', 'Duke Energy', 112000000000, 'Utilities'),
('MDLZ', 'Mondelez', 110900000000, 'Consumer Staples'),
('EQIX', 'Equinix', 109800000000, 'Real Estate'),
('SCHW', 'Charles Schwab', 108700000000, 'Financial Services'),
('TGT', 'Target', 107600000000, 'Consumer Discretionary'),
('NEE', 'NextEra Energy', 106500000000, 'Utilities'),
('ETN', 'Eaton', 105400000000, 'Industrials'),
('TJX', 'TJX Companies', 104300000000, 'Consumer Discretionary'),
('ADP', 'Automatic Data', 103200000000, 'Technology'),
('EMR', 'Emerson Electric', 102100000000, 'Industrials'),
('SLB', 'Schlumberger', 101000000000, 'Energy'),
('SBUX', 'Starbucks', 99900000000, 'Consumer Discretionary'),
('AMT', 'American Tower', 98800000000, 'Real Estate'),
('EL', 'Estée Lauder', 97700000000, 'Consumer Discretionary'),
('GM', 'General Motors', 96600000000, 'Consumer Discretionary'),
('CCI', 'Crown Castle', 95500000000, 'Real Estate'),
('USB', 'US Bancorp', 94400000000, 'Financial Services'),
('F', 'Ford', 93300000000, 'Consumer Discretionary'),
('ADM', 'Archer Daniels', 92200000000, 'Consumer Staples');

-- ============================================================================
-- 7. UTILITY FUNCTIONS AND VIEWS
-- ============================================================================

-- View for latest prices with percentage changes
CREATE VIEW latest_prices_with_changes AS
SELECT 
    dp1.symbol,
    dp1.trade_date,
    dp1.close_price,
    dp1.volume,
    COALESCE(
        ROUND(((dp1.close_price - dp2.close_price) / dp2.close_price * 100)::NUMERIC, 2), 
        0
    ) AS price_change_pct,
    COALESCE(
        ROUND(((dp1.volume - dp2.volume) / dp2.volume::DECIMAL * 100)::NUMERIC, 2),
        0
    ) AS volume_change_pct
FROM daily_prices dp1
LEFT JOIN daily_prices dp2 ON dp1.stock_id = dp2.stock_id 
    AND dp2.trade_date = (
        SELECT MAX(trade_date) 
        FROM daily_prices dp3 
        WHERE dp3.stock_id = dp1.stock_id 
        AND dp3.trade_date < dp1.trade_date
    )
WHERE dp1.trade_date = (
    SELECT MAX(trade_date) FROM daily_prices WHERE stock_id = dp1.stock_id
);

-- View for data completeness monitoring
CREATE VIEW data_completeness_report AS
SELECT 
    s.symbol,
    s.company_name,
    COUNT(dp.id) as days_of_data,
    MAX(dp.trade_date) as latest_date,
    MIN(dp.trade_date) as earliest_date,
    CASE 
        WHEN COUNT(dp.id) >= 250 THEN 'Complete'
        WHEN COUNT(dp.id) >= 200 THEN 'Mostly Complete'
        WHEN COUNT(dp.id) >= 100 THEN 'Partial'
        ELSE 'Insufficient'
    END as data_status
FROM stocks s
LEFT JOIN daily_prices dp ON s.id = dp.stock_id
GROUP BY s.id, s.symbol, s.company_name
ORDER BY s.market_cap DESC;

-- Function to get missing trading days for a symbol
CREATE OR REPLACE FUNCTION get_missing_trading_days(
    p_symbol VARCHAR(10),
    p_start_date DATE,
    p_end_date DATE
) RETURNS TABLE(missing_date DATE) AS $$
BEGIN
    RETURN QUERY
    WITH expected_dates AS (
        SELECT generate_series(p_start_date, p_end_date, '1 day'::interval)::DATE as trade_date
    ),
    weekdays AS (
        SELECT trade_date 
        FROM expected_dates 
        WHERE EXTRACT(DOW FROM trade_date) BETWEEN 1 AND 5
    ),
    existing_dates AS (
        SELECT DISTINCT dp.trade_date
        FROM daily_prices dp
        JOIN stocks s ON dp.stock_id = s.id
        WHERE s.symbol = p_symbol
    )
    SELECT w.trade_date
    FROM weekdays w
    LEFT JOIN existing_dates e ON w.trade_date = e.trade_date
    WHERE e.trade_date IS NULL
    ORDER BY w.trade_date;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- 8. BULK LOAD / PARTITION MAINTENANCE HELPERS
-- ============================================================================
-- Example: load daily_prices from CSV exported by the collector.
-- CSV should include: stock_id,symbol,trade_date,open_price,high_price,low_price,close_price,volume,adjusted_close
-- Ensure partitions exist for the trade_date values before running COPY to avoid errors.

-- Example COPY command (run from psql or a client with access to the CSV file):
-- \copy daily_prices(stock_id,symbol,trade_date,open_price,high_price,low_price,close_price,volume,adjusted_close) \
--     FROM '/path/to/daily_prices_2025_08.csv' WITH (FORMAT csv, HEADER true);

-- Helper function to create a monthly partition for `daily_prices`.
CREATE OR REPLACE FUNCTION create_monthly_partition(p_year INT, p_month INT)
RETURNS VOID AS $$
DECLARE
    partition_name TEXT := format('daily_prices_%s_%02s', p_year, p_month);
    start_date DATE := make_date(p_year, p_month, 1);
    end_date DATE := (start_date + INTERVAL '1 month')::DATE;
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_class c JOIN pg_namespace n ON c.relnamespace = n.oid
        WHERE c.relname = partition_name
    ) THEN
        EXECUTE format('CREATE TABLE %I PARTITION OF daily_prices FOR VALUES FROM (%L) TO (%L);',
            partition_name, start_date::TEXT, end_date::TEXT);
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Maintenance notes:
-- 1) Regularly run VACUUM ANALYZE on new partitions to keep planner stats up to date.
-- 2) Consider using pg_partman for automatic partition creation and retention policies.
-- 3) Use COPY for high-throughput ingestion, wrapped in transactions per-file to enable retries.
-- 4) For very large ingests, temporarily disable indexes on the target partition, load data, then reindex.

-- End of schema file.

-- ==========================================================================
-- 9. STAGING + UPSERT WORKFLOW (for efficient COPY + DB-side resolution)
-- ==========================================================================

-- Staging table that matches CSV provided by collector (no stock_id required)
CREATE TABLE IF NOT EXISTS daily_prices_staging (
    staging_id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    trade_date DATE NOT NULL,
    open_price NUMERIC(12,4),
    high_price NUMERIC(12,4),
    low_price NUMERIC(12,4),
    close_price NUMERIC(12,4),
    volume BIGINT,
    adjusted_close NUMERIC(12,4),
    ingestion_job_id UUID, -- optional grouping id
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_staging_symbol_date ON daily_prices_staging(symbol, trade_date);

-- Table to capture rows that could not be resolved to a stock_id (symbol not found)
CREATE TABLE IF NOT EXISTS daily_prices_staging_errors (
    id BIGSERIAL PRIMARY KEY,
    staging_id BIGINT,
    symbol VARCHAR(20),
    trade_date DATE,
    error_reason TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Upsert: move data from staging into daily_prices, resolving stock_id from stocks(symbol)
-- This SQL should be executed inside a transaction.
-- Steps (example):
-- 1) COPY daily_prices_staging(symbol,trade_date,open_price,high_price,low_price,close_price,volume,adjusted_close,ingestion_job_id) FROM '/path/to/file.csv' WITH (FORMAT csv, HEADER true);
-- 2) SELECT perform_daily_prices_upsert(); -- the function below

CREATE OR REPLACE FUNCTION perform_daily_prices_upsert() RETURNS TABLE(inserted INTEGER, updated INTEGER, failed INTEGER) AS $$
DECLARE
    v_inserted INTEGER := 0;
    v_updated INTEGER := 0;
    v_failed INTEGER := 0;
BEGIN
    -- 1) Insert/Update rows where symbol maps to a known stock_id
    WITH to_upsert AS (
        SELECT s.id as stock_id, st.symbol, st.trade_date, st.open_price, st.high_price, st.low_price, st.close_price, st.volume, st.adjusted_close
        FROM daily_prices_staging st
        JOIN stocks s ON s.symbol = st.symbol
    ), ins AS (
        INSERT INTO daily_prices (stock_id, symbol, trade_date, open_price, high_price, low_price, close_price, volume, adjusted_close, created_at)
        SELECT stock_id, symbol, trade_date, open_price, high_price, low_price, close_price, volume, adjusted_close, now()
        FROM to_upsert
        ON CONFLICT ON CONSTRAINT daily_prices_unique
        DO UPDATE SET
            open_price = EXCLUDED.open_price,
            high_price = EXCLUDED.high_price,
            low_price = EXCLUDED.low_price,
            close_price = EXCLUDED.close_price,
            volume = EXCLUDED.volume,
            adjusted_close = EXCLUDED.adjusted_close,
            created_at = now()
        RETURNING (xmax = 0) AS inserted_flag
    )
    SELECT COUNT(*) FILTER (WHERE inserted_flag) INTO v_inserted FROM ins;

    SELECT COUNT(*) FILTER (WHERE NOT inserted_flag) INTO v_updated FROM ins;

    -- 2) Capture staging rows that did not match any symbol
    INSERT INTO daily_prices_staging_errors (staging_id, symbol, trade_date, error_reason)
    SELECT st.staging_id, st.symbol, st.trade_date, 'symbol not found in stocks table'
    FROM daily_prices_staging st
    LEFT JOIN stocks s ON s.symbol = st.symbol
    WHERE s.id IS NULL;

    GET DIAGNOSTICS v_failed = ROW_COUNT;

    -- 3) Cleanup: delete processed rows from staging
    DELETE FROM daily_prices_staging st
    USING stocks s
    WHERE s.symbol = st.symbol;

    RETURN QUERY SELECT v_inserted, v_updated, v_failed;
END;
$$ LANGUAGE plpgsql;

-- Example usage notes (to include in runbook):
-- BEGIN;
-- COPY daily_prices_staging(symbol,trade_date,open_price,high_price,low_price,close_price,volume,adjusted_close,ingestion_job_id) FROM '/path/to/daily_prices_50.csv' WITH (FORMAT csv, HEADER true);
-- SELECT * FROM perform_daily_prices_upsert();
-- COMMIT;

-- After large loads run: VACUUM ANALYZE daily_prices;


