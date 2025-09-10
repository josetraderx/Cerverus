-- Cerverus PostgreSQL Initialization Script
-- This script sets up the initial database structure for Cerverus

-- Create databases
CREATE DATABASE IF NOT EXISTS cerverus;
CREATE DATABASE IF NOT EXISTS airflow;

-- Connect to cerverus database
\c cerverus;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS fraud_detection;
CREATE SCHEMA IF NOT EXISTS market_data;
CREATE SCHEMA IF NOT EXISTS audit;

-- Create tables for fraud detection
CREATE TABLE IF NOT EXISTS fraud_detection.transactions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(255) UNIQUE NOT NULL,
    amount DECIMAL(15,2) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    user_id VARCHAR(255),
    merchant_id VARCHAR(255),
    location VARCHAR(255),
    device_fingerprint VARCHAR(255),
    anomaly_score DECIMAL(5,4),
    is_fraud BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fraud_detection.anomalies (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(255) REFERENCES fraud_detection.transactions(transaction_id),
    algorithm VARCHAR(100) NOT NULL,
    score DECIMAL(5,4) NOT NULL,
    threshold DECIMAL(5,4) NOT NULL,
    features JSONB,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create tables for market data
CREATE TABLE IF NOT EXISTS market_data.securities (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) UNIQUE NOT NULL,
    name VARCHAR(255),
    sector VARCHAR(100),
    industry VARCHAR(100),
    exchange VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS market_data.price_history (
    id SERIAL PRIMARY KEY,
    security_id INTEGER REFERENCES market_data.securities(id),
    date DATE NOT NULL,
    open_price DECIMAL(10,2),
    high_price DECIMAL(10,2),
    low_price DECIMAL(10,2),
    close_price DECIMAL(10,2),
    volume BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(security_id, date)
);

-- Create audit table
CREATE TABLE IF NOT EXISTS audit.api_calls (
    id SERIAL PRIMARY KEY,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    user_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    request_body JSONB,
    response_status INTEGER,
    response_time_ms INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON fraud_detection.transactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON fraud_detection.transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_transactions_anomaly_score ON fraud_detection.transactions(anomaly_score);
CREATE INDEX IF NOT EXISTS idx_anomalies_transaction_id ON fraud_detection.anomalies(transaction_id);
CREATE INDEX IF NOT EXISTS idx_price_history_security_date ON market_data.price_history(security_id, date);
CREATE INDEX IF NOT EXISTS idx_api_calls_timestamp ON audit.api_calls(timestamp);

-- Create views for common queries
CREATE OR REPLACE VIEW fraud_detection.suspicious_transactions AS
SELECT
    t.*,
    a.algorithm,
    a.score,
    a.threshold
FROM fraud_detection.transactions t
LEFT JOIN fraud_detection.anomalies a ON t.transaction_id = a.transaction_id
WHERE t.anomaly_score > 0.8 OR t.is_fraud = TRUE;

-- Insert sample data for testing
INSERT INTO market_data.securities (symbol, name, sector, exchange) VALUES
('AAPL', 'Apple Inc.', 'Technology', 'NASDAQ'),
('GOOGL', 'Alphabet Inc.', 'Technology', 'NASDAQ'),
('MSFT', 'Microsoft Corporation', 'Technology', 'NASDAQ'),
('AMZN', 'Amazon.com Inc.', 'Consumer Discretionary', 'NASDAQ'),
('TSLA', 'Tesla Inc.', 'Consumer Discretionary', 'NASDAQ')
ON CONFLICT (symbol) DO NOTHING;

-- Create user and grant permissions
CREATE USER IF NOT EXISTS cerverus_user WITH PASSWORD 'cerverus_pass';
GRANT ALL PRIVILEGES ON DATABASE cerverus TO cerverus_user;
GRANT ALL PRIVILEGES ON SCHEMA fraud_detection TO cerverus_user;
GRANT ALL PRIVILEGES ON SCHEMA market_data TO cerverus_user;
GRANT ALL PRIVILEGES ON SCHEMA audit TO cerverus_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA fraud_detection TO cerverus_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA market_data TO cerverus_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA audit TO cerverus_user;

-- Grant permissions on future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA fraud_detection GRANT ALL ON TABLES TO cerverus_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA market_data GRANT ALL ON TABLES TO cerverus_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA audit GRANT ALL ON TABLES TO cerverus_user;