-- Crear tabla temporal para importar CSV
CREATE TEMP TABLE temp_daily_prices (
    symbol VARCHAR(10),
    trade_date DATE,
    open_price DECIMAL(12,4),
    high_price DECIMAL(12,4),
    low_price DECIMAL(12,4),
    close_price DECIMAL(12,4),
    volume BIGINT,
    adjusted_close VARCHAR(50)
);

-- Importar datos del CSV
\copy temp_daily_prices FROM '/tmp/daily_prices.csv' WITH (FORMAT csv, HEADER true);

-- Insertar en la tabla final con stock_id correcto
INSERT INTO daily_prices (stock_id, symbol, trade_date, open_price, high_price, low_price, close_price, volume)
SELECT 
    s.id as stock_id,
    t.symbol,
    t.trade_date,
    t.open_price,
    t.high_price,
    t.low_price,
    t.close_price,
    t.volume
FROM temp_daily_prices t
JOIN stocks s ON t.symbol = s.symbol
WHERE t.open_price > 0 AND t.high_price > 0 AND t.low_price > 0 AND t.close_price > 0;