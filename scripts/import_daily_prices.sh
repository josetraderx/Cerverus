#!/usr/bin/env bash
# Script to import collector CSV into PostgreSQL staging and run upsert.
# This is a single focused sequence (no choices):
# 1) create monthly partition (if needed)
# 2) COPY from local CSV into staging table
# 3) run upsert function inside a transaction

# ---------------------------
# How to use (pasos simples):
# 1) Edit the VARIABLES section below and set values.
# 2) Run this script from bash: ./scripts/import_daily_prices.sh
# ---------------------------

# ------- VARIABLES (Rellena aquí) -------
# PGHOST: database host (servidor), usa 'localhost' si PostgreSQL está en tu PC
# PGUSER: usuario de la base de datos (nombre que usas para entrar)
# PGDATABASE: nombre de la base de datos donde están las tablas
# PGPASSWORD: (opcional) contraseña; en sistemas seguros usa .pgpass o variable de entorno

PGHOST="localhost"        # <-- cambia si tu DB está en otra máquina
PGUSER="postgres"         # <-- tu usuario DB
PGDATABASE="cerverus"     # <-- nombre de la base de datos
CSV_PATH="C:/Users/DataBridge/Documents/workspace/Cerverus/data/ingestion/daily_prices_50.csv"

# Mes/año de ejemplo para crear partición. Ajusta si tu CSV tiene otras fechas.
PART_YEAR=2025
PART_MONTH=8

# ----------------------------------------

set -euo pipefail

export PGHOST PGUSER PGDATABASE

echo "[1/3] Creating monthly partition if needed (year=${PART_YEAR}, month=${PART_MONTH})"
psql "postgresql://${PGUSER}@${PGHOST}/${PGDATABASE}" -v ON_ERROR_STOP=1 <<'PSQL'
-- create partition helper (no-op if exists)
SELECT create_monthly_partition(:PART_YEAR::int, :PART_MONTH::int);
PSQL

echo "[2/3] Copying CSV into daily_prices_staging (from: ${CSV_PATH})"
# Use psql's \copy so the file is read from this client machine (no server-side file needed)
psql "postgresql://${PGUSER}@${PGHOST}/${PGDATABASE}" -v ON_ERROR_STOP=1 <<PSQL
\copy daily_prices_staging(symbol,trade_date,open_price,high_price,low_price,close_price,volume,adjusted_close,ingestion_job_id)
FROM '${CSV_PATH}' WITH (FORMAT csv, HEADER true);
PSQL

echo "[3/3] Running upsert (perform_daily_prices_upsert) inside a transaction"
psql "postgresql://${PGUSER}@${PGHOST}/${PGDATABASE}" -v ON_ERROR_STOP=1 <<'PSQL'
BEGIN;
SELECT * FROM perform_daily_prices_upsert();
COMMIT;
PSQL

echo "Import finished. Check daily_prices_staging_errors for any unresolved rows."

# Definitions rápidas (si no las conoces):
# - partition: sub-tabla por rango de fechas para mejorar rendimiento.
# - \copy: comando de psql que copia datos desde un archivo en tu máquina al servidor DB.
# - staging: tabla temporal/entrada donde subes datos sin afectar la tabla final.
# - upsert: operación que inserta si no existe o actualiza si ya existe.
