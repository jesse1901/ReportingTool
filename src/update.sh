#!/bin/bash
set -e

# Pfade
DB_DIR="/var/www/max-reports/ReportingTool/database"
PROD_DB="$DB_DIR/max-reports.duckdb"
STAGING_DB="$DB_DIR/max-reports-staging.duckdb"

# 1. Kopie erstellen
# Der Index (aus Schritt 1) wird mitkopiert!
if [ -f "$PROD_DB" ]; then
    cp "$PROD_DB" "$STAGING_DB"
fi

# 2. Updates laufen lassen
# slurm2sql sieht jetzt den Index und macht automatisch Updates statt Duplikate.
/usr/local/bin/slurm2sql --history-resume --duckdb "$STAGING_DB" -- -a

# GPU Script (muss Pfad als Argument annehmen!)
python3 /var/www/max-reports/ReportingTool/src/get_gpu.py

# 3. Swap
mv "$STAGING_DB" "$PROD_DB"
