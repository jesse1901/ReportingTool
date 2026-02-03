#!/bin/bash
set -e

# Pfade
BASE_DIR="/var/www/max-reports/ReportingTool/database"
POINTER_FILE="$BASE_DIR/current_db.txt"
DB_A="$BASE_DIR/max-reports_a.duckdb"
DB_B="$BASE_DIR/max-reports_b.duckdb"

# 1. Bestimmen, wer gerade aktiv ist
if [ ! -f "$POINTER_FILE" ]; then
    echo "a" > "$POINTER_FILE"
    CURRENT="a"
else
    CURRENT=$(cat "$POINTER_FILE")
fi

if [ "$CURRENT" == "a" ]; then
    TARGET_DB="$DB_B"
    NEXT_VAL="b"
    SOURCE_DB="$DB_A" # Wir kopieren A nach B als Basis für das Update
else
    TARGET_DB="$DB_A"
    NEXT_VAL="a"
    SOURCE_DB="$DB_B"
fi

# 3. Vorbereitung: Kopiere die aktuelle DB in die Ziel-DB als Basis
# (Nur nötig, wenn slurm2sql auf alten Daten aufbaut)
if [ -f "$SOURCE_DB" ]; then
    cp "$SOURCE_DB" "$TARGET_DB"
fi

/usr/local/bin/slurm2sql --history-resume --duckdb "$TARGET_DB" -- -a

python3 /var/www/max-reports/ReportingTool/src/get_gpu.py "$TARGET_DB"

echo "$NEXT_VAL" > "$POINTER_FILE"



