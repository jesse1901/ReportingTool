#!/bin/bash
set -e

# Pfade
BASE_DIR="/var/www/max-reports/ReportingTool/database"
POINTER_FILE="$BASE_DIR/current_db.txt"

# --- 0. SICHERHEITS-CHECK (Neu) ---
# Wenn keine Pointer-Datei da ist -> Sofortiger Abbruch!
if [ ! -f "$POINTER_FILE" ]; then
    echo "No Pointerfile: break to prevent dataloss"
    exit 1
fi

cleanup() {
    if [ -f "$POINTER_FILE" ]; then
        KEEP_DB=$(cat "$POINTER_FILE")
    else
        KEEP_DB=""
    fi
    
    if [ -n "$KEEP_DB" ]; then
        find "$BASE_DIR" -name "max-reports-*.duckdb" -type f ! -name "$KEEP_DB" -delete
        echo "Deleting databses, except $KEEP_DB"

    fi
}

trap cleanup EXIT


# 1. Neuen Namen generieren
TIMESTAMP=$(date +%s)
NEW_DB_NAME="max-reports-$TIMESTAMP.duckdb"
NEW_DB_PATH="$BASE_DIR/$NEW_DB_NAME"

# 2. Alte DB als Basis kopieren

LAST_DB_NAME=$(cat "$POINTER_FILE")
LAST_DB_PATH="$BASE_DIR/$LAST_DB_NAME"

if [ -f "$LAST_DB_PATH" ]; then
    cp "$LAST_DB_PATH" "$NEW_DB_PATH"
else
    echo "⚠️ Database in pointer file ($LAST_DB_PATH) does not exist"
    # Das Script läuft weiter und slurm2sql erstellt eine neue leere DB
fi

# Update Databse
/usr/local/bin/slurm2sql --history-resume --duckdb "$NEW_DB_PATH" -- -a
python3 /var/www/max-reports/ReportingTool/src/get_gpu.py "$NEW_DB_PATH"

# 4. Den "Zeiger" aktualisieren
echo "$NEW_DB_NAME" > "$POINTER_FILE"

# 5. Warten auf Streamlit (Watchdog Zeit geben)
sleep 5

