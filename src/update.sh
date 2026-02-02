#!/bin/bash
set -e

# Pfade
BASE_DIR="/var/www/max-reports/ReportingTool/database"
POINTER_FILE="$BASE_DIR/current_db.txt"

# 1. Neuen Namen generieren (z.B. max-reports-1706881234.duckdb)
TIMESTAMP=$(date +%s)
NEW_DB_NAME="max-reports-$TIMESTAMP.duckdb"
NEW_DB_PATH="$BASE_DIR/$NEW_DB_NAME"

# 2. Alte DB als Basis kopieren (falls vorhanden)
# Wir schauen in die Pointer-Datei, um die letzte DB zu finden
if [ -f "$POINTER_FILE" ]; then
    LAST_DB_NAME=$(cat "$POINTER_FILE")
    LAST_DB_PATH="$BASE_DIR/$LAST_DB_NAME"
    
    if [ -f "$LAST_DB_PATH" ]; then
        cp "$LAST_DB_PATH" "$NEW_DB_PATH"
    fi
fi

# Fallback: Falls gar keine DB existiert, wird slurm2sql eine leere erstellen.

# 3. Update auf der NEUEN Datei ausführen
/usr/local/bin/slurm2sql --history-resume --duckdb "$NEW_DB_PATH" -- -a
python3 /var/www/max-reports/ReportingTool/src/get_gpu.py "$NEW_DB_PATH"

# 4. Den "Zeiger" aktualisieren
# Wir schreiben den neuen Dateinamen in die Textdatei.
echo "$NEW_DB_NAME" > "$POINTER_FILE"

# 5. Aufräumen (Garbage Collection)
# Wir warten kurz, damit Streamlit den Wechsel mitbekommt (ca. 10 Sek),
# dann löschen wir alle alten .duckdb Dateien, außer der gerade erstellten.
sleep 15
find "$BASE_DIR" -name "max-reports-*.duckdb" -type f ! -name "$NEW_DB_NAME" -delete