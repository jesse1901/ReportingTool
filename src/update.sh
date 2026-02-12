#!/bin/bash
set -e

# Base directory where the script and dbs are located
BASE_DIR="/var/www/max-reports/ReportingTool"
DB_DIR="$BASE_DIR/database"
POINTER_FILE="$DB_DIR/current_db.txt"

# Database files
DB_A="$DB_DIR/max-reports-a.duckdb"
DB_B="$DB_DIR/max-reports_b.duckdb"

# Ensure the db directory exists
mkdir -p "$DB_DIR"

# Determine which database is current
if [ ! -f "$POINTER_FILE" ]; then
    echo "a" > "$POINTER_FILE"
fi
CURRENT=$(cat "$POINTER_FILE")

# Set target, next value, and source based on the current pointer
if [ "$CURRENT" == "a" ]; then
    TARGET_DB="$DB_B"
    NEXT_VAL="b"
    SOURCE_DB="$DB_A"
else
    TARGET_DB="$DB_A"
    NEXT_VAL="a"
    SOURCE_DB="$DB_B"
fi

# Create a temporary file for the new database to be written to.
# This avoids any process reading from a partially updated file.
TEMP_TARGET="${TARGET_DB}.tmp"

# If the source DB exists, copy it to the temporary target to start from there.
if [ -f "$SOURCE_DB" ]; then
    cp "$SOURCE_DB" "$TEMP_TARGET"
fi

# Run slurm2sql to update the history in the temporary database file.
# The double dash '--' separates slurm2sql options from sacct options.
/usr/local/bin/slurm2sql --history-resume --duckdb "$TEMP_TARGET" -- -a

# Run the python script to get GPU data and add it to the temporary database.
python3 "$BASE_DIR/src/get_gpu.py" "$TEMP_TARGET"

# ATOMIC SWAP:
# Move the completed temporary file to the final target destination.
# 'mv' is an atomic operation on most filesystems (like ext4, XFS).
# This means that any process trying to open the file will either get the old version or the new one,
# but never a corrupted or partial file. Processes that already have the old file open can continue reading from it
# without issue until they close it.
mv "$TEMP_TARGET" "$TARGET_DB"

# Update the pointer to indicate that the new database is now the current one.
echo "$NEXT_VAL" > "$POINTER_FILE"