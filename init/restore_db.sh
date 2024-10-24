#!/bin/bash
set -e

# Controlla se il database ha tabelle e se sono vuote
table_count=$(psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c \
"SELECT SUM(n_live_tup) FROM pg_stat_user_tables;")

if [[ -z "$table_count" || "$table_count" -eq 0 ]]; then
  echo "Database is empty or contains only empty tables, restoring from backup..."
  pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" /var/lib/postgresql/db_backup/dbDump
  echo "Database restore completed."
else
  echo "Database already initialized and contains data, skipping restore."
fi