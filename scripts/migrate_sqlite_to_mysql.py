"""Explicit schema setup and lossless, repeatable SQLite -> MySQL import.

Stop API/workers first. Run from the project root with python -m scripts.migrate_sqlite_to_mysql.
"""

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from config import Settings
from db.database import Database, TABLES


def source_connection(path):
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("BEGIN")
    return connection


def migrate(source_path, database):
    """All imported rows share one MySQL transaction; the source is read-only."""
    database.initialize()
    source = source_connection(source_path)
    try:
        source_tables = {row[0] for row in source.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
        known = {table.removeprefix("llm_") for table in TABLES}
        if source_tables - known:
            raise ValueError("Unrecognized SQLite tables: " + ", ".join(sorted(source_tables - known)))
        report = {}
        with database.connect() as target:
            metadata = target.execute(
                "SELECT TABLE_NAME, COLUMN_NAME, CHARACTER_MAXIMUM_LENGTH, COLUMN_KEY "
                "FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=DATABASE() "
                "ORDER BY TABLE_NAME, ORDINAL_POSITION"
            ).fetchall()
            columns = {table: [row for row in metadata if row["TABLE_NAME"] == table] for table in TABLES}
            # Preflight every source row before the first INSERT.
            for table in TABLES:
                old_name = table.removeprefix("llm_")
                if old_name not in source_tables:
                    continue
                source_columns = {row["name"] for row in source.execute(f'PRAGMA table_info("{old_name}")')}
                expected = {row["COLUMN_NAME"] for row in columns[table]}
                missing = expected - source_columns
                legacy_auth = {"label", "deleted_at"} if old_name == "auth_tokens" else set()
                if source_columns - expected or missing - legacy_auth:
                    raise ValueError(f"{old_name}: incompatible SQLite columns")
                for row in source.execute(f'SELECT * FROM "{old_name}"'):
                    for column in columns[table]:
                        name, maximum = column["COLUMN_NAME"], column["CHARACTER_MAXIMUM_LENGTH"]
                        value = row[name] if name in source_columns else None
                        if value is not None and maximum is not None and len(value) > maximum:
                            raise ValueError(f"{old_name}.{name}: value exceeds MySQL length {maximum}")
            for table in TABLES:
                old_name = table.removeprefix("llm_")
                if old_name not in source_tables:
                    continue
                names = [row["COLUMN_NAME"] for row in columns[table]]
                primary = [row["COLUMN_NAME"] for row in columns[table] if row["COLUMN_KEY"] == "PRI"]
                where = " AND ".join(f"`{name}`=%s" for name in primary)
                insert = f"INSERT INTO `{table}` (" + ",".join(f"`{name}`" for name in names) + ") VALUES ("
                insert += ",".join(["%s"] * len(names)) + ")"
                inserted = skipped = 0
                for row in source.execute(f'SELECT * FROM "{old_name}"'):
                    record = dict(row)
                    if old_name == "auth_tokens":
                        record.setdefault("label", "")
                        record.setdefault("deleted_at", None)
                    existing = target.execute(f"SELECT * FROM `{table}` WHERE " + where,
                                              [record[name] for name in primary]).fetchone()
                    if existing is not None:
                        if existing != record:
                            raise ValueError(f"{old_name}: primary key exists with different content; import rolled back")
                        skipped += 1
                    else:
                        target.execute(insert, [record[name] for name in names])
                        inserted += 1
                    saved = target.execute(f"SELECT * FROM `{table}` WHERE " + where,
                                           [record[name] for name in primary]).fetchone()
                    if saved != record:
                        raise ValueError(f"{old_name}: imported row differs from source; import rolled back")
                total = target.execute(f"SELECT COUNT(*) AS total FROM `{table}`").fetchone()["total"]
                report[old_name] = {"source": inserted + skipped, "inserted": inserted,
                                    "skipped": skipped, "target": total}
        return report
    finally:
        source.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="SQLite source file; never modified")
    parser.add_argument("--schema-only", action="store_true", help="only create/check llm_ tables")
    args = parser.parse_args()
    if bool(args.source) == args.schema_only:
        parser.error("provide exactly one of --source or --schema-only")
    database = Database(Settings.load())
    backup_path = None
    if args.source:
        # SQLite backup API also includes any committed WAL contents.
        backup_dir = database.data_dir / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup_path = backup_dir / f"app-before-mysql-{stamp}.sqlite3"
        source = source_connection(args.source)
        try:
            with sqlite3.connect(backup_path) as backup:
                source.backup(backup)
        finally:
            source.close()
        print("SQLite backup:", backup_path.resolve())
    database.create_schema()
    if args.source:
        report = migrate(args.source, database)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("14 llm_ tables ready; business tables unchanged")


if __name__ == "__main__":
    main()
