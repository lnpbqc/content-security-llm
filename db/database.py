"""MySQL transactions and explicit, namespaced schema management."""

import re
from contextlib import contextmanager
from pathlib import Path

import pymysql
from pymysql.constants import CLIENT

from config import Settings


SCHEMA = Path(__file__).with_name("mysql_schema.sql").read_text(encoding="utf-8")
TABLES = tuple(re.findall(r"CREATE TABLE IF NOT EXISTS (llm_\w+)", SCHEMA))


def _expression(value):
    """Normalize MySQL's metadata quoting for our literal defaults/checks."""
    if value is None:
        return None
    return re.sub(r"[\s`()\\]", "", value.lower().replace("_utf8mb4", ""))


class Connection(pymysql.connections.Connection):
    def execute(self, sql, parameters=None):
        """Keep the repositories' execute/fetch style with native MySQL SQL."""
        cursor = self.cursor()
        try:
            cursor.execute(sql, parameters)
        except BaseException:
            cursor.close()
            raise
        return cursor


class Database:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.data_dir = Path(settings.data_dir)

    @contextmanager
    def connect(self):
        connection = Connection(
            host=self.settings.mysql_host, port=self.settings.mysql_port,
            user=self.settings.mysql_user, password=self.settings.mysql_password,
            database=self.settings.mysql_database, charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor, autocommit=False,
            client_flag=CLIENT.FOUND_ROWS, connect_timeout=10, read_timeout=30,
            write_timeout=30,
            sql_mode="STRICT_ALL_TABLES,NO_ZERO_DATE,NO_ZERO_IN_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION",
            init_command="SET time_zone = '+00:00'",
        )
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def create_schema(self):
        """DDL is explicit; API and workers only check the schema."""
        with self.connect() as connection:
            for statement in SCHEMA.split(";"):
                if statement.strip():
                    connection.execute(statement)
        self.initialize()

    def initialize(self):
        """Reject missing/incompatible llm_ tables without altering any table."""
        with self.connect() as connection:
            columns = connection.execute(
                "SELECT TABLE_NAME, COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE, EXTRA, "
                "CHARACTER_MAXIMUM_LENGTH, COLLATION_NAME, COLUMN_DEFAULT "
                "FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = DATABASE()"
            ).fetchall()
            engines = connection.execute(
                "SELECT TABLE_NAME, ENGINE, TABLE_COLLATION FROM information_schema.TABLES "
                "WHERE TABLE_SCHEMA = DATABASE()"
            ).fetchall()
            indexes = connection.execute(
                "SELECT TABLE_NAME, INDEX_NAME, COLUMN_NAME, NON_UNIQUE, SEQ_IN_INDEX, SUB_PART "
                "FROM information_schema.STATISTICS WHERE TABLE_SCHEMA = DATABASE() "
                "ORDER BY TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX"
            ).fetchall()
            foreign = connection.execute(
                "SELECT TABLE_NAME, COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME "
                "FROM information_schema.KEY_COLUMN_USAGE WHERE TABLE_SCHEMA = DATABASE() "
                "AND REFERENCED_TABLE_NAME IS NOT NULL"
            ).fetchall()
            checks = connection.execute(
                "SELECT tc.TABLE_NAME, cc.CHECK_CLAUSE FROM information_schema.TABLE_CONSTRAINTS tc "
                "JOIN information_schema.CHECK_CONSTRAINTS cc "
                "ON cc.CONSTRAINT_SCHEMA=tc.CONSTRAINT_SCHEMA AND cc.CONSTRAINT_NAME=tc.CONSTRAINT_NAME "
                "WHERE tc.TABLE_SCHEMA=DATABASE() AND tc.CONSTRAINT_TYPE='CHECK' AND tc.ENFORCED='YES'"
            ).fetchall()
        by_table = {table: {} for table in TABLES}
        for column in columns:
            if column["TABLE_NAME"] in by_table:
                by_table[column["TABLE_NAME"]][column["COLUMN_NAME"]] = column
        engines = {row["TABLE_NAME"]: row for row in engines}
        actual_indexes = {}
        for row in indexes:
            key = (row["TABLE_NAME"], row["INDEX_NAME"], row["NON_UNIQUE"])
            actual_indexes.setdefault(key, []).append((row["COLUMN_NAME"], row["SUB_PART"]))
        actual_foreign = {(row["TABLE_NAME"], row["COLUMN_NAME"], row["REFERENCED_TABLE_NAME"],
                           row["REFERENCED_COLUMN_NAME"]) for row in foreign}
        actual_checks = {(row["TABLE_NAME"], _expression(row["CHECK_CLAUSE"])) for row in checks}
        errors = []
        for table, body in re.findall(r"CREATE TABLE IF NOT EXISTS (llm_\w+) \((.*?)\n\) ENGINE", SCHEMA, re.S):
            if table not in engines:
                errors.append(table + ": missing table")
                continue
            if engines[table]["ENGINE"] != "InnoDB" or engines[table]["TABLE_COLLATION"] != "utf8mb4_bin":
                errors.append(table + ": expected InnoDB/utf8mb4_bin")
            expected = {}
            required_indexes = []
            for name, type_, suffix in re.findall(
                    r"^    (\w+) (VARCHAR\(\d+\)|LONGTEXT|BIGINT|DOUBLE)([^\n]*)", body, re.M):
                expected[name] = type_.lower()
                column = by_table[table].get(name)
                nullable = "NO" if "NOT NULL" in suffix or "PRIMARY KEY" in suffix else "YES"
                if (column is None or column["COLUMN_TYPE"] != type_.lower()
                        or column["IS_NULLABLE"] != nullable
                        or (type_.startswith("VARCHAR") or type_ == "LONGTEXT")
                        and column["COLLATION_NAME"] != "utf8mb4_bin"
                        or column is not None and ("AUTO_INCREMENT" in suffix) != ("auto_increment" in column["EXTRA"])):
                    errors.append(table + "." + name + ": incompatible column")
                default = re.search(r"DEFAULT (\('[^']*'\)|'[^']*'|\d+)", suffix)
                if column is not None and _expression(column["COLUMN_DEFAULT"]) != _expression(default[1] if default else None):
                    errors.append(table + "." + name + ": incompatible default")
                if "PRIMARY KEY" in suffix:
                    required_indexes.append((0, [(name, None)]))
                    if actual_indexes.get((table, "PRIMARY", 0)) != [(name, None)]:
                        errors.append(table + ": missing primary key")
                if "UNIQUE" in suffix:
                    required_indexes.append((0, [(name, None)]))
            if set(by_table[table]) != set(expected):
                errors.append(table + ": unexpected/missing columns")
            for unique, parts in re.findall(r"^    (PRIMARY KEY|UNIQUE|INDEX \w+) ?\(([^\n]+)\)", body, re.M):
                fields = []
                for field in parts.split(","):
                    match = re.fullmatch(r"\s*(\w+)(?:\((\d+)\))?(?: DESC)?\s*", field)
                    fields.append((match[1], int(match[2]) if match[2] else None))
                required_indexes.append((1 if unique.startswith("INDEX") else 0, fields))
                if unique == "PRIMARY KEY" and actual_indexes.get((table, "PRIMARY", 0)) != fields:
                    errors.append(table + ": missing primary key")
            for unique, fields in required_indexes:
                if not any(t == table and non_unique == unique and values == fields
                           for (t, _, non_unique), values in actual_indexes.items()):
                    errors.append(table + ": missing index " + str(fields))
            for column, referenced in re.findall(r"FOREIGN KEY \((\w+)\) REFERENCES (llm_\w+)\(id\)", body):
                if (table, column, referenced, "id") not in actual_foreign:
                    errors.append(table + ": missing foreign key " + column)
            for expression in re.findall(r"CHECK\s*\((.*?)\)", body, re.S):
                if (table, _expression(expression)) not in actual_checks:
                    errors.append(table + ": missing/incompatible CHECK constraint")
        if errors:
            raise RuntimeError("MySQL schema check failed; run python -m scripts.init_mysql "
                               "before starting the service: " + "; ".join(errors))
        self.data_dir.mkdir(parents=True, exist_ok=True)
