"""Initialization reporting follows the schema, without modifying unrelated tables."""
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

from db.database import TABLES
from scripts import init_mysql


class InitMysqlTests(unittest.TestCase):
    def run_script(self, args, extras=()):
        database = MagicMock()
        connection = database.connect.return_value.__enter__.return_value
        connection.execute.return_value.fetchall.return_value = [
            {"TABLE_NAME": name} for name in (*TABLES, *extras)
        ]
        output = io.StringIO()
        with patch.object(init_mysql.Settings, "load"), patch.object(
            init_mysql, "Database", return_value=database
        ), redirect_stdout(output):
            init_mysql.main(args)
        return database, output.getvalue()

    def test_create_uses_declared_table_count(self):
        database, output = self.run_script([])
        database.create_schema.assert_called_once_with()
        self.assertIn(f"{len(TABLES)} required llm_ tables ready", output)

    def test_check_never_creates_tables(self):
        database, _ = self.run_script(["--check"])
        database.initialize.assert_called_once_with()
        database.create_schema.assert_not_called()

    def test_extra_table_is_reported_and_preserved(self):
        database, output = self.run_script(["--check"], ("llm_business_database_config",))
        self.assertIn("llm_business_database_config", output)
        self.assertIn(f"{len(TABLES) + 1} llm_ tables present", output)
        sql = database.connect.return_value.__enter__.return_value.execute.call_args.args[0]
        self.assertTrue(sql.startswith("SELECT "))


if __name__ == "__main__":
    unittest.main()
