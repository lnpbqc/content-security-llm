"""Create or check the llm_ tables declared in db/mysql_schema.sql."""

import argparse
from config import Settings
from db.database import Database, TABLES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Validate existing tables without executing CREATE TABLE")
    args = parser.parse_args(argv)
    database = Database(Settings.load())
    if args.check:
        database.initialize()
    else:
        database.create_schema()
    with database.connect() as connection:
        rows = connection.execute(
            "SELECT TABLE_NAME FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA=DATABASE() AND LEFT(TABLE_NAME, 4)='llm_' "
            "ORDER BY TABLE_NAME"
        ).fetchall()
    actual = {row["TABLE_NAME"] for row in rows}
    print("{} required llm_ tables ready; {} llm_ tables present".format(len(TABLES), len(actual)))
    extras = sorted(actual - set(TABLES))
    if extras:
        print("Additional tables preserved (not managed by this schema): " + ", ".join(extras))


if __name__ == "__main__":
    main()
