"""Create and validate this service's 14 llm_ tables in the configured MySQL database."""

from config import Settings
from db.database import Database


def main():
    database = Database(Settings.load())
    database.create_schema()
    print("14 llm_ tables ready")


if __name__ == "__main__":
    main()
