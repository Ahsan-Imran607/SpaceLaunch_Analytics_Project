
import csv
import sqlite3
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "valid_launches.csv"
DB_PATH = PROJECT_ROOT / "data" / "database" / "launches.db"

# Required CSV columns
REQUIRED_COLUMNS = [
    "launch_id",
    "launch_name",
    "launch_datetime",
    "launch_year",
    "launch_month",
    "provider",
    "rocket",
    "launch_location",
    "country",
    "status",
]


def load_data():
    # Check whether validated CSV exists
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Validated CSV not found: {CSV_PATH}\n"
            "Run validation.py first."
        )

    # Read CSV records
    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("CSV file is empty or has no headers.")

        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing CSV columns: {missing}")

        records = list(reader)

    # Create database folder
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Connect to SQLite database
    with sqlite3.connect(DB_PATH) as connection:

        # Create table
        connection.execute("""
            CREATE TABLE IF NOT EXISTS launches (
                launch_id TEXT PRIMARY KEY,
                launch_name TEXT NOT NULL,
                launch_datetime TEXT NOT NULL,
                launch_year INTEGER NOT NULL,
                launch_month INTEGER NOT NULL,
                provider TEXT NOT NULL,
                rocket TEXT NOT NULL,
                launch_location TEXT NOT NULL,
                country TEXT,
                status TEXT NOT NULL
            )
        """)

        # Insert records
        cursor = connection.executemany("""
            INSERT OR REPLACE INTO launches (
                launch_id, launch_name, launch_datetime,
                launch_year, launch_month, provider,
                rocket, launch_location, country, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            (
                row["launch_id"],
                row["launch_name"],
                row["launch_datetime"],
                int(row["launch_year"]),
                int(row["launch_month"]),
                row["provider"],
                row["rocket"],
                row["launch_location"],
                row["country"] or None,
                row["status"],
            )
            for row in records
        ])

        print(f"CSV records read: {len(records)}")
        print(f"New records inserted: {cursor.rowcount}")
        print(f"Database location: {DB_PATH}")


if __name__ == "__main__":
    load_data()
