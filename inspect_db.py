from __future__ import annotations

import sqlite3
from pathlib import Path


DB_PATH = Path("quantum_simulation.db")


def main() -> None:
    if not DB_PATH.is_file():
        raise FileNotFoundError(f"找不到資料庫：{DB_PATH.resolve()}")

    with sqlite3.connect(DB_PATH) as connection:
        tables = [
            row[0]
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                  AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            ).fetchall()
        ]

        print(f"資料庫：{DB_PATH.resolve()}")
        print(f"資料表：{tables}")

        for table in tables:
            escaped_table = table.replace('"', '""')

            count = connection.execute(
                f'SELECT COUNT(*) FROM "{escaped_table}"'
            ).fetchone()[0]

            columns = connection.execute(
                f'PRAGMA table_info("{escaped_table}")'
            ).fetchall()

            print("\n" + "=" * 80)
            print(f"資料表：{table}")
            print(f"資料筆數：{count}")
            print("欄位：")

            for column in columns:
                column_id, name, sql_type, not_null, default, primary_key = column
                print(
                    f"  {column_id:3d} | "
                    f"{name:35s} | "
                    f"{sql_type:10s} | "
                    f"PK={primary_key}"
                )

            sample = connection.execute(
                f'SELECT * FROM "{escaped_table}" LIMIT 1'
            ).fetchone()

            print("第一筆資料：")
            print(sample)


if __name__ == "__main__":
    main()