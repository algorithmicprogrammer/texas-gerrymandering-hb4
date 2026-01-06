from __future__ import annotations

import os
import duckdb


def connect_db(db_path: str | None = None) -> duckdb.DuckDBPyConnection:
    """
    Connect to DuckDB.

    - If db_path is None or ':memory:', use in-memory DB (no DB file written).
    - If db_path is a path, create parent dirs and persist.
    """
    if not db_path:
        db_path = ":memory:"

    if db_path != ":memory:":
        dir_ = os.path.dirname(db_path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)

    con = duckdb.connect(db_path)

    # best-effort performance settings
    try:
        con.execute("PRAGMA threads=4;")
        con.execute("PRAGMA memory_limit='6GB';")
    except Exception:
        pass

    return con

