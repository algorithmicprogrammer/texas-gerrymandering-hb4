from __future__ import annotations
import os
import duckdb

def connect_db(db_path: str) -> duckdb.DuckDBPyConnection:
    dir_ = os.path.dirname(db_path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=4;")
    con.execute("PRAGMA memory_limit='6GB';")
    return con
