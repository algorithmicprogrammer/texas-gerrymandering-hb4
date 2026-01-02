from __future__ import annotations

from pathlib import Path
from typing import Iterable

# Default tables to export. These should match tables created by your schema + stages.
DEFAULT_EXPORT_TABLES = [
    "geo_vtd",
    "election",
    "election_returns_vtd",
    "plan",
    "plan_district_vtd",
    "district_demo_vap",
    "district_returns",
    "opp_def",
    "district_opportunity",
    "plan_metrics",
    "ensemble_distribution",
    "plan_vs_ensemble",
    "ei_run",
    "ei_posterior_group",
    "ei_posterior_district",
]


def export_tables(
    con,
    out_dir: str,
    fmt: str = "parquet",
    tables: Iterable[str] | None = None,
) -> None:
    """
    Export DuckDB tables to files using DuckDB COPY.

    Parameters
    ----------
    con:
        DuckDB connection (from connect_db()).
    out_dir:
        Output directory.
    fmt:
        "parquet" or "csv".
    tables:
        Iterable of table names to export. If None, exports DEFAULT_EXPORT_TABLES.

    Notes
    -----
    - Skips tables that don't exist (useful if you didn't run all stages).
    - Writes one file per table: <table>.<fmt>
    """
    fmt = fmt.lower().strip()
    if fmt not in {"parquet", "csv"}:
        raise ValueError("fmt must be one of: 'parquet', 'csv'")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # What tables exist?
    try:
        existing = set(con.execute("SHOW TABLES").fetchdf()["name"].tolist())
    except Exception:
        # Fallback for older duckdb/python combos
        existing = {r[0] for r in con.execute("SHOW TABLES").fetchall()}

    if tables is None:
        tables = DEFAULT_EXPORT_TABLES

    exported = []
    for t in tables:
        t = t.strip()
        if not t or t not in existing:
            continue

        out_path = Path(out_dir) / f"{t}.{fmt}"

        if fmt == "parquet":
            con.execute(f"COPY {t} TO '{out_path.as_posix()}' (FORMAT PARQUET);")
        else:
            con.execute(
                f"COPY {t} TO '{out_path.as_posix()}' (FORMAT CSV, HEADER TRUE);"
            )

        exported.append(t)

    # Simple manifest so it's obvious it ran
    manifest = Path(out_dir) / "EXPORT_DONE.txt"
    with manifest.open("w", encoding="utf-8") as f:
        f.write(f"format={fmt}\n")
        f.write("tables_exported:\n")
        for t in exported:
            f.write(f" - {t}\n")
