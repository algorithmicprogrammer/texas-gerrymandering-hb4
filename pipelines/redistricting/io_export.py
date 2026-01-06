from __future__ import annotations

from pathlib import Path
import duckdb


def export_table(con: duckdb.DuckDBPyConnection, table: str, out_dir: Path, fmt: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        out_path = out_dir / f"{table}.parquet"
        con.execute(f"COPY (SELECT * FROM {table}) TO '{out_path.as_posix()}' (FORMAT PARQUET);")
        return out_path

    if fmt == "csv":
        out_path = out_dir / f"{table}.csv"
        con.execute(
            f"""COPY (SELECT * FROM {table}) TO '{out_path.as_posix()}'
                (HEADER, DELIMITER ',', QUOTE '"', ESCAPE '"');"""
        )
        return out_path

    raise ValueError(f"Unknown export format: {fmt}")


def export_outputs(con: duckdb.DuckDBPyConnection, out_dir: str, fmt: str = "parquet") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Export the main outputs your pipeline creates.
    tables = [
        "district_demo_vap",
        "district_returns",
        "district_opportunity",
        "plan_metrics",
        "ensemble_distribution",
        "plan_vs_ensemble",
        "ei_run",
        "ei_posterior_group",
        "ei_posterior_district",
    ]

    existing = set(
        con.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            """
        )
        .fetchdf()["table_name"]
        .tolist()
    )

    wrote_any = False
    for t in tables:
        if t not in existing:
            continue
        # skip empty tables
        try:
            n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except Exception:
            n = None
        if n == 0:
            continue

        fp = export_table(con, t, out, fmt)
        print(f"[export] {t} -> {fp}")
        wrote_any = True

    if not wrote_any:
        print("[export] No output tables were exported (tables missing or empty).")
