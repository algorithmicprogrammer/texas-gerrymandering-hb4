from __future__ import annotations
import duckdb

def sanity_checks(con: duckdb.DuckDBPyConnection) -> None:
    # 1) every (plan_id, vtd_geoid) appears once by PK, but check missing joins
    missing_geo = con.execute("""
        SELECT COUNT(*) AS n
        FROM plan_district_vtd pd
        LEFT JOIN geo_vtd g ON g.vtd_geoid = pd.vtd_geoid
        WHERE g.vtd_geoid IS NULL
    """).fetchone()[0]
    if missing_geo:
        raise ValueError(f"{missing_geo} VTD assignments missing from geo_vtd (bad vtd_geoid keys).")

    # 2) check each plan covers same number of VTDs (optional; helpful)
    counts = con.execute("""
        SELECT plan_id, COUNT(*) AS n_vtd
        FROM plan_district_vtd
        GROUP BY plan_id
        ORDER BY n_vtd
    """).df()
    if counts.empty:
        raise ValueError("No rows in plan_district_vtd.")
    if counts["n_vtd"].min() != counts["n_vtd"].max():
        # Not always fatal, but usually a data problem
        print("WARNING: not all plans assign the same number of VTDs.")
        print(counts.head())

    print("Sanity checks passed.")
