from __future__ import annotations

import os
import numpy as np
import pandas as pd
import duckdb

from .config import RACE_COLS_VAP, RACE_COLS_CVAP


def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".csv":
        df = pd.read_csv(path, dtype=str)
        return _auto_numeric(df)
    if ext == ".feather":
        return pd.read_feather(path)
    raise ValueError(f"Unsupported file type: {ext}")


def _auto_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c.endswith("_geoid") or c.endswith("_fips") or c in {"plan_id", "district_id", "election_id"}:
            continue
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df


def load_geo_vtd(con: duckdb.DuckDBPyConnection, path: str) -> None:
    df = read_table(path).copy()

    for c in ["vap_nh_native", "vap_nh_asian", "cvap_nh_native", "cvap_nh_asian"]:
        if c not in df.columns:
            df[c] = 0

    required = {"vtd_geoid", "vap_total", "cvap_total", *RACE_COLS_VAP, *RACE_COLS_CVAP}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"geo_vtd missing columns: {missing}")

    if "state_fips" not in df.columns:
        df["state_fips"] = "48"
    else:
        df["state_fips"] = df["state_fips"].astype(str)

    if "county_fips" not in df.columns:
        df["county_fips"] = df["vtd_geoid"].astype(str).str.slice(2, 5)
    else:
        df["county_fips"] = df["county_fips"].astype(str)

    if "county_name" not in df.columns:
        df["county_name"] = None
    if "area_km2" not in df.columns:
        df["area_km2"] = np.nan
    if "total_pop" not in df.columns:
        df["total_pop"] = pd.NA

    con.register("tmp_geo_vtd", df)
    con.execute(
        """
        INSERT OR REPLACE INTO geo_vtd
        SELECT
            vtd_geoid,
            COALESCE(state_fips, '48') AS state_fips,
            county_fips,
            county_name,
            area_km2,
            total_pop,
            vap_total,
            vap_nh_white,
            vap_nh_black,
            vap_hisp,
            vap_nh_asian,
            vap_nh_native,
            vap_other,
            cvap_total,
            cvap_nh_white,
            cvap_nh_black,
            cvap_hisp,
            cvap_nh_asian,
            cvap_nh_native,
            cvap_other
        FROM tmp_geo_vtd
        """
    )
    con.unregister("tmp_geo_vtd")


def load_election(con: duckdb.DuckDBPyConnection, path: str) -> None:
    df = read_table(path)
    required = {"election_id", "year", "office", "stage"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"election missing columns: {missing}")
    if "notes" not in df.columns:
        df = df.copy()
        df["notes"] = None
    con.register("tmp_election", df)
    con.execute(
        """
        INSERT OR REPLACE INTO election
        SELECT election_id, year, office, stage, notes
        FROM tmp_election
        """
    )
    con.unregister("tmp_election")


def load_election_returns_vtd(con: duckdb.DuckDBPyConnection, path: str) -> None:
    df = read_table(path).copy()

    alias_map = {
        "votes_gop": "votes_rep",
        "votes_r": "votes_rep",
        "rep_votes": "votes_rep",
        "votes_republican": "votes_rep",
    }
    for src, dst in alias_map.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})
            print(f"[io_load] Renamed {src} -> {dst}")

    if "votes_rep" not in df.columns:
        if {"votes_total", "votes_dem"}.issubset(df.columns):
            df["votes_total"] = pd.to_numeric(df["votes_total"], errors="coerce")
            df["votes_dem"] = pd.to_numeric(df["votes_dem"], errors="coerce")
            df["votes_rep"] = (df["votes_total"] - df["votes_dem"]).clip(lower=0)
            print("[io_load] Constructed votes_rep = max(votes_total - votes_dem, 0)")
        else:
            raise ValueError(
                "election_returns_vtd missing votes_rep and cannot be constructed "
                "(need votes_total and votes_dem)."
            )

    required = {"election_id", "vtd_geoid", "votes_total", "votes_dem", "votes_rep"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"election_returns_vtd missing columns: {missing}")

    if "votes_other" not in df.columns:
        df["votes_other"] = 0

    for c in ["votes_total", "votes_dem", "votes_rep", "votes_other"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["dem_share"] = np.where(
        df["votes_total"].notna() & (df["votes_total"] > 0),
        df["votes_dem"] / df["votes_total"],
        np.nan,
    )

    con.register("tmp_returns", df)
    con.execute(
        """
        INSERT OR REPLACE INTO election_returns_vtd
        SELECT election_id, vtd_geoid, votes_total, votes_dem, votes_rep, votes_other, dem_share
        FROM tmp_returns
        """
    )
    con.unregister("tmp_returns")


def load_plan(con: duckdb.DuckDBPyConnection, path: str) -> None:
    df = read_table(path).copy()
    if "plan_id" not in df.columns:
        raise ValueError("plan missing columns: {'plan_id'}")

    for c in ["cycle", "chamber", "ensemble_id"]:
        if c not in df.columns:
            df[c] = None

    if "plan_type" not in df.columns:
        ens = df["ensemble_id"].astype("string")
        is_blank = ens.isna() | (ens.str.strip() == "")
        df["plan_type"] = np.where(is_blank, "ENACTED", "ENSEMBLE")
        print("[io_load] Inferred plan_type from ensemble_id (blank -> ENACTED, else ENSEMBLE)")

    for c in ["generator", "seed", "constraints_json", "created_at"]:
        if c not in df.columns:
            df[c] = None

    con.register("tmp_plan", df)
    con.execute(
        """
        INSERT OR REPLACE INTO plan
        SELECT
            plan_id, plan_type, cycle, chamber, ensemble_id,
            generator, seed, constraints_json, created_at
        FROM tmp_plan
        """
    )
    con.unregister("tmp_plan")


def load_plan_district_vtd(con: duckdb.DuckDBPyConnection, path: str) -> None:
    df = read_table(path)
    required = {"plan_id", "vtd_geoid", "district_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"plan_district_vtd missing columns: {missing}")
    con.register("tmp_pdv", df)
    con.execute(
        """
        INSERT OR REPLACE INTO plan_district_vtd
        SELECT plan_id, vtd_geoid, district_id
        FROM tmp_pdv
        """
    )
    con.unregister("tmp_pdv")
