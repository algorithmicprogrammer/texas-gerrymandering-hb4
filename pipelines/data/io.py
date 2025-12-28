#!/usr/bin/env python3
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None

SUPPORTED_TABULAR = (".csv", ".tsv", ".parquet", ".pq", ".feather")
SUPPORTED_GEO = (".shp", ".gpkg", ".geojson", ".json", ".parquet", ".pq")

def mkdir_p(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def stdcols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def dataset_key(path: Path) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", path.stem.lower())

def is_geodf(obj) -> bool:
    return gpd is not None and hasattr(obj, "geometry") and getattr(obj, "geometry") is not None

def read_any(path: Path):
    ext = path.suffix.lower()
    if ext in (".parquet", ".pq"):
        if gpd is not None:
            try:
                return gpd.read_parquet(path)
            except Exception:
                return pd.read_parquet(path)
        return pd.read_parquet(path)
    if ext == ".feather":
        return pd.read_feather(path)
    if ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    if ext in SUPPORTED_GEO:
        if gpd is None:
            raise ImportError("geopandas required to read geospatial file: " + str(path))
        return gpd.read_file(path)
    # Treat .txt as a delimited table (PL94 exports are often | or tab delimited)
    if ext in [".txt"]:
        # Try to sniff delimiter from a sample
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            sample = f.read(8192)

        # Common delimiters for PL-style exports
        candidates = ["|", "\t", ",", ";"]

        # Try Sniffer first; fall back to candidates
        delim = None
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=candidates)
            delim = dialect.delimiter
        except Exception:
            # fallback heuristic: pick the delimiter that appears most often
            counts = {d: sample.count(d) for d in candidates}
            delim = max(counts, key=counts.get)

        # If no delimiter appears, fallback to fixed-width
        if sample.count(delim) == 0:
            return pd.read_fwf(path)

        return pd.read_csv(path, sep=delim, engine="python")

    raise ValueError(f"Unsupported input file type: {path}")

def ensure_crs(gdf):
    if gpd is None:
        raise ImportError("geopandas required for CRS operations")
    if gdf.crs is None:
        # Texas data is usually NAD83; leave as-is if unknown, but warn
        # User should set CRS upstream if missing.
        raise ValueError("GeoDataFrame has no CRS. Please set CRS on input data.")
    return gdf

def assert_projected_planar(gdf, name: str) -> None:
    if gpd is None:
        return
    crs = gdf.crs
    if crs and getattr(crs, "is_geographic", False):
        raise ValueError(f"{name} is in a geographic CRS. Reproject to a projected CRS before area computations.")

def write_parquet(df, path: Path) -> None:
    mkdir_p(path.parent)
    if is_geodf(df):
        df.to_parquet(path, index=False)
    else:
        df.to_parquet(path, index=False)
