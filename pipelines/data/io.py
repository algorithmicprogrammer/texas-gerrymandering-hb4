from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

# Tries to import GeoPandas.
try:
    import geopandas as gpd
# If anything goes wrong, gpd is set to None.
except Exception:
    gpd = None

# Two tuples of recognized file extensions for tabular and geospatial data, respectively.
SUPPORTED_TABULAR = (".csv", ".tsv", ".parquet", ".pq", ".feather")
SUPPORTED_GEO = (".shp", ".gpkg", ".geojson", ".json", ".parquet", ".pq")

def mkdir_p(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# Returns a standardized-column version of a DataFrame.
def stdcols(df: pd.DataFrame) -> pd.DataFrame:
    # Avoid mutating original DataFrame.
    df = df.copy()
    # Stripping whitespace and converting to lowercase.
    df.columns = [c.strip().lower() for c in df.columns]
    return df

# Produces a "safe" dataset identifier from a file name.
def dataset_key(path: Path) -> str:
    # Replaces any characters that are not a-z, _, or 0-9 with an underscore.
    return re.sub(r"[^a-z0-9_]+", "_", path.stem.lower())

# Detects whether an object behaves like a GeoDataFrame.
def is_geodf(obj) -> bool:
    # Checks whether geopandas is available and if the object has a geometry attribute that is not none.
    return gpd is not None and hasattr(obj, "geometry") and getattr(obj, "geometry") is not None

def read_any(path: Path):
    """"
    Reads many formats into either a Pandas DataFrame or a GeoPandas GeoDataFrame depending on extension and availability.

    Args:
        path: filepath

    Raises:
        ImportError: if Geopandas is missing in a geospatial file
        ValueError: if the input file type is not supported
    """

    # ext is the file extension in lowercase.
    ext = path.suffix.lower()

    # If the file type is parquet, it reads it into a GeoDataFrame if it loads with geometry; reads it into Pandas DataFrame otherwise.
    if ext in (".parquet", ".pq"):
        if gpd is not None:
            try:
                return gpd.read_parquet(path)
            except Exception:
                return pd.read_parquet(path)
        return pd.read_parquet(path)

    # Reads Apache Feather format into a Pandas DataFrame.
    if ext == ".feather":
        return pd.read_feather(path)

    # Reads CSV and TSV  into a Pandas DataFrame.
    if ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else ","
        return pd.read_csv(path, sep=sep)

    # if the file extension is in a supported geospatial format and hasn't already matched earlier branches
    if ext in SUPPORTED_GEO:
        # If GeoPandas is missing, raises an ImportError.
        if gpd is None:
            raise ImportError("geopandas required to read geospatial file: " + str(path))
        return gpd.read_file(path)

    # Special case for TXT files, which are common for Census demographics data.
    if ext in [".txt"]:
        # Reads 8192 characters to guess delimiter.
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
