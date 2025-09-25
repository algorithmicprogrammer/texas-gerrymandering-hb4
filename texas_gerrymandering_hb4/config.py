from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

PLANC2333_DIR = RAW_DATA_DIR / "PLANC2333"
PLANC2333_SHP_FILE = PLANC2333_DIR / "PLANC2333.shp"

GEOSPATIAL_DATA_DIR = PROCESSED_DATA_DIR / "geospatial"
CLEAN_DISTRICTS_PARQUET = GEOSPATIAL_DATA_DIR / "districts_clean.parquet"

CENSUS_DEMOGRAPHICS_DIR = RAW_DATA_DIR / "tx_pl2020_official"
CENSUS_DEMOGRAPHICS_TXT = CENSUS_DEMOGRAPHICS_DIR / "Blocks_Pop.txt"

CENSUS_GEO_DIR = RAW_DATA_DIR / "tl_2020_48_tabblock20"
CENSUS_GEO_SHP_FILE = CENSUS_GEO_DIR / "tl_2020_48_tabblock20.shp"

ELECTION_DATA_DIR = RAW_DATA_DIR / "2024-general-vtds-election-data"
GEN_ELECTION_CSV = ELECTION_DATA_DIR / "2024_General_Election_Returns.csv"

VTDS_GEO_DIR = RAW_DATA_DIR / "vtds_24pg"
VTDS_SHP_FILE= VTDS_GEO_DIR / "VTDs_24PG.shp"

CLEAN_ELECTION_RESULTS = INTERIM_DATA_DIR / "clean_vtd_election_results.csv"
CLEAN_VTD_GEO = INTERIM_DATA_DIR / "vtds_geo_clean.parquet"

TABULAR_DATA_DIR = PROCESSED_DATA_DIR / "tabular"
FINAL_CSV = TABULAR_DATA_DIR / "districts_final.csv"

RACE = ["pct_white", "pct_black", "pct_asian", "pct_hispanic"]

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
