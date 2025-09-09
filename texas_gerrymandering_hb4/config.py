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

PLANC2308_DIR = RAW_DATA_DIR / "PLANC2308"
PLANC2308_SHP_FILE = PLANC2308_DIR / "PLANC2308.shp"

CENSUS_DEMOGRAPHICS_DIR = RAW_DATA_DIR / "tx_pl"

ELECTION_DATA_DIR = RAW_DATA_DIR / "2024-general-vtds-election-data"
GEN_ELECTION_CSV = ELECTION_DATA_DIR / "2024_General_Election_Returns.csv"

CENSUS_API_KEY = "770829dff88eb1d68b4a0fdaa1ec8f5127ff893c"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
