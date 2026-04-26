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
IMAGES_DIR = PROJ_ROOT / "images"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


PLANC2333_DIR = RAW_DATA_DIR / "PLANC2333"
PLANC2333_SHP_FILE = PLANC2333_DIR / "PLANC2333.shp"

CENSUS_DEMOGRAPHICS_DIR = RAW_DATA_DIR / "tx_pl2020"
CENSUS_DEMOGRAPHICS_PL = CENSUS_DEMOGRAPHICS_DIR / "tx000022020.pl"
CENSUS_DEMOGRAPHICS_GEOHEADER = CENSUS_DEMOGRAPHICS_DIR / "txgeo2020.pl"

CENSUS_GEO_DIR = RAW_DATA_DIR / "tl_2020_48_tabblock20"
CENSUS_GEO_SHP_FILE = CENSUS_GEO_DIR / "tl_2020_48_tabblock20.shp"

ELECTION_DATA_DIR = RAW_DATA_DIR / "2024-general-vtds-election-data"
GEN_ELECTION_CSV = ELECTION_DATA_DIR / "2024_General_Election_Returns.csv"
DEM_PRIMARY_CSV = ELECTION_DATA_DIR/"2024_Democratic_Primary_Election_Returns.csv"
REP_PRIMARY_CSV = ELECTION_DATA_DIR/"2024_Republican_Primary_Election_Returns.csv"

VTDS_GEO_DIR = RAW_DATA_DIR / "vtds_24pg"
VTDS_SHP_FILE= VTDS_GEO_DIR / "VTDs_24PG.shp"

ACS_CSV_DIR = RAW_DATA_DIR / "CVAP_2020-2024_ACS_csv_files"
ACS_CSV_FILE = ACS_CSV_DIR / "Tract.csv"

CANDIDATE_RACE_PARTY = RAW_DATA_DIR / "Candidate_Race_Party.csv"
DROPPED_ELECS = RAW_DATA_DIR / "dropped_elecs.csv"
RECENCY_WEIGHTS = RAW_DATA_DIR / "recency_weights.csv"
TX_ELECTIONS = RAW_DATA_DIR / "TX_elections.csv"

PRECINCT_DATASET_PARQUET = PROCESSED_DATA_DIR/"tx_precincts_final.parquet"
PRECINCT_DATASET_CSV = PROCESSED_DATA_DIR/"tx_cvap_for_ei.csv"

PIPELINES_DIR = PROJ_ROOT / "pipelines"
ECOLOGICAL_INFERENCE_LAYER_DIR = PIPELINES_DIR / "ecological_inference_layer"

EI_OUTPUTS_DIR = ECOLOGICAL_INFERENCE_LAYER_DIR / "ei_outputs"
MEAN_PREC_VOTE_COUNTS_CSV = EI_OUTPUTS_DIR / "mean_prec_vote_counts.csv"
PREC_COUNT_QUANTS_CSV = EI_OUTPUTS_DIR / "prec_count_quants.csv"
STATEWIDE_RXC_EI_PREFERENCES_CSV = EI_OUTPUTS_DIR / "statewide_rxc_EI_preferences.csv"

RECOM_INPUTS_DIR = ECOLOGICAL_INFERENCE_LAYER_DIR / "recom_inputs"
INGROUP_WEIGHT_CSV_FILE = RECOM_INPUTS_DIR / "ingroup_weight.csv"
MEAN_PREC_VOTE_COUNTS_INPUT = RECOM_INPUTS_DIR / "mean_prec_vote_counts.csv"
PREC_COUNT_QUANTS_INPUT = RECOM_INPUTS_DIR / "prec_count_quants.csv"
STATEWIDE_RXC_EI_PREFERENCES_INPUT = RECOM_INPUTS_DIR / "statewide_rxc_EI_preferences.csv"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
