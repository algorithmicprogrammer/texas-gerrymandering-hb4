<p align="center">
# Code
 <a href="https://github.com/algorithmicprogrammer/texas-gerrymandering-hb4"><strong>Explore Repository »</strong></a>
    <br />
    <a href="https://github.com/algorithmicprogrammer/texas-gerrymandering-hb4/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/algorithmicprogrammer/texas-gerrymandering-hb4/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
</p>

## Getting Started

### Prerequisites
1. Install git (Debian/Ubuntu).
```
sudo apt install git
```

### Installation
1. Cloning the repository.
```
git clone https://github.com/algorithmicprogrammer/texas-gerrymandering-hb4.git
```

2. Installing requirements.
```
pip install -r requirements.txt
```

### Testing
Run tests.
```
pytests test
```

## Technologies Used
* pip
* mkdocs
* numpy
* scikit-learn
* pandas
* geopandas
* GeoJSON
* *duckdb
* GerryChain
* Shapely
* loguru
* python-dotenv
* pytest
* ruff
* tqdm
* typer

## Project Organization
```
├── LICENSE
├── Makefile
├── README.md
├── code
│   ├── datasets
    │   ├── 01_clean_district_shpfile.ipynb  # cleaning congressional district geospatial data
    │   ├── 02_clean_census_data.ipynb    # cleaning Census racial demographics data 
    │   ├── 03_clean_census_shpfile.ipynb    # cleaning census block geospatial data
    │   ├── 04_clean_vtd_election_results.ipynb   # cleaning election results data
    │   └── 05_clean_vtd_shpfile.ipynb   # cleaning precinct geospatial data
├── data
│   ├── interim
    │   ├── districts_clean.gpkg   # clean congressional district geospatial data
    │   └── tx_pl94_clean.parquet   # clean Census racial demographics data
│   ├── processed
│   └── raw
    │   ├── 2024-general-vtds-election-data  # precinct-level election results data
    │   ├── PLANC2333    # congressional districts geospatial data 
    │   ├── tl_2020_48_tabblock20    # census block geospatial data
    │   ├── tx_pl2020_official   # census racial demographics data
    │   └── vtds_24pg   # voting district geospatial data
├── docs
├── models
├── pyproject.toml
├── references
├── reports
│   └── figures
├── requirements.txt
├── setup.cfg
└── texas_gerrymandering_hb4
    ├── __init__.py
    ├── config.py
    ├── dataset.py
    ├── features.py
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py
    │   └── train.py
    └── plots.py
```