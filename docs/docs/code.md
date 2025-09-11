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
├── data
│   ├── interim
│   ├── processed
│   └── raw
    │   ├── 2024-general-vtds-election-data
    │   ├── PLANC2335
    │   └── vtds_24pg
├── docs
├── models
│   ├── 01_classification_linear_regression.ipynb
│   ├── 02_classification_logistic_regression.ipynb
│   ├── 03_classification_lda.ipynb
│   ├── 04_classification_random_forest.ipynb
│   ├── 05_classification_gradient_boost.ipynb
│   └── 06_clustering_kmeans.ipynb
├── notebooks
│   ├── 01_clean_district_shpfile.ipynb
│   ├── 02_clean_census_data.ipynb
│   ├── 03_clean_census_geography.ipynb
│   ├── 04_clean_election_data.ipynb
│   ├── 05_clean_census_geography.ipynb
│   ├── 06_merge_census_datasets.ipynb
│   ├── 07_merge_election_datasets.ipynb
│   └── 08_final_data_consolidation.ipynb
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