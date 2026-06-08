<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/texas_flag.jpg" alt="texas" width="80" height="80">
  </a>

  <h3 align="center">A Spatial Graph Framework for Conditional Finite-Sample,
Uncertainty-Aware Redistricting Ensemble Evaluation</h3>

  <p align="center">
    <a href="https://github.com/algorithmicprogrammer/texas-gerrymandering-hb4/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/algorithmicprogrammer/texas-gerrymandering-hb4/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#project-website">Project Website</a></li>
        <li><a href="#paper-rough-draft">Paper Rough Draft</a></li>
        <li><a href="#data-sources">Data Sources</a></li>
        <li><a href="#technologies-used">Technologies Used</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#testing">Testing</a></li>
      </ul>
    </li>
    <li><a href="#project-organization">Project Organization</a></li>
  </ol>
</details>

## About the Project
Evaluation of enacted redistricting plans is a spatial problem at
its core, as it requires transforming heterogeneous geographic,
demographic, and electoral data into a constrained spatial graph
partitioning, while carrying uncertainty forward from upstream
behavioral models. We present an auditable spatial graph framework for uncertainty-aware redistricting ensemble evaluation. The
framework harmonizes public redistricting inputs into a validated
precinct-level adjacency graph, estimates racially polarized voting
with Bayesian ecological inference, compresses posterior uncertainty into reusable scoring summaries, and compares enacted plans
to constrained alternatives using a reversible redistricting kernel
with Besag-Clifford finite-sample rank inference. We introduce
an expected-count minority-preferred-candidate opportunity functional that scores a plan as the expected number of districts in
which a protected group’s candidate of choice prevails, avoiding
discontinuities induced by threshold-based opportunity counts. In a
Texas congressional case study using PLANC2333, the enacted plan
attains the lower-tail resolution floor for Latino opportunity among
1,000 Besag-Clifford spoke plans, while Black opportunity is below
the ensemble mean but not a statistical outlier. The results show
how spatial data integration, graph construction, uncertainty propagation, constrained partition sampling, and finite-sample inference
can be combined into a reproducible workflow for high-stakes spatial decision support.

### Data Sources
<ul>
  <li>
    <a href="https://data.capitol.texas.gov/dataset/planc2335/resource/3552af40-54c1-45f2-9b02-b3c560bc0879">
    Texas Legislative Council Congressional District Geospatial Data (PLANC2333 Shapefile)
    </a>
      <ul>
        <li>The new district map's geospatial data is used for computing compactness scores, which are a widely-accepted indicator of gerrymandering.
        This shapefile contains the district geometries that will be compared against Census/election results data.
        </li>
      </ul>
  </li> 
  <li>
    <a href="https://redistrictingdatahub.org/data/about-our-data/pl-94171-dataset/">
    2020 Decennial Census RPL-94-171 Dataset
    </a>  
  </li>
    <ul>
      <li>This dataset, provided by Redistricting Data Hub, includes racial demogrpahics and population information from the 2020 Census.</li>
    </ul>
  <li>
    <a href="https://www2.census.gov/geo/tiger/TIGER2020PL/LAYER/tl_2020_48_tabblock20.zip">
    2020 Texas U.S. Census Blocks Geospatial Data (tl_2020_48_tabblock20 Shapefile)
    </a>
      <ul>
        <li>
        The geospatial data for the 2020 Texas U.S. Census blocks will be joined with the census block-level voter age population demographics from the 2020 
        Decennial Censusu Redistricting dataset.
        The census block geospatial data will be overlayed with the new district map's geospatial data to map racial composition in each congressional district.
        </li>
      </ul>
  </li>
  <li>
    <a href="https://data.capitol.texas.gov/dataset/comprehensive-election-datasets-compressed-format/resource/e1cd6332-6a7a-4c78-ad2a-852268f6c7a2">
    Texas Legislative Council 2024 Voting Districts General Election Data  
    </a>
      <ul>
        <li>Precinct-level election results data will be used to compute precinct-level votes by party. This dataset will be used to quantify partisanship in congressional districts; unfortunately, Texas voter registration does not record party affiliation, so this dataset is our most reliable source of partisanship information.</li>
      </ul>
  </li>
  <li>
    <a href="https://data.capitol.texas.gov/dataset/4d8298d0-d176-4c19-b174-42837027b73e/resource/906f47e4-4e39-4156-b1bd-4969be0b2780/download/vtds_24pg.zip">
      Texas Legislative Council 2024 Primary & General Elections Voting Districts Geospatial Data (vtds_24pg Shapefile)
    <a>
      <ul>
        <li>
          The Texas voting district geospatial data for the 2024 elections will be joined with the voting district election results. The voting district geospatial data will then be overlayed with 
          the new district map's geospatial data to map partisanship in each congressional district.
        </li>  
      </ul>  
  </li>
</ul>



### Technologies Used
<code><img height="27" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python"></code> 
<code><img height="27" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/pandas/pandas-original.svg" alt="pandas"></code> 
<code><img height="27" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/numpy/numpy-original.svg" alt="numpy"></code>
<code><img height="27" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/matplotlib/matplotlib-original.svg" alt="matplotlib"></code> 
<code><img height="27" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/duckdb/duckdb-original.svg" alt="duckdb"></code> 
<code><img height="27" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/sqlite/sqlite-original.svg" alt="sqlite"></code>
<code><img height="27" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/scikitlearn/scikitlearn-original.svg" alt="scikitlearn"></code> 
<code><img height="27" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/jupyter/jupyter-original-wordmark.svg" alt="jupyter"></code> 
<code><img height="27" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/markdown/markdown-original.svg" alt="markdown"></code>
<code><img height="27" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/latex/latex-original.svg" alt="latex"></code> 

## Getting Started
### Prerequisites
1. Install git (Debian/Ubuntu).
```
sudo apt install git
```

### Installation
1. Clone the repository.
```
git clone https://github.com/algorithmicprogrammer/texas-gerrymandering-hb4.git
```

2. Navigate to the cloned repository. 
```
cd texas-gerrymandering-hb4
```

3. Create a Python virtual environment:

* Command for MacOS/Linux:
```
python3 -m venv venv
```
* Command for Windows 11 Command Prompt:
```commandline
py -m venv venv
```

4. Activate virtual environment.

* Command for MacOS/Linux:
```
source venv/bin/activate
```

* Command for Windows 11 Command Prompt:
```commandline
venv\Scripts\activate.bat
```

5. Install requirements.
```
pip install -r requirements.txt
```

6. Run data engineering layer.


7. Run ecological inference layer.

* Command for Linux/MacOS:
```commandline
cd pipelines/ecological_inference_layer
python run_ei.py
```

* Command for Windows 11 Command Prompt:
```commandline

```

8. Run redistricting pipeline against ensemble.
* Command for Linux/MacOS:
```commandLine
python -m pipelines.redistricting.cli \
  --stage all \
  --geo-vtd data/processed/geo_vtd.parquet \
  --elections data/processed/elections.parquet \
  --returns data/processed/election_returns_vtd.parquet \
  --plans data/processed/plans.parquet \
  --plan-map data/processed/plan_district_vtd.parquet \
  --ensemble-plans data/processed/ensemble_plans.parquet \
  --ensemble-plan-map data/processed/ensemble_plan_district_vtd.parquet \
  --ensemble-id ENS_TXCD_2024_recom_v1 \
  --ei-election-id TX_SEN_2024_GEN \
  --ei-run-id EI_RUN_002 \
  --ei-draws 2000 \
  --ei-tune 3000 \
  --ei-chains 4 \
  --ei-target-accept 0.99 \
  --ei-max-treedepth 15 \
  --ei-seed 20240101 \
  --out-dir data/processed/redistricting_exports \
  --export-format parquet

```

### Testing
Run tests.
```
pytest tests/
```


## Project Organization
```
├── LICENSE
├── Makefile
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── data
│   └── raw                                      # original, immutable source data
│       ├── 2024-general-vtds-election-data/     # precinct-level election returns & VRTO (CSV)
│       ├── CVAP_2020-2024_ACS_csv_files/        # Citizen Voting Age Population (ACS)
│       │   └── Tract.csv
│       ├── PLANC2333/                           # congressional districts geospatial data (shapefile)
│       ├── tl_2020_48_tabblock20/               # 2020 census block geospatial data (shapefile)
│       ├── tx_pl2020/                           # 2020 Census PL 94-171 redistricting data
│       ├── vtds_24pg/                           # 2024 voting district geospatial data (shapefile)
│       ├── Candidate_Race_Party.csv
│       ├── TX_elections.csv
│       ├── dropped_elecs.csv
│       └── recency_weights.csv
│
├── docs                                         # MkDocs site published to the project website
│   ├── README.md
│   ├── mkdocs.yml
│   └── docs
│       ├── bayesian.md
│       ├── code.md
│       ├── data.md
│       ├── index.md
│       ├── models.md
│       └── references.md
│
├── images                                       # figures embedded in docs / README (PNG, JPG)
│
├── notebooks                                    # exploratory analysis & model development
│   ├── datasets
│   │   ├── data/artifacts/                      # intermediate CSV artifacts
│   │   ├── processed/
│   │   │   └── eda_vtds.ipynb
│   │   └── raw
│   │       ├── 01_eda_district_shpfile.ipynb    # congressional district geospatial EDA
│   │       ├── 02_eda_census_data.ipynb         # Census racial demographics EDA
│   │       ├── 03_eda_census_shpfile.ipynb      # census block geospatial EDA
│   │       ├── 05_eda_vtd_shapefile.ipynb       # precinct geospatial EDA
│   │       ├── 06_eda_final_dataset.ipynb       # consolidated dataset EDA
│   │       ├── 07_generate_racial_maps.ipynb    # racial composition maps
│   │       ├── eda_dem_primary_results.ipynb
│   │       └── eda_rep_primary_results.ipynb
│   └── models
│       └── kmeans_clustering
│           ├── 01_preprocess.ipynb
│           ├── 02_train.ipynb
│           ├── 03_evaluate.ipynb
│           ├── 04_map_generation.ipynb
│           └── artifacts/                        # scalers, models, cluster outputs
│
├── pipelines                                    # end-to-end data & modeling pipeline
│   ├── data_engineering_layer                   # ingest, join, validate, build datasets
│   │   ├── demographics.py
│   │   ├── ei_export.py
│   │   ├── elections.py
│   │   ├── ingest.py
│   │   ├── join.py
│   │   ├── nonzero_returns.py
│   │   ├── pipeline.py
│   │   ├── run_table3.py
│   │   ├── schema.py
│   │   ├── sensitivity.py
│   │   ├── spatial.py
│   │   ├── table3_diagnostics.py
│   │   ├── time_pipeline.py
│   │   └── validate.py
│   ├── ecological_inference_layer               # Bayesian ecological inference
│   │   ├── ablation_point_estimate_v_uncertainty.py
│   │   ├── build_calibration_data.py
│   │   ├── diagnose_empty_candidates.py
│   │   ├── generate_ei_results.py
│   │   ├── generate_tx_logit_params.py
│   │   ├── post_ei_processing.py
│   │   └── run_ei.py
│   └── ensemble_generation_layer                # redistricting ensemble generation
│       ├── besag_clifford_vra_opportunity.py
│       ├── run_functions.py
│       └── visualize_ensembles.py
│
├── reports
│   └── figures                                  # publication-ready figures (EPS)
│
├── tests                                        # test suite
│
└── texas_gerrymandering_hb4                     # shared project package
    ├── __init__.py
    ├── config.py
    └── plots.py
```

---
Made with ♥ by Algorithmic Programmer






