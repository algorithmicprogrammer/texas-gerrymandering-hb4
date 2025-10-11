<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/texas_flag.jpg" alt="texas" width="80" height="80">
  </a>

  <h3 align="center">A Quantitative Analysis of Negative Racial Gerrymandering in Texas</h3>

  <p align="center">
    An Illinois Institute of Technology study quantifying negative racial gerrymandering in Texas as a result of SB4.
    <br />
    <a href=""><strong>Explore Docs »</strong></a>
    <br />
    <br />
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
        <li><a href="#data-sources">Data Sources</a></li>
        <li><a href="#technologies-used">Technologies Used</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#testing">Testing</a></li>
      </ul>
    </li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About the Project
This experiment introduces a novel computational approach to detect illegal negative racial gerrymandering practices and applies it to Texas amid the mid-decade redistricting following the 2025 passage of HB4. Leveraging a reproducible workflow, we combine Texas Legislative Council shapefiles, U.S. Census Redistricting Data, U.S. Census TIGER/LINE shapefiles, and precinct-level election results to build a consolidated dataset providng detailed information on each Texas congressional district.
After an extensive data consolidation process via a custom pipeline, supervised and unsupervised machine learning methods are employed to examine the effect of racial features on Texas congressional districts. A linear regression classifier predicts district partisan outcome based on racial features, whereas a K-means clustering algorithm assesses how racial composition affects the clustering of the districts. Our results demonstrate that race plays a predominant role in shaping district party outcomes and clusters. This in turn provides evidence of illegal negative racial gerrymandering practices in Texas following the mid-decade redistrcting passed in HB4. We contribute a framework for evaluating negative racial gerrymandering with machine learning methods, which can be applied nationwide. 

## Website
<li>url: https://texasracialgerrymanderingstudy.vercel.app</li>
<li>password: marisoliit</li>

## Paper Rough Draft
Link: https://www.overleaf.com/read/ssvxhxnggkpx#7d91c3

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

3. Create a Python virtual environment.
```
python3 -m venv venv
```

4. Activate virtual environment (Linux/MacOS).
```
source venv/bin/activate
```

5. Install requirements.
```
pip install -r requirements.txt
```

6. Run ETL pipeline. The datasets are massive and the areal-weighted spatial joins are computationally expensive, so this could take 30+ minutes.
```
python etl_pipeline.py \
  --districts data/raw/PLANC2333/PLANC2333.shp \
  --census data/raw/tl_2020_48_tabblock20/tl_2020_48_tabblock20.shp \
  --vtds data/raw/vtds_24pg/VTDs_24PG.shp \
  --pl94 data/raw/tx_pl2020_official/Blocks_Pop.txt \
  --elections data/raw/2024-general-vtds-election-data/2024_General_Election_Returns.csv \
  --elections-office "U.S. Sen" \
  --data-processed-tabular data/processed/tabular \
  --data-processed-geospatial data/processed/geospatial \
  --sqlite data/warehouse/warehouse.db
```

### Testing
Run tests.
```
pytests test
```


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
    │   ├── models
    │   ├── linear_regression_classifier
            │   ├── 01_preprocess.ipynb    
            │   ├── 02_train.ipynb  
            │   └── 03_evaluate.ipynb 
    │   └── kmeans_clustering 
            │   ├── 01_preprocess.ipynb   
            │   ├── 02_train.ipynb  
            │   └── 03_evaluate.ipynb
├── data
│   ├── processed
│   └── raw
    │   ├── 2024-general-vtds-election-data  # precinct-level election results data
    │   ├── PLANC2333    # congressional districts geospatial data 
    │   ├── tl_2020_48_tabblock20    # census block geospatial data
            │   ├── tl_2020_48_tabblock.cpg    #identity character encoding
            │   ├── tl_2020_48_tabblock.dbf    #tabular attribute information
            │   ├── tl_2020_48_tabblock.prj    #Coordinate System information
            │   ├── tl_2020_48_tabblock.shp    # Feature Geometry
            │   ├── tl_2020_48_tabblock.shp.ea.iso    # International Organization for Standardization metadata in XML
            │   └── tl_2020_48_tabblock20.shp.iso.xml   # Entity and attribute of ISO 191 metadata in XML 
    │   ├── tx_pl2020_official   # census racial demographics data
    │   └── vtds_24pg   # voting district geospatial data
├── docs #docs to be published on project website
│   ├── docs
    │   ├── code.md  # readme regarding code
    │   ├── data.md    # readme regarding data
    │   ├── index.md   # readme for project
    │   ├── models.md   # readme for machine learning models
    │   └── references.md   # readme for references used
├── pyproject.toml
├── etl_pipeline.py
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

---
Made with ♥ by Algorithmic Programmer






