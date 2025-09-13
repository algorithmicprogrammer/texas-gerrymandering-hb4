# Data

## Data Sources
1. <a href="https://data.capitol.texas.gov/dataset/planc2333/resource/5712ebe1-d777-4d4a-b836-0534e17bca01">Texas Legislative Council Congressional District Geospatial Data</a>
    * This is the geospatial data for PLANC2333, which was enacted by the 89th Legislature. 
    * The new district map's geospatial data is used for computing compactness scores (Polsby-Popper, Reock, Convex-Hull), which are a widely-accepted indicator of gerrymandering.
            This shapefile contains the district geometries that will be compared against Census/election results data.
    * The raw dataset has 38 rows (congressional districts) and 2 columns (district number and geospatial coordinates).
    * The raw dataset is under the Creative Commons License (can be freely used, distributed, modified, and combined with other sources).
2.  <a href="https://redistrictingdatahub.org/dataset/texas-block-block-group-cdp-city-county-and-tract-pl-94-171-2020-official/">2020 Decennial Census Redistricting PL-94-171 Dataset</a>
    *  This API, provided by the U.S. Census Bureau, provides 2020 Texas voter age population breakdowns by race at the census-block level.
3.  <a href="https://www2.census.gov/geo/tiger/TIGER2020PL/LAYER/tl_2020_48_tabblock20.zip">2020 Texas U.S. Census Blocks Geospatial Data</a>
    *  The geospatial data for the 2020 Texas U.S. Census blocks will be joined with the census block-level voter age population demographics from the 2020
              Decennial Censusu Redistricting API.
              The census block geospatial data will be overlayed with the new district map's geospatial data to map racial composition in each congressional district.
    * The raw dataset has 668,757 rows and 18 columns. There are 295,195 missing values in the UACE20 and UATYPE20 columns.
4.  <a href="https://data.capitol.texas.gov/dataset/comprehensive-election-datasets-compressed-format/resource/e1cd6332-6a7a-4c78-ad2a-852268f6c7a2">Texas Legislative Council 2024 Voting Districts General Election Data</a>
    * Precinct-level election results data will be used to compute precinct-level votes by party. This dataset will be used to quantify partisanship in congressional districts; unfortunately, Texas voter registration does not record party affiliation, so this dataset is our most reliable source of partisanship information.
    * The raw dataset (2024_General_Election_Returns.csv) has 450,357 rows (candidate votes per precinct) and 10 columns (county, county code, VTD ID, county-VTD ID, VTD key, office, candidate name, candidate party, incumbent indicator, and votes).
    * The raw dataset is under the Creative Commons License (can be freely used, distributed, modified, and combined with other sources).
5.  <a href="https://data.capitol.texas.gov/dataset/4d8298d0-d176-4c19-b174-42837027b73e/resource/906f47e4-4e39-4156-b1bd-4969be0b2780/download/vtds_24pg.zip">Texas Legislative Council 2024 General Elections Voting Districts Geospatial Data</a>
    * The Texas voting district geospatial data for the 2024 election will be joined with the voting district election results. The voting district geospatial data will then be overlayed with
                the new district map's geospatial data to map partisanship in each congressional district.
    * The raw dataset has 9712 rows (precincts) and 9 columns (county ID, color code, VTD code, VTD key, county+VTD code, area, perimeter, and geometry).
    * The raw dataset is under the Creative Commons License (can be freely used, distributed, modified, and combined with other sources).
## Dataset Issues/Limitations
1. <a href="https://data.capitol.texas.gov/dataset/planc2335/resource/3552af40-54c1-45f2-9b02-b3c560bc0879">Texas Legislative Council Congressional District Geospatial Data</a>
2. <a href="https://api.census.gov/data/2020/dec/pl">2020 Decennial Census Redistricting PL-94-171 API</a>
   * Although this API is capable of providing Texas census block-level voter age population racial composition breakdowns, it does not provide any data on the citizen voting age population. In order to be able to register to vote in Texas, one has to be a US citizen (Voter Registration Eligibility in Texas | VoteTexas.gov, 2024).
   By including non-citizens,
   .
3. <a href="https://www2.census.gov/geo/tiger/TIGER2020PL/LAYER/tl_2020_48_tabblock20.zip">2020 Texas U.S. Census Blocks Geospatial Data</a>
4. <a href="https://data.capitol.texas.gov/dataset/comprehensive-election-datasets-compressed-format/resource/e1cd6332-6a7a-4c78-ad2a-852268f6c7a2">Texas Legislative Council 2024 Voting Districts General Election Data</a>
5. <a href="https://data.capitol.texas.gov/dataset/4d8298d0-d176-4c19-b174-42837027b73e/resource/906f47e4-4e39-4156-b1bd-4969be0b2780/download/vtds_24pg.zip">Texas Legislative Council 2024 General Elections Voting Districts Geospatial Data</a>

## Data Cleaning
1. <a href="https://data.capitol.texas.gov/dataset/planc2335/resource/3552af40-54c1-45f2-9b02-b3c560bc0879">Texas Legislative Council Congressional District Geospatial Data</a>
    * Turn shapefiles into geopackage files with GeoJSON.
    * Convert to the global web maps coordinate reference system.
    * Snake-case column names.
    * Check for and fix invalid geometry with GeoJSON and Shapely.
    * Sort and index districts/blocks.
    * To compute needed derived attributes (i.e. area, perimeter, compactness), convert to an equal area coordinate reference system (i.e. Texas Centric Albers Equal Area) with GeoJSON.
    * Compute needed derived attributes (i.e. area, perimeter, compactness scores).
2.  <a href="https://api.census.gov/data/2020/dec/pl">2020 Decennial Census Redistricting PL-94-171 API</a>
    *
    * Create a GeoID to join with census block geospatial data.
3.  <a href="https://www2.census.gov/geo/tiger/TIGER2020PL/LAYER/tl_2020_48_tabblock20.zip">2020 Texas U.S. Census Blocks Geospatial Data</a>
4.  <a href="https://data.capitol.texas.gov/dataset/comprehensive-election-datasets-compressed-format/resource/e1cd6332-6a7a-4c78-ad2a-852268f6c7a2">Texas Legislative Council 2024 Voting Districts General Election Data</a>
    * Filter to the top-of-the-ticket office (President).
    * Group data by precinct and compute vote shares.
    * Validate vote share totals against those on <a href="https://redistrictingdatahub.org/dataset/texas-2024-general-election-precinct-level-texas-vtd-results-and-boundaries/">Redistricting Data Hub</a>.
    * Map results onto VTD geospatial data in order to visualize partisanship across precincts. Join on vtd key attribute.
5.  <a href="https://data.capitol.texas.gov/dataset/4d8298d0-d176-4c19-b174-42837027b73e/resource/906f47e4-4e39-4156-b1bd-4969be0b2780/download/vtds_24pg.zip">Texas Legislative Council 2024 General Elections Voting Districts Geospatial Data</a>


## Data Merging
* Joining Election Results to Precinct
* Overlaying precincts and Census Geography on the Current District Map (PLANC2335)
* Aggregating Election & Census Data to Congressional Districts

## Generating Test Data
* GerryChain can generate alternative district maps using Markov chains (Overview of the Chain, 2022).