Texas Legislative Council
Research Division
FTP Directory


2020 CENSUS DATA 
https://data.capitol.texas.gov/dataset/2020-census-geography

This directory contains 2020 Census geographic and population data.


GEOGRAPHY FOLDER

The United States Census Bureau publishes geographic units used for tabulation of
the 2020 Census population data in the 2020 TIGER/Line Shapefile.  The geographic
units, which remain constant throughout the decade, include counties, census tracts,
block groups, and blocks.  Fields have been added so data formatted or published
by the council can be joined to the shapefile for analysis.

Each Shapefile (.shp) is in a compressed file (.zip) format.

Blocks.zip - Census Blocks
BlockGroups.zip - Block Groups
Tracts.zip - Census Tracts
Cities.zip - Census Places (Cities)
CDPs.zip - Census Designated Places
Counties.zip - Counties

The geographic data was extracted from the 2020 Census TIGER/Line Shapefile.


POPULATION FOLDER

In accordance with Public Law 94-171, the Census Bureau is required to provide the
states with the official census population numbers needed for redistricting,
including total and voting age population by race and ethnicity for every
census geographic level.

The council groups the population data into five race and ethnicity categories for
redistricting modeling and reporting:

ASIAN - Encompasses all people identifying themselves as Asian Indian, Chinese,
Filipino, Japanese, Korean, Vietnamese, or Other Asian on the census questionnaire,
even if they also identified themselves with other racial/ethnic groups.

BLACK - Encompasses all people identifying themselves as Black, African American,
or Negro on the census questionnaire, even if they also identified themselves with
other racial/ethnic groups.

HISPANIC - Encompasses all people identifying themselves as Hispanic, Latino, or
Spanish origin, whatever their race.

BLACK+HISPANIC - A combined total of all those identifying themselves as Black
and all those identifying themselves as Hispanic, adjusted so that those
identifying themselves as both Black and Hispanic are not counted twice.

ANGLO - Includes all people who selected White as their only race and did not
identify themselves as Hispanic, Latino, or Spanish origin.

NON-ANGLO - A combined total of all those not included in the Anglo group, adjusted
so that those identififying themselves with more than one group are not counted twice.
Anlgo plus Non-Anglo equals the total population.

VAP - Voting age population, persons in a geographic unit who are at least 18 years of age.


Each comma-separated values file (.csv) contains the 2020 Census population for each
geographic level in a compressed file (.zip) format.

BlocksPop.zip - Census Blocks 2020 Census Population

Join to geographic data on 'blkkey'.

trt (num) - Census Tract
block (num) - Census Block
anglo (num) - Anglo Population
anglovap (num) - Anglo Voting Age Population
hispanic (num) - Hispanic Population
hispanicvap (num) - Hispanic Voting Age Population
total (num) - Total Population
vap (num) - Voting Age Population
bh (num) - Black+Hispanic Population
bhvap (num) - Black+Hispanic Voting Age Population
black (num) - Black Population
blackvap (num) - Black Voting Age Population
asian (num) - Asian Population
asianvap (num) - Asian Voting Age Population
nanglo (num) - Non-Anglo Population
nanglovap (num) - Non-Anglo Voting Age Population
cntykey (num) - Unique code used to join to geographic data.
blkkey (num) - Unique code used to join to geographic data.
fips (num) - FIPS Census County Code


BlockGroupPop.zip - Census Block Groups 2020 Census Population

Join to geographic data on 'bgkey'.

bgkey (num) - Unique code used to join to geographic data.
cntykey (num) - Unique code used to join to geographic data.
fips (num) - FIPS Census CountyCode
trtkey (num) - Unique code used to join to geographic data.
anglo (num) - Anglo Population
anglovap (num) - Anglo Voting Age Population
hispanic (num) - Hispanic Population
hispanicvap (num) - Hispanic Voting Age Population
total (num) - Total Population
vap (num) - Voting Age Population
bh (num) - Black+Hispanic Population
bhvap (num) - Black+Hispanic Voting Age Population
black (num) - Black Population
blackvap (num) - Black Voting Age Population
asian (num) - Asian Population
asianvap (num) - Asian Voting Age Population
nanglo (num) - Non-Anglo Population
nanglovap (num) - Non-Anglo Voting Age Population
bg (num) - Block Group


TractsPop.zip - Census Tracts 2020 Census Population

Join to geographic data on 'trtkey'.

trtkey (num) - Unique code used to join to geographic data.
cntykey (num) - Unique code used to join to geographic data.
trt (num) - Census Tract
anglo (num) - Anglo Population
anglovap (num) - Anglo Voting Age Population
hispanic (num) - Hispanic Population
hispanicvap (num) - Hispanic Voting Age Population
total (num) - Total Population
black (num) - Black Population
blackvap (num) - Black Voting Age Population
asian (num) - Asian Population
asianvap (num) - Asian Voting Age Population
nanglo (num) - Non-Anglo Population
nanglovap (num) - Non-Anglo Voting Age Population
bh (num) - Black+Hispanic Population
vap (num) - Voting Age Population
bhvap (num) - Black+Hispanic Voting Age Population


CountiesPop.zip - Counties 2020 Census Population

Join to geographic data on 'cntykey'.

anglo (num) - Anglo Population
anglovap (num) - Anglo Voting Age Population
hispanic (num) - Hispanic Population
hispanicvap (num) - Hispanic Voting Age Population
total (num) - Total Population
vap (num) - Voting Age Population
bh (num) - Black+Hispanic Population
bhvap (num) - Black+Hispanic Voting Age Population
black (num) - Black Population
blackvap (num) - Black Voting Age Population
asian (num) - Asian Population
asianvap (num) - Asian Voting Age Population
nanglo (num) - Non-Anglo Population
nanglovap (num) - Non-Anglo Voting Age Population
cntykey (num) - Unique code used to join to geographic data.
fips (num) - FIPS Census County Code



Last modified on August 17, 2021.