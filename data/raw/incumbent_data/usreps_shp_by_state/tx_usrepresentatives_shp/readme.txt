2020 TX U.S. Representatives Incumbent Address

## RDH Date Retrieval
04/07/2021

## RDH Upload Date
04/10/2021

## Sources
The RDH received all incumbent address data from Dr. Carl E. Klarner (klarnerpolitics.org/bio-1).

## Fields

**Fields:**		**Descriptions:**
state			Name of state
sfips			Census state fips code
dno			Number of Congressional district+
vac			1 = No US Representative currently represents the district, 0 = else.
last			Last name of US Representative, no hyphens or spaces
first			First name of US Representative
middle			Middle name of US Representative
nick			Nickname of US Representative
suffix			Suffix of US Representative
last1			Last name components of US Representative when US Rep has or may have a compound last name
last2			Last name components of of US Representative when US Rep has or may have a compound last name
ltype1			Last name component type of US Representative in last1 
ltype2			Last name component type of state US Representative in last2
byear			Year of birth of US Representative
bmonth			Month of birth of US Representative
bday			Day of birth of US Representative
party			Political party of US Representative - Possible values: d: Democrat, r: Republican
gender			Possible values f: female, m: male, t: transgender (not observed)
race			Race of US Representative++
raceeth			Usually not coded (Null), sometimes gives more detailed statement of a legislator's heritage (i.e., "Persian" instead of "MENA" etc.).
outsidecd		Number of Congressional district the US Representative was found in as a registrant
res_line1		First line of US Representative's residence address
res_extrli		Extra line below first of US Representative's residence address
res_city			City/town of US Representative's residence address
res_state		State of US Representative's residence address
res_zip			Zip code of US Representative's residence address
res_zippl4		Last 4 of extended zip code for US Representative's residence address
res_ad_lat		US Representative's residence address latitude
res_ad_lon		US Representative's residence address longitude
geometry		GIS point geometry of US Representative's residence address 

## Processing Steps
The RDH processed all of the US Representative residence latitudes and longitudes in QGIS to create point geometries and exported the data as a shapefile. The shapefile was then processed in Python using the Pandas and Geopandas packages. Shapefile fields are limited to 10 characters, so the RDH renamed all fields that exceeded 10 characters to prepare for shapefile processing. 

Dr. Klarner's original dataset contained documentation fields explaining each row. The RDH removed documentation fields from the csv for simplicity, to reduce the number of fields. Relevant documentation is provided below in "Additional Notes". For more information, please reach out to the RDH help desk (info@redistrictingdatahub.org).

The original dataset from Dr. Klarner contained all US states in one file. The RDH processed all the the data, split the file by state and exported data for each state as a separate shapefile using Python and the GeoPandas library.

## Additional Notes
Of the 435 US Representative seats, the RDH has address data from Dr. Klarner for 422. 
At the time of data collection, three seats were vacant, and 10 addresses could not be found. States with missing address information include - 
AK, VT, IA, CO, MS, LA, NY, CA, FL, IL, LA, TX. 

Louisiana has two vacant seats, and Texas has one. 

The remaining 10 missing addresses are are missing because Dr. Klarner could not confirm with confidence that the individual matched the given address, or could not find any address information at all. 

Alaska and Vermont are at-large states. Because the one representative does not have address information available, a shapefile for these states has not been provided.


+Note that if a state only has one Congressional district (e.g., Wyoming), a "1" appears, not characters indicating an at-large district (i.e., "AL")

++Race codes below:
a: Asian, east-asian specifically, also Pacific Islands (not including Hawaiians)
b: African-American/black
c: Cuban (rarely coded)
h: Hispanic/latino (includes Cuban)
m: middle-eastern/North African
n: Native-American/Hawaiian
s: South-Asian: Afghanistan, Pakistan, India, Nepal, Bhutan, Sri Lanka, Bangladesh, not Burma
u: Race missing (one case in entire US)
w: white/non-minority
Multiple codes can be given for one person. Every Cuban also coded "h" so the code "c" always appears as "ch"

Please contact info@redistrictingdatahub.org for more information.
