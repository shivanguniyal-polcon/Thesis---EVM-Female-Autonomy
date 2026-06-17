# Data Sources and Descriptions

## 1. Demographic & Economic Control Data
These files provide the core baseline characteristics for each district, allowing you to control for geographic, social, and economic variation.

* **`pc91_pca_clean_pc91dist.csv`**
    * **Description:** The 1991 Primary Census Abstract (PCA). Used to calculate baseline demographics like Literacy Rate (`Lit_Pct`), Scheduled Caste percentage (`SC_Pct`), Scheduled Tribe percentage (`ST_Pct`), and the Child Sex Ratio proxy for Patriarchy.
    * **Source:** Socioeconomic High-resolution Rural-Urban Geographic Dataset (SHRUG) / Development Data Lab (DDL), derived from the 1991 Census of India.

* **`pc91_td_clean_pc91dist.csv`**
    * **Description:** The 1991 Town Directory. Used to calculate the percentage of the population living in urban areas (`Urban_Pct`).
    * **Source:** SHRUG / Development Data Lab (DDL).

* **`ec98_aggregated_1991_districts.csv`**
    * **Description:** The 1998 Economic Census. This is the crucial file that provides your primary moderator: Female Enterprise Density (`Fem_Enterprise_Pct`), serving as the proxy for economic agency.
    * **Source:** SHRUG / Development Data Lab (DDL), derived from the Ministry of Statistics and Programme Implementation (MoSPI).

## 2. Electoral Data (The Treatment & Dependent Variables)
These files contain the raw parliamentary voting statistics and act as the primary dependent variables (Turnout) and treatment indicators (EVM exposure) for the DiD models.

* **`1996_election_data_cleaned.csv`** 
    * **Description:** Pre-treatment baseline.

* **`1998_election_data_cleaned.csv`** 
    * **Description:** Placebo pre-trend check.

* **`1999_election_data_corrected.csv`** / **`1999_election_data_cleaned.csv`**
    * **Description:** Main EVM treatment rollout. PC-level electoral outcomes containing the counts of Female Electors and Female Voters. The 1999 file was used to definitively tag the 46 PCs that received EVM machines.
    * **Source:** Election Commission of India (ECI) Statistical Reports. *(Note: Cleaned structural versions of this data are frequently sourced via the Trivedi Centre for Political Data (TCPD) / Lok Dhaba database).*

## 3. Geographic & Spatial Engineering Data
These files were required to build the mathematical "Rosetta Stone" that linked the mismatched electoral boundaries with the census boundaries.

* **`shrid_loc_names.csv`** & **`shrid_pc91dist_key.csv`**
    * **Description:** Granular translation dictionaries mapping text-based location names to official numeric 1991 Census IDs.
    * **Source:** SHRUG / Development Data Lab (DDL).

* **`India-State-Districts-1991.geojsonl`**
    * **Description:** Raw spatial polygons representing the historical 1991 district borders.
    * **Source:** Likely an open-source mapping repository (such as Data{Meet}) or native SHRUG shapefiles.

* **`PC_2004_Data_from_ARCGIS.geojson`**
    * **Description:** Modern spatial boundaries for Parliamentary Constituencies (acting as a close proxy for 1999 borders), used in the geospatial overlay to cut intersection slices.
    * **Source:** ArcGIS / Survey of India / ECI spatial datasets.
