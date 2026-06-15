import pandas as pd

# 1. Load both SHRUG files
loc_names = pd.read_csv('/Users/ganeshchandrauniyal/Desktop/Thesis Script/shrug-shrid-keys-csv/shrid_loc_names.csv')
key_df = pd.read_csv('/Users/ganeshchandrauniyal/Desktop/Thesis Script/shrug-pc-keys-csv/shrid_pc91dist_key.csv')

# 2. Merge them together using 'shrid2' as the bridge
merged_key = pd.merge(
    loc_names[['shrid2', 'state_name', 'district_name']], 
    key_df, 
    on='shrid2', 
    how='inner'
)

# 3. Drop all the duplicate villages to create a clean, unique dictionary of Districts
master_mapping = merged_key.drop_duplicates(
    subset=['state_name', 'district_name', 'pc91_state_id', 'pc91_district_id']
)

# 4. Display the results
print("--- Master Mapping Created! ---")
print(master_mapping[['state_name', 'district_name', 'pc91_state_id', 'pc91_district_id']].head(10))

# 5. Save this master key for future use!
master_mapping.to_csv('/Users/ganeshchandrauniyal/Desktop/Thesis Script/Master_District_Mapping_1991.csv', index=False)

import pandas as pd
import geopandas as gpd
import json
from shapely.geometry import shape

# 1. Load Data
df_census = pd.read_csv('pc91_pca_clean_pc91dist.csv')
df_master_key = pd.read_csv('Master_District_Mapping_1991.csv')

features = []
with open('India-State-Districts-1991.geojsonl', 'r') as f:
    for line in f:
        item = json.loads(line)
        geom = shape(item['geometry'])
        features.append({'geometry': geom, **item['properties']})

gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")

# 2. Base Text Cleaning
gdf['district_clean'] = gdf['District'].str.lower().str.strip()
gdf['state_clean'] = gdf['State'].str.lower().str.strip()

# 3. State-Level Bridge (Historical to Modern)
state_replacements = {
    "orissa": "odisha",
    "jammu and kashmir": "jammu kashmir",
    "andaman and nicobar islands": "andaman nicobar islands",
    "dadra and nagar haveli": "dadra nagar haveli",
    "daman and diu": "daman diu",
    "delhi": "nct of delhi",
    "pondicherry": "puducherry"
}
gdf['state_clean'] = gdf['state_clean'].replace(state_replacements)

# 4. The Final Missing 10 District Bridge
district_replacements = {
    # Catching the edits you made + the SHRUG expectations
    "delhi": "north west",
    "imphal": "imphal west",
    "nellore": "sri potti sriramulu nellore",
    "spsr nellore": "sri potti sriramulu nellore",
    "cuddapah": "ysr kadapa",
    "kadapa": "ysr kadapa",
    "phulabani": "kandhamal",
    "dadra and nagar haveli": "dadra nagar haveli",
    "greater bombay": "mumbai suburban",
    "mumbai city": "mumbai suburban",
    "pondicherry": "puducherry",
    "andaman and nicobar islands": "south andaman",
    "calcutta": "kolkata"
}
gdf['district_clean'] = gdf['district_clean'].replace(district_replacements)

# 5. Bridge 1: Merge on District Name 
df_master_unique = df_master_key.drop_duplicates(subset=['district_name'])

gdf_with_ids = gdf.merge(
    df_master_unique,
    left_on='district_clean',
    right_on='district_name',
    how='left'
)

# 6. Bridge 2: Attach Census Data
# Drop J&K and the 'NaN' polygon so they don't break the integer conversion
gdf_clean = gdf_with_ids.dropna(subset=['pc91_district_id']).copy()

gdf_clean['pc91_state_id'] = gdf_clean['pc91_state_id'].astype(int)
gdf_clean['pc91_district_id'] = gdf_clean['pc91_district_id'].astype(int)

df_census['pc91_state_id'] = pd.to_numeric(df_census['pc91_state_id'], errors='coerce').fillna(0).astype(int)
df_census['pc91_district_id'] = pd.to_numeric(df_census['pc91_district_id'], errors='coerce').fillna(0).astype(int)

final_gdf = gdf_clean.merge(
    df_census,
    on=['pc91_state_id', 'pc91_district_id'],
    how='inner'
)

# 7. Validation
print(f"Total Polygons matched with Census Data: {len(final_gdf)}")
print("Expected Target: 451 to 452 (Excluding J&K)")

# Export the pristine map
final_gdf.to_file('Pristine_Census_Map_1991.geojson', driver='GeoJSON')

