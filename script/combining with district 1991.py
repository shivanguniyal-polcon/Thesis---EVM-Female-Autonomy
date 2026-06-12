import pandas as pd
import geopandas as gpd
import json
from shapely.geometry import shape

# 1. Load Data
df_census = pd.read_csv('/Users/ganeshchandrauniyal/Desktop/Thesis Script/shrug-pca91-csv/pc91_pca_clean_pc91dist.csv')
df_master_key = pd.read_csv('/Users/ganeshchandrauniyal/Desktop/Thesis Script/Master_District_Mapping_1991.csv')

features = []
with open('/Users/ganeshchandrauniyal/Desktop/Thesis Script/Data/India-State-Districts-1991.geojsonl', 'r') as f:
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

# 4. The COMPLETE District Bridge
district_replacements = {
    # Tamil Nadu
    "madras": "madurai",
    "chengalpattu mgr": "nagapattinam",
    "south arcot": "cuddalore",
    "north arcot ambedkar": "vellore",
    "tiruvannamalai sambuvarayar": "tiruvannamalai",
    "pasumpon muthuramalinga thevar": "sivaganga",
    "kamarajar": "karur",
    "chidambaranar": "coimbatore",
    "tirunelveli kattabomman": "tirunelveli",
    "dindigul anna": "dindigul",
    "periyar": "ariyalur",
    "tiruchchirappalli": "tiruchirappalli",
    "nilgiri": "the nilgiris",
    
    # North East
    "dhuburi": "dhubri",
    "dibang valley": "lower dibang valley",
    "marigaon": "morigaon",
    "sibsagar": "sivasagar",
    "north cachar hills": "cachar",
    "imphal": "imphal west",
    "chhimtuipui": "saiha",
    
    # Northern & Central
    "lahul and spiti": "lahul spiti",
    "dehra dun": "dehradun",
    "uttar kashi": "uttarkashi",
    "maharajganj": "mahrajganj",
    "nawadah": "nawada",
    "hazaribag": "hazaribagh",
    "raj nandgaon": "rajnandgaon",
    
    # West & South
    "greater bombay": "mumbai suburban",
    "mumbai city": "mumbai suburban",
    "rangareddi": "rangareddy",
    "nellore": "sri potti sriramulu nellore",
    "spsr nellore": "sri potti sriramulu nellore",
    "cuddapah": "ysr kadapa",
    "kadapa": "ysr kadapa",
    
    # East
    "west dinajpur": "uttar dinajpur",
    "medinipur": "purba medinipur",
    "calcutta": "kolkata",
    "phulabani": "kandhamal",
    
    # Union Territories
    "delhi": "north west",
    "dadra and nagar haveli": "dadra nagar haveli",
    "pondicherry": "puducherry",
    "andaman and nicobar islands": "south andaman"
}
gdf['district_clean'] = gdf['district_clean'].replace(district_replacements)

# 5. Bridge 1: Merge on District Name 
# Strictly deduplicating the SHRUG key prevents the 456 one-to-many geometry inflation
df_master_unique = df_master_key.drop_duplicates(subset=['district_name'])

gdf_with_ids = gdf.merge(
    df_master_unique,
    left_on='district_clean',
    right_on='district_name',
    how='left'
)

# 6. Bridge 2: Attach Census Data
# J&K districts will safely fall away here as NaNs
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
print("Expected Target: 452 (Excluding J&K)")

# Export the pristine map
final_gdf.to_file('/Users/ganeshchandrauniyal/Desktop/Thesis Script/Pristine_Census_Map_1991_Final.geojson', driver='GeoJSON')

# 1. Identify missing districts
all_districts = gdf['district_clean'].unique()
matched_districts = final_gdf['district_clean'].unique()

missing_districts = [d for d in all_districts if d not in matched_districts]

print(f"Number of missing districts: {len(missing_districts)}")
print("First 10 missing district names:")
print(missing_districts[:60])