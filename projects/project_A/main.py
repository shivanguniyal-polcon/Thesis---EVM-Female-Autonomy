import pandas as pd

# [CORE START]
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
master_mapping = merged_key.drop_duplicates(subset=['state_name', 'district_name', 'pc91_state_id', 'pc91_district_id']
)

# 4. Display the results
print(master_mapping[['state_name', 'district_name', 'pc91_state_id', 'pc91_district_id']].head(10))

# 5. Save this master key for future use!
master_mapping.to_csv('/Users/ganeshchandrauniyal/Desktop/Thesis Script/Master_District_Mapping_1991.csv', index=False)

[# [CORE END]
