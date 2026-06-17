# EVM & Female Autonomy Analytical Pipeline
# Copyright (C) 2026 Shivang Uniyal
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

import geopandas as gpd
import pandas as pd

# [CORE START]

#1. Load the pristine datasets
print("Loading geometries...")
dist_gdf = gpd.read_file('/Users/ganeshchandrauniyal/Desktop/Thesis Script/Pristine_Census_Map_1991_Final.geojson')
pc_gdf = gpd.read_file('/Users/ganeshchandrauniyal/Desktop/Thesis Script/Data/PC_2004_Data_from_ARCGIS.geojson')

#2. Project to Cylindrical Equal-Area (EPSG:6933)
# This is mandatory for accurate square-meter area calculations in India
print("Projecting to Equal-Area...")
dist_proj = dist_gdf.to_crs(epsg=6933)
pc_proj = pc_gdf.to_crs(epsg=6933)

#3. THE COOKIE CUTTER (Spatial Intersection)
# We use intersection to find where PCs and Districts overlap
intersections = gpd.overlay(pc_proj, dist_proj, how='intersection')

#4.Calculate the Area of the Slices
intersections['slice_area'] = intersections.geometry.area

#5. Calculate Total Mapped Area per PC
#We group by the PC to find out how much of its area successfully landed inside districts
pc_total_areas = intersections.groupby('pc_code')['slice_area'].sum().reset_index()
pc_total_areas.rename(columns={'slice_area': 'pc_total_mapped_area'}, inplace=True)

#Merge the total mapped area back into the intersections
intersections = intersections.merge(pc_total_areas, on='pc_code')

#6. Calculate the Normalized Weight
# (Slice Area / Total Mapped Area of that specific PC)
# This guarantees that the sum of weights for every PC equals exactly 1.0
intersections['pc_weight'] = intersections['slice_area'] / intersections['pc_total_mapped_area']

#VALIDATION
validation = intersections.groupby('pc_name')['pc_weight'].sum().reset_index()
print("\nWeight Validation Check (Every PC should sum to 1.0):")
print(validation.head())

#7. Create the Final Master Crosswalk
#Drop the heavy geometries and keep only the linking keys and weights
crosswalk_cols = [
    'st_name', 'pc_name', 'pc_code',        # 2004 PC Identifiers
    'pc91_state_id', 'pc91_district_id',    # 1991 District Identifiers
    'state_clean', 'district_clean',        # District Names
    'pc_weight'                             # The Multiplier
]

final_crosswalk = pd.DataFrame(intersections[crosswalk_cols])

#Sort for readability
final_crosswalk = final_crosswalk.sort_values(by=['st_name', 'pc_name', 'pc_weight'], ascending=[True, True, False])

# Export to CSV
print(f"\nSuccessfully created crosswalk with {len(final_crosswalk)} intersection slices.")
final_crosswalk.to_csv('/Users/ganeshchandrauniyal/Desktop/Thesis Script/PC2004_to_Dist1991_Weightage_Crosswalk.csv', index=False)

# [CORE END]
