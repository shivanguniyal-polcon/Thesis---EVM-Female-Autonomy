## Process Summary
Builds a geospatially accurate 1991 India census map by merging historical GeoJSON boundaries with CSV census statistics. After standardizing district names and linking geometries via a master mapping, it exports a validated GeoJSON containing 452 district polygons (excluding J&K) with attached census attributes.
### Summary Outcome (Data)
1. **Numbers**: Final output contains 452 district polygons matching 1991 census boundaries. All districts have valid pc91_state_id and pc91_district_id. Census data successfully merged via inner join on state/district IDs. J&K districts excluded due to missing census data.
2. **What it Proves**: Historical boundary geometries can be successfully harmonized with 1991 census identifiers through systematic name normalization. The extensive dictionary of historical→modern name changes (especially for Tamil Nadu, North East states, and reorganized states) enables accurate temporal bridging.
3. **What Still Needs Proving**: Micro-level boundary accuracy verification against original 1991 survey records. Some minor districts may require manual verification if automated name matching failed.
### Detailed Sequential Analysis
**Step 1: Load Data Sources (`pc91_pca_clean_pc91dist.csv`, `Master_District_Mapping_1991.csv`, `India-State-Districts-1991.geojsonl`)**
- **Data**:
  - `pc91_pca_clean_pc91dist.csv`: Cleaned 1991 census PCA data with district-level statistics
  - `Master_District_Mapping_1991.csv`: District ID mapping from Project A
  - `India-State-Districts-1991.geojsonl`: Historical district boundary geometries (one JSON object per line)
- **Inference**: Three complementary datasets provide geometry, ID mapping, and statistical attributes. GeoJSONL format requires line-by-line parsing rather than standard GeoJSON loading.
