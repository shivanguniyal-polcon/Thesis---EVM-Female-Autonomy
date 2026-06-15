## Process Summary
Creates a spatial crosswalk linking 2004 Parliamentary Constituencies to 1991 districts using area-based weights. By projecting boundary datasets to EPSG:6933 and performing a spatial intersection, it calculates and normalizes intersection areas. Ensuring each PC's weights sum to exactly 1.0, the pipeline exports a validated CSV crosswalk for proportional data allocation.
### Summary Outcome (Data)
1. **Numbers**: Final crosswalk contains one row per PC-district intersection slice. Each row includes pc_code, pc91_district_id, and pc_weight (0-1). Validation confirms Σ(pc_weight) = 1.0 for every PC within floating-point precision. Intersection produces expected number of slices based on geographic reality of boundary overlaps.
2. **What it Proves**: Spatial intersection methodology successfully bridges incompatible administrative boundary systems. Equal-area projection (EPSG:6933) ensures accurate weight calculations. Normalization guarantees no data loss or duplication when allocating PC-level statistics to 1991 districts.
3. **What Still Needs Proving**: Temporal validity of weight assumptions (2004→1991 allocation assumes population density uniformity within slices). Micro-level validation against actual constituency composition records would strengthen confidence.
### Detailed Sequential Analysis
**Step 1: Load Pristine Geospatial Datasets (`Pristine_Census_Map_1991_Final.geojson`, `PC_2004_Data_from_ARCGIS.geojson`)**
- **Data**:
  - `Pristine_Census_Map_1991_Final.geojson`: 1991 district boundaries from Project B (452 districts, excluding J&K)
  - `PC_2004_Data_from_ARCGIS.geojson`: 2004 Parliamentary Constituency boundaries from ARC-GIS source
  - Both contain polygon geometries with administrative identifiers
- **Inference**: Two temporally distinct boundary systems require spatial harmonization. Input quality depends on Projects A/B outputs and ARC-GIS source accuracy.
