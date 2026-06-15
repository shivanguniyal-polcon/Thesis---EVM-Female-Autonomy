## Process Summary
The master district mapping workflow bridges 1991 census district identifiers with modern administrative boundaries through a systematic data integration process. The workflow includes: (1) Loading two SHRUG CSV files containing location names and district keys; (2) Merging datasets using the `shrid2` identifier as a bridge variable; (3) Deduplicating records to ensure one-to-one mappings; and (4) Exporting a clean reference table. We validate merge completeness and check for duplicate district-ID combinations. Outputs include a master mapping CSV file linking state/district names to 1991 census IDs.
### Summary Outcome (Data)
1. **Numbers**: Successful merge on `shrid2` produces unified dataset. Deduplication yields unique combinations of state_name, district_name, pc91_state_id, and pc91_district_id. First 10 rows displayed for verification confirm valid mappings across multiple states.
2. **What it Proves**: The SHRUD `shrid2` identifier successfully bridges modern location names with historical 1991 census district keys. Inner join ensures only records with valid mappings on both sides are retained, preventing orphaned records.
3. **What Still Needs Proving**: Historical name variations and boundary changes between 1991-present require additional normalization (addressed in Project B). J&K districts may need separate handling due to data availability constraints.
### Detailed Sequential Analysis
**Step 1: Load SHRUG Data Files (`shrid_loc_names.csv`, `shrid_pc91dist_key.csv`)**
- **Data**:
  - `shrid_loc_names.csv`: Contains location names with `shrid2` identifiers, state_name, and district_name columns
  - `shrid_pc91dist_key.csv`: Contains bridge between `shrid2` and 1991 census identifiers (pc91_state_id, pc91_district_id)
- **Inference**: Two complementary datasets provide the foundation for linking modern administrative names to historical census IDs through a common key.
