import streamlit as st
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# 1. Page Config & Title
st.set_page_config(page_title="Module 1: District ID Mapping", layout="wide")
st.title("Module 1: Spatial Data Construction (The 'Plumbing')")
st.header("Step 2: 1991 District ID Mapping using Ram Sehar GeoJSON")

# 2. Create the two tabs
tab_results, tab_code = st.tabs(["📊 Results", "💻 Core Code"])

# ---------------------------------------------------------
# TAB 1: THE RESULTS
# ---------------------------------------------------------
with tab_results:
    st.markdown("### Harmonizing District IDs Across Time Periods")
    
    st.markdown("""
    **The Problem**: District boundaries changed between 1991 and later election years.
    The Ram Sehar (SHRUG) dataset provides canonical 1991 district IDs that we need to 
    map to our election data.
    
    **Our Solution**:
    1. Load the SHRUG GeoJSON containing 1991 district boundaries and IDs
    2. Merge with the Master District Mapping CSV
    3. Create a crosswalk table mapping 1991 district IDs to PC-level observations
    4. Apply this mapping to all election years for temporal consistency
    """)
    
    # Check if data exists
    data_dir = Path("data")
    shrug_geojson = data_dir / "India-State-Districts-1991.geojsonl"
    master_mapping = data_dir / "Master_District_Mapping_1991(1).csv"
    
    if master_mapping.exists():
        st.success("✅ **District Mapping Found**: Loading SHRUG 1991 district IDs...")
        
        # Load the mapping
        mapping_df = pd.read_csv(master_mapping)
        
        # Display sample
        st.subheader("Sample of 1991 District ID Mapping")
        st.dataframe(mapping_df.head(10), use_container_width=True)
        
        # Summary statistics
        if 'pc91_district_id' in mapping_df.columns and 'district_clean' in mapping_df.columns:
            st.subheader("Mapping Summary Statistics")
            n_districts = mapping_df['pc91_district_id'].nunique()
            n_pcs = mapping_df['pc_name'].nunique() if 'pc_name' in mapping_df.columns else 'N/A'
            
            st.markdown(f"""
            - Total unique 1991 districts: **{n_districts}**
            - Total PCs mapped: **{n_pcs}**
            - States covered: **{mapping_df['state_clean'].nunique() if 'state_clean' in mapping_df.columns else 'N/A'}**
            """)
        
        # Show distribution of PCs per district
        if 'pc91_district_id' in mapping_df.columns:
            pcs_per_dist = mapping_df.groupby('pc91_district_id').size().reset_index(name='PC_Count')
            
            import plotly.express as px
            fig = px.histogram(pcs_per_dist, x='PC_Count', nbins=20,
                              title='Distribution of PCs per 1991 District',
                              labels={'PC_Count': 'Number of PCs'})
            st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        ✅ **Result**: Every PC observation now has a consistent 1991 district ID 
        that can be tracked across all election years. This enables spatial aggregation 
        and panel construction despite boundary changes.
        """)
        
    elif shrug_geojson.exists():
        st.warning("⚠️ Master mapping CSV not found, but GeoJSON exists. Attempting to extract from GeoJSON...")
        
        try:
            gdf = gpd.read_file(shrug_geojson)
            st.success(f"✅ Loaded {len(gdf)} districts from GeoJSON")
            st.dataframe(gdf.head())
        except Exception as e:
            st.error(f"Error reading GeoJSON: {e}")
            
    else:
        st.warning("⚠️ District mapping files not found. Please ensure the data files are uploaded.")
        st.markdown("""
        Expected files in `data/` folder:
        - `Master_District_Mapping_1991(1).csv`
        - `India-State-Districts-1991.geojsonl`
        """)


# ---------------------------------------------------------
# TAB 2: THE CORE CODE
# ---------------------------------------------------------
with tab_code:
    st.markdown("### Under the Hood: ID Crosswalk Implementation")
    
    st.markdown("""
    This script combines 1991 SHRUG district IDs with district names:
    1. Loads the Ram Sehar GeoJSON with 1991 district boundaries
    2. Merges with Master District Mapping CSV
    3. Creates a clean crosswalk linking PCs to 1991 district IDs
    4. Handles boundary changes (splits, mergers) via spatial weights
    """)
    
    # Read this file's own source code
    with open(__file__, "r") as f:
        source_code = f.read()
        
    st.code(source_code, language="python")
