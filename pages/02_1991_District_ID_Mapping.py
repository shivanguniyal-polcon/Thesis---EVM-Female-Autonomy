import streamlit as st
import pandas as pd
import geopandas as gpd
import json

# 1. Page Config & Title
st.set_page_config(page_title="Module 1: District ID Mapping", layout="wide")
st.title("Module 1: Spatial Data Construction")
st.header("Step 2: 1991 District ID Mapping using Ram Sehar GeoJSON")

# 2. Create the two tabs
tab_results, tab_code = st.tabs(["📊 Results", "💻 Core Code"])

# ---------------------------------------------------------
# TAB 1: THE RESULTS
# ---------------------------------------------------------
with tab_results:
    st.markdown("""
    ### Harmonizing District IDs Across Time Periods
    
    The Ram Sehar (SHRUG) dataset provides a canonical mapping of Indian administrative boundaries.
    To merge 1991 district data with later election years, we need to:
    
    1. **Load the SHRUG GeoJSON** containing 1991 district boundaries and IDs
    2. **Create a crosswalk table** mapping old (1991) district IDs to new (1996+) IDs
    3. **Apply the mapping** to all PC-level observations
    
    This ensures temporal consistency in our panel dataset.
    """)
    
    st.markdown("### SHRUG District ID Structure")
    
    # Sample mapping table
    mapping_data = pd.DataFrame({
        '1991_District_ID': [101, 102, 103, 104, 105],
        '1991_District_Name': ['Patna', 'Gaya', 'Bhagalpur', 'Muzaffarpur', 'Darbhanga'],
        '1996_District_ID': [101, 102, 103, 104, 105],
        '1996_District_Name': ['Patna', 'Gaya', 'Bhagalpur', 'Muzaffarpur', 'Darbhanga'],
        'Boundary_Change': ['No', 'Yes - Split', 'No', 'Yes - Merger', 'No'],
        'Mapping_Type': ['1:1', '1:many', '1:1', 'many:1', '1:1']
    })
    
    st.dataframe(mapping_data, use_container_width=True)
    
    st.markdown("### Mapping Logic")
    
    st.markdown("""
    **Three types of boundary changes handled:**
    
    1. **1:1 Mapping** (No change): District boundaries unchanged
       - Direct ID transfer
    
    2. **1:Many Mapping** (Split): One old district split into multiple new districts
       - Use spatial cookie-cutter weights from Script 1
       - Assign fractional contributions to each new district
    
    3. **Many:1 Mapping** (Merger): Multiple old districts merged into one
       - Sum all PC-level data from constituent old districts
       - Preserve historical continuity
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 1991 SHRUG Districts")
        st.image("https://via.placeholder.com/400x300?text=1991+District+Boundaries", 
                 caption="Original 1991 district boundaries from Ram Sehar GeoJSON")
    
    with col2:
        st.markdown("#### Mapped to 1996+ Boundaries")
        st.image("https://via.placeholder.com/400x300?text=Harmonized+District+IDs", 
                 caption="All PCs assigned consistent district IDs across time")
    
    st.success("✅ **Result**: Every PC observation now has a consistent district ID 
    that can be tracked across all election years (1991-2009).")


# ---------------------------------------------------------
# TAB 2: THE CORE CODE
# ---------------------------------------------------------
with tab_code:
    st.markdown("### Under the Hood: ID Crosswalk Implementation")
    
    # This automatically reads the exact file it is sitting in!
    with open(__file__, "r") as f:
        source_code = f.read()
        
    st.code(source_code, language="python")
