import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import numpy as np

# 1. Page Config & Title
st.set_page_config(page_title="Module 1: Spatial Crosswalk", layout="wide")
st.title("Module 1: Spatial Data Construction")
st.header("Step 1: Cookie-Cutter Method for MAUP Resolution")

# 2. Create the two tabs
tab_results, tab_code = st.tabs(["📊 Results", "💻 Core Code"])

# ---------------------------------------------------------
# TAB 1: THE RESULTS
# ---------------------------------------------------------
with tab_results:
    st.markdown("""
    ### Solving the Modifiable Areal Unit Problem (MAUP)
    
    When district boundaries change over time (e.g., 1991 → 1996 → 2000), 
    comparing Polling Center (PC) level data becomes impossible without spatial harmonization.
    
    **The Cookie-Cutter Solution:**
    1. Intersect PC polygons with district boundaries for each time period
    2. Calculate the proportion of each PC's area falling into each district
    3. Weight PC-level outcomes by these proportions when aggregating to districts
    
    This ensures that a PC split across two districts contributes proportionally to both.
    """)
    
    # Create sample visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Before: Misaligned Boundaries")
        st.image("https://via.placeholder.com/400x300?text=PC+Polygons+Overlapping+Districts", 
                 caption="Polling Centers don't align with district boundaries")
    
    with col2:
        st.markdown("#### After: Cookie-Cutter Weights")
        st.image("https://via.placeholder.com/400x300?text=Weighted+PC+Assignments", 
                 caption="Each PC weighted by area proportion in each district")
    
    st.markdown("### Weight Calculation Formula")
    st.latex(r"""
    w_{pc,d} = \frac{\text{Area}(PC_{pc} \cap District_d)}{\text{Area}(PC_{pc})}
    """)
    
    st.markdown("""
    Where:
    - $w_{pc,d}$ = weight of polling center $pc$ in district $d$
    - $\text{Area}(PC_{pc} \cap District_d)$ = intersection area
    - $\sum_d w_{pc,d} = 1$ for each PC (weights sum to 1)
    """)
    
    # Sample data table
    st.markdown("### Example: PC Weight Assignments")
    sample_weights = pd.DataFrame({
        'PC_ID': ['PC001', 'PC002', 'PC003', 'PC004'],
        'District_1991': ['D1', 'D1', 'D2', 'D2'],
        'District_1996': ['D1', 'D1_new', 'D1_new', 'D2'],
        'Weight_1991': [1.0, 1.0, 1.0, 1.0],
        'Weight_1996': [0.7, 0.6, 0.8, 1.0]
    })
    st.dataframe(sample_weights, use_container_width=True)
    
    st.info("✅ **Result**: All subsequent analyses use these spatially-weighted PC assignments, ensuring comparability across time periods despite boundary changes.")


# ---------------------------------------------------------
# TAB 2: THE CORE CODE
# ---------------------------------------------------------
with tab_code:
    st.markdown("### Under the Hood: Geospatial Implementation")
    
    # This automatically reads the exact file it is sitting in!
    with open(__file__, "r") as f:
        source_code = f.read()
        
    st.code(source_code, language="python")
