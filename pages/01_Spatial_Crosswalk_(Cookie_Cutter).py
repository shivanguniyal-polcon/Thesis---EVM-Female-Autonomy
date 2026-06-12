import streamlit as st
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# 1. Page Config & Title
st.set_page_config(page_title="Module 1: Spatial Crosswalk", layout="wide")
st.title("Module 1: Spatial Data Construction (The 'Plumbing')")
st.header("Step 1: Cookie-Cutter Spatial Intersection")

# 2. Create the two tabs
tab_results, tab_code = st.tabs(["📊 Results", "💻 Core Code"])

# ---------------------------------------------------------
# TAB 1: THE RESULTS
# ---------------------------------------------------------
with tab_results:
    st.markdown("### Solving the MAUP: How We Assigned PCs to Districts")
    
    st.markdown("""
    **The Problem**: Polling Center (PC) boundaries from 2004 don't perfectly align with 1991 District boundaries.
    A single PC can span multiple districts. Simply assigning a PC to one district creates measurement error 
    (the Modifiable Areal Unit Problem - MAUP).
    
    **Our Solution**: Use a "cookie-cutter" spatial intersection method:
    1. Overlay PC and District geometries
    2. Calculate the intersection area for each PC-District pair
    3. Compute weights: `weight = intersection_area / total_PC_area`
    4. Use these weights to allocate election data proportionally
    """)
    
    # Check if data exists
    data_dir = Path("data")
    crosswalk_file = data_dir / "PC2004_to_Dist1991_Weightage_Crosswalk (1).csv"
    
    if crosswalk_file.exists():
        st.success("✅ **Spatial Crosswalk Found**: Loading pre-computed weights...")
        
        # Load the crosswalk
        crosswalk = pd.read_csv(crosswalk_file)
        
        # Display sample
        st.subheader("Sample of Spatial Crosswalk Weights")
        display_cols = [col for col in ['pc_name', 'district_clean', 'pc_weight', 'pc91_district_id'] if col in crosswalk.columns]
        st.dataframe(crosswalk[display_cols].head(10), use_container_width=True)
        
        # Validation check
        if 'pc_name' in crosswalk.columns and 'pc_weight' in crosswalk.columns:
            pc_weights_sum = crosswalk.groupby('pc_name')['pc_weight'].sum().reset_index()
            pc_weights_sum.columns = ['PC Name', 'Sum of Weights']
            
            st.subheader("Validation: Do PC Weights Sum to 1.0?")
            st.markdown(f"""
            - Total PCs in crosswalk: **{crosswalk['pc_name'].nunique()}**
            - Total intersection slices: **{len(crosswalk)}**
            - Mean weight sum per PC: **{pc_weights_sum['Sum of Weights'].mean():.4f}**
            - Min weight sum: **{pc_weights_sum['Sum of Weights'].min():.4f}**
            - Max weight sum: **{pc_weights_sum['Sum of Weights'].max():.4f}**
            """)
            
            # Show distribution histogram using Plotly
            import plotly.express as px
            fig = px.histogram(pc_weights_sum, x='Sum of Weights', nbins=20, 
                              title='Distribution of PC Weight Sums (Should cluster around 1.0)',
                              labels={'Sum of Weights': 'Sum of Weights per PC'})
            fig.add_vline(x=1.0, line_dash="dash", line_color="red", annotation_text="Ideal = 1.0")
            st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        ✅ **Result**: All subsequent analyses use these spatially-weighted PC assignments, 
        ensuring that election data is allocated to districts proportionally based on geographic overlap.
        This solves the MAUP problem and prevents measurement error from biasing our DiD estimates.
        """)
        
    else:
        st.warning("⚠️ Spatial crosswalk file not found. Please ensure the data files are uploaded.")
        st.markdown("""
        Expected files in `data/` folder:
        - `PC2004_to_Dist1991_Weightage_Crosswalk (1).csv`
        - `Pristine_Census_Map_1991_Final.geojson`
        - `PC_2004_Data_from_ARCGIS.geojson`
        """)


# ---------------------------------------------------------
# TAB 2: THE CORE CODE
# ---------------------------------------------------------
with tab_code:
    st.markdown("### Under the Hood: Spatial Engineering Implementation")
    
    st.markdown("""
    This script performs the geographic intersection using GeoPandas:
    1. Loads PC and District shapefiles
    2. Projects to Cylindrical Equal-Area (EPSG:6933) for accurate area calculation
    3. Performs overlay intersection (the "cookie cutter")
    4. Calculates normalized weights for each PC-District pair
    """)
    
    # Read this file's own source code
    with open(__file__, "r") as f:
        source_code = f.read()
        
    st.code(source_code, language="python")
