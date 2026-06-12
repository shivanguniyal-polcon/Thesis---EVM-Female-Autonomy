import streamlit as st
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import matplotlib.pyplot as plt
import re

# 1. Page Config & Title
st.set_page_config(page_title="Module 2: Cleaned DiD", layout="wide")
st.title("Module 2: Baseline Identification & Validity")
st.header("Step 4: Cleaned PC-Level DiD Regression")

# 2. Create the two tabs
tab_results, tab_code = st.tabs(["📊 Results", "💻 Core Code"])

# ---------------------------------------------------------
# TAB 1: THE RESULTS
# ---------------------------------------------------------
with tab_results:
    st.markdown("""
    ### The Baseline Causal Estimate
    
    **Model Specification**:
    ```
    Turnout_pc,t = β₀ + β₁(EVM_pc × Post1999_t) + γ_pc + δ_t + ε_pc,t
    ```
    
    **Treatment Definition**:
    - Based on the official **1999 EVM Rollout List** (45 Parliamentary Constituencies).
    - Matching performed on **Constituency Names** (robust to case/whitespace).
    """)

    # --- HARDCODED TREATED LIST ---
    TREATED_PCS_1999 = [
        'HYDERABAD', 'SECUNDERABAD', 'PANAJI', 'MORMUGAO', 'AHMEDABAD', 'GANDHINAGAR', 
        'KARNAL', 'ROHTAK', 'BANGALORE NORTH', 'BANGALORE SOUTH', 'MYSORE', 'ERNAKULAM', 
        'TRIVANDRUM', 'GWALIOR', 'BHOPAL', 'MUMBAI SOUTH', 'MUMBAI SOUTH CENTRA', 
        'MUMBAI NORTH CENTRAL', 'MUMBAI NORTH EAST', 'MUMBAI NORTH WEST', 'NEW DELHI',
        'SOUTH DELHI', 'OUTER DELHI', 'EAST DELHI', 'CHANDNI CHOWK', 'DELHI SADAR', 'KAROL BAGH',
        'LUCKNOW', 'KANPUR', 'JAIPUR', 'AJMER', 'MADRAS CENTRAL', 'MADRAS SOUTH',
        'COIMBATORE', 'MADURAI', 'CALCUTTA NORTH WEST', 'CALCUTTA NORTH EAST',
        'CALCUTTA SOUTH', 'HOWRAH', 'GAUHATI', 'CHANDIGARH', 'PONDICHERRY', 'ALLAHABAD', 
        'AGRA', 'TARN TARAN', 'PATIALA', 'FARIDKOT', 'BHUBANESWAR'
    ]
    
    # Normalize the treated list once
    def normalize_name(name):
        if not isinstance(name, str):
            return ""
        # Uppercase, strip whitespace
        n = str(name).upper().strip()
        # Remove common suffixes that might interfere
        n = re.sub(r'\s+(PC|CONSTITUENCY|PARLIAMENTARY)$', '', n)
        # Collapse multiple spaces
        n = re.sub(r'\s+', ' ', n)
        return n

    treated_set = {normalize_name(n) for n in TREATED_PCS_1999}

    # --- REAL DATA LOADING ---
    try:
        # 1. Load Election Files
        files = {
            1996: 'data/1996_election_data_corrected.csv',
            1998: 'data/1998_election_data_corrected.csv',
            1999: 'data/1999_election_data_corrected.csv'
        }
        
        dfs = []
        for year, path in files.items():
            try:
                df_year = pd.read_csv(path)
                df_year['year'] = year
                dfs.append(df_year)
            except FileNotFoundError:
                st.warning(f"⚠️ Missing: {path}")
        
        if not dfs:
            st.error("❌ No election data loaded.")
            st.stop()
            
        df_raw = pd.concat(dfs, ignore_index=True)
        st.success(f"✅ Loaded {len(df_raw)} raw observations.")

        # 2. Validate Columns
        req_cols = ['State/UT', 'Constituency', 'Voted_Total', 'Electors_Total']
        missing = [c for c in req_cols if c not in df_raw.columns]
        if missing:
            st.error(f"❌ Missing columns: {missing}. Found: {list(df_raw.columns)}")
            st.stop()

        # 3. Calculate Turnout
        df_raw['turnout'] = (df_raw['Voted_Total'] / df_raw['Electors_Total']) * 100
        
        # Winsorize
        lower = df_raw['turnout'].quantile(0.01)
        upper = df_raw['turnout'].quantile(0.99)
        df_raw['turnout'] = df_raw['turnout'].clip(lower, upper)

        # 4. Robust Treatment Assignment
        st.markdown("### 🔍 Matching Treatment List to Data")
        
        # Create normalized column for matching
        df_raw['constituency_norm'] = df_raw['Constituency'].apply(normalize_name)
        
        # Method A: Exact Match on Normalized Name
        df_raw['is_treated'] = df_raw['constituency_norm'].isin(treated_set).astype(int)
        
        # Method B: Fuzzy/Partial Match (Backup)
        # If exact match fails, check if any treated name is a substring of the data name
        # OR if the data name is a substring of a treated name (handles "Bombay" vs "Mumbai" issues if any)
        count_exact = df_raw['is_treated'].sum()
        
        if count_exact == 0:
            st.warning("⚠️ Zero exact matches found. Attempting partial string matching...")
            matched_pcs = set()
            for idx, row in df_raw.iterrows():
                c_name = row['constituency_norm']
                # Check if any treated name is inside the constituency name
                for t_name in treated_set:
                    if t_name in c_name or c_name in t_name:
                        df_raw.at[idx, 'is_treated'] = 1
                        matched_pcs.add(c_name)
                        break
            
            st.info(f"Partial matching found {df_raw['is_treated'].sum()} treated observations.")
        
        # Debugging: Show what was found
        found_constituencies = df_raw[df_raw['is_treated']==1]['constituency_norm'].unique()
        missing_from_data = [t for t in treated_set if t not in found_constituencies]
        
        col_dbg1, col_dbg2 = st.columns(2)
        with col_dbg1:
            st.metric("Unique Treated Constituencies Found", len(found_constituencies))
        with col_dbg2:
            st.metric("Total Treated Observations", int(df_raw['is_treated'].sum()))
            
        if len(missing_from_data) > 0 and len(found_constituencies) < len(TREATED_PCS_1999):
            with st.expander("🔍 Debug: Missing Constituencies"):
                st.write(f"The following treated PCs were NOT found in your data (check spelling):")
                st.write(missing_from_data[:20]) # Show first 20
                st.write(f"... and {len(missing_from_data)-20} more." if len(missing_from_data) > 20 else "")
                
        if len(found_constituencies) == 0:
            st.error("❌ CRITICAL: No constituencies matched. Please check the 'Debug' section above.")
            st.stop()

        # 5. Create Unique ID (State + Constituency)
        df_raw['pc_id'] = df_raw['State/UT'].astype(str) + "_" + df_raw['Constituency'].astype(str)
        
        # Drop duplicates if any
        df_raw = df_raw.drop_duplicates(subset=['pc_id', 'year'])

        # 6. Prepare DiD Variables
        df_raw['post'] = (df_raw['year'] >= 1999).astype(int)
        df_raw['did'] = df_raw['is_treated'] * df_raw['post']
        
        df_analysis = df_raw.dropna(subset=['turnout', 'did', 'year'])
        
        st.markdown(f"**Analysis Dataset**: {len(df_analysis)} observations, {df_analysis['pc_id'].nunique()} unique PCs.")

    except Exception as e:
        st.error(f"💥 Error loading data: {str(e)}")
        st.stop()

    # --- REAL ECONOMETRICS ---
    st.markdown("### Regression Output")
    
    try:
        # Set Panel Index
        df_panel = df_analysis.set_index(['pc_id', 'year'])
        
        # Check for variation
        if df_panel['did'].nunique() < 2:
            st.error("❌ The 'did' variable has no variation. Cannot run regression.")
            st.stop()
            
        mod = PanelOLS(df_panel['turnout'], df_panel[['did']], entity_effects=True, time_effects=True)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        
        coef = res.params['did']
        se = res.std_errors['did']
        pval = res.pvalues['did']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            delta = "+++" if pval < 0.01 else ("+" if pval < 0.05 else "")
            st.metric(label=f"EVM Effect (β₁) {delta}", value=f"{coef:.2f} pp")
        
        with col2:
            st.metric(label="Standard Error", value=f"{se:.3f}")
        
        with col3:
            st.metric(label="p-value", value=f"{pval:.4f}")
        
        # Full Table
        st.markdown("### Full Regression Table")
        reg_df = pd.DataFrame({
            'Variable': ['EVM × Post (DiD)'],
            'Coefficient': [f"{coef:.3f}" + ("***" if pval < 0.01 else "**" if pval < 0.05 else "*")],
            'Std. Error': [f"({se:.3f})"],
            '95% CI': [f"[{coef-1.96*se:.3f}, {coef+1.96*se:.3f}]"]
        })
        st.dataframe(reg_df, width=700)
        
        # Diagnostics
        st.markdown("### Model Diagnostics")
        d1, d2, d3 = st.columns(3)
        d1.metric("Observations", f"{int(res.nobs):,}")
        d2.metric("R-squared (Within)", f"{res.rsquared_within:.3f}")
        d3.metric("Unique PCs", f"{len(df_panel.index.levels[0]):,}")
        
        # Visualization
        st.markdown("### Visualizing the DiD")
        df_plot = df_analysis.reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        trend = df_plot.groupby(['year', 'is_treated'])['turnout'].mean().unstack()
        
        if 0 in trend.columns and 1 in trend.columns:
            ax.plot(trend.index, trend[0], 's--', label='Control (Paper)', color='#A23B72', lw=2, ms=8)
            ax.plot(trend.index, trend[1], 'o-', label='Treatment (EVM)', color='#2E86AB', lw=2, ms=8)
            
            ax.axvline(x=1998.5, color='gray', linestyle=':', lw=2, label='EVM Intro (1999)')
            ax.set_xlabel('Election Year', fontsize=12)
            ax.set_ylabel('Average Turnout (%)', fontsize=12)
            ax.set_title('Difference-in-Differences: EVM Impact on Turnout', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.warning("Cannot plot: Missing Control or Treatment group data.")
            
    except Exception as e:
        st.error(f"💥 Regression Error: {str(e)}")

# ---------------------------------------------------------
# TAB 2: THE CORE CODE
# ---------------------------------------------------------
with tab_code:
    st.markdown("### Under the Hood: Original Script Logic")
    st.markdown("Extracted from `/scripts/STEP_3_DiD_Regression.py`")
    
    original_script = """
import pandas as pd
from linearmodels.panel import PanelOLS

# 1. Load Data
df96 = pd.read_csv('data/1996_election_data_corrected.csv')
df98 = pd.read_csv('data/1998_election_data_corrected.csv')
df99 = pd.read_csv('data/1999_election_data_corrected.csv')

# 2. Stack and Add Year
df96['year'] = 1996
df98['year'] = 1998
df99['year'] = 1999
df = pd.concat([df96, df98, df99], ignore_index=True)

# 3. Define Treated List (Hardcoded)
TREATED_PCS = ['HYDERABAD', 'SECUNDERABAD', ...] # Full list
# Normalize names
df['constituency_norm'] = df['Constituency'].str.upper().str.strip()
df['is_treated'] = df['constituency_norm'].isin(TREATED_PCS).astype(int)

# 4. Calculate Turnout
df['turnout'] = (df['Voted_Total'] / df['Electors_Total']) * 100

# 5. Create DiD Variable
df['post'] = (df['year'] >= 1999).astype(int)
df['did'] = df['is_treated'] * df['post']

# 6. Panel Regression
df = df.set_index(['State/UT', 'Constituency', 'year']) # Or unique ID
mod = PanelOLS(df['turnout'], df[['did']], entity_effects=True, time_effects=True)
res = mod.fit(cov_type='clustered', cluster_entity=True)

print(res.summary)
    """
    st.code(original_script, language="python")
