import streamlit as st
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import matplotlib.pyplot as plt

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
    - A PC is treated (`is_treated=1`) **if and only if** its Constituency name matches the official 1999 EVM rollout list.
    - **No spatial codes used for treatment assignment**; purely name-based matching to ensure transparency.
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
            st.error("❌ No election data loaded. Check file paths.")
            st.stop()
            
        df_raw = pd.concat(dfs, ignore_index=True)
        st.success(f"✅ Loaded {len(df_raw)} raw observations from election files.")

        # 2. Validate Columns
        req_cols = ['State/UT', 'Constituency', 'Voted_Total', 'Electors_Total']
        missing = [c for c in req_cols if c not in df_raw.columns]
        if missing:
            st.error(f"❌ Missing columns in election data: {missing}")
            st.stop()

        # 3. Calculate Turnout
        df_raw['turnout'] = (df_raw['Voted_Total'] / df_raw['Electors_Total']) * 100
        
        # Winsorize Turnout (1st, 99th percentile)
        lower = df_raw['turnout'].quantile(0.01)
        upper = df_raw['turnout'].quantile(0.99)
        df_raw['turnout'] = df_raw['turnout'].clip(lower, upper)

        # 4. Assign Treatment (STRICT NAME MATCHING)
        # Normalize constituency names for robust matching (strip whitespace, uppercase)
        df_raw['Const_Norm'] = df_raw['Constituency'].astype(str).str.strip().str.upper()
        
        # Create boolean mask
        df_raw['is_treated'] = df_raw['Const_Norm'].isin(TREATED_PCS_1999).astype(int)
        
        n_treated_pcs = df_raw[df_raw['is_treated']==1]['Constituency'].nunique()
        n_total_pcs = df_raw['Constituency'].nunique()
        
        st.info(f"✅ Treatment Assigned: **{n_treated_pcs}** unique constituencies matched the EVM list out of **{n_total_pcs}** total.")
        
        # Optional: Show matched names for verification
        with st.expander("View Matched Treated Constituencies"):
            matched_names = df_raw[df_raw['is_treated']==1]['Constituency'].unique()
            st.write(sorted(matched_names))

        # 5. Prepare DiD Variables
        df_raw['post'] = (df_raw['year'] >= 1999).astype(int)
        df_raw['did'] = df_raw['is_treated'] * df_raw['post']
        
        # Create a unique PC ID for Panel Regression (State + Constituency)
        df_raw['pc_id'] = df_raw['State/UT'].astype(str) + "_" + df_raw['Constituency'].astype(str)
        
        # Filter for analysis
        df_analysis = df_raw.dropna(subset=['turnout', 'did', 'year', 'pc_id'])
        
        if len(df_analysis) == 0:
            st.error("❌ No data left after cleaning.")
            st.stop()

        st.markdown(f"**Analysis Dataset**: {len(df_analysis)} observations, {df_analysis['pc_id'].nunique()} unique PCs.")

    except Exception as e:
        st.error(f"💥 Error loading data: {str(e)}")
        st.stop()

    # --- REAL ECONOMETRICS ---
    st.markdown("### Regression Output")
    
    try:
        # Set Panel Index
        df_panel = df_analysis.set_index(['pc_id', 'year'])
        
        # Run PanelOLS
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
            if 1 not in trend.columns:
                st.error("No treated PCs found! Check if your Constituency names exactly match the TREATED_PCS_1999 list.")
            
    except Exception as e:
        st.error(f"💥 Regression Error: {str(e)}")
        st.info("Hint: Ensure 'pc_id' has enough variation.")

# ---------------------------------------------------------
# TAB 2: THE CORE CODE
# ---------------------------------------------------------
with tab_code:
    st.markdown("### Under the Hood: Original Script Logic")
    st.markdown("Extracted from `/scripts/STEP_3_DiD_Regression.py`")
    
    original_script = """
import pandas as pd
from linearmodels.panel import PanelOLS

# 1. Define Treated List (Hardcoded from Official Records)
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

# 2. Load Data
df96 = pd.read_csv('data/1996_election_data_corrected.csv')
df98 = pd.read_csv('data/1998_election_data_corrected.csv')
df99 = pd.read_csv('data/1999_election_data_corrected.csv')

# 3. Stack and Add Year
df96['year'] = 1996
df98['year'] = 1998
df99['year'] = 1999
df = pd.concat([df96, df98, df99], ignore_index=True)

# 4. Calculate Turnout
df['turnout'] = (df['Voted_Total'] / df['Electors_Total']) * 100

# 5. Assign Treatment (Strict Name Matching)
df['Const_Norm'] = df['Constituency'].str.strip().str.upper()
df['is_treated'] = df['Const_Norm'].isin(TREATED_PCS_1999).astype(int)

# 6. Create Panel ID
df['pc_id'] = df['State/UT'] + "_" + df['Constituency']
df['post'] = (df['year'] >= 1999).astype(int)
df['did'] = df['is_treated'] * df['post']

# 7. Panel Regression
df = df.set_index(['pc_id', 'year'])
mod = PanelOLS(df['turnout'], df[['did']], entity_effects=True, time_effects=True)
res = mod.fit(cov_type='clustered', cluster_entity=True)

print(res.summary)
    """
    st.code(original_script, language="python")
