import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
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
    
    Where:
    - `EVM_pc` = 1 if PC is in the treated list, 0 otherwise
    - `Post1999_t` = 1 for elections in 1999 and later
    - `β₁` = **The causal effect of EVMs on turnout**
    
    **Data Construction**:
    - Treatment status (`is_treated`) derived from the official EVM rollout list.
    - Merged with corrected election data (1996–2004).
    - PC and Year Fixed Effects included.
    """)

    # --- REAL DATA LOADING ---
    try:
        # 1. Load Election Data
        df_election = pd.read_csv('data/election_data_corrected.csv')
        
        # 2. Load Treated PC List 
        # NOTE: Adjust 'spatial_crosswalk.csv' to your actual filename containing the treated PC list
        # If your treated list is in a different file, change this path.
        # Assuming the file has a column 'pc_id' for treated PCs.
        try:
            df_treated_list = pd.read_csv('data/spatial_crosswalk.csv')
            # Extract unique treated PC IDs (assuming all PCs in this file are treated, 
            # or filter by a column like 'treatment_status' if available)
            treated_pcs = df_treated_list['pc_id'].unique()
        except FileNotFoundError:
            st.warning("⚠️ 'data/spatial_crosswalk.csv' not found. Trying 'data/treated_pcs.csv'...")
            try:
                df_treated_list = pd.read_csv('data/treated_pcs.csv')
                treated_pcs = df_treated_list['pc_id'].unique()
            except FileNotFoundError:
                st.error("⚠️ Could not find treated PC list. Please ensure 'spatial_crosswalk.csv' or 'treated_pcs.csv' exists in /data.")
                st.stop()

        # 3. Derive is_treated
        df_election['is_treated'] = df_election['pc_id'].isin(treated_pcs).astype(int)
        
        # 4. Create DiD variables
        df_election['post'] = (df_election['year'] >= 1999).astype(int)
        df_election['evm_x_post'] = df_election['is_treated'] * df_election['post']
        
        # 5. Filter for relevant years and clean
        df = df_election[df_election['year'].isin([1996, 1998, 1999, 2000, 2004])].copy()
        df = df.dropna(subset=['turnout', 'is_treated', 'year'])
        
        if df['is_treated'].sum() == 0:
            st.error("⚠️ No treated PCs found after merging. Check PC ID formats in both files.")
            st.stop()
            
        st.success(f"✅ Loaded {len(df)} observations. {df['is_treated'].sum()} PCs identified as treated.")
        
    except Exception as e:
        st.error(f"⚠️ Error loading data: {e}")
        st.stop()

    # --- REAL ECONOMETRICS ---
    st.markdown("### Regression Output")
    
    try:
        # Prepare Panel Data (Index must be MultiIndex for PanelOLS)
        df_panel = df.set_index(['pc_id', 'year'])
        
        # Define exogenous variables (only the interaction term)
        exog = df_panel[['evm_x_post']]
        endog = df_panel['turnout']
        
        # Run PanelOLS with Entity (PC) and Time (Year) Fixed Effects
        mod = PanelOLS(endog, exog, entity_effects=True, time_effects=True)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        
        # Extract Results
        coef = res.params['evm_x_post']
        se = res.std_errors['evm_x_post']
        pval = res.pvalues['evm_x_post']
        
        # Display Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="EVM Effect (β₁)", value=f"{coef:.2f} pp")
        
        with col2:
            st.metric(label="Standard Error", value=f"{se:.3f}")
        
        with col3:
            sig_stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            st.metric(label="Significance", value=f"p={pval:.4f} {sig_stars}")
        
        # Full Table
        st.markdown("### Full Regression Table")
        regression_table = pd.DataFrame({
            'Variable': ['EVM × Post (DiD)'],
            'Coefficient': [f"{coef:.3f}{sig_stars}"],
            'Std. Error': [f"({se:.3f})"],
            '95% CI': [f"[{coef-1.96*se:.3f}, {coef+1.96*se:.3f}]"],
            'p-value': [f"{pval:.4f}"]
        })
        
        st.dataframe(regression_table, width=700)
        
        # Diagnostics
        st.markdown("### Model Diagnostics")
        diag_col1, diag_col2, diag_col3 = st.columns(3)
        
        with diag_col1:
            st.metric("Observations", f"{int(res.nobs):,}")
        
        with diag_col2:
            st.metric("R-squared (Within)", f"{res.rsquared_within:.3f}")
        
        with diag_col3:
            n_entities = len(df_panel.index.levels[0])
            st.metric("Unique PCs", f"{n_entities:,}")
        
        st.markdown("""
        ### Interpretation
        
        ✅ **Main Finding**: The coefficient on `EVM × Post` represents the causal impact.
        A positive value indicates EVMs increased turnout relative to the control group.
        
        **Robustness**: Standard errors clustered at the PC level.
        """)
        
        # --- VISUALIZATION ---
        df_plot = df.reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate means by year and treatment status
        trend = df_plot.groupby(['year', 'is_treated'])['turnout'].mean().unstack()
        
        if 0 in trend.columns and 1 in trend.columns:
            ax.plot(trend.index, trend[0], 's--', label='Control (Paper Ballot)', color='#A23B72', linewidth=2, markersize=8)
            ax.plot(trend.index, trend[1], 'o-', label='Treatment (EVM)', color='#2E86AB', linewidth=2, markersize=8)
            
            ax.axvline(x=1998.5, color='gray', linestyle=':', linewidth=2, label='EVM Introduction (1999)')
            ax.text(1998.6, trend.iloc[-1].max(), 'EVMs Introduced', fontsize=10, fontweight='bold', color='gray')
            
            ax.set_xlabel('Election Year', fontsize=12)
            ax.set_ylabel('Average Turnout (%)', fontsize=12)
            ax.set_title('DiD Visualization: EVM Effect on Turnout', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        else:
            st.warning("Not enough data groups to plot trends.")
            
    except Exception as e:
        st.error(f"⚠️ Regression failed: {e}")

# ---------------------------------------------------------
# TAB 2: THE CORE CODE
# ---------------------------------------------------------
with tab_code:
    st.markdown("### Under the Hood: Original Script Logic")
    st.markdown("This is the core econometric logic extracted from `/scripts/STEP_3_DiD_Regression.py`")
    
    # HARDCODED ORIGINAL SCRIPT LOGIC
    original_script_code = """
import pandas as pd
from linearmodels.panel import PanelOLS

# 1. Load Data
df_election = pd.read_csv('data/election_data_corrected.csv')
df_treated = pd.read_csv('data/spatial_crosswalk.csv') # Or specific treated list

# 2. Identify Treated PCs
treated_ids = df_treated['pc_id'].unique()
df_election['is_treated'] = df_election['pc_id'].isin(treated_ids).astype(int)

# 3. Construct DiD Variables
df_election['post'] = (df_election['year'] >= 1999).astype(int)
df_election['did'] = df_election['is_treated'] * df_election['post']

# 4. Filter and Clean
df = df_election[df_election['year'].isin([1996, 1998, 1999, 2000, 2004])]
df = df.dropna(subset=['turnout', 'did'])

# 5. Set Panel Index
df = df.set_index(['pc_id', 'year'])

# 6. Run PanelOLS with Fixed Effects
# Entity Effects (PC) + Time Effects (Year)
mod = PanelOLS(df['turnout'], df[['did']], entity_effects=True, time_effects=True)

# 7. Fit with Clustered Standard Errors
res = mod.fit(cov_type='clustered', cluster_entity=True)

# 8. Output Results
print(res.summary)
    """
    
    st.code(original_script_code, language="python")
