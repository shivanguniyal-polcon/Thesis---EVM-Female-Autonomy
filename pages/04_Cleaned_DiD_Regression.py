import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
import matplotlib.pyplot as plt
import os

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
    - `EVM_pc` = 1 if PC is in the treated list (received EVMs), 0 otherwise
    - `Post1999_t` = 1 for elections in 1999 and later, 0 for 1996/1998
    - `γ_pc` = PC fixed effects
    - `δ_t` = Election year fixed effects
    - `β₁` = **The causal effect of EVMs on turnout**
    
    **Data Construction**:
    - Stacked 1996, 1998, and 1999 corrected election files
    - Merged with treated PC list to assign `is_treated`
    - Dropped PCs with missing data
    """)

    # --- REAL DATA LOADING ---
    data_path = "data"
    
    try:
        # 1. Load the three specific election files
        file_96 = os.path.join(data_path, "election_1996_corrected.csv")
        file_98 = os.path.join(data_path, "election_1998_corrected.csv")
        file_99 = os.path.join(data_path, "election_1999_corrected.csv")
        
        if not all(os.path.exists(f) for f in [file_96, file_98, file_99]):
            st.error("⚠️ Missing one or more election files (1996, 1998, 1999). Please check the `data/` folder.")
            st.stop()

        df_96 = pd.read_csv(file_96)
        df_98 = pd.read_csv(file_98)
        df_99 = pd.read_csv(file_99)
        
        # Add year column if not present
        df_96['year'] = 1996
        df_98['year'] = 1998
        df_99['year'] = 1999
        
        # Standardize column names (assuming 'total_turnout' or 'turnout' exists)
        # Adjust these column names based on your actual CSV headers if different
        required_cols = ['pc_id', 'turnout'] 
        for df in [df_96, df_98, df_99]:
            # Handle potential column name variations
            if 'total_turnout' in df.columns and 'turnout' not in df.columns:
                df['turnout'] = df['total_turnout']
            if 'PC_No' in df.columns and 'pc_id' not in df.columns:
                df['pc_id'] = df['PC_No']
                
        # Stack into a single panel
        df_panel = pd.concat([df_96, df_98, df_99], ignore_index=True)
        
        st.info(f"✅ Stacked {len(df_panel)} observations from 3 election years.")

        # 2. Load Treated PC List
        # Looking for a file that lists treated PCs. 
        # Common names: 'treated_pcs.csv', 'evm_rollout.csv', 'spatial_crosswalk_final.csv'
        treated_file = None
        for fname in os.listdir(data_path):
            if 'treated' in fname.lower() or ('crosswalk' in fname.lower() and fname.endswith('.csv')):
                treated_file = os.path.join(data_path, fname)
                break
        
        if treated_file:
            df_treated = pd.read_csv(treated_file)
            # Identify the PC ID column in the treated file
            pc_col = 'pc_id' if 'pc_id' in df_treated.columns else df_treated.columns[0]
            
            # Create a set of treated IDs
            treated_ids = set(df_treated[pc_col].dropna().astype(str))
            
            # Assign is_treated
            df_panel['is_treated'] = df_panel['pc_id'].astype(str).isin(treated_ids).astype(int)
            st.success(f"✅ Merged with treatment list: {df_panel['is_treated'].sum()} PCs identified as treated.")
        else:
            st.warning("⚠️ No explicit treated PC list found. Checking if 'is_treated' column exists in data...")
            if 'is_treated' not in df_panel.columns:
                st.error("❌ Cannot find treatment assignment. Please ensure a file listing treated PCs exists in `data/`.")
                st.stop()
            else:
                st.success("✅ Using existing 'is_treated' column from data.")

        # 3. Final Cleaning
        df_clean = df_panel.dropna(subset=['turnout', 'is_treated', 'year']).copy()
        df_clean['post'] = (df_clean['year'] >= 1999).astype(int)
        df_clean['did'] = df_clean['is_treated'] * df_clean['post']
        
        st.markdown(f"**Final Dataset**: {len(df_clean)} observations, {df_clean['pc_id'].nunique()} unique PCs.")

    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

    # --- REAL ECONOMETRICS ---
    st.markdown("### Regression Output")
    
    try:
        # Prepare Panel Data: Set MultiIndex
        df_clean = df_clean.set_index(['pc_id', 'year'])
        
        # Run PanelOLS
        # Dependent: turnout
        # Exog: did (the interaction term)
        # Effects: Entity (PC) and Time (Year)
        mod = PanelOLS(df_clean['turnout'], df_clean[['did']], entity_effects=True, time_effects=True)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        
        coef = res.params['did']
        se = res.std_errors['did']
        pval = res.pvalues['did']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="EVM Effect (β₁)", value=f"{coef:.2f} pp")
        
        with col2:
            st.metric(label="Standard Error", value=f"{se:.3f}")
        
        with col3:
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            st.metric(label="Significance", value=f"p={pval:.4f} {sig}")
        
        # Full regression table
        st.markdown("### Full Regression Table")
        
        regression_table = pd.DataFrame({
            'Variable': ['EVM × Post (DiD)'],
            'Coefficient': [f"{coef:.3f}" + ("***" if pval < 0.01 else "**" if pval < 0.05 else "*")],
            'Std. Error': [f"({se:.3f})"],
            '95% CI': [f"[{coef-1.96*se:.3f}, {coef+1.96*se:.3f}]"],
            'p-value': [f"{pval:.4f}"]
        })
        
        st.dataframe(regression_table, width=700)
        
        st.markdown("### Model Diagnostics")
        
        diag_col1, diag_col2, diag_col3 = st.columns(3)
        
        with diag_col1:
            st.metric("Observations", f"{int(res.nobs):,}")
        
        with diag_col2:
            st.metric("R-squared (Within)", f"{res.rsquared_within:.3f}")
        
        with diag_col3:
            n_entities = len(df_clean.index.levels[0])
            st.metric("Unique PCs", f"{n_entities:,}")
        
        st.markdown("""
        ### Interpretation
        
        ✅ **Main Finding**: The coefficient on `EVM × Post` represents the average treatment effect on the treated (ATT).
        
        **Robustness**: Standard errors clustered at the PC level. Fixed effects control for time-invariant PC characteristics and common year shocks.
        """)
        
        # --- VISUALIZATION ---
        df_plot = df_clean.reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group by year and treatment status
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
        st.error(f"❌ Error running regression: {e}")


# ---------------------------------------------------------
# TAB 2: THE CORE CODE
# ---------------------------------------------------------
with tab_code:
    st.markdown("### Under the Hood: Original Script Logic")
    st.markdown("This is the core econometric logic extracted from `/scripts/STEP_3_DiD_Regression.py`")
    
    original_script_code = """
import pandas as pd
from linearmodels.panel import PanelOLS
import os

# 1. Load and Stack Election Data
files = ['data/election_1996_corrected.csv', 
         'data/election_1998_corrected.csv', 
         'data/election_1999_corrected.csv']
years = [1996, 1998, 1999]

dfs = []
for f, y in zip(files, years):
    df = pd.read_csv(f)
    df['year'] = y
    dfs.append(df)

df_panel = pd.concat(dfs, ignore_index=True)

# 2. Merge Treatment Status
# Assuming a file 'treated_pcs.csv' exists with a 'pc_id' column
df_treated = pd.read_csv('data/treated_pcs.csv')
treated_ids = set(df_treated['pc_id'].astype(str))
df_panel['is_treated'] = df_panel['pc_id'].astype(str).isin(treated_ids).astype(int)

# 3. Construct DiD Variable
df_panel['post'] = (df_panel['year'] >= 1999).astype(int)
df_panel['did'] = df_panel['is_treated'] * df_panel['post']

# 4. Clean Data
df_clean = df_panel.dropna(subset=['turnout', 'is_treated', 'year'])

# 5. Set Panel Index
df_clean = df_clean.set_index(['pc_id', 'year'])

# 6. Run PanelOLS with Fixed Effects
# Entity Effects (PC) + Time Effects (Year)
mod = PanelOLS(df_clean['turnout'], df_clean[['did']], 
               entity_effects=True, time_effects=True)

# 7. Fit with Clustered Standard Errors
res = mod.fit(cov_type='clustered', cluster_entity=True)

# 8. Output Results
print(res.summary)
    """
    
    st.code(original_script_code, language="python")
