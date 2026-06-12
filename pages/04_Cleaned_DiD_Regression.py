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
    
    **Data Cleaning Applied**:
    - Stacked 1996, 1998, and 1999 corrected election files.
    - Merged with treated PC list to identify treatment status.
    - Winsorized turnout at 1st and 99th percentiles.
    """)

    # --- REAL DATA LOADING ---
    file_paths = {
        1996: 'data/1996_election_data_corrected.csv',
        1998: 'data/1998_election_data_corrected.csv',
        1999: 'data/1999_election_data_corrected.csv'
    }
    
    treated_file = 'data/treated_pcs_list.csv'  # Adjust filename if different
    
    dfs = []
    missing_files = []
    
    for year, path in file_paths.items():
        try:
            df_year = pd.read_csv(path)
            df_year['year'] = year
            dfs.append(df_year)
        except FileNotFoundError:
            missing_files.append(path)
    
    if missing_files:
        st.error(f"⚠️ Missing election files: {', '.join(missing_files)}. Please check the data/ folder.")
        st.stop()
    
    # Combine into panel
    df_panel = pd.concat(dfs, ignore_index=True)
    
    # --- FLEXIBLE COLUMN DETECTION ---
    # Find PC ID column
    pc_id_cols = [c for c in df_panel.columns if 'pc' in c.lower() and 'id' in c.lower()]
    if not pc_id_cols:
        # Fallback if naming is just 'pc' or 'station'
        pc_id_cols = [c for c in df_panel.columns if c.lower() in ['pc', 'station', 'polling_station']]
    
    if not pc_id_cols:
        st.error(f"⚠️ Could not find PC ID column. Available columns: {list(df_panel.columns)}")
        st.stop()
    
    pc_id_col = pc_id_cols[0]
    st.info(f"✅ Identified PC ID column: `{pc_id_col}`")
    
    # Find Turnout column
    turnout_cols = [c for c in df_panel.columns if 'turnout' in c.lower()]
    if not turnout_cols:
        # Fallback: calculate turnout if votes/electors exist
        if 'total_votes' in df_panel.columns and 'total_electors' in df_panel.columns:
            df_panel['turnout'] = (df_panel['total_votes'] / df_panel['total_electors']) * 100
            turnout_col = 'turnout'
            st.info("✅ Calculated turnout from votes/electors.")
        else:
            st.error("⚠️ Could not find turnout column or calculate it.")
            st.stop()
    else:
        turnout_col = turnout_cols[0]
    
    # Rename for consistency
    df_panel = df_panel.rename(columns={pc_id_col: 'pc_id', turnout_col: 'turnout'})
    
    # --- LOAD TREATED LIST ---
    try:
        df_treated = pd.read_csv(treated_file)
        # Detect ID column in treated list
        t_pc_cols = [c for c in df_treated.columns if 'pc' in c.lower() and 'id' in c.lower()]
        if not t_pc_cols:
             t_pc_cols = [c for c in df_treated.columns if c.lower() in ['pc', 'station']]
        
        if t_pc_cols:
            df_treated = df_treated.rename(columns={t_pc_cols[0]: 'pc_id'})
            treated_ids = set(df_treated['pc_id'].astype(str))
            
            # Create treatment indicator
            df_panel['is_treated'] = df_panel['pc_id'].astype(str).isin(treated_ids).astype(int)
            st.success(f"✅ Loaded {len(treated_ids)} treated PCs.")
        else:
            st.warning("⚠️ Could not find PC ID column in treated list. Assuming no treatment data.")
            df_panel['is_treated'] = 0
            
    except FileNotFoundError:
        st.warning(f"⚠️ Treated list `{treated_file}` not found. Cannot identify treatment group.")
        df_panel['is_treated'] = 0

    # Check if we have variation
    if df_panel['is_treated'].sum() == 0:
        st.error("⚠️ No treated PCs found in the dataset. Check ID matching between election data and treated list.")
        st.stop()

    # --- PREPARE DiD VARIABLES ---
    df_panel['post'] = (df_panel['year'] >= 1999).astype(int)
    df_panel['did'] = df_panel['is_treated'] * df_panel['post']
    
    # Drop missing
    df_panel = df_panel.dropna(subset=['turnout', 'is_treated', 'did'])
    
    # Winsorize
    lower = df_panel['turnout'].quantile(0.01)
    upper = df_panel['turnout'].quantile(0.99)
    df_panel['turnout'] = df_panel['turnout'].clip(lower, upper)

    st.markdown("### Regression Output")
    
    # --- RUN REGRESSION ---
    try:
        # Set Index for PanelOLS
        df_panel = df_panel.set_index(['pc_id', 'year'])
        
        # Model: Turnout ~ DID + EntityEffects + TimeEffects
        mod = PanelOLS(df_panel['turnout'], df_panel[['did']], entity_effects=True, time_effects=True)
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
        
        st.markdown("### Full Regression Table")
        regression_table = pd.DataFrame({
            'Variable': ['EVM × Post (DiD)'],
            'Coefficient': [f"{coef:.3f}{sig}"],
            'Std. Error': [f"({se:.3f})"],
            '95% CI': [f"[{coef-1.96*se:.3f}, {coef+1.96*se:.3f}]"]
        })
        st.dataframe(regression_table, width=700)
        
        st.markdown("### Model Diagnostics")
        d1, d2, d3 = st.columns(3)
        d1.metric("Observations", f"{int(res.nobs):,}")
        d2.metric("R-squared (Within)", f"{res.rsquared_within:.3f}")
        d3.metric("Clusters (PCs)", f"{int(res.nobs / len(df_panel.index.levels[1])):,}")
        
        # --- PLOT ---
        df_plot = df_panel.reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        trend = df_plot.groupby(['year', 'is_treated'])['turnout'].mean().unstack()
        
        if 0 in trend.columns and 1 in trend.columns:
            ax.plot(trend.index, trend[0], 's--', label='Control (Paper)', color='#A23B72', linewidth=2)
            ax.plot(trend.index, trend[1], 'o-', label='Treatment (EVM)', color='#2E86AB', linewidth=2)
            ax.axvline(x=1998.5, color='gray', linestyle=':', linewidth=2)
            ax.text(1998.6, trend.iloc[-1].max(), 'EVMs Introduced', fontsize=10, fontweight='bold')
            ax.set_xlabel('Election Year')
            ax.set_ylabel('Average Turnout (%)')
            ax.set_title('DiD Visualization: EVM Effect on Turnout')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.warning("Not enough data groups to plot trends.")
            
    except Exception as e:
        st.error(f"Error running regression: {e}")


# ---------------------------------------------------------
# TAB 2: THE CORE CODE
# ---------------------------------------------------------
with tab_code:
    st.markdown("### Under the Hood: Original Script Logic")
    st.markdown("Extracted from `/scripts/STEP_3_DiD_Regression.py`")
    
    original_script_code = """
import pandas as pd
from linearmodels.panel import PanelOLS

# 1. Load and Stack Data
df_96 = pd.read_csv('data/1996_election_data_corrected.csv')
df_98 = pd.read_csv('data/1998_election_data_corrected.csv')
df_99 = pd.read_csv('data/1999_election_data_corrected.csv')

df_96['year'], df_98['year'], df_99['year'] = 1996, 1998, 1999
df = pd.concat([df_96, df_98, df_99], ignore_index=True)

# 2. Merge Treatment Status
treated = pd.read_csv('data/treated_pcs_list.csv')
df = df.merge(treated[['pc_id']], on='pc_id', how='left', indicator=True)
df['is_treated'] = (df['_merge'] == 'both').astype(int)

# 3. Define DiD Variables
df['post'] = (df['year'] >= 1999).astype(int)
df['did'] = df['is_treated'] * df['post']

# 4. Run PanelOLS
df = df.set_index(['pc_id', 'year'])
mod = PanelOLS(df['turnout'], df[['did']], entity_effects=True, time_effects=True)
res = mod.fit(cov_type='clustered', cluster_entity=True)

print(res.summary)
    """
    
    st.code(original_script_code, language="python")
