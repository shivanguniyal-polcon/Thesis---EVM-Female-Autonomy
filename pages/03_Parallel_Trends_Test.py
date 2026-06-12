import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import plotly.graph_objects as go
import os

# 1. Page Config & Title
st.set_page_config(page_title="Module 2: Parallel Trends", layout="wide")
st.title("Module 2: Baseline Identification & Validity")
st.header("Step 3: Parallel Trends Test (1996 vs 1998)")

# 2. Create the two tabs
tab_results, tab_code = st.tabs(["📊 Results", "💻 Core Code"])

# ---------------------------------------------------------
# TAB 1: THE RESULTS
# ---------------------------------------------------------
with tab_results:
    st.markdown("""
    ### The Critical Validity Check for Difference-in-Differences
    
    **Research Question**: Did EVM and paper-ballot PCs follow similar trends 
    in female turnout *before* the introduction of EVMs in 1999?
    
    **Why This Matters**: The parallel trends assumption is the cornerstone of DiD.
    If treatment and control groups were already on different trajectories before 
    the intervention, we cannot attribute post-treatment differences to the treatment itself.
    
    **Test Period**: 1996 vs 1998 (both pre-EVM elections)
    """)
    
    # The Verified 46 EVM PCs List
    TREATED_PCS_1999 = [
        'HYDERABAD', 'SECUNDERABAD', 'PANAJI', 'MORMUGAO', 'AHMEDABAD', 'GANDHINAGAR', 
        'KARNAL', 'ROHTAK', 'BANGALORE NORTH', 'BANGALORE SOUTH', 'MYSORE', 'ERNAKULAM', 
        'TRIVANDRUM', 'GWALIOR', 'BHOPAL', 'MUMBAI SOUTH', 'MUMBAI SOUTH CENTRA', 
        'MUMBAI NORTH CENTRAL', 'MUMBAI NORTH EAST', 'MUMBAI NORTH WEST', 'NEW DELHI',
        'SOUTH DELHI', 'OUTER DELHI', 'EAST DELHI', 'CHANDNI CHOWK', 'DELHI SADAR', 'KAROL BAGH',
        'LUCKNOW', 'KANPUR', 'PATNA', 'RANCHI', 'JAIPUR', 'AJMER', 'MADRAS CENTRAL', 'MADRAS NORTH',
        'MADRAS SOUTH', 'COIMBATORE', 'MADURAI', 'CALCUTTA NORTH WEST', 'CALCUTTA NORTH EAST',
        'CALCUTTA SOUTH', 'HOWRAH', 'GAUHATI', 'CHANDIGARH', 'PONDICHERRY', 'ALLAHABAD', 
        'AGRA', 'TARN TARAN', 'PATIALA', 'FARIDKOT', 'BHUBANESWAR'
    ]
    
    # Helper function to clean and calculate turnout
    def process_year(file_path, year):
        if not os.path.exists(file_path):
            return None
        
        df = pd.read_csv(file_path)
        df['pc_name_clean'] = (
            df['Constituency']
            .str.upper()
            .str.replace(r'\s*\(SC\)', '', regex=True)
            .str.replace(r'\s*\(ST\)', '', regex=True)
            .str.split(' NO :').str[0]
            .str.strip()
        )
        df['EVM'] = df['pc_name_clean'].isin(TREATED_PCS_1999).astype(int)
        
        # Calculate female turnout
        if 'Voted_Female' in df.columns and 'Electors_Female' in df.columns:
            df['Female_Turnout'] = (df['Voted_Female'] / df['Electors_Female']) * 100
        else:
            df['Female_Turnout'] = np.nan
            
        df['Year'] = year
        return df[['pc_name_clean', 'EVM', 'Female_Turnout', 'Year']]
    
    # Load Pre-Treatment Data (1996 and 1998)
    data_dir = "data"
    
    file_1996 = os.path.join(data_dir, '1996_election_data_corrected.csv')
    file_1998 = os.path.join(data_dir, '1998_election_data_corrected.csv')
    file_1999 = os.path.join(data_dir, '1999_election_data_corrected.csv')
    
    if os.path.exists(file_1996) and os.path.exists(file_1998):
        df_1996 = process_year(file_1996, 1996)
        df_1998 = process_year(file_1998, 1998)
        df_1999 = process_year(file_1999, 1999) if os.path.exists(file_1999) else None
        
        if df_1996 is not None and df_1998 is not None:
            # Statistical Test: Did the gap change between 1996 and 1998?
            merged_pre = pd.merge(df_1996, df_1998, on='pc_name_clean', suffixes=('_1996', '_1998'))
            merged_pre = merged_pre.dropna(subset=['Female_Turnout_1996', 'Female_Turnout_1998'])
            merged_pre['Delta_Turnout_98_96'] = merged_pre['Female_Turnout_1998'] - merged_pre['Female_Turnout_1996']
            
            # Regress the change in turnout on the EVM dummy
            parallel_model = smf.ols('Delta_Turnout_98_96 ~ EVM_1996', data=merged_pre).fit(cov_type='HC3')
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Mean Change (Paper PCs)", 
                    value=f"{merged_pre[merged_pre['EVM_1996']==0]['Delta_Turnout_98_96'].mean():.2f}%"
                )
            
            with col2:
                st.metric(
                    label="Mean Change (EVM PCs)", 
                    value=f"{merged_pre[merged_pre['EVM_1996']==1]['Delta_Turnout_98_96'].mean():.2f}%"
                )
            
            st.markdown("### Regression Results: Pre-Treatment Trend Test")
            
            results_df = pd.DataFrame({
                'Variable': ['EVM (Treatment)', 'Constant'],
                'Coefficient': [parallel_model.params['EVM_1996'], parallel_model.params['Intercept']],
                'Std. Error': [parallel_model.bse['EVM_1996'], parallel_model.bse['Intercept']],
                'P-value': [parallel_model.pvalues['EVM_1996'], parallel_model.pvalues['Intercept']]
            })
            
            st.dataframe(results_df, width="stretch")
            
            # Interpretation
            if parallel_model.pvalues['EVM_1996'] > 0.05:
                st.success(f"""
                ✅ **RESULT: Parallel Trends Assumption HOLDS**
                
                - Coefficient: {parallel_model.params['EVM_1996']:.4f}
                - P-value: {parallel_model.pvalues['EVM_1996']:.4f} (> 0.05)
                
                Fail to reject null hypothesis. EVM and Paper PCs changed at the same rate before 1999.
                """)
            else:
                st.error(f"""
                ⚠️ **RESULT: Trends Diverging Before Treatment**
                
                - Coefficient: {parallel_model.params['EVM_1996']:.4f}
                - P-value: {parallel_model.pvalues['EVM_1996']:.4f} (< 0.05)
                
                Reject null hypothesis. Trends were diverging BEFORE 1999!
                """)
            
            # Create visualization with Plotly
            if df_1999 is not None:
                panel_df = pd.concat([df_1996, df_1998, df_1999], ignore_index=True)
                years_to_plot = [1996, 1998, 1999]
            else:
                panel_df = pd.concat([df_1996, df_1998], ignore_index=True)
                years_to_plot = [1996, 1998]
            
            trends = panel_df.groupby(['Year', 'EVM'])['Female_Turnout'].mean().reset_index()
            
            # Create interactive plot
            evm_trends = trends[trends['EVM'] == 1]
            paper_trends = trends[trends['EVM'] == 0]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=evm_trends['Year'], 
                y=evm_trends['Female_Turnout'],
                mode='lines+markers',
                name='EVM PCs (Treatment)',
                line=dict(color='#6A0DAD', width=3),
                marker=dict(size=10)
            ))
            
            fig.add_trace(go.Scatter(
                x=paper_trends['Year'], 
                y=paper_trends['Female_Turnout'],
                mode='lines+markers',
                name='Paper Ballot PCs (Control)',
                line=dict(color='#d3d3d3', width=3, dash='dash'),
                marker=dict(size=10)
            ))
            
            # Add vertical line for treatment
            if df_1999 is not None:
                fig.add_shape(
                    type="line",
                    x0=1998.5, y0=trends['Female_Turnout'].min(),
                    x1=1998.5, y1=trends['Female_Turnout'].max(),
                    line=dict(color="red", width=2, dash="dash"),
                )
                fig.add_annotation(
                    x=1998.7, y=trends['Female_Turnout'].max(),
                    text="EVM Introduction (1999)",
                    showarrow=False,
                    font=dict(size=12, color="red")
                )
            
            fig.update_layout(
                title='Parallel Trend Test: Female Turnout by Voting Technology (1996-1999)',
                xaxis_title='General Election Year',
                yaxis_title='Average Female Voter Turnout (%)',
                legend_title='Technology in 1999',
                height=500,
                width="stretch",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, width="stretch")
            
            st.markdown("""
            ### Interpretation
            
            The visual test shows whether EVM and paper-ballot constituencies followed 
            parallel trajectories in female turnout before the 1999 election. 
            
            **Key Insights**:
            - Flat or parallel lines in 1996-1998 validate the research design
            - Divergence only after 1999 supports causal interpretation
            - Pre-trends test is more credible than post-hoc controls
            """)
        else:
            st.error("Could not load election data. Please check the data files.")
    else:
        st.warning("""
        **Data Files Not Found**
        
        Please ensure the following files exist in the `data/` folder:
        - `1996_election_data_corrected.csv`
        - `1998_election_data_corrected.csv`
        - `1999_election_data_corrected.csv` (optional for extended plot)
        
        Required columns: `Constituency`, `Voted_Female`, `Electors_Female`
        """)


# ---------------------------------------------------------
# TAB 2: THE CORE CODE
# ---------------------------------------------------------
with tab_code:
    st.markdown("### Under the Hood: Original Econometric Implementation")
    
    st.info("""
    **Note**: This is the original analysis script from the `/script` folder, 
    not the Streamlit wrapper code. This demonstrates the actual research logic.
    """)
    
    # Read the original script file
    original_script_path = "script/ STEP 2- PARALLEL TREND TEST (1996 vs 1998).py"
    
    if os.path.exists(original_script_path):
        with open(original_script_path, "r") as f:
            source_code = f.read()
        
        st.code(source_code, language="python")
    else:
        st.warning(f"""
        Original script not found at `{original_script_path}`.
        
        Please ensure the script files are in the `script/` folder.
        """)
        
        # Fallback: show current file code
        with open(__file__, "r") as f:
            source_code = f.read()
        st.code(source_code, language="python")
