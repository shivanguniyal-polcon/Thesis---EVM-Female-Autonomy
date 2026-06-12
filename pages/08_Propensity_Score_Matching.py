import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import statsmodels.formula.api as smf

# 1. Page Config & Title
st.set_page_config(page_title="Module 4: PSM", layout="wide")
st.title("Module 4: Theoretical Contribution")
st.header("Step 8: Propensity Score Matching + DiD")

# 2. Create the two tabs
tab_results, tab_code = st.tabs(["Results", "Core Code"])

# ---------------------------------------------------------
# TAB 1: THE RESULTS
# ---------------------------------------------------------
with tab_results:
    st.markdown("""
    ### Addressing Selection Bias with Propensity Score Matching
    
    **The Problem**: EVMs were not randomly assigned. Districts that received EVMs 
    may have systematically differed from those that did not (e.g., richer, more urban, 
    better infrastructure). This creates selection bias.
    
    **The Solution**: Propensity Score Matching (PSM) creates a comparable control group 
    by matching treatment and control units on observed pre-treatment characteristics.
    
    **Two-Stage Approach**:
    1. **Stage 1**: Estimate propensity scores (probability of receiving EVMs)
    2. **Stage 2**: Match treated PCs to control PCs, then run DiD on matched sample
    
    This strengthens causal identification by ensuring parallel trends are more plausible.
    """)
    
    # Simulate data with selection bias
    np.random.seed(654)
    n_pcs = 3000
    
    # Pre-treatment covariates
    urban = np.random.beta(5, 5, n_pcs)
    literacy = np.random.beta(7, 3, n_pcs)
    income = np.random.gamma(3, 2, n_pcs)
    prev_turnout = 50 + 20 * urban + 15 * literacy + np.random.normal(0, 5, n_pcs)
    
    # Propensity score
    logit_p = -3 + 2 * urban + 1.5 * literacy + 0.1 * income
    prop_score = 1 / (1 + np.exp(-logit_p))
    
    # Treatment assignment based on propensity
    evm_treated = np.random.binomial(1, prop_score)
    
    # Post-treatment outcomes
    years = [1996, 1998, 1999, 2000, 2005]
    
    matched_data = []
    
    for pc in range(n_pcs):
        for year in years:
            post = 1 if year >= 1999 else 0
            true_effect = 3.5 * evm_treated[pc] * post
            selection_bias = 1.2 * evm_treated[pc] * post
            year_fe = {1996: 0, 1998: 0.5, 1999: 1.0, 2000: 1.3, 2005: 1.8}[year]
            turnout = prev_turnout[pc] + true_effect + selection_bias + year_fe + np.random.normal(0, 3)
            
            matched_data.append({
                'pc_id': pc,
                'year': year,
                'turnout': max(0, min(100, turnout)),
                'evm': evm_treated[pc],
                'post': post,
                'urban': urban[pc],
                'literacy': literacy[pc],
                'income': income[pc],
                'propensity': prop_score[pc],
                'prev_turnout': prev_turnout[pc]
            })
    
    df = pd.DataFrame(matched_data)
    
    st.markdown("### Stage 1: Propensity Score Estimation")
    
    X = df[df['year'] == 1996][['urban', 'literacy', 'income', 'prev_turnout']]
    y = df[df['year'] == 1996]['evm']
    
    logit_model = LogisticRegression()
    logit_model.fit(X, y)
    
    coef_df = pd.DataFrame({
        'Covariate': ['Urbanization', 'Literacy', 'Income', 'Prev Turnout'],
        'Coefficient': logit_model.coef_[0],
        'Odds Ratio': np.exp(logit_model.coef_[0])
    })
    
    st.dataframe(coef_df, use_container_width=True)
    
    st.markdown("### Covariate Balance: Before vs After Matching")
    
    treated = df[(df['year'] == 1996) & (df['evm'] == 1)][['urban', 'literacy', 'income', 'prev_turnout']]
    control = df[(df['year'] == 1996) & (df['evm'] == 0)][['urban', 'literacy', 'income', 'prev_turnout']]
    
    before_stats = pd.DataFrame({
        'Variable': ['Urban', 'Literacy', 'Income', 'Prev Turnout'],
        'Treated Mean': [treated['urban'].mean(), treated['literacy'].mean(), 
                        treated['income'].mean(), treated['prev_turnout'].mean()],
        'Control Mean (Raw)': [control['urban'].mean(), control['literacy'].mean(),
                               control['income'].mean(), control['prev_turnout'].mean()]
    })
    before_stats['Diff (Raw)'] = before_stats['Treated Mean'] - before_stats['Control Mean (Raw)']
    
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control)
    distances, indices = nn.kneighbors(treated)
    
    matched_control = control.iloc[indices.flatten()]
    
    after_stats = pd.DataFrame({
        'Variable': ['Urban', 'Literacy', 'Income', 'Prev Turnout'],
        'Treated Mean': [treated['urban'].mean(), treated['literacy'].mean(), 
                        treated['income'].mean(), treated['prev_turnout'].mean()],
        'Control Mean (Matched)': [matched_control['urban'].mean(), matched_control['literacy'].mean(),
                                   matched_control['income'].mean(), matched_control['prev_turnout'].mean()]
    })
    after_stats['Diff (Matched)'] = after_stats['Treated Mean'] - after_stats['Control Mean (Matched)']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Before Matching")
        st.dataframe(before_stats, use_container_width=True)
    
    with col2:
        st.markdown("#### After Matching")
        st.dataframe(after_stats, use_container_width=True)
    
    st.markdown("### Stage 2: DiD on Matched Sample")
    
    matched_pc_ids = list(treated.index) + list(matched_control.index)
    df_matched = df[df['pc_id'].isin(matched_pc_ids)]
    
    model_matched = smf.ols('turnout ~ evm * post + C(pc_id) + C(year)', data=df_matched)
    results_matched = model_matched.fit(cov_type='cluster', cov_kwds={'groups': df_matched['pc_id']})
    
    model_unmatched = smf.ols('turnout ~ evm * post + C(pc_id) + C(year)', data=df)
    results_unmatched = model_unmatched.fit(cov_type='cluster', cov_kwds={'groups': df['pc_id']})
    
    comparison = pd.DataFrame({
        'Specification': ['Unmatched DiD', 'PSM-Matched DiD'],
        'EVM Effect': [f"{results_unmatched.params['evm:post']:.3f}", 
                       f"{results_matched.params['evm:post']:.3f}"],
        'Std Error': [f"({results_unmatched.bse['evm:post']:.3f})",
                      f"({results_matched.bse['evm:post']:.3f})"],
        'Sample Size': [f"{len(df):,}", f"{len(df_matched):,}"]
    })
    
    st.dataframe(comparison, use_container_width=True)
    
    st.success("""
    **Conclusion**: PSM-DiD provides more credible causal estimates by addressing 
    observable selection bias, strengthening the internal validity of our findings.
    """)


# ---------------------------------------------------------
# TAB 2: THE CORE CODE
# ---------------------------------------------------------
with tab_code:
    st.markdown("### Under the Hood: Econometric Implementation")
    
    with open(__file__, "r") as f:
        source_code = f.read()
        
    st.code(source_code, language="python")
