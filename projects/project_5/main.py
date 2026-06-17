# EVM & Female Autonomy Analytical Pipeline
# Copyright (C) 2026 Shivang Uniyal
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

import os
import warnings
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.diagnostic import linear_reset
from scipy.stats import normaltest, jarque_bera, skew, kurtosis
import statsmodels.stats.stattools as stattools
import statsmodels.stats.diagnostic as diag
import scipy.stats as stats
import statsmodels.stats.outliers_influence as influence
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.discrete.discrete_model import Logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess

warnings.filterwarnings('ignore')

BASE_DIR = "/Users/ganeshchandrauniyal/Desktop/Thesis Script"
CSV_DIR = os.path.join(BASE_DIR, "CSV_OUTPUTS")
os.makedirs(CSV_DIR, exist_ok=True)

# [CORE START]

TREATED_PCS = [
    'HYDERABAD', 'SECUNDERABAD', 'PANAJI', 'MORMUGAO', 'AHMEDABAD', 'GANDHINAGAR',
    'KARNAL', 'ROHTAK', 'BANGALORE NORTH', 'BANGALORE SOUTH', 'MYSORE',
    'ERNAKULAM', 'TRIVANDRUM', 'GWALIOR', 'BHOPAL', 'MUMBAI SOUTH', 
    'MUMBAI SOUTH CENTRAL', 'MUMBAI NORTH CENTRAL', 'MUMBAI NORTH EAST', 
    'MUMBAI NORTH WEST', 'BHUBANESWAR', 'TARN TARAN', 'PATIALA', 'FARIDKOT',
    'JAIPUR', 'AJMER', 'MADRAS CENTRAL', 'MADRAS SOUTH', 'COIMBATORE', 'MADURAI',
    'LUCKNOW', 'ALLAHABAD', 'KANPUR', 'AGRA', 'CALCUTTA NORTH WEST', 
    'CALCUTTA NORTH EAST', 'CALCUTTA SOUTH', 'CHANDIGARH', 'NEW DELHI', 
    'SOUTH DELHI', 'OUTER DELHI', 'EAST DELHI', 'CHANDNI CHOWK', 'DELHI SADAR', 
    'KAROL BAGH', 'PUDUCHERRY'
]

CONTAMINATED_1998_PCS = [
    'NEW DELHI', 'SOUTH DELHI', 'OUTER DELHI', 'EAST DELHI', 
    'CHANDNI CHOWK', 'DELHI SADAR', 'KAROL BAGH', 'GWALIOR', 'BHOPAL', 'JAIPUR', 'AJMER'
]

def export_ols_to_csv(model, model_name, save_path):
    ci = model.conf_int()
    res_df = pd.DataFrame({
        'Model': model_name,
        'Variable': model.params.index,
        'Coefficient': model.params.values,
        'Std_Error': model.bse.values,
        'P_Value': model.pvalues.values,
        'CI_Lower': ci[0].values,
        'CI_Upper': ci[1].values,
        'N': int(model.nobs),
        'R2': model.rsquared,
        'Adj_R2': model.rsquared_adj,
        'F_Stat': model.fvalue,
        'F_PVal': model.f_pvalue,
        'AIC': model.aic,
        'BIC': model.bic
    })
    res_df.to_csv(save_path, index=False)
    print(f"Saved regression results to: {save_path}")

def aggregate_year(file_name, year):
    df = pd.read_csv(os.path.join(BASE_DIR, file_name))
    df['pc_name_clean'] = df['Constituency Clean'].str.upper().fillna('')
    df['pc_name_clean'] = df['pc_name_clean'].replace({
        'PONDICHERRY': 'PUDUCHERRY',
        'MUMBAI SOUTH CENTRA': 'MUMBAI SOUTH CENTRAL'
    })
    
    if year == 1998:
        df = df[~df['pc_name_clean'].isin(CONTAMINATED_1998_PCS)].copy()
        
    m = pd.merge(cw, df, on='pc_name_clean', how='inner')
    weight_col = 'pc_weight_relative' if 'pc_weight_relative' in m.columns else 'pc_weight'
    
    m['alloc_electors_f'] = m['Electors_Female'] * m[weight_col]
    m['alloc_voters_f'] = m['Voted_Female'] * m[weight_col]

    dist = m.groupby(['pc91_state_id', 'pc91_district_id', 'state_clean']).agg({
        'alloc_electors_f': 'sum', 'alloc_voters_f': 'sum'
    }).reset_index()

    dist['Female_Turnout'] = (dist['alloc_voters_f'] / dist['alloc_electors_f']) * 100
    dist['Year'] = year
    return dist

def compute_evm_exposure(file_name_1999):
    df = pd.read_csv(os.path.join(BASE_DIR, file_name_1999))
    df['pc_name_clean'] = df['Constituency Clean'].str.upper().fillna('')
    df['pc_name_clean'] = df['pc_name_clean'].replace({
        'PONDICHERRY': 'PUDUCHERRY',
        'MUMBAI SOUTH CENTRA': 'MUMBAI SOUTH CENTRAL'
    })
    df['is_evm'] = df['pc_name_clean'].isin(TREATED_PCS).astype(int)
    
    m = pd.merge(cw, df, on='pc_name_clean', how='inner')
    weight_col = 'pc_weight_relative' if 'pc_weight_relative' in m.columns else 'pc_weight'
    
    m['alloc_electors_f'] = m['Electors_Female'] * m[weight_col]
    m['treated_electors_f'] = m['alloc_electors_f'] * m['is_evm']

    dist = m.groupby(['pc91_state_id', 'pc91_district_id', 'state_clean']).agg({
        'alloc_electors_f': 'sum', 'treated_electors_f': 'sum'
    }).reset_index()

    dist['EVM_Exposure_Cont'] = dist['treated_electors_f'] / dist['alloc_electors_f']
    dist['EVM_Exposure_Binary'] = (dist['EVM_Exposure_Cont'] >= 0.01).astype(int)
    
    merged_treated = m[m['is_evm'] == 1]['pc_name_clean'].unique()
    print(f"Total treated PCs in TREATED_PCS list: {len(TREATED_PCS)}")
    print(f"Treated PCs that successfully merged: {len(merged_treated)}")
    missing_treated = [pc for pc in TREATED_PCS if pc not in merged_treated]
    if missing_treated:
        print(f"Treated PCs that FAILED to merge: {missing_treated}")
    else:
        print("ALL TREATED PCs SUCCESSFULLY HARMONIZED AND MERGED!")
        
    return dist[['pc91_state_id', 'pc91_district_id', 'state_clean', 'EVM_Exposure_Cont', 'EVM_Exposure_Binary']]

def test_all_functional_forms(df, target_var, vars_to_test, base_controls):
    results = []
    df_local = df.copy() 
    for var in vars_to_test:
        df_local[f'{var}_sq'] = df_local[var] ** 2
        df_local[f'{var}_log'] = np.log(df_local[var].clip(lower=1e-6)) 
        df_local[f'{var}_ihs'] = np.arcsinh(df_local[var])
        other_controls = [c for c in base_controls if c != var]
        control_str = " + ".join(other_controls) if other_controls else "1"
        pval_sq = np.nan  
        try:
            mod_lin = smf.ols(f"{target_var} ~ {var} + {control_str}", data=df_local).fit()
            bic_lin = mod_lin.bic
        except Exception:
            bic_lin = np.inf
        try:
            mod_quad = smf.ols(f"{target_var} ~ {var} + {var}_sq + {control_str}", data=df_local).fit()
            bic_quad = mod_quad.bic
            pval_sq = mod_quad.pvalues[f'{var}_sq']
        except Exception:
            bic_quad, pval_sq = np.inf, np.nan
        try:
            mod_log = smf.ols(f"{target_var} ~ {var}_log + {control_str}", data=df_local).fit()
            bic_log = mod_log.bic
        except Exception:
            bic_log = np.inf
        try:
            mod_ihs = smf.ols(f"{target_var} ~ {var}_ihs + {control_str}", data=df_local).fit()
            bic_ihs = mod_ihs.bic
        except Exception:
            bic_ihs = np.inf
        models = {'Linear': bic_lin, 'Quadratic': bic_quad, 'Log': bic_log, 'IHS': bic_ihs}
        best_model = min(models, key=models.get)
        results.append({
            'Variable': var, 'Best_Form': best_model,
            'BIC_Linear': round(bic_lin, 1), 'BIC_Quad': round(bic_quad, 1),
            'BIC_Log': round(bic_log, 1), 'BIC_IHS': round(bic_ihs, 1),
            'Quad_p-val': round(pval_sq, 4) if not np.isnan(pval_sq) else "-"
        })
    return pd.DataFrame(results)

def main():
    global cw
    cw = pd.read_csv(os.path.join(BASE_DIR, "PC2004_to_Dist1991_Weightage_Crosswalk (1).csv"))
    cw['pc_name_clean'] = cw['Constituency Clean'].str.upper()
    cw['pc_name_clean'] = cw['pc_name_clean'].replace({
        'MUMBAI SOUTH CENTRA': 'MUMBAI SOUTH CENTRAL',
        'PONDICHERRY': 'PUDUCHERRY'
    })

    census = pd.read_csv(os.path.join(BASE_DIR, "shrug-pca91-csv/pc91_pca_clean_pc91dist.csv"))
    td = pd.read_csv(os.path.join(BASE_DIR, "shrug-td91-csv/pc91_td_clean_pc91dist.csv"))
    ec98 = pd.read_csv(os.path.join(BASE_DIR, "Data/Economic Census 1991 Data/ec98_aggregated_1991_districts.csv"))

    census['Lit_Pct'] = (census['pc91_pca_p_lit'] / census['pc91_pca_tot_p']) * 100
    census['SC_Pct'] = (census['pc91_pca_p_sc'] / census['pc91_pca_tot_p']) * 100
    census['ST_Pct'] = (census['pc91_pca_p_st'] / census['pc91_pca_tot_p']) * 100

    td_merged = pd.merge(td, census[['pc91_state_id', 'pc91_district_id', 'pc91_pca_tot_p']], 
                         on=['pc91_state_id', 'pc91_district_id'], how='left')
    td_merged['Urban_Pct'] = (td_merged['pc91_td_p_7andup'] / td_merged['pc91_pca_tot_p']) * 100
    td_merged['Urban_Pct'] = td_merged['Urban_Pct'].clip(upper=100)

    urban_subset = td_merged[['pc91_state_id', 'pc91_district_id', 'Urban_Pct']].drop_duplicates()
    ec98['Fem_Enterprise_Pct'] = (ec98['ec98_count_own_f'] / ec98['ec98_count_all']) * 100

    panel = pd.concat([
        aggregate_year("1996_election_data_cleaned.csv", 1996),
        aggregate_year("1998_election_data_cleaned.csv", 1998),
        aggregate_year("1999_election_data_cleaned.csv", 1999)
    ], ignore_index=True)

    panel = pd.merge(panel, census[['pc91_state_id', 'pc91_district_id', 'Lit_Pct', 'SC_Pct', 'ST_Pct']], on=['pc91_state_id', 'pc91_district_id'], how='left')
    panel = pd.merge(panel, urban_subset, on=['pc91_state_id', 'pc91_district_id'], how='left')
    panel = pd.merge(panel, ec98[['pc91_state_id', 'pc91_district_id', 'Fem_Enterprise_Pct']], on=['pc91_state_id', 'pc91_district_id'], how='left')
    panel = panel.dropna()

    wide = panel.pivot_table(
        index=['pc91_state_id', 'pc91_district_id', 'state_clean', 
               'Lit_Pct', 'SC_Pct', 'ST_Pct', 'Fem_Enterprise_Pct', 'Urban_Pct'],
        columns='Year', 
        values='Female_Turnout'    ).reset_index()

    wide = wide.rename(columns={1996: 'Turnout_96', 1998: 'Turnout_98', 1999: 'Turnout_99'})
    wide = wide.dropna(subset=['Turnout_96', 'Turnout_98', 'Turnout_99'])

    evm_exposure = compute_evm_exposure("1999_election_data_cleaned.csv")
    wide = pd.merge(wide, evm_exposure, on=['pc91_state_id', 'pc91_district_id', 'state_clean'], how='left')
    wide['EVM_Exposure_Cont'] = wide['EVM_Exposure_Cont'].fillna(0)
    wide['EVM_Exposure_Binary'] = wide['EVM_Exposure_Binary'].fillna(0).astype(int)

    print(f"Final Clean Sample: {len(wide)} districts.")
    print(f"Treated Districts (EVM Exposure > 0): {(wide['EVM_Exposure_Cont'] > 0).sum()}")

    wide['ST_Pct_ihs'] = np.arcsinh(wide['ST_Pct'])
    wide['Fem_Enterprise_Pct_ihs'] = np.arcsinh(wide['Fem_Enterprise_Pct'])
    wide['Urban_Pct_ihs'] = np.arcsinh(wide['Urban_Pct'])

    agency_mean = wide['Fem_Enterprise_Pct_ihs'].mean()
    wide['Agency_Centered'] = wide['Fem_Enterprise_Pct_ihs'] - agency_mean
    print(f"Mean-centering Agency. Original IHS Mean: {agency_mean:.4f}")

    wide['EVM_Post_Agency'] = wide['EVM_Exposure_Cont'] * wide['Agency_Centered']

    final_formula = (
        "Turnout_99 ~ EVM_Exposure_Cont + Agency_Centered + EVM_Post_Agency + Urban_Pct_ihs + "
        "Turnout_96 + Lit_Pct + SC_Pct + ST_Pct_ihs + C(state_clean)"
    )

    final_model = smf.ols(final_formula, data=wide).fit(cov_type='HC1')

    print("--- FINAL MODEL COEFFICIENTS (Mean-Centered) ---")
    res_df = pd.DataFrame({
        'Coefficient': final_model.params,
        'Std. Error': final_model.bse,
        'P-Value': final_model.pvalues,
    })
    print(res_df.to_string(float_format="%.4f"))

    print("MARGINAL EFFECTS OF EVM ACROSS AGENCY DISTRIBUTION")
    print(f"{'Percentile':<12} | {'Raw Agency %':<15} | {'Centered IHS':<15} | {'Marginal Effect':<18} | {'P-Value'}")
    beta_evm = final_model.params['EVM_Exposure_Cont']
    beta_int = final_model.params['EVM_Post_Agency']
    cov_matrix = final_model.cov_params()
    var_evm = cov_matrix.loc['EVM_Exposure_Cont', 'EVM_Exposure_Cont']
    var_int = cov_matrix.loc['EVM_Post_Agency', 'EVM_Post_Agency']
    cov_evm_int = cov_matrix.loc['EVM_Exposure_Cont', 'EVM_Post_Agency']

    for p in [0.10, 0.25, 0.50, 0.75, 0.90]:
        raw_val = wide['Fem_Enterprise_Pct'].quantile(p)
        ihs_val = np.arcsinh(raw_val)
        centered_val = ihs_val - agency_mean
        me = beta_evm + beta_int * centered_val
        se_me = np.sqrt(var_evm + (centered_val**2)*var_int + 2*centered_val*cov_evm_int)
        z_score = me / se_me
        p_val = 2 * (1 - stats.norm.cdf(abs(z_score)))
        print(f"{int(p*100):<12} | {raw_val:<15.2f} | {centered_val:<15.4f} | {me:<18.4f} | {p_val:.4f}")

    print("GENERATING MAIN MODEL FOREST PLOT")
    key_vars = ['EVM_Exposure_Cont', 'Agency_Centered', 'EVM_Post_Agency']
    coefs = [final_model.params[var] for var in key_vars]
    conf_ints = [final_model.conf_int().loc[var] for var in key_vars]
    lower_bounds = [ci[0] for ci in conf_ints]
    upper_bounds = [ci[1] for ci in conf_ints]
    errors = [[c - l, u - c] for c, l, u in zip(coefs, lower_bounds, upper_bounds)]

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(key_vars))
    ax.errorbar(coefs, y_pos, xerr=np.array(errors).T, fmt='o', color='#6A0DAD', 
                ecolor='gray', capsize=5, markersize=8)
    ax.axvline(0, color='red', linestyle='--', lw=1.5, label='Zero Effect')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(['EVM Exposure (Main)', 'Economic Agency (Main)', 'EVM x Agency (Interaction)'], fontsize=11)
    ax.set_xlabel('Coefficient Estimate (with 95% CI)', fontsize=12)
    ax.set_title('Main Causal Model: Key Coefficients', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, "Main_Model_Forest_Plot.png"), dpi=300)
    plt.show()

    print("COVARIATE BALANCE (SELECTION TEST)")
    covariates = ['Lit_Pct', 'SC_Pct', 'ST_Pct', 'Fem_Enterprise_Pct', 'Urban_Pct', 'Turnout_96']
    for cov in covariates:
        mod = smf.ols(f"{cov} ~ EVM_Exposure_Cont", data=wide).fit(cov_type='HC3')
        print(f"{cov}: Beta={mod.params['EVM_Exposure_Cont']:.4f} (p={mod.pvalues['EVM_Exposure_Cont']:.3f})")

    print("GENERATING COVARIATE BALANCE LOVE PLOT")
    balance_coefs, balance_pvals = [], []
    for cov in covariates:
        mod = smf.ols(f"{cov} ~ EVM_Exposure_Cont", data=wide).fit(cov_type='HC3')
        balance_coefs.append(mod.params['EVM_Exposure_Cont'])
        balance_pvals.append(mod.pvalues['EVM_Exposure_Cont'])

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(covariates))
    colors = ['#d9534f' if p < 0.05 else '#5cb85c' for p in balance_pvals]

    ax.barh(y_pos, balance_coefs, color=colors, edgecolor='black', alpha=0.8)
    ax.axvline(0, color='black', linestyle='-', lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(covariates, fontsize=11)
    ax.set_xlabel('Coefficient of EVM Exposure on Covariate', fontsize=12)
    ax.set_title('Covariate Balance Test (Selection on Observables)\n(Red = p < 0.05, Green = p >= 0.05)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, "Covariate_Balance_Love_Plot.png"), dpi=300)
    plt.show()

    print("FRISCH-WAUGH-LOVELL (FWL) THEOREM")
    controls = "Turnout_96 + Lit_Pct + SC_Pct + ST_Pct_ihs + Agency_Centered + Urban_Pct_ihs + EVM_Exposure_Cont + C(state_clean)"
    res_y = smf.ols(f"Turnout_99 ~ {controls}", data=wide).fit().resid
    res_x = smf.ols(f"EVM_Post_Agency ~ {controls}", data=wide).fit().resid
    fwl_df = pd.DataFrame({'Res_Y': res_y, 'Res_X': res_x})
    fwl_mod = smf.ols("Res_Y ~ Res_X", data=fwl_df).fit()
    print(f"FWL Interaction Beta: {fwl_mod.params['Res_X']:.4f}")

    print("GENERATING FWL PARTIAL REGRESSION PLOT")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(fwl_df['Res_X'], fwl_df['Res_Y'], alpha=0.5, edgecolors='k', color='#6A0DAD', s=40)

    x_range = np.linspace(fwl_df['Res_X'].min(), fwl_df['Res_X'].max(), 100)
    intercept = fwl_mod.params.get('Intercept', 0.0)
    y_hat = intercept + fwl_mod.params['Res_X'] * x_range
    ax.plot(x_range, y_hat, color='red', lw=2.5, label=f'FWL Beta = {fwl_mod.params["Res_X"]:.4f}')

    ax.set_xlabel('EVM x Agency (Residualized)', fontsize=12)
    ax.set_ylabel('Turnout 1999 (Residualized)', fontsize=12)
    ax.set_title('Frisch-Waugh-Lovell (FWL) Partial Regression Plot', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, "FWL_Partial_Regression_Plot.png"), dpi=300)
    plt.show()

    print("GENERATING FWL PARTIAL RESIDUAL PLOTS FOR ALL COVARIATES")
    continuous_covariates = [
        'EVM_Exposure_Cont', 'Agency_Centered', 'Urban_Pct_ihs', 
        'Turnout_96', 'Lit_Pct', 'SC_Pct', 'ST_Pct_ihs'
    ]

    n_cols = 3
    n_rows = int(np.ceil(len(continuous_covariates) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten()

    for i, cov in enumerate(continuous_covariates):
        ax = axes[i]
        beta_j = final_model.params[cov]
        partial_resid = final_model.resid + (beta_j * wide[cov])
        ax.scatter(wide[cov], partial_resid, alpha=0.4, edgecolors='k', color='#6A0DAD', s=30, label='Partial Residuals')
        x_range = np.linspace(wide[cov].min(), wide[cov].max(), 100)
        m, b = np.polyfit(wide[cov], partial_resid, 1)
        ax.plot(x_range, m * x_range + b, color='red', lw=2.5, linestyle='--', label=f'Linear Fit (β={beta_j:.3f})')
        lowess_line = lowess(partial_resid, wide[cov], frac=0.6)
        ax.plot(lowess_line[:, 0], lowess_line[:, 1], color='orange', lw=3, label='LOWESS (Non-linear check)')
        ax.set_xlabel(cov, fontsize=12, fontweight='bold')
        ax.set_ylabel('Partial Residuals (FWL Adjusted)', fontsize=12)
        ax.set_title(f'FWL Partial Residual Plot: {cov}', fontsize=13)
        ax.grid(True, linestyle=':', alpha=0.6)
        if i == 0:
            ax.legend(loc='best', fontsize=9)

    for j in range(len(continuous_covariates), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('Frisch-Waugh-Lovell Diagnostics: Partial Residual (Component-Plus-Residual) Plots\n(Visualizing isolated effects after partialling out controls)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, "FWL_Partial_Residuals_Grid.png"), dpi=300, bbox_inches='tight')
    print("Plot saved as 'FWL_Partial_Residuals_Grid.png'.")
    plt.show()

    print("LEAVE-ONE-OUT (STATE LEVEL)")
    interaction_terms = [p for p in final_model.params.index if 'EVM' in p and ('Agency' in p or 'Fem' in p or 'Post' in p) and p != 'EVM_Exposure_Cont']
    target_param = interaction_terms[0] if interaction_terms else 'EVM_Post_Agency'
    print(f"Target interaction parameter: '{target_param}'")

    states = wide['state_clean'].unique()
    loo_results = []
    for st in states:
        temp_df = wide[wide['state_clean'] != st]
        if temp_df['EVM_Exposure_Cont'].sum() > 0 and (temp_df['EVM_Exposure_Cont'] == 0).sum() > 0:
            try:
                mod = smf.ols(final_formula, data=temp_df).fit(cov_type='HC1')
                if target_param in mod.params.index:
                    loo_results.append({'Dropped_State': st, 'Beta': mod.params[target_param]})
            except Exception:
                pass 
            
    loo_df = pd.DataFrame(loo_results)
    baseline_beta = final_model.params[target_param]
    print(f"Baseline Interaction Beta: {baseline_beta:.4f}")
    if not loo_df.empty:
        print(f"LOO Mean Beta: {loo_df['Beta'].mean():.4f}")
        print(f"LOO Std Dev: {loo_df['Beta'].std():.4f}")
    else:
        print("LOO returned no results. Check if states have enough variation.")

    print("GENERATING LEAVE-ONE-OUT SENSITIVITY PLOT")
    if not loo_df.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(loo_df))
        ax.errorbar(loo_df['Beta'], y_pos, xerr=0, fmt='o', color='gray', markersize=7, label='LOO Estimate (State Dropped)')
        ax.axvline(baseline_beta, color='#6A0DAD', linestyle='--', lw=2.5, label=f'Baseline Estimate ({baseline_beta:.4f})')
        ax.axvline(0, color='red', linestyle=':', lw=1.5, label='Zero Effect')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(loo_df['Dropped_State'], fontsize=10)
        ax.set_xlabel('Interaction Coefficient (EVM x Agency)', fontsize=12)
        ax.set_title('Leave-One-Out Sensitivity Analysis (Dropping One State at a Time)', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.legend(loc='lower right')
        plt.tight_layout()
        fig.savefig(os.path.join(BASE_DIR, "LOO_Sensitivity_Plot.png"), dpi=300)
        plt.show()

    print("SUB-SAMPLE HETEROGENEITY")
    median_agency_orig = wide['Fem_Enterprise_Pct'].median()
    high_agency = wide[wide['Fem_Enterprise_Pct'] >= median_agency_orig]
    low_agency = wide[wide['Fem_Enterprise_Pct'] < median_agency_orig]
    sub_formula = "Turnout_99 ~ EVM_Exposure_Cont + Agency_Centered + EVM_Post_Agency + Urban_Pct_ihs + Turnout_96 + Lit_Pct + C(state_clean)"
    mod_high = smf.ols(sub_formula, data=high_agency).fit(cov_type='HC3')
    mod_low = smf.ols(sub_formula, data=low_agency).fit(cov_type='HC3')
    print(f"High Agency Beta: {mod_high.params['EVM_Post_Agency']:.4f} (p={mod_high.pvalues['EVM_Post_Agency']:.3f})")
    print(f"Low Agency Beta:  {mod_low.params['EVM_Post_Agency']:.4f} (p={mod_low.pvalues['EVM_Post_Agency']:.3f})")

    print("GENERATING SUB-SAMPLE HETEROGENEITY PLOT")
    sub_betas = [mod_high.params['EVM_Post_Agency'], mod_low.params['EVM_Post_Agency']]
    sub_ci_high = mod_high.conf_int().loc['EVM_Post_Agency']
    sub_ci_low = mod_low.conf_int().loc['EVM_Post_Agency']
    sub_errors = [
        [sub_betas[0] - sub_ci_high[0], sub_ci_high[1] - sub_betas[0]],
        [sub_betas[1] - sub_ci_low[0], sub_ci_low[1] - sub_betas[1]]
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ['High Agency (>= Median)', 'Low Agency (< Median)']
    x_pos = np.arange(len(labels))

    ax.errorbar(sub_betas, x_pos, xerr=np.array(sub_errors).T, fmt='s', color='#6A0DAD', 
                ecolor='black', capsize=6, markersize=10)
    ax.axvline(0, color='red', linestyle='--', lw=1.5)

    ax.set_yticks(x_pos)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel('Interaction Effect (EVM x Agency)', fontsize=12)
    ax.set_title('Heterogeneity by Baseline Female Enterprise Density', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, "Subsample_Heterogeneity_Plot.png"), dpi=300)
    plt.show()

    print("TABLE 1: SUMMARY STATISTICS OF KEY VARIABLES")
    summary_cols = ['Turnout_99', 'Turnout_96', 'EVM_Exposure_Cont', 'Fem_Enterprise_Pct', 'Urban_Pct', 'Lit_Pct']
    summary_cols = [c for c in summary_cols if c in wide.columns]
    print(wide[summary_cols].describe().T[['count', 'mean', 'std', 'min', 'max']].to_string(float_format="%.3f"))

    print("TABLE 2: FINAL REGRESSION RESULTS & DIAGNOSTICS")
    model_standard = smf.ols(final_formula, data=wide).fit()
    model_robust = smf.ols(final_formula, data=wide).fit(cov_type='HC1')

    n_obs = int(model_standard.nobs)
    df_model = int(model_standard.df_model)
    df_resid = int(model_standard.df_resid)
    r2 = model_standard.rsquared
    r2_adj = model_standard.rsquared_adj
    f_stat = model_standard.fvalue
    f_pval = model_standard.f_pvalue
    aic = model_standard.aic
    bic = model_standard.bic
    cond_num = model_standard.condition_number
    dw_stat = stattools.durbin_watson(model_standard.resid)
    omnibus_stat, omnibus_pval = normaltest(model_standard.resid)
    jb_stat, jb_pval = jarque_bera(model_standard.resid)
    jb_skew = skew(model_standard.resid)
    jb_kurt = kurtosis(model_standard.resid)
    bp_test = diag.het_breuschpagan(model_standard.resid, model_standard.model.exog)
    bp_stat, bp_pval, bp_f, bp_fpval = bp_test

    print(f"{'Metric':<35} | {'Value':<20}")
    print(f"{'Observations (N)':<35} | {n_obs:<20}")
    print(f"{'Degrees of Freedom (Model)':<35} | {df_model:<20}")
    print(f"{'Degrees of Freedom (Residual)':<35} | {df_resid:<20}")
    print(f"{'R-squared':<35} | {r2:<20.4f}")
    print(f"{'Adjusted R-squared':<35} | {r2_adj:<20.4f}")
    print(f"{'F-statistic':<35} | {f_stat:<20.4f}")
    print(f"{'Prob (F-statistic)':<35} | {f_pval:<20.4e}")
    print(f"{'Akaike Info Criterion (AIC)':<35} | {aic:<20.2f}")
    print(f"{'Bayesian Info Criterion (BIC)':<35} | {bic:<20.2f}")
    print(f"{'Condition Number':<35} | {cond_num:<20.2f}")
    print(f"{'Durbin-Watson Statistic':<35} | {dw_stat:<20.4f}")
    print(f"{'Omnibus K^2 (Normality)':<35} | {omnibus_stat:<20.4f}")
    print(f"{'Prob (Omnibus)':<35} | {omnibus_pval:<20.4e}")
    print(f"{'Jarque-Bera (JB)':<35} | {jb_stat:<20.4f}")
    print(f"{'Prob (JB)':<35} | {jb_pval:<20.4e}")
    print(f"{'Skewness':<35} | {jb_skew:<20.4f}")
    print(f"{'Kurtosis':<35} | {jb_kurt:<20.4f}")
    print(f"{'Breusch-Pagan (Heteroskedasticity)':<35} | {bp_stat:<20.4f}")
    print(f"{'Prob (Breusch-Pagan)':<35} | {bp_pval:<20.4e}")

    print("HC1 Coeffcients FOR REPORTING")
    try:
        print(model_robust.summary2().tables[1])
    except Exception:
        print(res_df.to_string(float_format="%.4f"))

    print("VIF FOR CONTINUOUS REGRESSORS (Excluding State FEs) - note VIF>5")
    vif_vars = ['EVM_Exposure_Cont', 'Agency_Centered', 'EVM_Post_Agency', 'Urban_Pct_ihs', 
                'Turnout_96', 'Lit_Pct', 'SC_Pct', 'ST_Pct_ihs']
    X_vif = wide[vif_vars].copy()
    X_vif = sm.add_constant(X_vif)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    print(vif_data[vif_data["Variable"] != "const"].to_string(index=False))

    print("RE-ESTIMATING WITH STATE-CLUSTERED STANDARD ERRORS")
    model_clustered = smf.ols(final_formula, data=wide).fit(
        cov_type='cluster', cov_kwds={'groups': wide['state_clean']}
    )
    print(f"Clustered EVM_Exposure_Cont:{model_clustered.params['EVM_Exposure_Cont']:.4f} (p={model_clustered.pvalues['EVM_Exposure_Cont']:.4f})")
    print(f"Clustered EVM_Post_Agency:{model_clustered.params['EVM_Post_Agency']:.4f} (p={model_clustered.pvalues['EVM_Post_Agency']:.4f})")

    print("--- GENERATING VIF DIAGNOSTICS PLOT ---")
    vif_plot_df = vif_data[vif_data["Variable"]!= "const"].copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#d9534f' if v > 10 else '#f0ad4e' if v > 5 else '#5cb85c' for v in vif_plot_df['VIF']]

    ax.barh(vif_plot_df['Variable'], vif_plot_df['VIF'], color=colors, edgecolor='black', alpha=0.85)
    ax.axvline(5, color='orange', linestyle='--', lw=2, label='Moderate Concern (VIF = 5)')
    ax.axvline(10, color='red', linestyle='--', lw=2, label='Severe Multicollinearity (VIF = 10)')

    ax.set_xlabel('Variance Inflation Factor (VIF)', fontsize=12)
    ax.set_title('Multicollinearity Diagnostics (Excluding State FEs)', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, "VIF_Diagnostics_Plot.png"), dpi=300)
    plt.show()

    print("GENERATING COOK'S DISTANCE PLOT")
    ols_for_influence = smf.ols(final_formula, data=wide).fit()
    influence_obj = influence.OLSInfluence(ols_for_influence)
    cooks_d, _ = influence_obj.cooks_distance

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(range(len(cooks_d)), cooks_d, alpha=0.6, edgecolors='k', s=50, color='#6A0DAD')
    ax.axhline(4 / len(wide), color='red', linestyle='--', lw=2, label='Threshold (4/N)')
    ax.set_xlabel('Observation Index (District)')
    ax.set_ylabel("Cook's Distance")
    ax.set_title("Influence Diagnostics: Checking for High-Leverage Outliers")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(BASE_DIR,"Cooks_Distance_Plot.png"), dpi=300)
    print("Plot saved as 'Cooks_Distance_Plot.png'.")
    plt.show()

    print("GENERATING MARGINAL EFFECTS PLOT")
    agency_range = np.linspace(wide['Fem_Enterprise_Pct'].min(), wide['Fem_Enterprise_Pct'].quantile(0.95), 100)
    ihs_range = np.arcsinh(agency_range)
    centered_range = ihs_range - agency_mean
    me_range = beta_evm + beta_int * centered_range
    se_range = np.sqrt(var_evm + (centered_range**2)*var_int + 2*centered_range*cov_evm_int)
    ci_lower = me_range - 1.96 * se_range
    ci_upper = me_range + 1.96 * se_range

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(agency_range, me_range, color='#6A0DAD', lw=3, label='Marginal Effect of EVM')
    ax.fill_between(agency_range, ci_lower, ci_upper, color='#6A0DAD', alpha=0.2, label='95% Confidence Interval')
    ax.axhline(0, color='red', linestyle='--', lw=2, label='Zero Effect')
    ax.set_title('Heterogeneous Effects: EVM Impact on Female Turnout by Economic Agency', fontsize=14, fontweight='bold')
    ax.set_xlabel('Female Enterprise Density (%)', fontsize=12)
    ax.set_ylabel('Marginal Effect on Female Turnout (Percentage Points)', fontsize=12)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, "Marginal_Effects_Plot.png"), dpi=300)
    print("Plot saved as 'Marginal_Effects_Plot.png'!")
    plt.show()

    print("ROBUSTNESS: EXCLUDING COOK'S DISTANCE OUTLIERS")
    threshold = 4/ len(wide)
    outlier_indices = cooks_d[cooks_d > threshold].index
    outlier_info = wide.loc[outlier_indices, ['state_clean', 'EVM_Exposure_Cont', 'Fem_Enterprise_Pct', 'Turnout_99']]
    print("High-Leverage Outliers Identified:")
    print(outlier_info)
    wide_no_outliers = wide.drop(index=outlier_indices)
    print(f"Re-estimating model with {len(wide_no_outliers)} districts (dropped {len(outlier_indices)} outliers)...")
    model_no_outliers = smf.ols(final_formula, data=wide_no_outliers).fit(cov_type='HC1')
    print(f"EVM_Exposure_Cont:{model_no_outliers.params['EVM_Exposure_Cont']:.4f} (p={model_no_outliers.pvalues['EVM_Exposure_Cont']:.4f})")
    print(f"EVM_Post_Agency: {model_no_outliers.params['EVM_Post_Agency']:.4f} (p={model_no_outliers.pvalues['EVM_Post_Agency']:.4f})")

    print("ROBUSTNESS: PROPENSITY SCORE WEIGHTING (IPW via sklearn)")
    features = ['Lit_Pct', 'Urban_Pct_ihs', 'ST_Pct_ihs', 'SC_Pct','Turnout_96']
    X_ps = wide[features].copy()
    y_ps = wide['EVM_Any']
    scaler = StandardScaler()
    X_ps_scaled = scaler.fit_transform(X_ps)
    log_reg = LogisticRegression(max_iter=1000, C=1.0)
    log_reg.fit(X_ps_scaled, y_ps)

    wide['pscore'] = np.clip(log_reg.predict_proba(X_ps_scaled)[:, 1], 0.05, 0.95)
    wide['ipw_weight'] = (wide['EVM_Any'] / wide['pscore']) + ((1 - wide['EVM_Any']) / (1 - wide['pscore']))
    wide['IPW_Interaction'] = wide['EVM_Any'] * wide['Agency_Centered']
    try:
        model_ipw = smf.wls(ipw_formula, data=wide, weights=wide['ipw_weight']).fit(cov_type='HC1')
        print(f"IPW Main Effect (EVM_Any): {model_ipw.params['EVM_Any']:.4f} (p={model_ipw.pvalues['EVM_Any']:.4f})")
        print(f"IPW Interaction (EVM*Agency):{model_ipw.params['IPW_Interaction']:.4f} (p={model_ipw.pvalues['IPW_Interaction']:.4f})")
    except Exception as e:
        print(f"WLS estimation failed: {e}")

    print("--- GENERATING IPW OVERLAP PLOT ---")
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=wide, x='pscore', hue='EVM_Any', fill=True, common_norm=False, palette=['#d3d3d3', '#6A0DAD'])
    plt.title('Propensity Score Distribution (Common Support Check)')
    plt.xlabel('Propensity Score (Probability of EVM Adoption)')
    plt.ylabel('Density')
    plt.axvline(0.05, color='red', linestyle='--', label='Trimming Bounds (0.05 - 0.95)')
    plt.axvline(0.95, color='red', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "IPW_Overlap_Plot.png"), dpi=300)
    print("Saved 'IPW_Overlap_Plot.png'.")
    plt.show()

    print("ROBUSTNESS: IPW WITH TRIMMED WEIGHT")
    p1 = wide['ipw_weight'].quantile(0.01)
    p99 = wide['ipw_weight'].quantile(0.99)
    wide['ipw_weight_trimmed'] = wide['ipw_weight'].clip(lower=p1, upper=p99)
    print(f"Original Weight Range: {wide['ipw_weight'].min():.2f} to {wide['ipw_weight'].max():.2f}")
    print(f"Trimmed Weight Range (1%-99%): {p1:.2f} to {p99:.2f}")
    try:
        model_ipw_trimmed = smf.wls(ipw_formula, data=wide, weights=wide['ipw_weight_trimmed']).fit(cov_type='HC1')
        print(f"Trimmed IPW Main Effect:{model_ipw_trimmed.params['EVM_Any']:.4f} (p={model_ipw_trimmed.pvalues['EVM_Any']:.4f})")
        print(f"Trimmed IPW Interaction:  {model_ipw_trimmed.params['IPW_Interaction']:.4f} (p={model_ipw_trimmed.pvalues['IPW_Interaction']:.4f})")
    except Exception as e:
        print(f"Trimmed WLS failed: {e}")

    print("DIAGNOSTIC: TREATMENT SPARSITY & OVERLAP")
    treated_count = (wide['EVM_Any'] == 1).sum()
    print(f"Total districts with ANY EVM exposure: {treated_count} out of {len(wide)} ({(treated_count/len(wide))*100:.1f}%)")
    print("\nDistribution of Treated Districts by State:")
    state_dist = wide[wide['EVM_Any'] == 1]['state_clean'].value_counts()
    print(state_dist)
    print("\nStates with ZERO Treated Districts:")
    zero_treatment_states = set(wide['state_clean'].unique()) - set(state_dist.index)
    print(f"{len(zero_treatment_states)} states have no EVM exposure in this sample.")

    print("1998 PRE-TREND TEST (Parallel Trends Validation)")
    if 'Turnout_98' not in wide.columns:
        print("Turnout_98 not found. Ensure the 1998 rename patch was applied to the pivot.")
    else:
        wide['Delta_9698'] = wide['Turnout_98'] - wide['Turnout_96']
        pretrend_formula = (
            "Delta_9698 ~ EVM_Exposure_Cont + Agency_Centered + EVM_Post_Agency + "
            "Urban_Pct_ihs + Lit_Pct + SC_Pct + ST_Pct_ihs + C(state_clean)"
        )
        
        df_pretrend = wide.dropna(subset=['Delta_9698'])
        print(f"Pre-trend sample N: {len(df_pretrend)}")
        
        try:
            pretrend_mod = smf.ols(pretrend_formula, data=df_pretrend).fit(cov_type='HC1')
            
            print("\nDependent Variable: Change in Female Turnout (1996 -> 1998)")
            print(f"EVM_Exposure_Cont (Main):{pretrend_mod.params['EVM_Exposure_Cont']:.4f} (p={pretrend_mod.pvalues['EVM_Exposure_Cont']:.3f})")
            print(f"EVM*Agency (Interaction):{pretrend_mod.params['EVM_Post_Agency']:.4f} (p={pretrend_mod.pvalues['EVM_Post_Agency']:.3f})")
            
            if pretrend_mod.pvalues['EVM_Post_Agency'] > 0.10:
                print("PASS: Pre-trends are parallel. The conditional parallel trends assumption holds.")
                print("The heterogeneous effect of EVMs was NOT present before the 1999 rollout.")
            else:
                print("WARNING: Significant pre-trend detected.")
                print("Differential trends by agency level existed before the 1999 EVM rollout.")
        except Exception as e:
            print(f"Pre-trend test failed: {e}")

        print(f"Districts in main model: {len(wide)}")
        print(f"Districts in pre-trend test: {wide['Delta_9698'].notna().sum()}")
        print(f"Districts DROPPED from pre-trend: {wide['Delta_9698'].isna().sum()}")
        print(wide[wide['Delta_9698'].isna()][['state_clean', 'EVM_Exposure_Cont']].value_counts('state_clean'))

        print("GENERATING PRE-TREND PARALLEL TRENDS PLO")
        try:
            actual_beta = final_model.params['EVM_Post_Agency']
            actual_ci = final_model.conf_int().loc['EVM_Post_Agency']
            
            placebo_beta = pretrend_mod.params['EVM_Post_Agency']
            placebo_ci = pretrend_mod.conf_int().loc['EVM_Post_Agency']
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            ax.errorbar(placebo_beta, 0, 
                        xerr=[[placebo_beta - placebo_ci[0]], [placebo_ci[1] - placebo_beta]], 
                        fmt='o', color='gray', markersize=10, capsize=5, label='Pre-Trend (1996-1998)')
            
            ax.errorbar(actual_beta, 1, 
                        xerr=[[actual_beta - actual_ci[0]], [actual_ci[1] - actual_beta]], 
                        fmt='o', color='#6A0DAD', markersize=10, capsize=5, label='Treatment Effect (1996-1999)')
            
            ax.axvline(0, color='red', linestyle='--', lw=1.5, label='Zero Effect')
            
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['1998 (Placebo)', '1999 (Actual)'], fontsize=12)
            ax.set_xlabel('Coefficient of EVM x Agency Interaction', fontsize=12)
            ax.set_title('Parallel Trends / Pre-Trend Validation', fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            ax.legend()
            plt.tight_layout()
            fig.savefig(os.path.join(BASE_DIR, "Parallel_Trends_PreTrend_Plot.png"), dpi=300)
            plt.show()
        except NameError:
            print("Pre-trend model not estimated yet. Run the pre-trend phase first.")

    print("ECONOMIC AGENCY vs. PATRIARCHY (Z-SCORED)")
    fem_col = 'pc91_pca_f_06'   
    male_col = 'pc91_pca_m_06'  

    if fem_col in census.columns and male_col in census.columns:
        sex_data = census[['pc91_state_id', 'pc91_district_id', fem_col, male_col]].copy()
        sex_data['Sex_Ratio']= (sex_data[fem_col] / sex_data[male_col].replace(0, np.nan)) * 1000
        
        wide = pd.merge(wide, sex_data[['pc91_state_id', 'pc91_district_id', 'Sex_Ratio']], 
                        on=['pc91_state_id', 'pc91_district_id'], how='left')
        
        wide['Sex_Ratio'] = wide['Sex_Ratio'].fillna(wide['Sex_Ratio'].mean())
        sex_mean = wide['Sex_Ratio'].mean()
        wide['Patriarchy_Centered'] = wide['Sex_Ratio'] - sex_mean
        
        print(f"\nChild Sex Ratio (0-6) Mean: {sex_mean:.1f} females per 1000 males")
        
        wide['Agency_Z'] = wide['Agency_Centered'] / wide['Agency_Centered'].std()
        wide['Patriarchy_Z'] = wide['Patriarchy_Centered'] / wide['Patriarchy_Centered'].std()
        
        wide['EVM_Post_Agency_Z'] = wide['EVM_Exposure_Cont'] * wide['Agency_Z']
        wide['EVM_Post_Patriarchy_Z'] = wide['EVM_Exposure_Cont'] * wide['Patriarchy_Z']
        
        horse_race_formula = (
            "Turnout_99 ~ EVM_Exposure_Cont + Agency_Z + Patriarchy_Z + "
            "EVM_Post_Agency_Z + EVM_Post_Patriarchy_Z + Urban_Pct_ihs + Turnout_96 + "
            "Lit_Pct + SC_Pct + ST_Pct_ihs + C(state_clean)"
        )
        
        horse_race_model = smf.ols(horse_race_formula, data=wide).fit(cov_type='HC1')
        
        print("Z-SCORED HORSE RACE RESULTS (Dep Var: Turnout_99)")
        print("Note: Coefficients represent the effect of a 1 Standard Deviation increase in the moderator.")        
        beta_eco = horse_race_model.params['EVM_Post_Agency_Z']
        p_eco = horse_race_model.pvalues['EVM_Post_Agency_Z']
        
        beta_soc = horse_race_model.params['EVM_Post_Patriarchy_Z']
        p_soc = horse_race_model.pvalues['EVM_Post_Patriarchy_Z']
        
        print(f"Economic Agency (1 SD increase):Beta = {beta_eco:.4f}  | p-value = {p_eco:.4f}")
        print(f"Cultural Patriarchy (1 SD increase):Beta = {beta_soc:.4f}  | p-value = {p_soc:.4f}")
        
        print("\nMECHANISM INTERPRETATION")
        if p_eco < 0.05 and p_soc > 0.10:
            print("ECONOMIC WINS: EVMs empower women primarily through economic independence/bargaining power, overcoming even deep-rooted demographic patriarchy.")
        elif p_soc < 0.05 and p_eco > 0.10:
            print("PATRIARCHY WINS: The deep-rooted cultural environment (Sex Ratio) dictates EVM efficacy more than individual enterprise.")
        elif p_eco < 0.05 and p_soc < 0.10:
            print("DUAL BURDEN: Both economic capacity and a less patriarchal environment are independent, binding constraints.")
        else:
            print("Neither mechanism is individually significant when controlling for the other.")
            
        corr = wide['Agency_Z'].corr(wide['Patriarchy_Z'])
        print(f"\nCorrelation between Z-scored Agency and Patriarchy: {corr:.3f}")

        print("GENERATING MECHANISM HORSE RACE PLOT")
        key_vars_race = ['EVM_Post_Agency_Z', 'EVM_Post_Patriarchy_Z']
        labels_race = ['Economic Agency\n(Female Enterprise)', 'Cultural Patriarchy\n(Child Sex Ratio)']
        
        coefs_race = [horse_race_model.params[var] for var in key_vars_race]
        ci_race = [horse_race_model.conf_int().loc[var] for var in key_vars_race]
        
        lower_race = [ci[0] for ci in ci_race]
        upper_race = [ci[1] for ci in ci_race]
        
        xerr_lower = [c - l for c, l in zip(coefs_race, lower_race)]
        xerr_upper = [u - c for c, u in zip(coefs_race, upper_race)]
        
        colors_race = ['#6A0DAD' if p < 0.05 else '#d3d3d3' for p in [p_eco, p_soc]]

        max_abs_coef = max([abs(c) for c in coefs_race]) if coefs_race else 0
        buffer = 0.02 if max_abs_coef < 1 else 0.1

        fig, ax = plt.subplots(figsize=(9, 5))
        y_pos = np.arange(len(key_vars_race))
        
        for i in range(len(key_vars_race)):
            ax.errorbar(coefs_race[i], y_pos[i], 
                        xerr=np.array([[xerr_lower[i]], [xerr_upper[i]]]), 
                        fmt='o', color=colors_race[i], 
                        ecolor='gray', capsize=6, markersize=12, elinewidth=2.5)
            
            p_val = [p_eco, p_soc][i]
            star = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.10 else 'n.s.'
            
            text_x = coefs_race[i] + xerr_upper[i] + buffer 
            ax.text(text_x, y_pos[i], f" {star}", va='center', fontsize=16, fontweight='bold', color=colors_race[i])
            
        ax.axvline(0, color='red', linestyle='--', lw=2, label='Zero Effect')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_race, fontsize=13, fontweight='bold')
        ax.set_xlabel('Standardized Interaction Effect (Impact of 1 SD increase in Moderator)', fontsize=12)
        ax.set_title('The Mechanism Horse Race: What drives the EVM effect?\n(Z-Scored for direct magnitude comparison)', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', linestyle=':', alpha=0.7)
        ax.legend(loc='lower right')
            
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, "Mechanism_Horse_Race_Plot.png"), dpi=300)
        print("Saved 'Mechanism_Horse_Race_Plot.png'")
        plt.show()

    # [CORE END]

    print("EXPORTING MASTER APPENDIX CSVs FOR THESIS REPLICATION")
    master_data_path = os.path.join(CSV_DIR, "Master_Dataset_Final.csv")
    wide.to_csv(master_data_path, index=False)
    print(f"1/5 Saved Master Dataset ({len(wide)} obs) to: {master_data_path}")

    summary_vars = [
        'Turnout_99', 'Turnout_96', 'EVM_Exposure_Cont', 
        'Fem_Enterprise_Pct', 'Agency_Centered', 'Sex_Ratio', 'Patriarchy_Centered',
        'Lit_Pct', 'SC_Pct', 'ST_Pct', 'Urban_Pct'
    ]
    summary_vars = [v for v in summary_vars if v in wide.columns]

    desc_stats = wide[summary_vars].describe().T
    desc_stats['Median'] = wide[summary_vars].median()
    desc_stats = desc_stats[['count', 'mean', 'std', 'min', 'Median', 'max']]
    desc_stats_path = os.path.join(CSV_DIR, "Table1_Summary_Statistics.csv")
    desc_stats.to_csv(desc_stats_path)
    print(f"2/5 Saved Summary Statistics to: {desc_stats_path}")

    corr_vars = [v for v in ['EVM_Exposure_Cont', 'Agency_Centered', 'Patriarchy_Centered', 
                             'Lit_Pct', 'SC_Pct', 'ST_Pct', 'Urban_Pct', 'Turnout_96'] if v in wide.columns]
    corr_matrix = wide[corr_vars].corr()
    corr_path = os.path.join(CSV_DIR, "Correlation_Matrix_Continuous.csv")
    corr_matrix.to_csv(corr_path)
    print(f"3/5 Saved Correlation Matrix to: {corr_path}")

    def get_model_stats(model, model_name):
        ci = model.conf_int()
        rows = []
        for var in model.params.index:
            rows.append({
                'Model': model_name,
                'Variable': var,
                'Coefficient': model.params[var],
                'Std_Error': model.bse[var],
                'P_Value': model.pvalues[var],
                'CI_Lower': ci.loc[var, 0],
                'CI_Upper': ci.loc[var, 1],
                'N': int(model.nobs),
                'R2': model.rsquared,
                'Adj_R2': model.rsquared_adj
            })
        return pd.DataFrame(rows)

    reg_tables = []
    try: reg_tables.append(get_model_stats(final_model, "1_Final_ANCOVA_HC1"))
    except Exception: pass
    try: reg_tables.append(get_model_stats(model_clustered, "2_Final_ANCOVA_Clustered"))
    except Exception: pass
    try: reg_tables.append(get_model_stats(pretrend_mod, "3_PreTrend_Placebo"))
    except Exception: pass
    try: reg_tables.append(get_model_stats(horse_race_model, "4_Mechanism_HorseRace"))
    except Exception: pass

    master_reg_path = os.path.join(CSV_DIR, "Master_Regression_Coefficients.csv")
    pd.concat(reg_tables, ignore_index=True).to_csv(master_reg_path, index=False)
    print(f"4/5 Saved Master Regression Table to: {master_reg_path}")

    robustness_rows = []
    try:
        for _, row in vif_data[vif_data["Variable"] != "const"].iterrows():
            robustness_rows.append({'Category': 'VIF', 'Metric': row['Variable'], 'Value': row['VIF']})
    except Exception: pass

    try:
        robustness_rows.append({'Category': 'Pre-Trend', 'Metric': 'EVMxAgency_Placebo_Beta', 'Value': pretrend_mod.params['EVM_Post_Agency']})
        robustness_rows.append({'Category': 'Pre-Trend', 'Metric': 'EVMxAgency_Placebo_PVal', 'Value': pretrend_mod.pvalues['EVM_Post_Agency']})
    except Exception: pass

    try:
        robustness_rows.append({'Category': 'Influence', 'Metric': 'CooksD_Outliers_Dropped', 'Value': len(outlier_indices)})
    except Exception: pass

    try:
        robustness_rows.append({'Category': 'Mechanism', 'Metric': 'Agency_Patriarchy_Corr', 'Value': corr})
    except Exception: pass

    robustness_path = os.path.join(CSV_DIR, "Robustness_and_Diagnostics_Summary.csv")
    pd.DataFrame(robustness_rows).to_csv(robustness_path, index=False)
    print(f"5/5 Saved Robustness Summary to: {robustness_path}")

    print("ALL THESIS APPENDIX CSVs SUCCESSFULLY GENERATED IN 'CSV_OUTPUTS' FOLDER!")

if __name__ == "__main__":
    main()
