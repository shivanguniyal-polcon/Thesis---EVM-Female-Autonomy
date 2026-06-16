import os
import warnings
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

BASE_DIR = "/Users/ganeshchandrauniyal/Desktop/Thesis Script"
CSV_DIR = os.path.join(BASE_DIR, "CSV_OUTPUTS")
os.makedirs(CSV_DIR, exist_ok=True)

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

# [CORE START]

def main():
    TREATED_PCS_1999 = [
        'HYDERABAD', 'SECUNDERABAD', 'PANAJI', 'MORMUGAO', 'AHMEDABAD', 'GANDHINAGAR',
        'KARNAL', 'ROHTAK', 'BANGALORE NORTH', 'BANGALORE SOUTH', 'MYSORE',
        'ERNAKULAM', 'TRIVANDRUM', 'GWALIOR', 'BHOPAL', 'MUMBAI SOUTH', 
        'MUMBAI SOUTH CENTRAL', 'MUMBAI NORTH CENTRAL', 'MUMBAI NORTH EAST', 'MUMBAI NORTH WEST', 
        'BHUBANESWAR', 'TARN TARAN', 'PATIALA', 'FARIDKOT', 'JAIPUR', 'AJMER', 
        'MADRAS CENTRAL', 'MADRAS SOUTH', 'COIMBATORE', 'MADURAI', 'LUCKNOW', 
        'ALLAHABAD', 'KANPUR', 'AGRA', 'CALCUTTA NORTH WEST', 'CALCUTTA NORTH EAST', 
        'CALCUTTA SOUTH', 'CHANDIGARH', 'NEW DELHI', 'SOUTH DELHI', 'OUTER DELHI', 
        'EAST DELHI', 'CHANDNI CHOWK', 'DELHI SADAR', 'KAROL BAGH', 'PUDUCHERRY'
    ]

    print("STEP 3: HETEROGENEOUS EFFECTS & ECONOMIC AGENCY")

    df_1999 = pd.read_csv(os.path.join(BASE_DIR, "1999_election_data_corrected.csv"))
    df_1999['pc_name_clean'] = df_1999['Constituency'].str.split(' NO :').str[0].str.strip().str.upper()
    df_1999['pc_name_clean'] = df_1999['pc_name_clean'].replace({
        'MUMBAI SOUTH CENTRA': 'MUMBAI SOUTH CENTRAL', 
        'PONDICHERRY': 'PUDUCHERRY'
    })
    df_1999['EVM'] = df_1999['pc_name_clean'].isin(TREATED_PCS_1999).astype(int)

    cw = pd.read_csv(os.path.join(BASE_DIR, "PC2004_to_Dist1991_Weightage_Crosswalk (1).csv"))
    cw['pc_name_clean'] = cw['Constituency Clean'].str.upper().replace({
        'MUMBAI SOUTH CENTRA': 'MUMBAI SOUTH CENTRAL', 
        'PONDICHERRY': 'PUDUCHERRY'
    })

    m = pd.merge(cw, df_1999, on='pc_name_clean', how='inner')
    weight_col = 'pc_weight_relative' if 'pc_weight_relative' in m.columns else 'pc_weight'

    m['alloc_electors_f'] = m['Electors_Female'] * m[weight_col]
    m['alloc_voters_f'] = m['Voted_Female'] * m[weight_col]
    m['alloc_treated_electors_f'] = m['alloc_electors_f'] * m['EVM']

    dist_electoral = m.groupby(['pc91_state_id', 'pc91_district_id', 'state_clean']).agg({
        'alloc_electors_f': 'sum', 
        'alloc_voters_f': 'sum', 
        'alloc_treated_electors_f': 'sum'
    }).reset_index()

    dist_electoral['Female_Turnout_Dist'] = (dist_electoral['alloc_voters_f'] / dist_electoral['alloc_electors_f']) * 100
    dist_electoral['EVM_Exposure'] = dist_electoral['alloc_treated_electors_f'] / dist_electoral['alloc_electors_f']
    print(f"Aggregated {len(dist_electoral)} districts.")

    census = pd.read_csv(os.path.join(BASE_DIR, "shrug-pca91-csv/pc91_pca_clean_pc91dist.csv"))
    td = pd.read_csv(os.path.join(BASE_DIR, "shrug-td91-csv/pc91_td_clean_pc91dist.csv"))

    census['Lit_Pct'] = (census['pc91_pca_p_lit'] / census['pc91_pca_tot_p'].replace(0, np.nan)) * 100
    census['SC_Pct'] = (census['pc91_pca_p_sc'] / census['pc91_pca_tot_p'].replace(0, np.nan)) * 100
    census['ST_Pct'] = (census['pc91_pca_p_st'] / census['pc91_pca_tot_p'].replace(0, np.nan)) * 100

    td_merged = pd.merge(td, census[['pc91_state_id', 'pc91_district_id', 'pc91_pca_tot_p']], 
                         on=['pc91_state_id', 'pc91_district_id'], how='left')
    td_merged['Urban_Pct'] = (td_merged['pc91_td_p_7andup'] / td_merged['pc91_pca_tot_p'].replace(0, np.nan)) * 100
    td_merged['Urban_Pct'] = td_merged['Urban_Pct'].clip(upper=100)

    ec98 = pd.read_csv(os.path.join(BASE_DIR, "Data/Economic Census 1991 Data/ec98_aggregated_1991_districts.csv"))
    ec98['Fem_Enterprise_Pct'] = (ec98['ec98_count_own_f'] / ec98['ec98_count_all'].replace(0, np.nan)) * 100
    ec98['Fem_Enterprise_Pct'] = ec98['Fem_Enterprise_Pct'].fillna(0)
    ec98['Agency_IHS'] = np.arcsinh(ec98['Fem_Enterprise_Pct'])

    final_df = pd.merge(dist_electoral, census[['pc91_state_id', 'pc91_district_id', 'Lit_Pct', 'SC_Pct', 'ST_Pct']], 
                        on=['pc91_state_id', 'pc91_district_id'], how='left')
    final_df = pd.merge(final_df, td_merged[['pc91_state_id', 'pc91_district_id', 'Urban_Pct']], 
                        on=['pc91_state_id', 'pc91_district_id'], how='left')
    final_df = pd.merge(final_df, ec98[['pc91_state_id', 'pc91_district_id', 'Fem_Enterprise_Pct', 'Agency_IHS']], 
                        on=['pc91_state_id', 'pc91_district_id'], how='left')

    final_df = final_df.dropna()
    print(f"Final Analytical Sample: {len(final_df)} districts.")

    agency_mean = final_df['Agency_IHS'].mean()
    final_df['Agency_Centered'] = final_df['Agency_IHS'] - agency_mean

    controls = "Lit_Pct + SC_Pct + ST_Pct + Urban_Pct + C(state_clean)"
    mod_main = smf.ols(f"Female_Turnout_Dist ~ EVM_Exposure + Agency_Centered + {controls}", data=final_df).fit(cov_type='HC1')
    mod_interact = smf.ols(f"Female_Turnout_Dist ~ EVM_Exposure * Agency_Centered + {controls}", data=final_df).fit(cov_type='HC1')

    res_table = summary_col([mod_main, mod_interact], stars=True, float_format='%0.3f',
                            model_names=['(1) Main Effects', '(2) Interaction Model'],
                            info_dict={'N': lambda x: "{0:d}".format(int(x.nobs)), 
                                       'R2': lambda x: "{0:.3f}".format(x.rsquared)})
    print(res_table)

    beta_evm = mod_interact.params['EVM_Exposure']
    int_term = [p for p in mod_interact.params.index if 'EVM_Exposure' in p and 'Agency_Centered' in p][0] 
    beta_int = mod_interact.params[int_term]

    cov_matrix = mod_interact.cov_params()
    var_evm = cov_matrix.loc['EVM_Exposure', 'EVM_Exposure']
    var_int = cov_matrix.loc[int_term, int_term]
    cov_evm_int = cov_matrix.loc['EVM_Exposure', int_term]

    print(f"{'Percentile':<12} | {'Raw Agency %':<15} | {'Marginal Effect':<18} | {'P-Value'}")
    print("-" * 65)

    for p in [0.10, 0.25, 0.50, 0.75, 0.90]:
        raw_val = final_df['Fem_Enterprise_Pct'].quantile(p)
        ihs_val = np.arcsinh(raw_val)
        centered_val = ihs_val - agency_mean
        me = beta_evm + beta_int * centered_val
        se_me = np.sqrt(var_evm + (centered_val**2)*var_int + 2*centered_val*cov_evm_int)
        z_score = me / se_me
        p_val = 2 * (1 - stats.norm.cdf(abs(z_score)))
        print(f"{int(p*100)}th{'':<8} | {raw_val:<15.2f} | {me:<18.4f} | {p_val:.4f}")

    agency_range = np.linspace(final_df['Fem_Enterprise_Pct'].min(), final_df['Fem_Enterprise_Pct'].quantile(0.95), 100)
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

    ax.set_title('Heterogeneous Effects: EVM Impact by Baseline Female Economic Agency\n(1991 District-Level Analysis)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Female Enterprise Density (%)', fontsize=12)
    ax.set_ylabel('Marginal Effect on Female Turnout (Percentage Points)', fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "Step3_Marginal_Effects_Plot.png"), dpi=300)
    print("Saved 'Step3_Marginal_Effects_Plot.png'")
    plt.show()

    median_agency = final_df['Fem_Enterprise_Pct'].median()
    high_agency = final_df[final_df['Fem_Enterprise_Pct'] >= median_agency]
    low_agency = final_df[final_df['Fem_Enterprise_Pct'] < median_agency]

    sub_formula = f"Female_Turnout_Dist ~ EVM_Exposure + {controls}"
    mod_high = smf.ols(sub_formula, data=high_agency).fit(cov_type='HC1')
    mod_low = smf.ols(sub_formula, data=low_agency).fit(cov_type='HC1')

    sub_betas = [mod_high.params['EVM_Exposure'], mod_low.params['EVM_Exposure']]
    sub_ci_high = mod_high.conf_int().loc['EVM_Exposure']
    sub_ci_low = mod_low.conf_int().loc['EVM_Exposure']
    sub_errors = [
        [sub_betas[0] - sub_ci_high[0], sub_ci_high[1] - sub_betas[0]],
        [sub_betas[1] - sub_ci_low[0], sub_ci_low[1] - sub_betas[1]]
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ['High Agency (>= Median)', 'Low Agency (< Median)']
    x_pos = np.arange(len(labels))

    ax.errorbar(sub_betas, x_pos, xerr=np.array(sub_errors).T, fmt='s', color='#6A0DAD', 
                ecolor='black', capsize=6, markersize=10, elinewidth=2)
    ax.axvline(0, color='red', linestyle='--', lw=2)

    ax.set_yticks(x_pos)
    ax.set_yticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_xlabel('Effect of EVM Exposure on Female Turnout', fontsize=12)
    ax.set_title('Subsample Analysis: High vs. Low Economic Agency Districts', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "Step3_Subsample_Heterogeneity.png"), dpi=300)
    print("Saved 'Step3_Subsample_Heterogeneity.png'")
    plt.show()

    print("STEP 3 COMPLETE")

    # [CORE END]


    
    final_df.to_csv(os.path.join(CSV_DIR, "Step3_District_Agency_Data.csv"), index=False)
    export_ols_to_csv(mod_main, "Main_Effects", os.path.join(CSV_DIR, "Step3_Model_MainEffects.csv"))
    export_ols_to_csv(mod_interact, "Interaction_Model", os.path.join(CSV_DIR, "Step3_Model_Interaction.csv"))

    me_rows = []
    for p in [0.10, 0.25, 0.50, 0.75, 0.90]:
        raw_val = final_df['Fem_Enterprise_Pct'].quantile(p)
        ihs_val = np.arcsinh(raw_val)
        centered_val = ihs_val - agency_mean
        me = beta_evm + beta_int * centered_val
        se_me = np.sqrt(var_evm + (centered_val**2)*var_int + 2*centered_val*cov_evm_int)
        z_score = me / se_me
        p_val = 2 * (1 - stats.norm.cdf(abs(z_score)))
        me_rows.append({
            'Percentile': p, 'Raw_Agency_Pct': raw_val, 'Centered_IHS': centered_val,
            'Marginal_Effect': me, 'Std_Error': se_me, 'Z_Score': z_score, 'P_Value': p_val
        })
    pd.DataFrame(me_rows).to_csv(os.path.join(CSV_DIR, "Step3_Marginal_Effects.csv"), index=False)

    sub_res = pd.DataFrame({
        'Sample': ['High_Agency', 'Low_Agency'],
        'EVM_Coefficient': [mod_high.params['EVM_Exposure'], mod_low.params['EVM_Exposure']],
        'P_Value': [mod_high.pvalues['EVM_Exposure'], mod_low.pvalues['EVM_Exposure']],
        'N': [int(mod_high.nobs), int(mod_low.nobs)]
    })
    sub_res.to_csv(os.path.join(CSV_DIR, "Step3_Subsample_Results.csv"), index=False)

if __name__ == "__main__":
    main()
