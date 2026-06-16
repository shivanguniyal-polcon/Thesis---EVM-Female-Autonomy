import os
import warnings
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels.stats.stattools as stattools
import statsmodels.stats.diagnostic as diag
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

    CONTAMINATED_1998_PCS = [
        'NEW DELHI', 'SOUTH DELHI', 'OUTER DELHI', 'EAST DELHI', 
        'CHANDNI CHOWK', 'DELHI SADAR', 'KAROL BAGH', 'GWALIOR', 'BHOPAL', 'JAIPUR', 'AJMER'
    ]

    print("STEP 4: DIFFERENCE-IN-DIFFERENCES & PRE-TREND VALIDATION")

    def aggregate_year(file_name, year):
        df = pd.read_csv(os.path.join(BASE_DIR, file_name))
        
        if 'Constituency Clean' in df.columns:
            df['pc_name_clean'] = df['Constituency Clean'].str.upper()
        elif 'Constituency' in df.columns:
            df['pc_name_clean'] = df['Constituency'].str.split(' NO :').str[0].str.strip().str.upper()
        else:
            raise ValueError(f"No constituency column found in {file_name}")
            
        df['pc_name_clean'] = df['pc_name_clean'].replace({
            'MUMBAI SOUTH CENTRA': 'MUMBAI SOUTH CENTRAL', 
            'PONDICHERRY': 'PUDUCHERRY'
        })
        
        if year == 1998:
            df = df[~df['pc_name_clean'].isin(CONTAMINATED_1998_PCS)].copy()
            
        df['is_evm'] = df['pc_name_clean'].isin(TREATED_PCS_1999).astype(int)
        
        cw = pd.read_csv(os.path.join(BASE_DIR, "PC2004_to_Dist1991_Weightage_Crosswalk (1).csv"))
        cw['pc_name_clean'] = cw['Constituency Clean'].str.upper().replace({
            'MUMBAI SOUTH CENTRA': 'MUMBAI SOUTH CENTRAL', 
            'PONDICHERRY': 'PUDUCHERRY'
        })
        
        m = pd.merge(cw, df, on='pc_name_clean', how='inner')
        weight_col = 'pc_weight_relative' if 'pc_weight_relative' in m.columns else 'pc_weight'
        
        m['alloc_electors_f'] = m['Electors_Female'] * m[weight_col]
        m['alloc_voters_f'] = m['Voted_Female'] * m[weight_col]
        m['alloc_treated_electors_f'] = m['alloc_electors_f'] * m['is_evm']
        
        dist = m.groupby(['pc91_state_id', 'pc91_district_id', 'state_clean']).agg({
            'alloc_electors_f': 'sum', 
            'alloc_voters_f': 'sum', 
            'alloc_treated_electors_f': 'sum'
        }).reset_index()
        
        dist[f'Turnout_{year}'] = (dist['alloc_voters_f'] / dist['alloc_electors_f']) * 100
        dist[f'EVM_Exposure_{year}'] = dist['alloc_treated_electors_f'] / dist['alloc_electors_f']
        
        return dist[['pc91_state_id', 'pc91_district_id', 'state_clean', f'Turnout_{year}', f'EVM_Exposure_{year}']]

    print("Aggregating 1996, 1998, and 1999 Elections to 1991 Districts...")
    dist_96 = aggregate_year("1996_election_data_cleaned.csv", 1996)
    dist_98 = aggregate_year("1998_election_data_cleaned.csv", 1998)
    dist_99 = aggregate_year("1999_election_data_corrected.csv", 1999)

    panel = pd.merge(dist_96, dist_98, on=['pc91_state_id', 'pc91_district_id', 'state_clean'], how='inner')
    panel = pd.merge(panel, dist_99, on=['pc91_state_id', 'pc91_district_id', 'state_clean'], how='inner')

    panel['EVM_Exposure'] = panel['EVM_Exposure_1999']
    print(f"Balanced Panel Created: {len(panel)} districts across 3 election cycles.")

    print("Merging Demographics & Economic Agency...")
    census = pd.read_csv(os.path.join(BASE_DIR, "shrug-pca91-csv/pc91_pca_clean_pc91dist.csv"))
    td = pd.read_csv(os.path.join(BASE_DIR, "shrug-td91-csv/pc91_td_clean_pc91dist.csv"))
    ec98 = pd.read_csv(os.path.join(BASE_DIR, "Data/Economic Census 1991 Data/ec98_aggregated_1991_districts.csv"))

    census['Lit_Pct'] = (census['pc91_pca_p_lit'] / census['pc91_pca_tot_p'].replace(0, np.nan)) * 100
    census['SC_Pct'] = (census['pc91_pca_p_sc'] / census['pc91_pca_tot_p'].replace(0, np.nan)) * 100
    census['ST_Pct'] = (census['pc91_pca_p_st'] / census['pc91_pca_tot_p'].replace(0, np.nan)) * 100

    td_merged = pd.merge(td, census[['pc91_state_id', 'pc91_district_id', 'pc91_pca_tot_p']], on=['pc91_state_id', 'pc91_district_id'], how='left')
    td_merged['Urban_Pct'] = (td_merged['pc91_td_p_7andup'] / td_merged['pc91_pca_tot_p'].replace(0, np.nan)) * 100
    td_merged['Urban_Pct'] = td_merged['Urban_Pct'].clip(upper=100)

    ec98['Fem_Enterprise_Pct'] = (ec98['ec98_count_own_f'] / ec98['ec98_count_all'].replace(0, np.nan)) * 100
    ec98['Agency_IHS'] = np.arcsinh(ec98['Fem_Enterprise_Pct'].fillna(0))

    panel = pd.merge(panel, census[['pc91_state_id', 'pc91_district_id', 'Lit_Pct', 'SC_Pct', 'ST_Pct']], on=['pc91_state_id', 'pc91_district_id'], how='left')
    panel = pd.merge(panel, td_merged[['pc91_state_id', 'pc91_district_id', 'Urban_Pct']], on=['pc91_state_id', 'pc91_district_id'], how='left')
    panel = pd.merge(panel, ec98[['pc91_state_id', 'pc91_district_id', 'Agency_IHS']], on=['pc91_state_id', 'pc91_district_id'], how='left')

    panel = panel.dropna()

    agency_mean = panel['Agency_IHS'].mean()
    panel['Agency_Centered'] = panel['Agency_IHS'] - agency_mean

    panel['Delta_PreTrend'] = panel['Turnout_1998'] - panel['Turnout_1996']
    panel['Delta_DiD'] = panel['Turnout_1999'] - panel['Turnout_1996']

    print(f"Final Analytical Sample: {len(panel)} districts.")

    def print_full_diagnostics(model, model_name):
        print(f"\nDIAGNOSTICS: {model_name}")
        n_obs = int(model.nobs)
        r2 = model.rsquared
        r2_adj = model.rsquared_adj
        f_stat = model.fvalue
        f_pval = model.f_pvalue
        aic = model.aic
        bic = model.bic
        cond_num = model.condition_number
        dw_stat = stattools.durbin_watson(model.resid)
        
        jb_stat, jb_pval, jb_skew, jb_kurt = stattools.jarque_bera(model.resid)
        
        bp_test = diag.het_breuschpagan(model.resid, model.model.exog)
        bp_stat, bp_pval, _, _ = bp_test
        
        print(f"{'Metric':<35} | {'Value':<20}")
        print(f"{'Observations (N)':<35} | {n_obs:<20}")
        print(f"{'R-squared':<35} | {r2:<20.4f}")
        print(f"{'Adjusted R-squared':<35} | {r2_adj:<20.4f}")
        print(f"{'F-statistic (Overall Significance)':<35} | {f_stat:<20.4f}")
        print(f"{'Prob (F-statistic)':<35} | {f_pval:<20.4e}")
        print(f"{'Akaike Info Criterion (AIC)':<35} | {aic:<20.2f}")
        print(f"{'Bayesian Info Criterion (BIC)':<35} | {bic:<20.2f}")
        print(f"{'Condition Number (Multicollinearity)':<35} | {cond_num:<20.2f}")
        print(f"{'Durbin-Watson (Autocorrelation)':<35} | {dw_stat:<20.4f}")
        print(f"{'Breusch-Pagan (Heteroskedasticity)':<35} | {bp_stat:<20.4f}")
        print(f"{'Prob (Breusch-Pagan)':<35} | {bp_pval:<20.4e}")
        if bp_pval < 0.05:
            print("  Heteroskedasticity detected. Robust SEs (HC1) applied.")
        print(f"{'Jarque-Bera (Normality of Residuals)':<35} | {jb_stat:<20.4f}")
        print(f"{'Prob (Jarque-Bera)':<35} | {jb_pval:<20.4e}")
        print(f"{'Skewness':<35} | {jb_skew:<20.4f}")
        print(f"{'Kurtosis':<35} | {jb_kurt:<20.4f}")
        if jb_pval < 0.05:
            print("  Residuals are non-normal. OLS unbiased, but rely on HC1 SEs.")

    print("\nESTIMATING CAUSAL MODELS...")
    controls = "Lit_Pct + SC_Pct + ST_Pct + Urban_Pct + C(state_clean)"
    interaction = "EVM_Exposure * Agency_Centered"

    formula_pre = f"Delta_PreTrend ~ {interaction} + {controls}"
    mod_pre = smf.ols(formula_pre, data=panel).fit(cov_type='HC1')

    formula_did = f"Delta_DiD ~ {interaction} + {controls}"
    mod_did = smf.ols(formula_did, data=panel).fit(cov_type='HC1')

    formula_ancova = f"Turnout_1999 ~ {interaction} + Turnout_1996 + {controls}"
    mod_ancova = smf.ols(formula_ancova, data=panel).fit(cov_type='HC1')

    print_full_diagnostics(mod_pre, "PLACEBO PRE-TREND (1996-1998)")
    print_full_diagnostics(mod_did, "MAIN DiD TREATMENT (1996-1999)")
    print_full_diagnostics(mod_ancova, "ANCOVA ROBUSTNESS (1999 + Baseline)")

    print("GENERATING CAUSAL VALIDATION PLOTS...")

    int_term = [p for p in mod_did.params.index if 'EVM_Exposure' in p and 'Agency_Centered' in p][0]

    coefs = [mod_pre.params[int_term], mod_did.params[int_term]]
    ci_pre = mod_pre.conf_int().loc[int_term]
    ci_did = mod_did.conf_int().loc[int_term]
    errors = [
        [coefs[0] - ci_pre[0], ci_pre[1] - coefs[0]],
        [coefs[1] - ci_did[0], ci_did[1] - coefs[1]]
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = [0, 1]
    labels = ['Pre-Trend (1996-1998) [Placebo]', 'Treatment (1996-1999) [Actual DiD]']

    ax.errorbar(coefs, x_pos, xerr=np.array(errors).T, fmt='o', color='#6A0DAD', 
                ecolor='gray', capsize=8, markersize=12, elinewidth=2.5)
    ax.axvline(0, color='red', linestyle='--', lw=2, label='Zero Effect (Parallel Trends)')

    ax.set_yticks(x_pos)
    ax.set_yticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_xlabel('Coefficient of EVM x Agency Interaction', fontsize=12)
    ax.set_title('Causal Validation: Heterogeneous Effect Only Emerges Post-1999\n(Event Study / Pre-Trend Test)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.legend(fontsize=11)
    ax.grid(axis='x', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "Step4_PreTrend_Event_Study.png"), dpi=300)
    print("Saved 'Step4_PreTrend_Event_Study.png'")
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(mod_did.fittedvalues, mod_did.resid, alpha=0.5, edgecolors='k', color='#6A0DAD', s=30)
    axes[0].axhline(0, color='red', linestyle='--', lw=2)
    axes[0].set_xlabel('Fitted Values (Predicted Change in Turnout)', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].set_title('Residuals vs. Fitted\n(Visual check for Breusch-Pagan Heteroskedasticity)', fontsize=13, fontweight='bold')
    axes[0].grid(True, linestyle=':', alpha=0.6)

    sm.qqplot(mod_did.resid, line='s', ax=axes[1], color='#6A0DAD', markerfacecolor='#6A0DAD', markeredgecolor='k')
    axes[1].set_title('Q-Q Plot of Residuals\n(Visual check for Jarque-Bera Normality)', fontsize=13, fontweight='bold')
    axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "Step4_Diagnostic_Residual_Plots.png"), dpi=300)
    print("Saved 'Step4_Diagnostic_Residual_Plots.png'")
    plt.show()
    # [CORE END]

    panel.to_csv(os.path.join(CSV_DIR, "Step4_Panel_Data.csv"), index=False)

    export_ols_to_csv(mod_pre, "Placebo_PreTrend", os.path.join(CSV_DIR, "Step4_Model_PreTrend.csv"))
    export_ols_to_csv(mod_did, "DiD_Treatment", os.path.join(CSV_DIR, "Step4_Model_DiD.csv"))
    export_ols_to_csv(mod_ancova, "ANCOVA_Robustness", os.path.join(CSV_DIR, "Step4_Model_ANCOVA.csv"))

    diag_rows = []
    for m, name in zip([mod_pre, mod_did, mod_ancova], ["PreTrend", "DiD", "ANCOVA"]):
        jb_stat, jb_pval, jb_skew, jb_kurt = stattools.jarque_bera(m.resid)
        bp_test = diag.het_breuschpagan(m.resid, m.model.exog)
        diag_rows.append({
            'Model': name, 'N': int(m.nobs), 'R2': m.rsquared, 'Adj_R2': m.rsquared_adj,
            'F_Stat': m.fvalue, 'F_PVal': m.f_pvalue, 'AIC': m.aic, 'BIC': m.bic,
            'Condition_Number': m.condition_number,
            'Durbin_Watson': stattools.durbin_watson(m.resid),
            'BP_Stat': bp_test[0], 'BP_PVal': bp_test[1],
            'JB_Stat': jb_stat, 'JB_PVal': jb_pval, 'Skewness': jb_skew, 'Kurtosis': jb_kurt
        })
    pd.DataFrame(diag_rows).to_csv(os.path.join(CSV_DIR, "Step4_Full_Diagnostics.csv"), index=False)

if __name__ == "__main__":
    main()
