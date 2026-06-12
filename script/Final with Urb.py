import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from statsmodels.stats.diagnostic import linear_reset
from scipy.stats import normaltest, jarque_bera, skew, kurtosis
import statsmodels.stats.stattools as stattools
import statsmodels.stats.diagnostic as diag

warnings.filterwarnings('ignore')

# ==========================================
# PHASE 1: DATA INGESTION & SPATIAL CROSSWALK
# ==========================================
print("--- PHASE 1: LOADING DATA & SPATIAL AGGREGATION ---")
BASE_DIR = "/Users/ganeshchandrauniyal/Desktop/Thesis Script"

# 1. Load Crosswalk
cw = pd.read_csv(os.path.join(BASE_DIR, "PC2004_to_Dist1991_Weightage_Crosswalk (1).csv"))
cw['pc_name_clean'] = cw['pc_name'].str.upper().str.strip()

# 2. Load 1991 PCA Census, Town Directory (TD), and 1998 Economic Census
census = pd.read_csv(os.path.join(BASE_DIR, "shrug-pca91-csv/pc91_pca_clean_pc91dist.csv"))
td = pd.read_csv(os.path.join(BASE_DIR, "shrug-td91-csv/pc91_td_clean_pc91dist.csv"))
ec98 = pd.read_csv(os.path.join(BASE_DIR, "Data/Economic Census 1991 Data/ec98_aggregated_1991_districts.csv"))

# 3. Calculate Covariates
census['Lit_Pct'] = (census['pc91_pca_p_lit'] / census['pc91_pca_tot_p']) * 100
census['SC_Pct'] = (census['pc91_pca_p_sc'] / census['pc91_pca_tot_p']) * 100
census['ST_Pct'] = (census['pc91_pca_p_st'] / census['pc91_pca_tot_p']) * 100

# 🆕 NEW URBANIZATION LOGIC
# Merge TD with PCA to get the total population denominator
td_merged = pd.merge(td, census[['pc91_state_id', 'pc91_district_id', 'pc91_pca_tot_p']], 
                     on=['pc91_state_id', 'pc91_district_id'], how='left')
# Calculate Urban Population Percentage
td_merged['Urban_Pct'] = (td_merged['pc91_td_p_7andup'] / td_merged['pc91_pca_tot_p']) * 100
urban_subset = td_merged[['pc91_state_id', 'pc91_district_id', 'Urban_Pct']].drop_duplicates()

ec98['Fem_Enterprise_Pct'] = (ec98['ec98_count_own_f'] / ec98['ec98_count_all']) * 100

# 4. Treated PCs List
TREATED_PCS = [
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

# 5. Aggregation Function
def aggregate_year(file_name, year):
    df = pd.read_csv(os.path.join(BASE_DIR, file_name))
    df['pc_name_clean'] = df['Constituency'].str.split(' NO :').str[0].str.upper().str.strip()
    df['is_evm'] = df['pc_name_clean'].isin(TREATED_PCS).astype(int)
    
    m = pd.merge(cw, df, on='pc_name_clean', how='inner')
    m['alloc_electors_f'] = m['Electors_Female'] * m['pc_weight_relative']
    m['alloc_voters_f'] = m['Voted_Female'] * m['pc_weight_relative']
    m['evm_exposure'] = m['is_evm'] * m['dist_weight_relative'] # Intensive dosage

    dist = m.groupby(['pc91_state_id', 'pc91_district_id', 'state_clean']).agg({
        'alloc_electors_f': 'sum', 'alloc_voters_f': 'sum', 'evm_exposure': 'sum'
    }).reset_index()

    dist['Female_Turnout'] = (dist['alloc_voters_f'] / dist['alloc_electors_f']) * 100
    dist['EVM_Exposure'] = dist['evm_exposure']
    dist['Year'] = year
    return dist

# 6. Build Panel
panel = pd.concat([
    aggregate_year("1996_election_data_corrected.csv", 1996),
    aggregate_year("1998_election_data_corrected.csv", 1998),
    aggregate_year("1999_election_data_corrected.csv", 1999)
], ignore_index=True)

# 7. Merge All Covariates
panel = pd.merge(panel, census[['pc91_state_id', 'pc91_district_id', 'Lit_Pct', 'SC_Pct', 'ST_Pct']], on=['pc91_state_id', 'pc91_district_id'], how='left')
panel = pd.merge(panel, urban_subset, on=['pc91_state_id', 'pc91_district_id'], how='left') # 🆕 Merged Urban_Pct
panel = pd.merge(panel, ec98[['pc91_state_id', 'pc91_district_id', 'Fem_Enterprise_Pct']], on=['pc91_state_id', 'pc91_district_id'], how='left')
panel = panel.dropna()

# 8. Create DDD Variables
panel['Post'] = (panel['Year'] == 1999).astype(int)
panel['EVM_Post'] = panel['EVM_Exposure'] * panel['Post']
panel['Post_Agency'] = panel['Post'] * panel['Fem_Enterprise_Pct']
panel['EVM_Post_Agency'] = panel['EVM_Exposure'] * panel['Post'] * panel['Fem_Enterprise_Pct']

# 9. Pivot for Cross-Sectional ANCOVA/DDD
wide = panel.pivot_table(index=['pc91_state_id', 'pc91_district_id', 'state_clean', 'EVM_Exposure', 
                                'Lit_Pct', 'SC_Pct', 'ST_Pct', 'Fem_Enterprise_Pct', 'Urban_Pct'],
                         columns='Year', values='Female_Turnout').reset_index()
wide = wide.rename(columns={1996: 'Turnout_96', 1999: 'Turnout_99'}).dropna()
wide['EVM_Post_Agency'] = wide['EVM_Exposure'] * wide['Fem_Enterprise_Pct']
print(f"Final Clean Sample: {len(wide)} districts.")


# ==========================================
# SPECIFICATION HORSE RACE FUNCTION
# ==========================================
def test_all_functional_forms(df, target_var, vars_to_test, base_controls):
    results = []
    for var in vars_to_test:
        df[f'{var}_sq'] = df[var] ** 2
        df[f'{var}_log'] = np.log1p(df[var]) 
        df[f'{var}_ihs'] = np.arcsinh(df[var])
        
        other_controls = [c for c in base_controls if c != var]
        control_str = " + ".join(other_controls) if other_controls else "1"
        
        try:
            mod_lin = smf.ols(f"{target_var} ~ {var} + {control_str}", data=df).fit()
            bic_lin = mod_lin.bic
        except: bic_lin = np.inf
            
        try:
            mod_quad = smf.ols(f"{target_var} ~ {var} + {var}_sq + {control_str}", data=df).fit()
            bic_quad = mod_quad.bic
            pval_sq = mod_quad.pvalues[f'{var}_sq']
        except: bic_quad, pval_sq = np.inf, 1.0
            
        try:
            mod_log = smf.ols(f"{target_var} ~ {var}_log + {control_str}", data=df).fit()
            bic_log = mod_log.bic
        except: bic_log = np.inf
            
        try:
            mod_ihs = smf.ols(f"{target_var} ~ {var}_ihs + {control_str}", data=df).fit()
            bic_ihs = mod_ihs.bic
        except: bic_ihs = np.inf
            
        models = {'Linear': bic_lin, 'Quadratic': bic_quad, 'Log': bic_log, 'IHS': bic_ihs}
        best_model = min(models, key=models.get)
        
        results.append({
            'Variable': var,
            'Best_Form': best_model,
            'BIC_Linear': round(bic_lin, 1),
            'BIC_Quad': round(bic_quad, 1),
            'BIC_Log': round(bic_log, 1),
            'BIC_IHS': round(bic_ihs, 1),
            'Quad_p-val': round(pval_sq, 4) if pval_sq != np.inf else "-"
        })
    return pd.DataFrame(results)

print("\n--- RUNNING SPECIFICATION HORSE RACE ---")
target = 'Turnout_99'
# 🆕 Added Urban_Pct to the test list
vars_to_test = ['Lit_Pct', 'SC_Pct', 'ST_Pct', 'Fem_Enterprise_Pct', 'Urban_Pct'] 
base_controls = ['Turnout_96', 'EVM_Exposure']

summary_df = test_all_functional_forms(wide, target, vars_to_test, base_controls)
print("\n--- FUNCTIONAL FORM RECOMMENDATIONS (Lowest BIC Wins) ---")
print(summary_df.to_string(index=False))


# ==========================================
# PHASE 2: MAIN CAUSAL MODEL (Cross-Sectional DDD)
# ==========================================
print("\n--- PHASE 2: MAIN CAUSAL MODEL ---")

# Apply transformations based on typical results (IHS for skewed percentages)
wide['ST_Pct_ihs'] = np.arcsinh(wide['ST_Pct'])
wide['Fem_Enterprise_Pct_ihs'] = np.arcsinh(wide['Fem_Enterprise_Pct'])
wide['Urban_Pct_ihs'] = np.arcsinh(wide['Urban_Pct']) # 🆕 Transformed Urbanization

# 🆕 Updated Formula to include Urban_Pct_ihs
final_formula = (
    "Turnout_99 ~ EVM_Exposure + Fem_Enterprise_Pct_ihs + Urban_Pct_ihs + "
    "EVM_Exposure:Fem_Enterprise_Pct_ihs + Turnout_96 + "
    "Lit_Pct + SC_Pct + ST_Pct_ihs + C(state_clean)"
)

final_model = smf.ols(final_formula, data=wide).fit(cov_type='HC1')

print("\n--- FINAL MODEL COEFFICIENTS ---")
res_df = pd.DataFrame({
    'Coefficient': final_model.params,
    'Std. Error': final_model.bse,
    'P-Value': final_model.pvalues,
    'Conf. Int Lower': final_model.conf_int()[0],
    'Conf. Int Upper': final_model.conf_int()[1]
})
print(res_df.to_string(float_format="%.4f"))


# ==========================================
# PHASE 3: COVARIATE BALANCE (SELECTION TEST)
# ==========================================
print("\n--- PHASE 3: COVARIATE BALANCE (SELECTION TEST) ---")
# 🆕 Added Urban_Pct to balance test
covariates = ['Lit_Pct', 'SC_Pct', 'ST_Pct', 'Fem_Enterprise_Pct', 'Urban_Pct', 'Turnout_96']
for cov in covariates:
    mod = smf.ols(f"{cov} ~ EVM_Exposure", data=wide).fit(cov_type='HC3')
    print(f"{cov}: Beta={mod.params['EVM_Exposure']:.4f} (p={mod.pvalues['EVM_Exposure']:.3f})")


# ==========================================
# PHASE 4: FRISCH-WAUGH-LOVELL (FWL) THEOREM
# ==========================================
print("\n--- PHASE 4: FRISCH-WAUGH-LOVELL (FWL) THEOREM ---")
wide['EVM_Post_Agency'] = wide['EVM_Exposure'] * wide['Fem_Enterprise_Pct_ihs']

# 🆕 Added Urban_Pct_ihs to controls
controls = "Turnout_96 + Lit_Pct + SC_Pct + ST_Pct_ihs + Fem_Enterprise_Pct_ihs + Urban_Pct_ihs + EVM_Exposure + C(state_clean)"
res_y = smf.ols(f"Turnout_99 ~ {controls}", data=wide).fit().resid
res_x = smf.ols(f"EVM_Post_Agency ~ {controls}", data=wide).fit().resid

fwl_df = pd.DataFrame({'Res_Y': res_y, 'Res_X': res_x})
fwl_mod = smf.ols("Res_Y ~ Res_X", data=fwl_df).fit()
print(f"FWL Interaction Beta: {fwl_mod.params['Res_X']:.4f}")


# ==========================================
# PHASE 5: LEAVE-ONE-OUT (STATE LEVEL)
# ==========================================
print("\n--- PHASE 5: LEAVE-ONE-OUT (STATE LEVEL) ---")
target_param = 'EVM_Exposure:Fem_Enterprise_Pct_ihs'
states = wide['state_clean'].unique()
loo_results = []

for st in states:
    temp_df = wide[wide['state_clean'] != st]
    if temp_df['EVM_Exposure'].sum() > 0 and (temp_df['EVM_Exposure'] == 0).sum() > 0:
        try:
            mod = smf.ols(final_formula, data=temp_df).fit(cov_type='HC1')
            loo_results.append({'Dropped_State': st, 'Beta': mod.params[target_param]})
        except: pass 
            
loo_df = pd.DataFrame(loo_results)
baseline_beta = final_model.params[target_param]
print(f"Baseline Interaction Beta: {baseline_beta:.4f}")
if not loo_df.empty:
    print(f"LOO Mean Beta: {loo_df['Beta'].mean():.4f}")
    print(f"LOO Std Dev: {loo_df['Beta'].std():.4f}")


# ==========================================
# PHASE 6: SUB-SAMPLE HETEROGENEITY
# ==========================================
print("\n--- PHASE 6: SUB-SAMPLE HETEROGENEITY ---")
median_agency = wide['Fem_Enterprise_Pct'].median()
high_agency = wide[wide['Fem_Enterprise_Pct'] >= median_agency]
low_agency = wide[wide['Fem_Enterprise_Pct'] < median_agency]

# 🆕 Added Urban_Pct_ihs to sub-sample formulas
sub_formula = "Turnout_99 ~ EVM_Exposure + EVM_Post_Agency + Urban_Pct_ihs + Turnout_96 + Lit_Pct + C(state_clean)"
mod_high = smf.ols(sub_formula, data=high_agency).fit(cov_type='HC3')
mod_low = smf.ols(sub_formula, data=low_agency).fit(cov_type='HC3')

print(f"High Agency Beta: {mod_high.params['EVM_Post_Agency']:.4f} (p={mod_high.pvalues['EVM_Post_Agency']:.3f})")
print(f"Low Agency Beta:  {mod_low.params['EVM_Post_Agency']:.4f} (p={mod_low.pvalues['EVM_Post_Agency']:.3f})")


# ==========================================
# TABLE 1 & 2: SUMMARY STATS & DIAGNOSTICS
# ==========================================
print("\n" + "="*70)
print("  TABLE 1: SUMMARY STATISTICS OF KEY VARIABLES")
print("="*70)
summary_cols = ['Turnout_99', 'Turnout_96', 'EVM_Exposure', 'Fem_Enterprise_Pct', 'Urban_Pct', 'Lit_Pct']
summary_cols = [c for c in summary_cols if c in wide.columns]
print(wide[summary_cols].describe().T[['count', 'mean', 'std', 'min', 'max']].to_string(float_format="%.3f"))

print("\n" + "="*70)
print("  TABLE 2: FINAL REGRESSION RESULTS & DIAGNOSTICS")
print("="*70)

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
print("-" * 60)
print(f"{'Observations (N)':<35} | {n_obs:<20}")
print(f"{'Degrees of Freedom (Model)':<35} | {df_model:<20}")
print(f"{'Degrees of Freedom (Residual)':<35} | {df_resid:<20}")
print("-" * 60)
print(f"{'R-squared':<35} | {r2:<20.4f}")
print(f"{'Adjusted R-squared':<35} | {r2_adj:<20.4f}")
print(f"{'F-statistic':<35} | {f_stat:<20.4f}")
print(f"{'Prob (F-statistic)':<35} | {f_pval:<20.4e}")
print("-" * 60)
print(f"{'Akaike Info Criterion (AIC)':<35} | {aic:<20.2f}")
print(f"{'Bayesian Info Criterion (BIC)':<35} | {bic:<20.2f}")
print(f"{'Condition Number':<35} | {cond_num:<20.2f}")
print("-" * 60)
print(f"{'Durbin-Watson Statistic':<35} | {dw_stat:<20.4f}")
print("-" * 60)
print(f"{'Omnibus K^2 (Normality)':<35} | {omnibus_stat:<20.4f}")
print(f"{'Prob (Omnibus)':<35} | {omnibus_pval:<20.4e}")
print(f"{'Jarque-Bera (JB)':<35} | {jb_stat:<20.4f}")
print(f"{'Prob (JB)':<35} | {jb_pval:<20.4e}")
print(f"{'Skewness':<35} | {jb_skew:<20.4f}")
print(f"{'Kurtosis':<35} | {jb_kurt:<20.4f}")
print("-" * 60)
print(f"{'Breusch-Pagan (Heteroskedasticity)':<35} | {bp_stat:<20.4f}")
print(f"{'Prob (Breusch-Pagan)':<35} | {bp_pval:<20.4e}")
print("=" * 60)

print("\n--- ROBUST COEFFICIENTS (HC1) FOR REPORTING ---")
try:
    print(model_robust.summary2().tables[1])
except:
    print(res_df.to_string(float_format="%.4f"))