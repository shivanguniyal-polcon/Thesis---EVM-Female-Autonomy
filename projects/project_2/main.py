import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')
BASE_DIR = "/Users/ganeshchandrauniyal/Desktop/Thesis Script"

import os
CSV_DIR = os.path.join(BASE_DIR, "CSV_OUTPUTS")
os.makedirs(CSV_DIR, exist_ok=True)

# Helper function to export any statsmodels OLS result to CSV
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
    print(f"✅ Saved regression results to: {save_path}")

# [CORE START]
# Define the 46 Treated PCs based on ECI 1999 EVM rollout data
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

print("STEP 2: SPATIAL PROJECTION & DEMOGRAPHIC CONTROLS")

df_1999 = pd.read_csv(os.path.join(BASE_DIR, "1999_election_data_corrected.csv"))
df_1999['pc_name_clean'] = df_1999['Constituency'].str.split(' NO :').str[0].str.strip().str.upper()
df_1999['pc_name_clean'] = df_1999['pc_name_clean'].replace({
    'MUMBAI SOUTH CENTRA': 'MUMBAI SOUTH CENTRAL',
    'PONDICHERRY': 'PUDUCHERRY'
})

df_1999['EVM'] = df_1999['pc_name_clean'].isin(TREATED_PCS_1999).astype(int)
df_1999['Female_Turnout'] = (df_1999['Voted_Female'] / df_1999['Electors_Female']) * 100
print(f"Loaded {len(df_1999)} PCs. Tagged {df_1999['EVM'].sum()}EVM PCs.")

cw = pd.read_csv(os.path.join(BASE_DIR, "PC2004_to_Dist1991_Weightage_Crosswalk (1).csv"))
cw['pc_name_clean'] = cw['Constituency Clean'].str.upper()
cw['pc_name_clean'] = cw['pc_name_clean'].replace({
    'MUMBAI SOUTH CENTRA': 'MUMBAI SOUTH CENTRAL',
    'PONDICHERRY': 'PUDUCHERRY'
})

m = pd.merge(cw, df_1999, on='pc_name_clean', how='inner')
weight_col = 'pc_weight_relative' if 'pc_weight_relative' in m.columns else 'pc_weight'

m['alloc_electors_f'] = m['Electors_Female'] * m[weight_col]
m['alloc_voters_f'] = m['Voted_Female'] * m[weight_col]
m['alloc_treated_electors_f'] = m['alloc_electors_f'] * m['EVM']

dist_electoral = m.groupby(['pc91_state_id', 'pc91_district_id', 'state_clean']).agg({'alloc_electors_f': 'sum', 
    'alloc_voters_f': 'sum',
    'alloc_treated_electors_f': 'sum'}).reset_index()

dist_electoral['Female_Turnout_Dist'] = (dist_electoral['alloc_voters_f']/dist_electoral['alloc_electors_f']) * 100
dist_electoral['EVM_Exposure'] = dist_electoral['alloc_treated_electors_f']/ dist_electoral['alloc_electors_f']
print(f"Aggregated data into {len(dist_electoral)} 1991 districts.")

census = pd.read_csv(os.path.join(BASE_DIR, "shrug-pca91-csv/pc91_pca_clean_pc91dist.csv"))
td = pd.read_csv(os.path.join(BASE_DIR, "shrug-td91-csv/pc91_td_clean_pc91dist.csv"))

census['Lit_Pct'] = (census['pc91_pca_p_lit'] / census['pc91_pca_tot_p'].replace(0, np.nan)) * 100
census['SC_Pct'] = (census['pc91_pca_p_sc'] / census['pc91_pca_tot_p'].replace(0, np.nan)) * 100
census['ST_Pct'] = (census['pc91_pca_p_st'] / census['pc91_pca_tot_p'].replace(0, np.nan)) * 100

td_merged = pd.merge(td, census[['pc91_state_id', 'pc91_district_id', 'pc91_pca_tot_p']], on=['pc91_state_id', 'pc91_district_id'], how='left')
td_merged['Urban_Pct'] = (td_merged['pc91_td_p_7andup'] / td_merged['pc91_pca_tot_p'].replace(0, np.nan)) * 100
td_merged['Urban_Pct'] = td_merged['Urban_Pct'].clip(upper=100)
print("Demographic covariates calculated.")

final_df = pd.merge(dist_electoral, census[['pc91_state_id', 'pc91_district_id', 'Lit_Pct', 'SC_Pct', 'ST_Pct']], on=['pc91_state_id', 'pc91_district_id'], how='left')
final_df = pd.merge(final_df, td_merged[['pc91_state_id', 'pc91_district_id', 'Urban_Pct']], 
                    on=['pc91_state_id', 'pc91_district_id'], how='left')

initial_n = len(final_df)
final_df = final_df.dropna()
print(f"Final Analytical Sample: {len(final_df)} districts (Dropped {initial_n - len(final_df)}")

print("Using various specifications and runnuing 3 models")
mod1 = smf.ols("Female_Turnout_Dist~EVM_Exposure", data=final_df).fit(cov_type='HC1')
mod2 = smf.ols("Female_Turnout_Dist ~ EVM_Exposure + Lit_Pct + SC_Pct + ST_Pct + Urban_Pct", data=final_df).fit(cov_type='HC1')
mod3 = smf.ols("Female_Turnout_Dist ~ EVM_Exposure + Lit_Pct + SC_Pct + ST_Pct + Urban_Pct + C(state_clean)", data=final_df).fit(cov_type='HC1')

res_table = summary_col([mod1, mod2, mod3], stars=True, float_format='%0.3f',
                        model_names=['(1) No Controls', '(2) Demographics', '(3) State FEs'],
                        info_dict={'N': lambda x: "{0:d}".format(int(x.nobs)),
                                   'R2': lambda x: "{0:.3f}".format(x.rsquared)})
print(res_table)

models = [mod1, mod2, mod3]
model_labels = ['(1) No Controls', '(2) Demographics', '(3) State FEs']
coefs = [m.params['EVM_Exposure'] for m in models]
conf_ints = [m.conf_int().loc['EVM_Exposure'] for m in models]

lower_bounds = [ci[0] for ci in conf_ints]
upper_bounds = [ci[1] for ci in conf_ints]
errors = [[c - l, u - c] for c, l, u in zip(coefs, lower_bounds, upper_bounds)]

fig, ax = plt.subplots(figsize=(9, 5))
y_pos = np.arange(len(models))

ax.errorbar(coefs, y_pos, xerr=np.array(errors).T, fmt='o', color='#6A0DAD',ecolor='gray', capsize=6, markersize=10, elinewidth=2)
ax.axvline(0, color='red', linestyle='--', lw=2, label='Zero Effect')

ax.set_yticks(y_pos)
ax.set_yticklabels(model_labels, fontsize=12, fontweight='bold')
ax.set_xlabel('Coefficient of EVM Exposure on Female Turnout (Percentage Points)', fontsize=12)
ax.set_title('Robustness of EVM Effect Across Model Specifications\n(1991 District-Level Analysis)', fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.legend(fontsize=11)
ax.grid(axis='x', linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "Step2_EVM_Coef_Forest_Plot.png"), dpi=300)
print("Saved 'Step2_EVM_Coef_Forest_Plot.png'")
plt.show()

print("Covariate Balance")
covariates = ['Lit_Pct', 'SC_Pct', 'ST_Pct', 'Urban_Pct']
balance_coefs, balance_pvals = [], []

for cov in covariates:
    balance_mod = smf.ols(f"{cov} ~ EVM_Exposure", data=final_df).fit(cov_type='HC3')
    balance_coefs.append(balance_mod.params['EVM_Exposure'])
    balance_pvals.append(balance_mod.pvalues['EVM_Exposure'])
    
    significance = "Significant" if balance_mod.pvalues['EVM_Exposure'] < 0.05 else "Balanced"
    print(f"{cov}: Beta={balance_mod.params['EVM_Exposure']:.4f} (p={balance_mod.pvalues['EVM_Exposure']:.3f}) {significance}")

fig, ax = plt.subplots(figsize=(8, 5))
y_pos = np.arange(len(covariates))
colors = ['#d9534f' if p < 0.05 else '#5cb85c' for p in balance_pvals]

ax.barh(y_pos, balance_coefs, color=colors, edgecolor='black', alpha=0.85)
ax.axvline(0, color='black', linestyle='-', lw=1.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(['Literacy %', 'Scheduled Caste %', 'Scheduled Tribe %', 'Urbanization %'], fontsize=12)
ax.set_xlabel('Correlation with EVM Exposure', fontsize=12)
ax.set_title('Covariate Balance Test\n(Does EVM rollout correlate with baseline demographics?)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "Step2_Covariate_Balance_Plot.png"), dpi=300)
print("Saved 'Step2_Covariate_Balance_Plot.png'")
plt.show()

# [CORE END]

final_df.to_csv(os.path.join(CSV_DIR, "Step2_District_Level_Data.csv"), index=False)

export_ols_to_csv(mod1, "No_Controls", os.path.join(CSV_DIR, "Step2_Model1_NoControls.csv"))
export_ols_to_csv(mod2, "Demographics", os.path.join(CSV_DIR, "Step2_Model2_Demographics.csv"))
export_ols_to_csv(mod3, "State_FEs", os.path.join(CSV_DIR, "Step2_Model3_StateFEs.csv"))

balance_rows = []
for cov in covariates:
    bm = smf.ols(f"{cov} ~ EVM_Exposure", data=final_df).fit(cov_type='HC3')
    balance_rows.append({
        'Covariate': cov, 
        'Beta_on_EVM': bm.params['EVM_Exposure'], 
        'Std_Error': bm.bse['EVM_Exposure'],
        'P_Value': bm.pvalues['EVM_Exposure']
    })
pd.DataFrame(balance_rows).to_csv(os.path.join(CSV_DIR, "Step2_Covariate_Balance.csv"), index=False)

if name == "main":
main()
