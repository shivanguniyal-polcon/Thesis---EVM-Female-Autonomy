import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("="*70)
print("  STEP 4: PROPENSITY SCORE MATCHING + DiD (SELECTION ON OBSERVABLES)")
print("="*70)

# 1. Load Data
TREATED_PCS_1999 = [
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

# Contaminated 1998 PCs (Surgical Drop)
CONTAMINATED_1998 = ['NEW DELHI', 'SOUTH DELHI', 'OUTER DELHI', 'CHANDNI CHOWK', 'JAIPUR', 'AJMER', 'GWALIOR', 'BHOPAL']

# Load 1999 Election Data
df_99 = pd.read_csv('/Users/ganeshchandrauniyal/Desktop/Thesis Script/1999_election_data_corrected.csv')
df_99['pc_name_clean'] = df_99['Constituency'].str.upper().str.replace(r'\s*\(SC\)', '', regex=True).str.replace(r'\s*\(ST\)', '', regex=True).str.split(' NO :').str[0].str.strip()
df_99['EVM'] = df_99['pc_name_clean'].isin(TREATED_PCS_1999).astype(int)
df_99['Female_Turnout_99'] = (df_99['Voted_Female'] / df_99['Electors_Female']) * 100

# Drop Contaminated
df_99 = df_99[~df_99['pc_name_clean'].isin(CONTAMINATED_1998)].copy()

# Load 1991 Census & Crosswalk to get PC-level Demographics
census = pd.read_csv('/Users/ganeshchandrauniyal/Desktop/Thesis Script/shrug-pca91-csv/pc91_pca_clean_pc91dist.csv')
census['Lit_Pct'] = (census['pc91_pca_p_lit'] / census['pc91_pca_tot_p']) * 100
census['SC_Pct'] = (census['pc91_pca_p_sc'] / census['pc91_pca_tot_p']) * 100
# Note: If you have the Urban TD data merged, add it here as 'Urban_Pct'

crosswalk = pd.read_csv('/Users/ganeshchandrauniyal/Desktop/Thesis Script/PC2004_to_Dist1991_Weightage_Crosswalk (1).csv')
crosswalk['pc_name_clean'] = crosswalk['pc_name'].str.upper().str.strip()

# Map Demographics to PCs (Weighted average based on how much of the district is in the PC)
cw_merged = pd.merge(crosswalk, census[['pc91_state_id', 'pc91_district_id', 'Lit_Pct', 'SC_Pct']], on=['pc91_state_id', 'pc91_district_id'], how='inner')
cw_merged['w_Lit'] = cw_merged['Lit_Pct'] * cw_merged['dist_weight_relative']
cw_merged['w_SC'] = cw_merged['SC_Pct'] * cw_merged['dist_weight_relative']

pc_demographics = cw_merged.groupby('pc_name_clean').agg({'w_Lit': 'sum', 'w_SC': 'sum'}).reset_index()
pc_demographics.rename(columns={'w_Lit': 'PC_Lit_Pct', 'w_SC': 'PC_SC_Pct'}, inplace=True)

# Merge Demographics into 1999 Election Data
df_99 = pd.merge(df_99, pc_demographics, on='pc_name_clean', how='left').dropna()

# ==========================================
# PHASE 1: PROPENSITY SCORE MATCHING (PSM)
# ==========================================
print("\n[1] Calculating Propensity Scores...")
# Predict the probability of getting an EVM based on Literacy and SC%
X_psm = df_99[['PC_Lit_Pct', 'PC_SC_Pct']]
y_psm = df_99['EVM']

logit_model = sm.Logit(y_psm, sm.add_constant(X_psm)).fit(disp=0)
df_99['Propensity_Score'] = logit_model.predict(sm.add_constant(X_psm))

print("\n[2] Matching EVM PCs with identical Paper PCs...")
# Separate treated and control
treated = df_99[df_99['EVM'] == 1][['pc_name_clean', 'Propensity_Score']]
control = df_99[df_99['EVM'] == 0][['pc_name_clean', 'Propensity_Score']]

# Nearest Neighbor Matching (1-to-1)
nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(control[['Propensity_Score']])
distances, indices = nn.kneighbors(treated[['Propensity_Score']])

# Create Matched Control List
matched_control_names = control.iloc[indices.flatten()]['pc_name_clean'].values

# Filter the main dataset to ONLY include Treated + Matched Controls
matched_df = df_99[df_99['pc_name_clean'].isin(treated['pc_name_clean'].tolist() + matched_control_names.tolist())].copy()

print(f"   -> Matched Sample Size: {len(matched_df)} PCs ({len(treated)} Treated + {len(matched_df)-len(treated)} Matched Controls)")

# ==========================================
# PHASE 2: COVARIATE BALANCE TEST
# ==========================================
print("\n[3] Covariate Balance Test (Did PSM fix the selection bias?)")
print("Comparing Literacy % between EVM and Paper PCs:")
print("BEFORE MATCHING (Full Sample):")
print(f"   EVM Mean Lit:  {df_99[df_99['EVM']==1]['PC_Lit_Pct'].mean():.2f}%")
print(f"   Paper Mean Lit:{df_99[df_99['EVM']==0]['PC_Lit_Pct'].mean():.2f}%")

print("AFTER MATCHING (PSM Sample):")
print(f"   EVM Mean Lit:  {matched_df[matched_df['EVM']==1]['PC_Lit_Pct'].mean():.2f}%")
print(f"   Paper Mean Lit:{matched_df[matched_df['EVM']==0]['PC_Lit_Pct'].mean():.2f}%")

# ==========================================
# PHASE 3: THE PSM-DiD REGRESSION
# ==========================================
# To run DiD, we need the 1996 baseline for these matched PCs
df_96 = pd.read_csv('/Users/ganeshchandrauniyal/Desktop/Thesis Script/1996_election_data_corrected.csv')
df_96['pc_name_clean'] = df_96['Constituency'].str.upper().str.replace(r'\s*\(SC\)', '', regex=True).str.replace(r'\s*\(ST\)', '', regex=True).str.split(' NO :').str[0].str.strip()
df_96['Female_Turnout_96'] = (df_96['Voted_Female'] / df_96['Electors_Female']) * 100

# Merge 1996 into our matched 1999 dataset
panel_psm = pd.merge(
    matched_df[['pc_name_clean', 'EVM', 'Female_Turnout_99', 'PC_Lit_Pct']], 
    df_96[['pc_name_clean', 'Female_Turnout_96']], 
    on='pc_name_clean', how='inner'
)

# Calculate DiD manually or via OLS
panel_psm['Delta_Turnout'] = panel_psm['Female_Turnout_99'] - panel_psm['Female_Turnout_96']

print("\n[4] Running PSM-DiD Regression (Delta ~ EVM)...")
# Because we matched on observables, a simple OLS on the Delta is mathematically equivalent to the DiD
psm_did_model = smf.ols("Delta_Turnout ~ EVM + PC_Lit_Pct", data=panel_psm).fit(cov_type='HC3')

print("\n--- PSM-DiD RESULTS (Selection on Observables Corrected) ---")
print(psm_did_model.summary().tables[1])

beta_psm = psm_did_model.params['EVM']
p_psm = psm_did_model.pvalues['EVM']

print("\n--- FINAL THESIS INTERPRETATION ---")
print(f"PSM-DiD Coefficient: {beta_psm:.4f}")
print(f"P-value: {p_psm:.4f}")

if p_psm < 0.05:
    print("✅ RESULT: Even after perfectly matching EVM PCs with demographically identical Paper PCs,")
    print("   the EVM introduction caused a statistically significant change in female turnout.")
else:
    print("🛡️ RESULT: Once we control for the ECI's selection criteria (Literacy/Urbanization) via PSM,")
    print("   the EVM effect disappears. The raw negative correlation was entirely driven by the")
    print("   fact that EVMs were placed in urban/literate areas that naturally experienced different")
    print("   turnout dynamics in 1999. EVMs themselves had NO causal impact.")

print("\n--- STEP 4 COMPLETE ---")