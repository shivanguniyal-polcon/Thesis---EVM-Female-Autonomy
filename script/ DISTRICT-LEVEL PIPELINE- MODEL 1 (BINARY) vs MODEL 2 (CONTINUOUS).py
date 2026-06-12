import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors
import os
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("  DISTRICT-LEVEL PIPELINE: MODEL 1 (BINARY) vs MODEL 2 (CONTINUOUS)")
print("="*70)

# ==========================================
# PHASE 1: SPATIAL AGGREGATION (PC -> DISTRICT)
# ==========================================
print("\n[PHASE 1] Aggregating PC Election Data to District Level...")

DATA_DIR = "/Users/ganeshchandrauniyal/Desktop/Thesis Script"
WEIGHTS_PATH = "/Users/ganeshchandrauniyal/Desktop/Thesis Script/PC2004_to_Dist1991_Weightage_Crosswalk (1).csv"
CENSUS_PATH = "/Users/ganeshchandrauniyal/Desktop/Thesis Script/shrug-pca91-csv/pc91_pca_clean_pc91dist.csv"

# 1. Load Crosswalk
crosswalk = pd.read_csv(WEIGHTS_PATH)
crosswalk['pc_name_clean'] = crosswalk['pc_name'].str.upper().str.strip()

# 2. Define Treatment & Contamination Lists
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

CONTAMINATED_1998_PCS = ['NEW DELHI', 'SOUTH DELHI', 'OUTER DELHI', 'CHANDNI CHOWK', 'JAIPUR', 'AJMER', 'GWALIOR', 'BHOPAL']

def aggregate_to_district(file_name, year, cw_df, treated_pcs):
    path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(path): path = os.path.join(DATA_DIR, 'Data', file_name)
    df = pd.read_csv(path)
    
    # Clean PC Names
    df['pc_name_clean'] = (df['Constituency'].str.upper()
                           .str.replace(r'\s*\(SC\)', '', regex=True)
                           .str.replace(r'\s*\(ST\)', '', regex=True)
                           .str.split(' NO :').str[0].str.strip())
    
    # Surgically drop contaminated 1998 PCs
    df = df[~df['pc_name_clean'].isin(CONTAMINATED_1998_PCS)]
    
    # Merge with Crosswalk
    merged = pd.merge(cw_df, df, on='pc_name_clean', how='inner')
    
    # Calculate Allocated Counts (Extensive Variables)
    merged['alloc_electors'] = merged['Electors_Female'] * merged['pc_weight_relative']
    merged['alloc_voters'] = merged['Voted_Female'] * merged['pc_weight_relative']
    
    # Tag EVM & Calculate Treated Electors
    merged['is_evm'] = merged['pc_name_clean'].isin(treated_pcs).astype(int)
    merged['treated_electors'] = merged['alloc_electors'] * merged['is_evm']
    
    # Aggregate to District Level
    dist_agg = merged.groupby(['pc91_state_id', 'pc91_district_id']).agg({
        'alloc_electors': 'sum',
        'alloc_voters': 'sum',
        'treated_electors': 'sum'
    }).reset_index()
    
    # Calculate District Turnout
    dist_agg['Female_Turnout'] = (dist_agg['alloc_voters'] / dist_agg['alloc_electors']) * 100
    
    # MODEL 1: Binary Intent-to-Treat (1 if ANY part of district got EVM)
    dist_agg['Binary_EVM'] = (dist_agg['treated_electors'] > 0).astype(int)
    
    # MODEL 2: Continuous Elector-Weighted Dosage (0.0 to 1.0)
    dist_agg['Continuous_EVM'] = dist_agg['treated_electors'] / dist_agg['alloc_electors']
    
    dist_agg['Year'] = year
    return dist_agg

# Build the District Panel
dist_96 = aggregate_to_district('1996_election_data_corrected.csv', 1996, crosswalk, TREATED_PCS_1999)
dist_98 = aggregate_to_district('1998_election_data_corrected.csv', 1998, crosswalk, TREATED_PCS_1999)
dist_99 = aggregate_to_district('1999_election_data_corrected.csv', 1999, crosswalk, TREATED_PCS_1999)

panel = pd.concat([dist_96, dist_98, dist_99], ignore_index=True)
panel['Post_1999'] = (panel['Year'] == 1999).astype(int)
print(f"   -> District Panel Built: {len(panel)} observations ({len(panel) // 3} districts).")
# ==========================================
# PHASE 2: PARALLEL TRENDS TEST (Script 2)
# ==========================================
print("\n[PHASE 2] District-Level Parallel Trends Test (1996 to 1998)...")

# Pivot to get 1996 and 1998 side-by-side
pivot_pre = panel[panel['Year'].isin([1996, 1998])].pivot(
    index=['pc91_state_id', 'pc91_district_id', 'Binary_EVM', 'Continuous_EVM'],
    columns='Year', values='Female_Turnout'
).reset_index()

pivot_pre['Delta_98_96'] = pivot_pre[1998] - pivot_pre[1996]

# Test Model 1 (Binary)
trend_m1 = smf.ols('Delta_98_96 ~ Binary_EVM', data=pivot_pre).fit(cov_type='HC3')
# Test Model 2 (Continuous)
trend_m2 = smf.ols('Delta_98_96 ~ Continuous_EVM', data=pivot_pre).fit(cov_type='HC3')

print(f"Model 1 (Binary) Pre-Trend Coef:    {trend_m1.params['Binary_EVM']:.4f} (p={trend_m1.pvalues['Binary_EVM']:.3f})")
print(f"Model 2 (Continuous) Pre-Trend Coef: {trend_m2.params['Continuous_EVM']:.4f} (p={trend_m2.pvalues['Continuous_EVM']:.3f})")
print("   -> (Both p-values should be > 0.05 to prove parallel trends hold)")

# ==========================================
# PHASE 3: DIFFERENCE-IN-DIFFERENCES (Script 3)
# ==========================================
print("\n[PHASE 3] District-Level DiD Regression (1996, 1998, 1999)...")

# Model 1: Binary DiD
did_m1 = smf.ols("Female_Turnout ~ Binary_EVM + C(Year) + Binary_EVM:Post_1999", data=panel).fit(cov_type='HC3')
# Model 2: Continuous DiD
did_m2 = smf.ols("Female_Turnout ~ Continuous_EVM + C(Year) + Continuous_EVM:Post_1999", data=panel).fit(cov_type='HC3')

print("\n--- MODEL 1: BINARY INTENT-TO-TREAT DiD ---")
print(f"Coefficient (Binary_EVM:Post_1999): {did_m1.params['Binary_EVM:Post_1999']:.4f}")
print(f"P-value: {did_m1.pvalues['Binary_EVM:Post_1999']:.3f}")

print("\n--- MODEL 2: CONTINUOUS ELECTOR-WEIGHTED DOSAGE DiD ---")
print(f"Coefficient (Continuous_EVM:Post_1999): {did_m2.params['Continuous_EVM:Post_1999']:.4f}")
print(f"P-value: {did_m2.pvalues['Continuous_EVM:Post_1999']:.3f}")

# ==========================================
# PHASE 4: PROPENSITY SCORE MATCHING (Script 4)
# ==========================================
print("\n[PHASE 4] District-Level PSM + DiD (Selection on Observables)...")

# 1. Load 1991 Census Controls
census = pd.read_csv(CENSUS_PATH)
census['Lit_Pct'] = (census['pc91_pca_p_lit'] / census['pc91_pca_tot_p']) * 100
census['SC_Pct'] = (census['pc91_pca_p_sc'] / census['pc91_pca_tot_p']) * 100
census['ST_Pct'] = (census['pc91_pca_p_st'] / census['pc91_pca_tot_p']) * 100
census_subset = census[['pc91_state_id', 'pc91_district_id', 'Lit_Pct', 'SC_Pct', 'ST_Pct']]

# 2. Merge Controls to 1999 Cross-Section for PSM Matching
panel_99 = panel[panel['Year'] == 1999].copy()
psm_df = pd.merge(panel_99, census_subset, on=['pc91_state_id', 'pc91_district_id'], how='inner').dropna()

# 3. Calculate Propensity Scores (PSM requires a binary treatment, so we use Model 1)
X_psm = psm_df[['Lit_Pct', 'SC_Pct', 'ST_Pct']]
y_psm = psm_df['Binary_EVM']
logit = sm.Logit(y_psm, sm.add_constant(X_psm)).fit(disp=0)
psm_df['pscore'] = logit.predict(sm.add_constant(X_psm))

# 4. Nearest Neighbor Matching (1-to-1)
treated = psm_df[psm_df['Binary_EVM'] == 1][['pc91_district_id', 'pscore']]
control = psm_df[psm_df['Binary_EVM'] == 0][['pc91_district_id', 'pscore']]

nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(control[['pscore']])
distances, indices = nn.kneighbors(treated[['pscore']])
matched_control_ids = control.iloc[indices.flatten()]['pc91_district_id'].values

# 5. Filter Full Panel to ONLY Matched Districts
matched_ids = treated['pc91_district_id'].tolist() + matched_control_ids.tolist()
panel_psm = panel[panel['pc91_district_id'].isin(matched_ids)].copy()

print(f"   -> Matched Sample: {len(matched_ids)} districts ({len(treated)} Treated + {len(matched_control_ids)} Controls)")

# 6. Run DiD on the Matched Sample
psm_did_m1 = smf.ols("Female_Turnout ~ Binary_EVM + C(Year) + Binary_EVM:Post_1999", data=panel_psm).fit(cov_type='HC3')
psm_did_m2 = smf.ols("Female_Turnout ~ Continuous_EVM + C(Year) + Continuous_EVM:Post_1999", data=panel_psm).fit(cov_type='HC3')

print("\n--- PSM-MATCHED MODEL 1 (BINARY) ---")
print(f"Coefficient: {psm_did_m1.params['Binary_EVM:Post_1999']:.4f} (p={psm_did_m1.pvalues['Binary_EVM:Post_1999']:.3f})")

print("\n--- PSM-MATCHED MODEL 2 (CONTINUOUS) ---")
print(f"Coefficient: {psm_did_m2.params['Continuous_EVM:Post_1999']:.4f} (p={psm_did_m2.pvalues['Continuous_EVM:Post_1999']:.3f})")

print("\n" + "="*70)
print("  PIPELINE COMPLETE.")
print("="*70)
