import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

print("="*70)
print("  SCRIPT 5: THE AGENCY INTERACTION (DDD MODEL)")
print("  Testing how Female Economic Agency moderates the EVM effect")
print("="*70)

# ==========================================
# PHASE 1: DATA LOADING & PANEL RECONSTRUCTION
# ==========================================
DATA_DIR = "/Users/ganeshchandrauniyal/Desktop/Thesis Script"
WEIGHTS_PATH = os.path.join(DATA_DIR, "PC2004_to_Dist1991_Weightage_Crosswalk (1).csv")
EC98_PATH = os.path.join(DATA_DIR, "Data/Economic Census 1991 Data/ec98_aggregated_1991_districts.csv")

crosswalk = pd.read_csv(WEIGHTS_PATH)
crosswalk['pc_name_clean'] = crosswalk['pc_name'].str.upper().str.strip()

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
    df['pc_name_clean'] = (df['Constituency'].str.upper().str.replace(r'\s*\(SC\)', '', regex=True)
                           .str.replace(r'\s*\(ST\)', '', regex=True).str.split(' NO :').str[0].str.strip())
    df = df[~df['pc_name_clean'].isin(CONTAMINATED_1998_PCS)]
    merged = pd.merge(cw_df, df, on='pc_name_clean', how='inner')
    merged['alloc_electors'] = merged['Electors_Female'] * merged['pc_weight_relative']
    merged['alloc_voters'] = merged['Voted_Female'] * merged['pc_weight_relative']
    merged['is_evm'] = merged['pc_name_clean'].isin(treated_pcs).astype(int)
    merged['treated_electors'] = merged['alloc_electors'] * merged['is_evm']
    
    dist_agg = merged.groupby(['pc91_state_id', 'pc91_district_id']).agg({
        'alloc_electors': 'sum', 'alloc_voters': 'sum', 'treated_electors': 'sum'
    }).reset_index()
    
    dist_agg['Female_Turnout'] = (dist_agg['alloc_voters'] / dist_agg['alloc_electors']) * 100
    dist_agg['Binary_EVM'] = (dist_agg['treated_electors'] > 0).astype(int)
    dist_agg['Continuous_EVM'] = dist_agg['treated_electors'] / dist_agg['alloc_electors']
    dist_agg['Year'] = year
    return dist_agg

print("Rebuilding District Panel...")
panel = pd.concat([
    aggregate_to_district('1996_election_data_corrected.csv', 1996, crosswalk, TREATED_PCS_1999),
    aggregate_to_district('1998_election_data_corrected.csv', 1998, crosswalk, TREATED_PCS_1999),
    aggregate_to_district('1999_election_data_corrected.csv', 1999, crosswalk, TREATED_PCS_1999)
], ignore_index=True)

# ==========================================
# PHASE 2: MERGE ECONOMIC CENSUS (AGENCY DATA)
# ==========================================
print("Merging 1998 Economic Census (Female Agency)...")
ec98 = pd.read_csv(EC98_PATH)
ec98.columns = ec98.columns.str.strip() # Clean hidden spaces
ec98['pc91_state_id'] = pd.to_numeric(ec98['pc91_state_id'], errors='coerce')
ec98['pc91_district_id'] = pd.to_numeric(ec98['pc91_district_id'], errors='coerce')

# Dynamically find the Enterprise column if the exact name varies
if 'Fem_Enterprise_Pct' not in ec98.columns:
    # Fallback: look for columns containing 'enterprise' or calculate from raw counts if available
    ent_cols = [c for c in ec98.columns if 'enterprise' in c.lower() or 'own_f' in c.lower()]
    if ent_cols:
        ec98['Fem_Enterprise_Pct'] = ec98[ent_cols[0]]
    else:
        raise ValueError("Could not find Female Enterprise column in EC98 data.")

ec98_subset = ec98[['pc91_state_id', 'pc91_district_id', 'Fem_Enterprise_Pct']].drop_duplicates()
panel = pd.merge(panel, ec98_subset, on=['pc91_state_id', 'pc91_district_id'], how='left')
panel['Fem_Enterprise_Pct'] = panel['Fem_Enterprise_Pct'].fillna(0)

# ==========================================
# PHASE 3: DDD MODEL SPECIFICATION
# ==========================================
print("Constructing DDD Interactions...")
panel['entity_id'] = panel['pc91_state_id'].astype(str) + "_" + panel['pc91_district_id'].astype(str)
panel['Post'] = (panel['Year'] == 1999).astype(int)

# Model 1 (Binary) Interactions
panel['EVM_Post'] = panel['Binary_EVM'] * panel['Post']
panel['Post_Agency'] = panel['Post'] * panel['Fem_Enterprise_Pct']
panel['EVM_Post_Agency'] = panel['Binary_EVM'] * panel['Post'] * panel['Fem_Enterprise_Pct']

# Model 2 (Continuous) Interactions
panel['Cont_EVM_Post'] = panel['Continuous_EVM'] * panel['Post']
panel['Cont_EVM_Post_Agency'] = panel['Continuous_EVM'] * panel['Post'] * panel['Fem_Enterprise_Pct']

# ==========================================
# PHASE 4: ESTIMATION (WITH FIXED EFFECTS)
# ==========================================
print("\nEstimating DDD Models (District & Year Fixed Effects)...")
# Note: C(entity_id) absorbs the main effects of EVM and Agency.
# C(Year) absorbs the main effect of Post.

formula_m1 = "Female_Turnout ~ EVM_Post + Post_Agency + EVM_Post_Agency + C(Year) + C(entity_id)"
formula_m2 = "Female_Turnout ~ Cont_EVM_Post + Post_Agency + Cont_EVM_Post_Agency + C(Year) + C(entity_id)"

# Fit with Robust Standard Errors (HC3)
res_m1 = smf.ols(formula_m1, data=panel).fit(cov_type='HC3')
res_m2 = smf.ols(formula_m2, data=panel).fit(cov_type='HC3')

print("\n--- MODEL 1: BINARY DDD (Intent-to-Treat) ---")
print(f"EVM_Post (Baseline EVM Effect):      {res_m1.params['EVM_Post']:.4f} (p={res_m1.pvalues['EVM_Post']:.3f})")
print(f"EVM_Post_Agency (The DDD Moderator): {res_m1.params['EVM_Post_Agency']:.4f} (p={res_m1.pvalues['EVM_Post_Agency']:.3f})")

print("\n--- MODEL 2: CONTINUOUS DDD (Dosage) ---")
print(f"Cont_EVM_Post (Baseline EVM Effect):      {res_m2.params['Cont_EVM_Post']:.4f} (p={res_m2.pvalues['Cont_EVM_Post']:.3f})")
print(f"Cont_EVM_Post_Agency (The DDD Moderator): {res_m2.params['Cont_EVM_Post_Agency']:.4f} (p={res_m2.pvalues['Cont_EVM_Post_Agency']:.3f})")

# ==========================================
# PHASE 5: MARGINAL EFFECTS VISUALIZATION
# ==========================================
print("\nGenerating Marginal Effects Plot...")

# We use Model 2 (Continuous) for the plot as it represents the true dosage
beta_evm = res_m2.params['Cont_EVM_Post']
beta_dd = res_m2.params['Cont_EVM_Post_Agency']

# 🛠️ FIX: Use .bse instead of .std_errors
se_evm = res_m2.bse['Cont_EVM_Post']
se_dd = res_m2.bse['Cont_EVM_Post_Agency']

agency_range = np.linspace(panel['Fem_Enterprise_Pct'].min(), panel['Fem_Enterprise_Pct'].max(), 100)
marginal_effects = beta_evm + beta_dd * agency_range

# Calculate Confidence Intervals
se_margins = np.sqrt(se_evm**2 + (agency_range**2) * se_dd**2)
ci_lower = marginal_effects - 1.96 * se_margins
ci_upper = marginal_effects + 1.96 * se_margins

plt.figure(figsize=(10, 6))
plt.plot(agency_range, marginal_effects, color='purple', lw=3, label='Marginal Effect of EVM Dosage')
plt.fill_between(agency_range, ci_lower, ci_upper, color='purple', alpha=0.2, label='95% Confidence Interval')
plt.axhline(0, color='red', linestyle='--', lw=2)
plt.title('Heterogeneous Treatment Effects: How Female Economic Agency Moderates EVM Impact', fontsize=14, fontweight='bold')
plt.xlabel('1998 Female Enterprise Density (Economic Agency)', fontsize=12)
plt.ylabel('Marginal Effect on Female Turnout (%)', fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("  SCRIPT 5 COMPLETE.")
print("="*70)