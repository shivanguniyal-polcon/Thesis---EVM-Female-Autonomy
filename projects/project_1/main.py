import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

import os
BASE_DIR = "/Users/ganeshchandrauniyal/Desktop/Thesis Script"
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
print("  STEP 1: THE RAW CORRELATION (EVM vs. FEMALE TURNOUT)")
#1. Load the 1999 Election Data
df_1999 = pd.read_csv('/Users/ganeshchandrauniyal/Desktop/Thesis Script/1999_election_data_corrected.csv')

#2.Clean PC names to match our EVM list
df_1999['pc_name_clean'] = df_1999['Constituency'].str.split(' NO :').str[0].str.strip().str.upper()

#3.Treated PCs based on ECI data
TREATED_PCS_1999 = [
    # Andhra Pradesh (2)
    'HYDERABAD', 'SECUNDERABAD',
    # Goa (2)
    'PANAJI', 'MORMUGAO',
    # Gujarat (2)
    'AHMEDABAD', 'GANDHINAGAR',
    # Haryana (2)
    'KARNAL', 'ROHTAK',
    # Karnataka (3)
    'BANGALORE NORTH', 'BANGALORE SOUTH', 'MYSORE',
    # Kerala (2)
    'ERNAKULAM', 'TRIVANDRUM',
    # Madhya Pradesh (2)
    'GWALIOR', 'BHOPAL',
    # Maharashtra (5)
    'MUMBAI SOUTH', 'MUMBAI SOUTH CENTRA', 'MUMBAI NORTH CENTRAL', 'MUMBAI NORTH EAST', 'MUMBAI NORTH WEST',
    # Orissa (1)
    'BHUBANESWAR',
    # Punjab (3)
    'TARN TARAN', 'PATIALA', 'FARIDKOT',
    # Rajasthan (2)
    'JAIPUR', 'AJMER',
    # Tamil Nadu (4)
    'MADRAS CENTRAL', 'MADRAS SOUTH', 'COIMBATORE', 'MADURAI',
    # Uttar Pradesh (4)
    'LUCKNOW', 'ALLAHABAD', 'KANPUR', 'AGRA',
    # West Bengal (3)
    'CALCUTTA NORTH WEST', 'CALCUTTA NORTH EAST', 'CALCUTTA SOUTH',
    # Chandigarh (1)
    'CHANDIGARH',
    # Delhi (7)
    'NEW DELHI', 'SOUTH DELHI', 'OUTER DELHI', 'EAST DELHI', 'CHANDNI CHOWK', 'DELHI SADAR', 'KAROL BAGH',
    # Pondicherry (1)
    'PONDICHERRY'
]

df_1999['EVM'] = df_1999['pc_name_clean'].isin(TREATED_PCS_1999).astype(int)

# Debug: Show which PCs were matched
matched_count = df_1999['EVM'].sum()
print(f"Total EVM PCs identified: {matched_count}")

if matched_count < 47: # We expect 46 treated PCs based on our list, but just to be sure put 47 as the threshold
    missing = set(TREATED_PCS_1999) - set(df_1999[df_1999['EVM']==1]['pc_name_clean'])
    print(f"\nMissing EVM PCs ({len(missing)}): {missing}")

# 4. Calculate Female Voter Turnout (%)
df_1999['Female_Turnout'] = (df_1999['Voted_Female'] / df_1999['Electors_Female']) * 100

# 5. Descriptive Statistics
print("\n[1] DESCRIPTIVE STATISTICS")
print(df_1999.groupby('EVM')['Female_Turnout'].describe())

# 6. T-Test (Difference in Means)
evm_turnout = df_1999[df_1999['EVM'] == 1]['Female_Turnout']
paper_turnout = df_1999[df_1999['EVM'] == 0]['Female_Turnout']
t_stat, p_val = stats.ttest_ind(evm_turnout, paper_turnout)
print(f"\n[2] T-TEST (Difference in Means)")
print(f"EVM Mean Turnout:{evm_turnout.mean():.2f}%")
print(f"Paper Mean Turnout:{paper_turnout.mean():.2f}%")
print(f"Raw Difference:{evm_turnout.mean() - paper_turnout.mean():.2f}%")
print(f"P-value:  {p_val:.4f}")

# 7. Simple Univariate OLS Regression (Turnout = Beta_0 + Beta_1*EVM)
print("\n[3] SIMPLE OLS REGRESSION (Female_Turnout ~ EVM)")
model = smf.ols("Female_Turnout ~ EVM", data=df_1999).fit() #a simple regression without controls or fixed effects, purely to show the raw correlation.
print(model.summary().tables[1])

# 8. Visualization
plt.figure(figsize=(8, 5))
sns.boxplot(x='EVM', y='Female_Turnout', data=df_1999, hue='EVM', palette=['#d3d3d3', '#6A0DAD'], legend=False)
plt.title('Raw Female Turnout: Paper Ballots vs. EVMs (1999 Lok Sabha)', fontsize=14, fontweight='bold')
plt.xlabel('Voting Technology', fontsize=12)
plt.ylabel('Female Voter Turnout (%)', fontsize=12)
plt.xticks([0, 1], ['Paper Ballots (0)', 'EVMs (1)'])
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()

# [CORE END]

print("\n Completed Step 1; now we run a check on whether all PCs are accounted for:")

# 1. Descriptive Statistics by EVM Status
desc_stats = df_1999.groupby('EVM')['Female_Turnout'].describe().reset_index()
desc_stats.to_csv(os.path.join(CSV_DIR, "Step1_Descriptive_Stats.csv"), index=False)

# 2. T-Test Results
ttest_df = pd.DataFrame({
    'Metric': ['EVM_Mean', 'Paper_Mean', 'Raw_Difference', 'T_Statistic', 'P_Value'],
    'Value': [evm_turnout.mean(), paper_turnout.mean(), 
              evm_turnout.mean() - paper_turnout.mean(), t_stat, p_val]
})
ttest_df.to_csv(os.path.join(CSV_DIR, "Step1_TTest_Results.csv"), index=False)

# 3. Simple OLS Regression
export_ols_to_csv(model, "Simple_OLS_No_Controls", os.path.join(CSV_DIR, "Step1_Simple_OLS.csv"))

# 4. Cleaned PC-Level Dataset
df_1999.to_csv(os.path.join(CSV_DIR, "Step1_Cleaned_PC_Data.csv"), index=False)

# After creating pc_name_clean, check which ones matched
print(f"\nTotal PCs in dataset: {len(df_1999)}")
print(f"Total EVM PCs tagged: {df_1999['EVM'].sum()}")

# Find which treated PCs were found
found_pcs = df_1999[df_1999['EVM'] == 1]['pc_name_clean'].unique()
print(f"\nFound {len(found_pcs)} EVM PCs in data:")
print(sorted(found_pcs))

# Find which treated PCs are MISSING
missing_pcs = set(TREATED_PCS_1999) - set(found_pcs)
print(f"\nMissing {len(missing_pcs)} PC(s) from data:")
print(sorted(missing_pcs))

# Show what the actual names look like in the data for similar PCs
if missing_pcs:
    print("\nLooking for similar names in the dataset for missing PCs")
    for missing in missing_pcs:
        # Search for partial matches
        similar = df_1999[df_1999['pc_name_clean'].str.contains(missing.split()[0], case=False, na=False)]['pc_name_clean'].unique()
        if len(similar) > 0:
            print(f"\n'{missing}' might be named:")
            print(f"{similar}")
