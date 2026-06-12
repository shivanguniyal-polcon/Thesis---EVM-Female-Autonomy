import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("="*60)
print("  STEP 3: SURGICALLY CLEANED DIFF-IN-DIFF (DiD)")
print("="*60)

# 1. The Verified 46 EVM PCs List (1999 Treatment)
TREATED_PCS_1999 = [
    'HYDERABAD', 'SECUNDERABAD', 'PANAJI', 'MORMUGAO', 'AHMEDABAD', 'GANDHINAGAR', 
    'KARNAL', 'ROHTAK', 'BANGALORE NORTH', 'BANGALORE SOUTH', 'MYSORE', 'ERNAKULAM', 
    'TRIVANDRUM', 'GWALIOR', 'BHOPAL', 'MUMBAI SOUTH', 'MUMBAI SOUTH CENTRA', 
    'MUMBAI NORTH CENTRAL', 'MUMBAI NORTH EAST', 'MUMBAI NORTH WEST', 'NEW DELHI',
    'SOUTH DELHI', 'OUTER DELHI', 'EAST DELHI', 'CHANDNI CHOWK', 'DELHI SADAR', 'KAROL BAGH',
    'LUCKNOW', 'KANPUR', 'PATNA', 'RANCHI', 'JAIPUR', 'AJMER', 'MADRAS CENTRAL', 'MADRAS SOUTH',
    'COIMBATORE', 'MADURAI', 'CALCUTTA NORTH WEST', 'CALCUTTA NORTH EAST',
    'CALCUTTA SOUTH', 'HOWRAH', 'GAUHATI', 'CHANDIGARH', 'PONDICHERRY', 'ALLAHABAD', 
    'AGRA', 'TARN TARAN', 'PATIALA', 'FARIDKOT', 'BHUBANESWAR'
]

# 🛡️ THE SURGICAL SCALPEL: The 8 PCs contaminated by 1998 State Assembly EVMs
CONTAMINATED_1998_PCS = [
    'NEW DELHI', 'SOUTH DELHI', 'OUTER DELHI', 'CHANDNI CHOWK', 
    'JAIPUR', 'AJMER', 'GWALIOR', 'BHOPAL'
]

# Helper to load and clean
def load_clean(file_name, year):
    path = f'/Users/ganeshchandrauniyal/Desktop/Thesis Script/{file_name}'
    if not os.path.exists(path):
        path = f'/Users/ganeshchandrauniyal/Desktop/Thesis Script/Data/{file_name}'
    
    df = pd.read_csv(path)
    df['pc_name_clean'] = (
        df['Constituency'].str.upper()
        .str.replace(r'\s*\(SC\)', '', regex=True)
        .str.replace(r'\s*\(ST\)', '', regex=True)
        .str.split(' NO :').str[0].str.strip()
    )
    
    # Tag 1999 Treatment
    df['EVM_1999'] = df['pc_name_clean'].isin(TREATED_PCS_1999).astype(int)
    
    # 🛡️ Tag 1998 Contamination
    df['Contaminated_1998'] = df['pc_name_clean'].isin(CONTAMINATED_1998_PCS).astype(int)
    
    df['Female_Turnout'] = (df['Voted_Female'] / df['Electors_Female']) * 100
    df['Year'] = year
    return df

# 2. Build the Panel Dataset (1996, 1998, 1999)
print("Building Panel Dataset...")
df_96 = load_clean('1996_election_data_corrected.csv', 1996)
df_98 = load_clean('1998_election_data_corrected.csv', 1998)
df_99 = load_clean('1999_election_data_corrected.csv', 1999)

panel = pd.concat([df_96, df_98, df_99], ignore_index=True)

# 🛡️ SURGICAL DROP: Remove the 8 contaminated PCs entirely from the sample.
# This ensures our "Pre-Treatment" period (1996 & 1998) is 100% EVM-free for everyone.
initial_pcs = panel['pc_name_clean'].nunique()
panel_clean = panel[panel['Contaminated_1998'] == 0].copy()
final_pcs = panel_clean['pc_name_clean'].nunique()

print(f"   -> Dropped {initial_pcs - final_pcs} PCs contaminated by 1998 State Elections.")
print(f"   -> Final Clean Sample: {final_pcs} PCs ({len(panel_clean)} total observations).")

# 3. Define Treatment Variables for the Clean Sample
panel_clean['Post_1999'] = (panel_clean['Year'] == 1999).astype(int)

# 4. Run the Clean DiD Regression
print("\nRunning Surgically Cleaned DiD Regression...")
model_did = smf.ols(
    "Female_Turnout ~ EVM_1999 + C(Year) + EVM_1999:Post_1999", 
    data=panel_clean
).fit(cov_type='HC3') 

print("\n--- SURGICALLY CLEANED DiD RESULTS ---")
print(model_did.summary().tables[1])

# 5. Interpret the Coefficient
beta_did = model_did.params['EVM_1999:Post_1999']
p_did = model_did.pvalues['EVM_1999:Post_1999']

print("\n--- INTERPRETATION ---")
print(f"Clean DiD Coefficient (EVM_1999:Post_1999): {beta_did:.4f}")
print(f"P-value: {p_did:.4f}")

# 6. Visual Proof: Re-run Parallel Trends on the Cleaned Sample
print("\nGenerating Cleaned Parallel Trends Plot...")
trends = panel_clean.groupby(['Year', 'EVM_1999'])['Female_Turnout'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(data=trends, x='Year', y='Female_Turnout', hue='EVM_1999', 
             marker='o', linewidth=3, markersize=10, palette=['#d3d3d3', '#6A0DAD'])

plt.title('Cleaned Parallel Trend Test (16 ACs / 8 PCs Surgically Removed)\n1996-1998 Pre-Treatment is Now 100% EVM-Free', 
          fontsize=14, fontweight='bold')
plt.xlabel('General Election Year', fontsize=12)
plt.ylabel('Average Female Voter Turnout (%)', fontsize=12)
plt.xticks([1996, 1998, 1999], fontsize=12)
plt.legend(title='Technology in 1999', labels=['Paper Ballots', 'EVMs'], fontsize=11, title_fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.axvline(x=1998.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
plt.text(1998.6, plt.ylim()[1] - 1, ' 1999 EVM Treatment', color='red', fontsize=11, verticalalignment='top')

plt.tight_layout()
plt.show()

print("\n--- STEP 3 COMPLETE ---")