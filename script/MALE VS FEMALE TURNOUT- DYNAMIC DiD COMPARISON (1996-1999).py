import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("  MALE VS FEMALE TURNOUT: DYNAMIC DiD COMPARISON (1996-1999)")
print("="*70)

# 1. Load Data
DATA_DIR = "/Users/ganeshchandrauniyal/Desktop/Thesis Script"
years = [1996, 1998, 1999]
dfs = []

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

for year in years:
    file_path = os.path.join(DATA_DIR, f"{year}_election_data_corrected.csv")
    df = pd.read_csv(file_path)
    
    # Clean PC names
    df['pc_name_clean'] = df['Constituency'].str.split(' NO :').str[0].str.strip().str.upper()
    
    # Calculate Turnouts
    df['Female_Turnout'] = (df['Voted_Female'] / df['Electors_Female']) * 100
    df['Male_Turnout'] = (df['Voted_Male'] / df['Electors_Male']) * 100
    
    # Treatment Group Dummy (1 if in treated group, 0 otherwise)
    df['EVM_Group'] = df['pc_name_clean'].isin(TREATED_PCS).astype(int)
    
    df['Year'] = year
    dfs.append(df[['pc_name_clean', 'Year', 'EVM_Group', 'Female_Turnout', 'Male_Turnout']])

# Combine into panel
panel = pd.concat(dfs, ignore_index=True)

# Create Post dummy (1 for 1999, 0 for 1996/1998)
panel['Post'] = (panel['Year'] == 1999).astype(int)

# Interaction term (DiD Estimator)
panel['EVM_Post'] = panel['EVM_Group'] * panel['Post']

# 2. Run DiD Models
print("\n--- MODEL 1: FEMALE TURNOUT DiD ---")
# Using C(Year) absorbs the 'Post' dummy, avoiding the dummy variable trap
model_f = smf.ols("Female_Turnout ~ EVM_Post + EVM_Group + C(Year)", data=panel).fit(cov_type='HC3')
print(model_f.summary().tables[1])

print("\n--- MODEL 2: MALE TURNOUT DiD ---")
model_m = smf.ols("Male_Turnout ~ EVM_Post + EVM_Group + C(Year)", data=panel).fit(cov_type='HC3')
print(model_m.summary().tables[1])

# 3. Comparison
print("\n--- COEFFICIENT COMPARISON (The Mechanism Test) ---")
print(f"EVM Effect on Female Turnout: {model_f.params['EVM_Post']:.4f} (p={model_f.pvalues['EVM_Post']:.4f})")
print(f"EVM Effect on Male Turnout:   {model_m.params['EVM_Post']:.4f} (p={model_m.pvalues['EVM_Post']:.4f})")

# Create a long-format dataset with gender as a variable
panel_long = pd.melt(
    panel[['pc_name_clean', 'Year', 'EVM_Group', 'Post', 'Female_Turnout', 'Male_Turnout']],
    id_vars=['pc_name_clean', 'Year', 'EVM_Group', 'Post'],
    value_vars=['Female_Turnout', 'Male_Turnout'],
    var_name='Gender',
    value_name='Turnout'
)
panel_long['Female'] = (panel_long['Gender'] == 'Female_Turnout').astype(int)
panel_long['EVM_Post_Female'] = panel_long['EVM_Group'] * panel_long['Post'] * panel_long['Female']

# Test if the EVM effect differs by gender
model_gender = smf.ols(
    "Turnout ~ EVM_Group*Post*Female + C(Year) + C(pc_name_clean)", 
    data=panel_long
).fit(cov_type='HC3')

print(f"Gender Interaction (EVM_Post_Female): {model_gender.params['EVM_Post_Female']:.4f} (p={model_gender.pvalues['EVM_Post_Female']:.3f})")
# If p > 0.10, you fail to reject that the effects are equal