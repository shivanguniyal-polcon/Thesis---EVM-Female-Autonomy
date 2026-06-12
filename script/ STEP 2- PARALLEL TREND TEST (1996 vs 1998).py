import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("="*60)
print("  STEP 2: PARALLEL TREND TEST (1996 vs 1998)")
print("="*60)

# 1. The Verified 46 EVM PCs List
TREATED_PCS_1999 = [
    'HYDERABAD', 'SECUNDERABAD', 'PANAJI', 'MORMUGAO', 'AHMEDABAD', 'GANDHINAGAR', 
    'KARNAL', 'ROHTAK', 'BANGALORE NORTH', 'BANGALORE SOUTH', 'MYSORE', 'ERNAKULAM', 
    'TRIVANDRUM', 'GWALIOR', 'BHOPAL', 'MUMBAI SOUTH', 'MUMBAI SOUTH CENTRA', 
    'MUMBAI NORTH CENTRAL', 'MUMBAI NORTH EAST', 'MUMBAI NORTH WEST', 'NEW DELHI',
    'SOUTH DELHI', 'OUTER DELHI', 'EAST DELHI', 'CHANDNI CHOWK', 'DELHI SADAR', 'KAROL BAGH',
    'LUCKNOW', 'KANPUR', 'PATNA', 'RANCHI', 'JAIPUR', 'AJMER', 'MADRAS CENTRAL', 'MADRAS NORTH',
    'MADRAS SOUTH', 'COIMBATORE', 'MADURAI', 'CALCUTTA NORTH WEST', 'CALCUTTA NORTH EAST',
    'CALCUTTA SOUTH', 'HOWRAH', 'GAUHATI', 'CHANDIGARH', 'PONDICHERRY', 'ALLAHABAD', 
    'AGRA', 'TARN TARAN', 'PATIALA', 'FARIDKOT', 'BHUBANESWAR'
]

# Helper function to clean and calculate turnout
def process_year(file_name, year):
    # Check both main directory and Data subdirectory
    for path in [f'/Users/ganeshchandrauniyal/Desktop/Thesis Script/{file_name}', 
                 f'/Users/ganeshchandrauniyal/Desktop/Thesis Script/Data/{file_name}']:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
            
    df['pc_name_clean'] = (
        df['Constituency']
        .str.upper()
        .str.replace(r'\s*\(SC\)', '', regex=True)
        .str.replace(r'\s*\(ST\)', '', regex=True)
        .str.split(' NO :').str[0]
        .str.strip()
    )
    df['EVM'] = df['pc_name_clean'].isin(TREATED_PCS_1999).astype(int)
    df['Female_Turnout'] = (df['Voted_Female'] / df['Electors_Female']) * 100
    df['Year'] = year
    return df[['pc_name_clean', 'EVM', 'Female_Turnout', 'Year']]

# 2. Load Pre-Treatment Data (1996 and 1998)
print("Loading 1996 and 1998 data...")
df_1996 = process_year('1996_election_data_corrected.csv', 1996)
df_1998 = process_year('1998_election_data_corrected.csv', 1998)
df_1999 = process_year('1999_election_data_corrected.csv', 1999)

# 3. Statistical Test: Did the gap change between 1996 and 1998?
print("\n[1] STATISTICAL PARALLEL TREND TEST (1996 to 1998)")
# Merge 1996 and 1998 to calculate the change
merged_pre = pd.merge(df_1996, df_1998, on='pc_name_clean', suffixes=('_1996', '_1998'))
merged_pre['Delta_Turnout_98_96'] = merged_pre['Female_Turnout_1998'] - merged_pre['Female_Turnout_1996']

# Regress the change in turnout on the EVM dummy
# If the coefficient is insignificant, it means EVM and Paper PCs changed at the exact same rate before 1999.
parallel_model = smf.ols('Delta_Turnout_98_96 ~ EVM_1996', data=merged_pre).fit()

print(f"Mean Change in Turnout (1996-1998) for Paper PCs: {merged_pre[merged_pre['EVM_1996']==0]['Delta_Turnout_98_96'].mean():.2f}%")
print(f"Mean Change in Turnout (1996-1998) for EVM PCs:   {merged_pre[merged_pre['EVM_1996']==1]['Delta_Turnout_98_96'].mean():.2f}%")
print(f"\nRegression Coefficient for EVM: {parallel_model.params['EVM_1996']:.4f}")
print(f"P-value:                        {parallel_model.pvalues['EVM_1996']:.4f}")
if parallel_model.pvalues['EVM_1996'] > 0.05:
    print("✅ RESULT: Fail to reject null hypothesis. Parallel trends assumption HOLDS.")
else:
    print("⚠️ RESULT: Reject null hypothesis. Trends were diverging BEFORE 1999!")

# 4. Visual Test: Plotting all three years
print("\n[2] VISUAL PARALLEL TREND PLOT (1996, 1998, 1999)")
# Combine all years
panel_df = pd.concat([df_1996, df_1998, df_1999], ignore_index=True)
trends = panel_df.groupby(['Year', 'EVM'])['Female_Turnout'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(data=trends, x='Year', y='Female_Turnout', hue='EVM', 
             marker='o', linewidth=3, markersize=10, palette=['#d3d3d3', '#6A0DAD'])

# Formatting
plt.title('Parallel Trend Test: Female Turnout by Voting Technology (1996-1999)', fontsize=14, fontweight='bold')
plt.xlabel('General Election Year', fontsize=12)
plt.ylabel('Average Female Voter Turnout (%)', fontsize=12)
plt.xticks([1996, 1998, 1999], fontsize=12)
plt.legend(title='Technology in 1999', labels=['Paper Ballots', 'EVMs'], fontsize=11, title_fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)

# Add a vertical line for the 1999 Treatment
plt.axvline(x=1998.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='EVM Introduction (1999)')
plt.text(1998.6, plt.ylim()[1] - 1, ' Treatment', color='red', fontsize=11, verticalalignment='top')

plt.tight_layout()
plt.show()

print("\n--- STEP 2 COMPLETE ---")