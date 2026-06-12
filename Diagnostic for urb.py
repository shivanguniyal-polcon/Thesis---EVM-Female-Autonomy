import pandas as pd
import numpy as np
import os

BASE_DIR = "/Users/ganeshchandrauniyal/Desktop/Thesis Script"
pca_path = os.path.join(BASE_DIR, "shrug-pca91-csv/pc91_pca_clean_pc91dist.csv")
td_path = os.path.join(BASE_DIR, "shrug-td91-csv/pc91_td_clean_pc91dist.csv")

print("="*60)
print("  URBANIZATION DIAGNOSTIC & FIX SCRIPT")
print("="*60)

# 1. Load Data
pca = pd.read_csv(pca_path)
td = pd.read_csv(td_path)

# 2. Diagnostic: Check Dtypes and Missing Columns
print("\n[DIAGNOSTIC 1] Column & Dtype Check:")
print(f"PCA ID Dtypes: {pca[['pc91_state_id', 'pc91_district_id']].dtypes.to_dict()}")
print(f"TD ID Dtypes:  {td[['pc91_state_id', 'pc91_district_id']].dtypes.to_dict()}")

if 'pc91_pca_tot_p' not in pca.columns:
    raise KeyError("❌ Denominator 'pc91_pca_tot_p' is missing from the PCA file!")
if 'pc91_td_p_7andup' not in td.columns:
    raise KeyError("❌ Numerator 'pc91_td_p_7andup' is missing from the TD file!")

# 3. Fix: Standardize Merge Keys (The usual suspect for silent merge failures)
# Convert to string, strip whitespace, and remove leading zeros to force exact matches
for df in [pca, td]:
    df['pc91_state_id'] = df['pc91_state_id'].astype(str).str.strip().str.lstrip('0')
    df['pc91_district_id'] = df['pc91_district_id'].astype(str).str.strip().str.lstrip('0')
    # Handle edge case where stripping '0' leaves an empty string (e.g. '00' -> '')
    df['pc91_state_id'] = df['pc91_state_id'].replace('', '0')
    df['pc91_district_id'] = df['pc91_district_id'].replace('', '0')

# 4. Merge
print("\n[DIAGNOSTIC 2] Merge Cardinality:")
print(f"PCA rows: {len(pca)} | TD rows: {len(td)}")
merged = pd.merge(pca, td, on=['pc91_state_id', 'pc91_district_id'], how='inner')
print(f"✅ Successfully merged rows: {len(merged)}")

if len(merged) == 0:
    raise ValueError("❌ Merge resulted in 0 rows! The IDs still do not match.")

# 5. Calculate & Clean
print("\n[DIAGNOSTIC 3] Math & Anomalies:")
# Replace 0s in denominator with NaN to avoid DivisionByZero warnings
denom = merged['pc91_pca_tot_p'].replace(0, np.nan)
merged['Urban_Pct'] = (merged['pc91_td_p_7andup'] / denom) * 100

print(f"NaNs generated (missing pop data): {merged['Urban_Pct'].isna().sum()}")
print(f"Values > 100% (spillover anomalies): {(merged['Urban_Pct'] > 100).sum()}")

# Cap at 100% and fill NaNs with 0
merged['Urban_Pct'] = merged['Urban_Pct'].clip(upper=100.0).fillna(0.0)

# 6. Export
out_path = os.path.join(BASE_DIR, "urban_pct_mapping.csv")
merged[['pc91_state_id', 'pc91_district_id', 'Urban_Pct']].to_csv(out_path, index=False)
print(f"\n✅ SUCCESS: Saved clean mapping to {out_path}")
print("\nFinal Distribution:")
print(merged['Urban_Pct'].describe())