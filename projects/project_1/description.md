### Process Summary:**
The analysis initiates with raw electoral returns from 543 Parliamentary Constituencies (PCs). The workflow involves merging 1991 and 1999 turnout datasets, filtering for the 46 treated PCs where EVMs were introduced, and constructing a binary treatment indicator. The core statistical engine applies a two-sample T-test to compare mean female turnout between treated and control groups, followed by a univariate OLS regression to quantify the raw "EVM penalty" without covariate adjustments. Visualizations include a boxplot distribution comparison and a map of treated constituencies.

### Summary Outcome (Data)
1. **Numbers**: N=543 PCs; Raw Mean Diff = -6.86 pp (EVM=-41.3%, Non-EVM=-48.2%); T-test p=0.00038; OLS β=-6.86 (SE=1.89, p=0.0004), R²=0.018.
2. **What it Proves**: A statistically significant, unadjusted negative association exists between EVM presence and female turnout in the raw data.
3. **What Still Needs Proving**: Whether this effect is causal or driven by omitted variable bias (demographics, state fixed effects, pre-trends).

---

### Detailed Sequential Analysis

**Step 1: Data Validation (`Step1_Data_Validation.csv`)**
- **Data**: Confirms 46 treated PCs and 497 control PCs. Total rows: 543. Missingness check: 0% missing on key variables (`female_turnout`, `treatment_status`).
- **Inference**: The sample is complete and balanced in terms of data availability, ready for statistical testing.

**Step 2: Descriptive Statistics (`Step2_Descriptive_Stats.csv`)**
- **Data**: 
  - Treated Mean Female Turnout: 41.3% (SD=14.1).
  - Control Mean Female Turnout: 48.2% (SD=12.4).
- **Inference**: The raw gap is substantial (~7 percentage points).

**Step 3: T-Test Results (`Step3_TTest_Results.csv`)**
- **Data**: t-statistic = -3.59, p-value = 0.00038.
- **Inference**: The probability of observing this gap by random chance is <0.05. The null hypothesis (no difference) is rejected.

**Step 4: Univariate OLS (`Step4_Univariate_OLS.csv`)**
- **Data**: Coefficient on `treatment` = -6.86, Standard Error = 1.89, p-value = 0.0004.
- **Inference**: Confirms the T-test result in a regression framework. The "raw penalty" is robustly estimated at ~6.9 percentage points.

**Visualizations**
- `boxplot_comparison.png`: Visually demonstrates the distribution shift. The median line for EVM constituencies is visibly lower than the non-EVM box. The interquartile ranges show overlap, but the central tendency difference is clear.
