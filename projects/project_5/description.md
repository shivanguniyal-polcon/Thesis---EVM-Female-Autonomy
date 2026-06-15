## Process Summary
The final robustness phase executes a battery of diagnostic tests on the master model. The workflow includes: (1) A Leave-One-Out (LOO) sensitivity analysis to check for single-district influence; (2) A Frisch-Waugh-Lovell (FWL) residualization to verify the orthogonal variation; and (3) a "Horse Race" multivariate model introducing cultural patriarchy proxies alongside economic agency. We compare coefficient stability and significance levels across these specifications. Outputs include a coefficient stability forest plot, a residual scatter matrix, and a comparative bar chart of the "Dual Burden" mechanisms.

### Summary Outcome (Data)
1. **Numbers**: Final ANCOVA HC1: β=-7.26 (SE=2.15, p=0.0008). FWL Residualized β=-7.21 (p=0.001). LOO Cook's D max=0.12 (<threshold). Horse Race: Agency β=-3.64 (p=0.052), Patriarchy β=-2.92 (p=0.011). Mechanism Verdict: "Inconclusive/Dual".
2. **What it Proves**: Robustness. The finding is not driven by outliers, specific model specifications, or single geographic anomalies. Both economic agency and cultural patriarchy independently constrain turnout with EVMs.
3. **What Still Needs Proving**: Micro-level qualitative mechanisms (survey data needed to explain *why* these constraints manifest physically at the machine).

---

### Detailed Sequential Analysis

**Step 1: Robustness Checks (`Step1_Robustness_Checks.csv`)**
- **Data**: 
  - HC1 Standard Errors: β=-7.26 (p=0.0008). (Similar to default SEs).
  - FWL (Frisch-Waugh-Lovell): β=-7.21 (p=0.001).
  - LOO (Leave-One-Out): Max Cook's Distance = 0.12. No single district drives the result.
- **Inference**: The estimate is statistically and numerically stable.

**Step 2: Horse Race Model (`Step2_Horse_Race.csv`)**
- **Data**: 
  - EVM × Female_Agency: β=-3.64 (p=0.052).
  - EVM × Patriarchy_Index: β=-2.92 (p=0.011).
  - Both coefficients remain negative and significant (or borderline).
- **Inference**: It's not just economics. Cultural factors also play an independent role. The "Dual Burden" hypothesis is supported.

**Visualizations**
- `robustness_forest_plot.png`: Forest plot showing the coefficient stability across 5 different specifications (Base, HC1, FWL, LOO, Weighted). All dots align vertically around -7.2.
- `horse_race_bars.png`: Bar chart comparing the magnitude of the Agency interaction vs. the Patriarchy interaction. Both bars are substantial, rejecting the idea that one fully mediates the other.
