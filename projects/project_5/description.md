# Project 5: Robustness, Diagnostics & Mechanism Analysis

## Overview
Comprehensive robustness checks and mechanism exploration. Tests alternative specifications, addresses potential biases, and disentangles competing theoretical channels explaining the EVM effect.

## Data Files

### `Step5_Master_Dataset_Final.csv`
Final analytical dataset with all transformations: IHS-transformed variables, mean-centered moderators, Z-scored mechanisms, propensity scores, and IPW weights. The complete replication file.

### `Step5_Table1_Summary_Statistics.csv`
Descriptive statistics for all analysis variables: means, standard deviations, min/max, medians. Essential for understanding sample composition and variable distributions.

### `Step5_Correlation_Matrix_Continuous.csv`
Pearson correlation matrix among continuous predictors. Diagnoses multicollinearity concerns that could inflate standard errors or destabilize estimates.

### `Step5_Main_Model_Forest_Plot.png`
Key coefficient visualization from the final ANCOVA model showing EVM main effect, agency main effect, and their interaction with 95% confidence intervals.

### `Step5_Marginal_Effects_Plot.png`
Marginal effects of EVM across the agency distribution, computed from the interaction model with HC1 robust standard errors.

### `Step5_IPW_Overlap_Plot.png`
Propensity score overlap assessment. Shows treatment and control group distributions after weighting—verifies common support for causal inference.

### `Step5_LOO_Sensitivity_Plot.png`
Leave-one-out sensitivity analysis. Tests whether any single district drives the results by iteratively re-estimating the model excluding each observation.

### `Step5_VIF_Diagnostics_Plot.png`
Variance Inflation Factor diagnostics. Quantifies multicollinearity severity; VIF > 10 signals problematic collinearity requiring attention.

### `Step5_Cooks_Distance_Plot.png`
Cook's distance influence diagnostics. Identifies high-leverage observations that disproportionately affect coefficient estimates.

### `Step5_FWL_Partial_Regression_Plot.png`
Frish-Waugh-Lovell partial regression plot. Visualizes the relationship between EVM and turnout after partialling out all controls—isolates the clean causal variation.

### `Step5_FWL_Partial_Residuals_Grid.png`
Grid of partial residual plots for all predictors. Diagnostic tool assessing functional form assumptions and identifying non-linearities.

### `Step5_Covariate_Balance_Love_Plot.png`
Extended covariate balance tests including pre-treatment turnout. Verifies that treatment assignment is uncorrelated with observables after weighting.

### `Step5_Subsample_Heterogeneity_Plot.png`
Heterogeneity analysis across different subsamples (e.g., urban/rural, high/low literacy). Tests external validity and boundary conditions.

### `Step5_Parallel_Trends_PreTrend_Plot.png`
Formal parallel trends validation comparing pre-trend (1996-98) vs. treatment effect (1996-99) coefficients with confidence intervals.

### `Step5_Mechanism_Horse_Race_Plot.png`
Mechanism comparison: Economic Agency vs. Cultural Patriarchy (child sex ratio). Z-scored coefficients allow direct magnitude comparison to identify the dominant channel.

### `Step5_Mechanism_Regression.csv` / `Step5_Mechanism_Summary.csv`
Horse race regression results pitting economic agency against deep-rooted patriarchy measures. Tests whether EVMs work through economic empowerment or cultural change channels.

### `Step5_Robustness_and_Diagnostics_Summary.csv`
Consolidated diagnostic metrics: VIF values, pre-trend test statistics, outlier counts, mechanism correlations. Single reference sheet for quality assurance.

## Key Insight
Robust causal claims survive multiple specification tests, show no concerning diagnostics, and have plausible mechanistic pathways. This comprehensive battery establishes credibility: the EVM effect is not an artifact of model choice, outliers, or omitted variables, and operates through economically interpretable channels.
