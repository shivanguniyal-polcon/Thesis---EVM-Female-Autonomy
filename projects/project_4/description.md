# Project 4: Difference-in-Differences & Pre-Trend Validation

## Overview
Implements a quasi-experimental Difference-in-Differences design using 1996, 1998, and 1999 election cycles. Validates the parallel trends assumption—the cornerstone of causal inference in DiD designs.

## Data Files

### `Step4_Panel_Data.csv`
Balanced panel dataset with three time periods per district. Contains turnout measures for 1996 (pre-pre), 1998 (pre), and 1999 (post), enabling within-district comparisons over time.

### `Step4_Model_PreTrend.csv`
Placebo test regression: `ΔTurnout_96-98 ~ EVM × Agency`. Tests whether differential trends existed before treatment. A null result validates the parallel trends assumption.

### `Step4_Model_DiD.csv`
Main DiD specification: `ΔTurnout_96-99 ~ EVM × Agency`. Estimates the causal effect by comparing changes in treated vs. control districts from pre to post.

### `Step4_Model_ANCOVA.csv`
ANCOVA robustness model: `Turnout_99 ~ EVM × Agency + Turnout_96 + Controls`. Alternative specification controlling for baseline turnout directly, often more statistically efficient than pure DiD.

### `Step4_Full_Diagnostics.csv`
Comprehensive diagnostic tests for all models: R², F-statistics, AIC/BIC, condition numbers (multicollinearity), Durbin-Watson (autocorrelation), Breusch-Pagan (heteroskedasticity), and Jarque-Bera (normality).

## Visualizations

### `Step4_PreTrend_Event_Study.png`
Event study plot comparing the interaction coefficient for the placebo period (1996-98) vs. treatment period (1996-99). The pre-trend should be indistinguishable from zero; the treatment effect should be significant.

### `Step4_Diagnostic_Residual_Plots.png`
Residual diagnostics: (1) Residuals vs. Fitted values checking for heteroskedasticity patterns, (2) Q-Q plot assessing normality of residuals. Validates OLS assumptions underlying inference.

## Key Insight
Credible causal claims require passing the pre-trend test. If the interaction term is insignificant in 1996-98 but significant in 1996-99, this rules out pre-existing differential trends as an alternative explanation—strengthening the case that EVMs caused the turnout increase.
