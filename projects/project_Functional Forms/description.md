# Functional Forms and Variable Transformations

This document details the functional forms, mathematical transformations, and variable naming conventions used in the primary causal models and mechanism horse race regressions. 

## 1. Main Causal Model (ANCOVA) Variables

These variables constitute the baseline Difference-in-Differences / ANCOVA specification estimating the heterogeneous effect of EVM exposure on female voter turnout.

| Raw Variable in Dataset | Transformation Applied | Variable Name in Formula | Econometric Justification |
| :--- | :--- | :--- | :--- |
| **EVM_Exposure_Cont** | **Linear** (Untransformed) | `EVM_Exposure_Cont` | Standard continuous treatment variable. Kept linear to ensure coefficients remain easily interpretable as percentage-point changes. |
| **Turnout_96** | **Linear** (Untransformed) | `Turnout_96` | Baseline control. The linear assumption is standard and statistically optimal for ANCOVA baseline adjustments. |
| **Lit_Pct** | **Linear** (Untransformed) | `Lit_Pct` | Bounded between 0-100. Diagnostic BIC tests confirm it is well-behaved and does not require non-linear compression. |
| **SC_Pct** | **Linear** (Untransformed) | `SC_Pct` | Bounded between 0-100. Standard demographic control with a well-behaved distribution in this sample. |
| **ST_Pct** | **Inverse Hyperbolic Sine (IHS)** | `ST_Pct_ihs` | Handles severe right-skewness and safely accommodates districts with a 0% Scheduled Tribe population without requiring arbitrary constant additions. |
| **Urban_Pct** | **Inverse Hyperbolic Sine (IHS)** | `Urban_Pct_ihs` | Handles right-skewness and safely accommodates districts with 0% urban population. *(Note: Raw values are clipped at 100% prior to transformation).* |
| **Fem_Enterprise_Pct** | **IHS + Mean-Centered** | `Agency_Centered` | IHS handles zeros/skew in enterprise density. Mean-centering ensures the main `EVM_Exposure_Cont` coefficient is interpretable as the treatment effect at the *sample average* level of economic agency. |
| **state_clean** | **Categorical Dummies** | `C(state_clean)` | Absorbs all time-invariant, state-level unobserved heterogeneity (State Fixed Effects). |

---

## 2. Mechanism Horse Race Variables

These variables are introduced in the final specification to directly compare the magnitude of the "Economic Agency" mechanism against the "Deep-Rooted Patriarchy" mechanism. They are standardized to ensure their coefficients are directly comparable.

| Raw Variable in Dataset | Transformation Applied | Variable Name in Formula | Econometric Justification |
| :--- | :--- | :--- | :--- |
| **Agency_Centered** | **Z-Score Standardization** | `Agency_Z` | Divides the mean-centered IHS variable by its standard deviation. A "1 unit" coefficient change now represents a 1 Standard Deviation (SD) increase in economic agency. |
| **Sex_Ratio** | **Mean-Centered + Z-Scored** | `Patriarchy_Z` | Centers the child sex ratio (females per 1000 males) around its mean, then scales by its SD. This matches the magnitude of `Agency_Z`, allowing for a direct "horse race" comparison of standardized effect sizes. |

---

## 3. Interaction Terms

Interaction terms are constructed dynamically in the script to test heterogeneous treatment effects.

| Interaction Term | Formula Name | Construction Logic |
| :--- | :--- | :--- |
| **EVM × Economic Agency** | `EVM_Post_Agency` | `EVM_Exposure_Cont` $\times$ `Agency_Centered` |
| **EVM × Z-Scored Agency** | `EVM_Post_Agency_Z` | `EVM_Exposure_Cont` $\times$ `Agency_Z` |
| **EVM × Z-Scored Patriarchy**| `EVM_Post_Patriarchy_Z` | `EVM_Exposure_Cont` $\times$ `Patriarchy_Z` |
