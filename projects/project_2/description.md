# Project 2: Spatial Projection & Demographic Controls

## Overview
Projects constituency-level data to 1991 district boundaries and introduces demographic covariates to address selection bias. Tests whether the EVM effect persists after accounting for observable differences.

## Data Files

### `Step2_District_Level_Data.csv`
Analytical dataset aggregated to 1991 district boundaries using spatial crosswalks. Contains district-level female turnout, continuous EVM exposure measures, and demographic controls.

### `Step2_Covariate_Balance.csv`
Balance tests examining whether EVM rollout correlates with baseline demographics. Each row tests if a covariate (literacy, urbanization, caste composition) predicts treatment assignment.

### `Step2_Model1_NoControls.csv`
Baseline regression: `Female_Turnout ~ EVM_Exposure` at district level. Replicates Step 1 correlation with spatially aggregated data.

### `Step2_Model2_Demographics.csv`
Multivariate regression adding demographic controls: literacy rate, Scheduled Caste/Tribe percentages, and urbanization. Tests robustness to observable confounders.

### `Step2_Model3_StateFEs.csv`
Full specification with state fixed effects. Absorbs unobserved state-level heterogeneity—policy environments, cultural factors, administrative quality.

## Visualizations

### `Step2_EVM_Coef_Forest_Plot.png`
Forest plot displaying the EVM coefficient across three model specifications. Demonstrates coefficient stability as controls are progressively added.

### `Step2_Covariate_Balance_Plot.png`
Love plot showing correlations between EVM exposure and covariates. Green bars indicate balanced covariates; red signals potential selection concerns.

## Key Insight
If the EVM coefficient remains stable across specifications and covariates are balanced, this supports the identifying assumption that treatment assignment is quasi-random conditional on observables and state fixed effects.
