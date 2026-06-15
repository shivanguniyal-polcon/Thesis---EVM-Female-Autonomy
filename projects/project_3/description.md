# Project 3: Heterogeneous Effects & Economic Agency

## Overview
Investigates whether the EVM effect varies by baseline female economic empowerment. Tests the hypothesis that technology reduces voting costs more for women in economically empowered positions.

## Data Files

### `Step3_District_Agency_Data.csv`
Extended analytical dataset incorporating 1998 Economic Census data. Contains female enterprise density, IHS-transformed agency measures, and mean-centered interaction terms.

### `Step3_Model_MainEffects.csv`
Additive model with EVM exposure and economic agency as separate predictors. Assumes uniform EVM effect across all districts regardless of baseline agency.

### `Step3_Model_Interaction.csv`
Interaction model: `Turnout ~ EVM × Agency + Controls`. The coefficient on the interaction term captures heterogeneous treatment effects—whether EVMs matter more where women already have economic footholds.

### `Step3_Marginal_Effects.csv`
Marginal effects of EVM exposure computed at different percentiles of the agency distribution. Shows how the treatment effect magnitude changes from low- to high-agency districts.

### `Step3_Subsample_Results.csv`
Split-sample analysis comparing EVM effects in above-median vs. below-median agency districts. Provides intuitive interpretation of heterogeneity.

## Visualizations

### `Step3_Marginal_Effects_Plot.png`
Continuous marginal effects plot showing the EVM coefficient (with 95% CI) across the full range of female enterprise density. Slope indicates how effects amplify or diminish with agency.

### `Step3_Subsample_Heterogeneity.png`
Coefficient comparison between high- and low-agency subsamples. Visual test of whether the effect is concentrated in economically empowered areas.

## Key Insight
A positive, significant interaction suggests complementarities: EVMs amplify existing economic agency rather than acting as a universal equalizer. This informs targeted policy—technology alone may not overcome deep structural barriers.
