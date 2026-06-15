# Project 1: Raw Correlation Analysis

## Overview
Establishes the foundational relationship between Electronic Voting Machine (EVM) adoption and female voter turnout in India's 1999 Lok Sabha elections.

##Data Files

### `Step1_Cleaned_PC_Data.csv`
Constituency-level electoral data with EVM treatment assignment. Contains raw elector counts, voted counts by gender, and derived female turnout percentages for 1999.

### `Step1_Descriptive_Stats.csv`
Summary statistics comparing female turnout distributions between EVM and paper ballot constituencies—means, standard deviations, and quartile breakdowns.

### `Step1_Simple_OLS.csv`
Univariate regression results: `Female_Turnout ~ EVM`. Reports the raw correlation coefficient without controls—the starting point for causal inference.

### `Step1_TTest_Results.csv`
Two-sample t-test comparing mean female turnout between treated (EVM) and control (paper ballot) groups. Tests whether the raw difference is statistically distinguishable from zero.

## Visualizations

### `Step1_Raw EVM 1999.png`
Boxplot visualization showing the distribution of female turnout across paper ballot vs. EVM constituencies. Provides immediate visual evidence of the treatment effect magnitude.

## Key Insight
The naive comparison reveals a positive association between EVM deployment and female participation—a pattern that motivates deeper investigation but requires rigorous controls to establish causality.
