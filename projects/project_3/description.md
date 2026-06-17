### Process Summary
To explore heterogeneity, district-level female economic agency is interacted with the EVM treatment. The model calculates marginal effects at key percentiles, visualized via an interaction plot showing EVM's conditional impact across agency levels.

**Data Transformation (Inverse Hyperbolic Sine & Mean-Centering)**:$$\text{Agency\_Centered}_i=\text{arcsinh}(\text{Fem\_Enterprise\_Pct}_i)-\overline{\text{arcsinh}(\text{Fem\_Enterprise\_Pct})}$$

**The Interaction Model**:$$\text{Turnout}_{is}=\beta_0+\beta_1\text{EVM\_Exposure}_{is}+\beta_2\text{Agency\_Centered}_{is}+\beta_3(\text{EVM\_Exposure}_{is}\times\text{Agency\_Centered}_{is})+\gamma\mathbf{X}_{is}+\alpha_s+\epsilon_{is}$$Marginal 

**Effect of EVM Exposure**:$$\frac{\partial(\text{Turnout})}{\partial(\text{EVM\_Exposure})}=\beta_1+\beta_3\times\text{Agency\_Centered}$$

### Summary Outcome (Data)
1. **Numbers**: N=426 districts. Interaction Term (EVM × Agency): $\beta$=-6.875 (SE=3.13, p=0.028). Marginal Effect at 90th Percentile Agency: -8.47 pp (p=0.004). Marginal Effect at 10th Percentile: +3.46 pp (p=0.388).
2. **What it Proves**: Heterogeneity. The negative effect is not uniform; it is concentrated almost entirely in high-agency districts. In low-agency districts, EVMs had no detectable negative impact, and potentially a slight (though statistically insignificant) positive trend.
3. **What Still Needs Proving**: We need to ensure this interaction wasn't present before EVMs were introduced (Placebo test).

### Detailed Sequential Analysis

**Step 1: Interaction Model (`Step3_Model_Interaction.csv`)**
- **Data**: 
   - Main Effect (EVM): $\beta$= -2.66 (p=0.23, insignificant).
   - Interaction (EVM × Female_Agency): $\beta$=-6.875 (p=0.028).
   - Adjusted R²=0.71.
- **Inference**: The main effect disappears; the action is entirely in the interaction. Higher economic agency predicts a significantly stronger negative response to EVMs.

**Step 2: Marginal Effects Calculation (`Step3_Marginal_Effects.csv`)**
- **Data**: 
  - At 10th %ile Agency: Effect = +3.46 pp (CI: [-4.39, 11.31], p=0.388).
  - At 50th %ile Agency: Effect = -2.45 pp (CI: [-6.85, 1.94], p=0.274).
  - At 90th %ile Agency: Effect = -8.47 pp (CI: [-14.26, -2.69], p=0.004).
- **Inference**: A clear, statistically significant gradient exists. The negative effect of EVMs scales directly with female economic agency, only becoming statistically significant in districts above the median (75th percentile and above).

**Visualizations**
- `Step3_Marginal_Effects_Plot.jpg`: This line graph depicts a downward-sloping marginal effect curve, demonstrating that the negative impact of EVM introduction on female voter turnout becomes progressively more severe as a district's baseline female enterprise density increases.
- `Step3_Subsample_Heterogeneity.png:`: This coefficient plot reveals that EVM exposure significantly decreased female turnout in districts with high economic agency, whereas the effect in low agency districts was slightly positive but statistically indistinguishable from zero.
- `Step3_Discrete Marginal Effect.png`: This forest plot illustrates that the negative impact of EVM introduction on female voter turnout only becomes statistically significant at higher levels of female enterprise density (the 75th and 90th percentiles), while effects at the median and below remain indistinguishable from zero.
