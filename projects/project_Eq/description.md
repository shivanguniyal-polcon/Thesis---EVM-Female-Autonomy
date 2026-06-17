# Econometric Specifications

## Step 1: Raw EVM Correlation (1999)
This model tests the unconditional, baseline relationship between the introduction of Electronic Voting Machines (EVMs) and female voter turnout.

$$\text{Turnout}_i=\beta_0+\beta_1\text{EVM}_i+\epsilon_i$$

* $\text{Turnout}_i$: Female voter turnout (%) in district $i$ in 1999.
* $\text{EVM}_i$: A binary indicator (1 if EVM used, 0 if paper ballots).
* $\epsilon_i$: Error term.

---

## Step 2: Spatial Projection & Demographic Controls
This model isolates the EVM effect by controlling for baseline demographics and unobserved state-level characteristics. 

$$\text{Turnout}_{is}=\beta_0+\beta_1\text{EVM\_Exposure}_{is}+\gamma\mathbf{X}_{is}+\alpha_s+\epsilon_{is}$$

* $\text{EVM\_Exposure}_{is}$: The continuous proportion of female electors exposed to EVMs in district $i$ in state $s$.
* $\mathbf{X}_{is}$: A vector of 1991 demographic controls (Literacy %, SC %, ST %, Urbanization %).
* $\alpha_s$: State fixed effects to control for time-invariant cultural/administrative differences across states.

---

## Step 3: Heterogeneous Effects & Economic Agency
This phase tests whether the impact of EVMs is conditional upon a district's baseline female economic independence.

**Data Transformation (Inverse Hyperbolic Sine & Mean-Centering):**
$$\text{Agency\_Centered}_i=\text{arcsinh}(\text{Fem\_Enterprise\_Pct}_i)-\overline{\text{arcsinh}(\text{Fem\_Enterprise\_Pct})}$$

**The Interaction Model:**
$$\text{Turnout}_{is}=\beta_0+\beta_1\text{EVM\_Exposure}_{is}+\beta_2\text{Agency\_Centered}_{is}+\beta_3(\text{EVM\_Exposure}_{is}\times\text{Agency\_Centered}_{is})+\gamma\mathbf{X}_{is}+\alpha_s+\epsilon_{is}$$

**Marginal Effect of EVM Exposure:**
The actual effect of EVMs at any specific level of economic agency is calculated as the first derivative with respect to EVM Exposure:
$$\frac{\partial(\text{Turnout})}{\partial(\text{EVM\_Exposure})}=\beta_1+\beta_3\times\text{Agency\_Centered}$$

---

## Step 4: Difference-in-Differences (DiD) & Pre-Trend Validation
This phase validates causality longitudinally across the 1996, 1998, and 1999 election cycles using the continuous EVM exposure ratio.

**1. Placebo Pre-Trend Test (1996 to 1998):**
Tests if differential trends existed *before* the 1999 rollout (the interaction term $\beta_3$ must equal zero for causality to hold).
$$\Delta\text{Turnout}_{is}^{1996\rightarrow1998}=\beta_0+\beta_1\text{EVM\_Exposure}_{is}+\beta_2\text{Agency}_{is}+\beta_3(\text{EVM\_Exposure}_{is}\times\text{Agency}_{is})+\gamma\mathbf{X}_{is}+\alpha_s+\epsilon_{is}$$

**2. Main Difference-in-Differences (1996 to 1999):**
$$\Delta\text{Turnout}_{is}^{1996\rightarrow1999}=\beta_0+\beta_1\text{EVM\_Exposure}_{is}+\beta_2\text{Agency}_{is}+\beta_3(\text{EVM\_Exposure}_{is}\times\text{Agency}_{is})+\gamma\mathbf{X}_{is}+\alpha_s+\epsilon_{is}$$

**3. Cross-Sectional ANCOVA (Robustness):**
Often statistically more powerful than DiD, this models 1999 Turnout while strictly controlling for the 1996 baseline turnout ($\delta$).
$$\text{Turnout}_{is}^{1999}=\beta_0+\beta_1\text{EVM\_Exposure}_{is}+\beta_2\text{Agency}_{is}+\beta_3(\text{EVM\_Exposure}_{is}\times\text{Agency}_{is})+\delta\text{Turnout}_{is}^{1996}+\gamma\mathbf{X}_{is}+\alpha_s+\epsilon_{is}$$

---

## Step 5: Mechanism Horse Race (Z-Scored)
This final model pits Economic Agency against Cultural Patriarchy (Child Sex Ratio) to determine the true underlying mechanism, using Z-scores for direct magnitude comparison.

**Z-Score Standardization:**
$$Z_i=\frac{X_i-\mu}{\sigma}$$

**The Dual Interaction Model:**
$$\text{Turnout}_{is}^{1999}=\beta_0+\beta_1\text{EVM\_Exposure}_{is}+\beta_2\text{Agency\_Z}_{is}+\beta_3\text{Patriarchy\_Z}_{is}+\beta_4(\text{EVM\_Exposure}_{is}\times\text{Agency\_Z}_{is})+\beta_5(\text{EVM\_Exposure}_{is}\times\text{Patriarchy\_Z}_{is})+\delta\text{Turnout}_{is}^{1996}+\gamma\mathbf{X}_{is}+\alpha_s+\epsilon_{is}$$

* $\beta_4$: Effect of a 1 Standard Deviation increase in Economic Agency on the EVM Penalty.
* $\beta_5$: Effect of a 1 Standard Deviation increase in Cultural Patriarchy on the EVM Penalty.
