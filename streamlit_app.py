import streamlit as st

st.set_page_config(page_title="Spatial Engineering & Causal Inference", layout="wide")

st.title("🗳️ EVMs and Electoral Outcomes in Bihar (1996-2009)")
st.markdown("### A Spatial Difference-in-Differences Analysis of Voting Technology Impact")

st.markdown("""
### 📖 Abstract

This thesis examines the causal impact of Electronic Voting Machines (EVMs) on electoral outcomes in Bihar, India, 
using a quasi-experimental Difference-in-Differences (DiD) design across five assembly elections (1996-2009).

By leveraging the staggered rollout of EVMs and constructing a novel spatial dataset that solves the 
Modifiable Areal Unit Problem (MAUP), this research identifies how voting technology affects:
- **Turnout rates** (overall and by gender)
- **Electoral competition** (vote shares, margins)
- **Incumbency advantages**

The analysis employs advanced econometric techniques including:
- Parallel trends validation
- Propensity Score Matching (PSM)
- Triple Difference (DDD) models testing Female Economic Agency moderation
- Urbanization-controlled robustness checks
""")

st.markdown("---")

st.sidebar.markdown("""
### 📚 Thesis Modules

**Module 1: Spatial Data Construction**
Solving the MAUP via cookie-cutter geospatial intersections.

**Module 2: Baseline Identification**
Proving parallel trends and running the surgically cleaned DiD.

**Module 3: Core Causal Estimates**
Testing placebo mechanisms and continuous treatment exposures.

**Module 4: Theoretical Contribution**
DDD interactions testing the moderation of Female Economic Agency.

**Module 5: Master Synthesis**
The fully robust, urbanization-controlled master model.
""")

st.markdown("""
---

### 🗂️ Navigation

Use the sidebar to navigate through the 5 thematic modules of this thesis.
Each module contains:
- **📊 Results**: Interactive visualizations and econometric output
- **💻 Core Code**: Full transparency of the analytical pipeline

### 📋 Key Findings Preview

| Module | Key Question | Method |
|--------|--------------|--------|
| 1 | How do we compare PCs across changing district boundaries? | Spatial Cookie-Cutter + ID Crosswalk |
| 2 | Did EVM and paper-ballot PCs follow similar pre-trends? | Parallel Trends Test (1996 vs 1998) |
| 3 | What is the causal effect of EVMs on turnout? | Cleaned DiD Regression |
| 4 | Do effects differ by gender? Is there a placebo? | Dynamic DiD (Male vs Female) |
| 5 | Does female economic agency moderate EVM effects? | Triple Difference (DDD) Model |

---

### 🛠️ Technical Stack

- **Spatial Analysis**: GeoPandas, Shapely, Cookie-cutter method for MAUP resolution
- **Econometrics**: Statsmodels, Linearmodels (PanelOLS), PSM
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Framework**: Streamlit for interactive academic presentation

---

*This thesis demonstrates both substantive contributions to political economy 
and methodological rigor in spatial econometrics.*
""")
