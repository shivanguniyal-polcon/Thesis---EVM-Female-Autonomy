import streamlit as st

# 1. Page Config & Title
st.set_page_config(
    page_title="EVMs & Electoral Outcomes in India (1996-2009)",
    page_icon="🗳️",
    layout="wide"
)

# 2. Main Title & Abstract
st.title("🗳️ EVMs and Electoral Outcomes in India (1996-1999)")
st.markdown("### A Spatial Difference-in-Differences Analysis of Voting Technology Impact")

st.markdown("""
### 📖 Abstract

This thesis examines the causal impact of Electronic Voting Machines (EVMs) on electoral outcomes across **major Indian constituencies**, 
using a quasi-experimental Difference-in-Differences (DiD) design covering the critical transition period (1996-2009).

By leveraging the **staggered national rollout** of EVMs—starting with 45 key parliamentary constituencies in 1999 (including Mumbai, Delhi, Bangalore, Hyderabad, Chennai, Kolkata, and others)—and constructing a novel spatial dataset that solves the **Modifiable Areal Unit Problem (MAUP)**, this research identifies how voting technology affects:
- **Turnout rates** (overall and by gender)
- **Electoral competition** (vote shares, margins)
- **Incumbency advantages**

The analysis employs advanced econometric techniques including:
- **Parallel trends validation** (Pre-treatment checks: 1996 vs 1998)
- **Propensity Score Matching (PSM)** (Addressing selection bias in rollout)
- **Triple Difference (DDD) models** (Testing Female Economic Agency moderation)
- **Urbanization-controlled robustness checks** (Spatial heterogeneity)
""")

st.markdown("---")

# 3. Sidebar Navigation Guide
st.sidebar.markdown("""
### 📚 Thesis Modules

**Module 1: Spatial Data Construction**
Solving the MAUP via cookie-cutter geospatial intersections.
- *Scripts*: Spatial Crosswalk, District ID Mapping

**Module 2: Baseline Identification**
Proving parallel trends and running the surgically cleaned DiD.
- *Scripts*: Parallel Trends Test (1996/1998), Cleaned DiD Regression

**Module 3: Core Causal Estimates**
Testing placebo mechanisms and continuous treatment exposures.
- *Scripts*: Male vs Female Turnout, Binary vs Continuous Exposure

**Module 4: Theoretical Contribution**
DDD interactions testing the moderation of Female Economic Agency.
- *Scripts*: Agency Interaction (DDD), Propensity Score Matching

**Module 5: Master Synthesis**
The fully robust, urbanization-controlled master model.
- *Scripts*: Final Master Model with Urbanization Controls
""")

# 4. Navigation & Findings Preview
st.markdown("""
### 🗂️ Navigation

Use the sidebar to navigate through the **9 scripts** organized into **5 thematic modules**.
Each script page contains two tabs:
- **📊 Results**: Interactive visualizations, maps, and econometric output.
- **💻 Core Code**: Full transparency of the analytical pipeline (original script logic).

### 📋 Key Findings Preview

| Module | Key Question | Method |
|--------|--------------|--------|
| **1** | How do we compare PCs across changing district boundaries? | Spatial Cookie-Cutter + ID Crosswalk |
| **2** | Did EVM and paper-ballot PCs follow similar pre-trends? | Parallel Trends Test (1996 vs 1998) |
| **3** | What is the causal effect of EVMs on turnout? | Cleaned DiD Regression |
| **3** | Do effects differ by gender? Is there a placebo? | Dynamic DiD (Male vs Female) |
| **4** | Does female economic agency moderate EVM effects? | Triple Difference (DDD) Model |
| **5** | Are effects heterogeneous by urbanization? | Master Model with Urban Controls |

---

### 🛠️ Technical Stack

- **Spatial Analysis**: GeoPandas, Shapely, Cookie-cutter method for MAUP resolution
- **Data Engineering**: Pandas, NumPy (Panel construction, ID harmonization)
- **Econometrics**: Statsmodels, Linearmodels (PanelOLS), Scikit-Learn (PSM)
- **Visualization**: Matplotlib, Seaborn, Plotly (Interactive charts)
- **Framework**: Streamlit (Interactive academic presentation)

---

*This thesis demonstrates both substantive contributions to political economy 
and methodological rigor in spatial econometrics, utilizing a comprehensive dataset 
spanning India's most significant urban and rural constituencies.*
""")
