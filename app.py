import streamlit as st
import pandas as pd
import os
import glob
import re
import json

import re

def inject_glowing_numbers(text):
    """Wraps numbers in HTML spans, but safely ignores Markdown code and math blocks."""
    if not text: 
        return text
        
    pattern = r'(?<![\w/])(?:(-?\d+\.\d+%?)|(-?\d+%?)(?!\.\s|\)\s))(?!\w)'
    
    def replacer(m):
        num = m.group(1) if m.group(1) is not None else m.group(2)
        return f'<span class="glow-number">{num}</span>'
        
    # 1. Protect inline code (backticks `)
    # Splitting by ` creates a list where even indices are outside code, odd are inside.
    backtick_parts = text.split('`')
    for i in range(0, len(backtick_parts), 2):
        backtick_parts[i] = re.sub(pattern, replacer, backtick_parts[i])
    text = '`'.join(backtick_parts)
    
    # 2. Protect math blocks ($$...$$ and $...$)
    # Split by $$ for block math first
    math_block_parts = text.split('$$')
    for i in range(0, len(math_block_parts), 2):
        # Then protect inline math ($) within the non-block parts
        inline_math_parts = math_block_parts[i].split('$')
        for j in range(0, len(inline_math_parts), 2):
            inline_math_parts[j] = re.sub(pattern, replacer, inline_math_parts[j])
        math_block_parts[i] = '$'.join(inline_math_parts)
        
    return '$$'.join(math_block_parts)

# --- DISABLE COPYING & TEXT SELECTION ---
st.markdown("""
<style>
    /* 1. Disable text selection and copying globally */
    * {
        -webkit-user-select: none !important; /* Safari/Chrome */
        -moz-user-select: none !important;    /* Firefox */
        -ms-user-select: none !important;     /* IE/Edge */
        user-select: none !important;         /* Standard */
    }

    /* 2. RE-ENABLE copying for Code Blocks and Dataframes (Highly Recommended) */
    /* Delete this section if you want to block copying literally everywhere */
    pre, code, [data-testid="stDataFrame"] {
        -webkit-user-select: text !important;
        -moz-user-select: text !important;
        -ms-user-select: text !important;
        user-select: text !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Function for Extracting Core Code ---
def extract_core_code(file_path):
    """
    Extracts code between '# [CORE START]' and '# [CORE END]' markers.
    Filters out streamlit commands and plt.show() to prevent layout breaking.
    """
    START_MARKER = "# [CORE START]"
    END_MARKER = "# [CORE END]"
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        return f"# Error reading file: {e}", False
        
    core_lines = []
    in_core_block = False
    has_markers = False
    
    # Regex to identify streamlit commands or plot shows to hide
    skip_patterns = [
        r'^\s*st\.',          # Lines starting with st. (streamlit)
        r'^\s*plt\.show\(\)', # Lines with plt.show()
    ]
    
    for line in lines:
        # Check markers first
        if START_MARKER in line:
            in_core_block = True
            has_markers = True
            continue
        elif END_MARKER in line:
            in_core_block = False
            continue
            
        if in_core_block:
            # Filter out UI-breaking lines if we are in core block
            should_skip = False
            for pattern in skip_patterns:
                if re.search(pattern, line):
                    should_skip = True
                    break
            
            if not should_skip:
                core_lines.append(line)
            
    if has_markers and core_lines:
        return "".join(core_lines), True
    else:
        # Fallback: Return full code (but still filter UI commands for safety in dashboard)
        clean_lines = []
        for line in lines:
            should_skip = False
            for pattern in skip_patterns:
                if re.search(pattern, line):
                    should_skip = True
                    break
            if not should_skip:
                clean_lines.append(line)
        return "".join(clean_lines), False

def natural_sort_key(s):
    """
    Helper to sort strings with numbers and letters naturally.
    Ensures project_A, project_B, project_C come before project_1, project_2...
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def parse_description_content(content):
    """
    Parses the description.md content into three parts:
    1. Process Summary (Text before 'Summary of Outcome')
    2. Outcome Summary (The structured Numbers/Proves/Needs Proving section)
    3. Detailed Analysis (Everything from 'Detailed Sequential Analysis' onwards)
    """
    lines = content.split('\n')
    
    process_summary_lines = []
    outcome_summary_lines = []
    detailed_analysis_lines = []
    
    state = 'process' # 'process', 'outcome', 'detailed'
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped_line = line.strip()
        
        # Detect Outcome Section Start
        if "Summary of Outcome" in stripped_line or stripped_line.startswith("### Numbers"):
            state = 'outcome'
            outcome_summary_lines.append(line)
            i += 1
            continue
            
        # Detect Detailed Analysis Section Start
        if "Detailed Sequential Analysis" in stripped_line or stripped_line.startswith("## Detailed"):
            state = 'detailed'
            detailed_analysis_lines.append(line)
            i += 1
            continue
            
        if state == 'process':
            process_summary_lines.append(line)
        elif state == 'outcome':
            outcome_summary_lines.append(line)
        elif state == 'detailed':
            detailed_analysis_lines.append(line)
            
        i += 1

    process_md = "\n".join(process_summary_lines).strip()
    outcome_md = "\n".join(outcome_summary_lines).strip()
    detailed_md = "\n".join(detailed_analysis_lines).strip()
    
    return process_md, outcome_md, detailed_md

def render_detailed_analysis(md_content):
    """
    Renders the detailed analysis markdown, ensuring Steps become subheaders.
    """
    if not md_content:
        st.info("No detailed analysis available.")
        return

    # Apply the Python injection function here as well!
    st.markdown(inject_glowing_numbers(md_content), unsafe_allow_html=True)


# --- 1. Page Configuration (MUST come first) ---
st.set_page_config(
    page_title="Project Dashboard",
    page_icon="👩🏻‍💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS FOR HIGHLIGHTED NUMBERS (COLOR ONLY) ---
st.markdown("""
<style>
    /* Simple color change for numbers - updated for modern Streamlit data-testid */
    [data-testid="stMarkdown"] .glow-number, 
    [data-testid="stExpander"] .glow-number,
    .stMarkdown .glow-number, 
    .stExpander .glow-number {
        color: #00FF41 !important; /* Bright neon green */
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Header ---
st.title("Female Agency and Voting Technology​")
st.markdown("A Case for Economic Agency moderating the effect of EVMs on Female Voter Turnout during 1999 Lok Sabha Elections in India​")

# --- 3. Directory Setup ---
BASE_DIR = "./projects"

# Custom display names for projects (Folder Name -> Display Title)
PROJECT_DISPLAY_NAMES = {
    "project_A": "Script A: Master District Mapping (1991)",
    "project_B": "Script B: Pristine Census Map Creation",
    "project_C": "Script C: PC2004-District1991 Weightage Crosswalk",
    "project_1": "Script 1: Raw EVM Correlation (1999)​",
    "project_2": "Script 2: Spatial Projection & Demographic Controls (1999 Lok Sabha)​",
    "project_3": "Script 3: Heterogeneous Effects & Economic Agency​",
    "project_4": "Script 4: Difference-in-Differences & Pre-Trend Validation​",
    "project_5": "Script 5: Causal Validation & Mechanism Horse Race​",
    "project_Final": "Ultimate Findings: The Dual Burden of Electoral Technology",
}

if not os.path.exists(BASE_DIR):
    st.warning(f"Please create a folder named `{BASE_DIR}` and add your project subfolders to it.")
    st.stop()

# Get all project folders and sort them naturally
all_folders = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]
project_folders = sorted(all_folders, key=natural_sort_key)

# Create a mapping for display: use custom name if available, else fallback to formatted folder name
def get_display_name(folder_name):
    return PROJECT_DISPLAY_NAMES.get(folder_name, folder_name.replace('_', ' ').title())

display_mapping = {get_display_name(f): f for f in project_folders}
display_options = list(display_mapping.keys())

if not project_folders:
    st.info("No projects found. Add a subfolder to the `projects` directory to get started!")
    st.stop()

# --- 4. Sidebar Navigation ---
st.sidebar.header("Navigation")
selected_display = st.sidebar.selectbox("Choose a script to view:", display_options)
selected_project = display_mapping[selected_display]  # Get actual folder name
project_path = os.path.join(BASE_DIR, selected_project)

# --- 5. Project Header & Description Parsing ---
# Display the custom display name as the header
st.header(f"{selected_display}")

description_file = os.path.join(project_path, "description.md")
if os.path.exists(description_file):
    with open(description_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    process_summary, outcome_summary, detailed_analysis = parse_description_content(content)
    
    # 1. Always Visible: Process Summary
    # 1. Always Visible: Process Summary
    if process_summary:
        st.markdown(inject_glowing_numbers(process_summary), unsafe_allow_html=True)
        st.markdown("---") # Separator

# 2. Always Visible: Outcome Summary (Data)
    if outcome_summary:
        st.subheader("Summary of Outcome (Data)")
        st.markdown(inject_glowing_numbers(outcome_summary), unsafe_allow_html=True)
        st.markdown("---") # Separator

    # 3. Collapsible: Detailed Sequential Analysis
    if detailed_analysis:
        with st.expander("Detailed Sequential Analysis (Click to Expand)", expanded=False):
            render_detailed_analysis(detailed_analysis)
else:
    st.info("No `description.md` found for this project.")

# --- 6. Main Content Tabs ---
# Added '🗺️ Maps (GeoJSON)' tab
tab_data, tab_plots, tab_maps, tab_code = st.tabs([
    "Data & Outputs", 
    "Charts & Visuals", 
    "Geographic Maps",
    "Core Code Logic"
])

# --- TAB 1: DATA ---
with tab_data:
    csv_files = glob.glob(os.path.join(project_path, "*.csv"))
    csv_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    if csv_files:
        selected_csv = st.selectbox("Select a dataset to preview:", csv_files, key="csv_select")
        try:
            df = pd.read_csv(selected_csv)
            st.dataframe(df, use_container_width=True, height=400)
            
            with open(selected_csv, "rb") as f:
                st.download_button(
                    label="Download Dataset",
                    data=f,
                    file_name=os.path.basename(selected_csv),
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
    else:
        st.info("No CSV files found in this project.")

# --- TAB 2: PLOTS ---
with tab_plots:
    png_files = glob.glob(os.path.join(project_path, "*.png"))
    png_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    if png_files:
        cols_per_row = 3
        for i in range(0, len(png_files), cols_per_row):
            cols = st.columns(cols_per_row)
            batch = png_files[i:i+cols_per_row]
            
            for j, png in enumerate(batch):
                with cols[j]:
                    st.image(png, caption=os.path.basename(png), use_container_width=True)
                    with open(png, "rb") as f:
                        st.download_button(
                            label="Download Image",
                            data=f,
                            file_name=os.path.basename(png),
                            mime="image/png",
                            key=f"dl_img_{i}_{j}"
                        )
    else:
        st.info("No PNG visualizations found in this project.")

# --- TAB 3: MAPS (GEOJSON) ---
with tab_maps:
    geojson_files = glob.glob(os.path.join(project_path, "*.geojson"))
    # Also check for .json files that might be geojson
    json_files = glob.glob(os.path.join(project_path, "*.json"))
    # Combine and filter unique
    all_geo = list(set(geojson_files + json_files))
    all_geo.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    if all_geo:
        st.info("Select a GeoJSON file to inspect its structure or download it for use in mapping tools.")
        selected_geo = st.selectbox("Select a map file:", all_geo, key="geo_select")
        
        try:
            with open(selected_geo, "r", encoding="utf-8") as f:
                geo_data = json.load(f)
            
            # Display basic stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Feature Type", geo_data.get("type", "Unknown"))
            with col2:
                features = geo_data.get("features", [])
                st.metric("Number of Features", len(features))
            
            # Show a sample of properties if available
            if features and "properties" in features[0]:
                st.write("**Sample Properties (First Feature):**")
                st.json(features[0]["properties"])
            
            # Option to view full raw JSON (collapsible)
            with st.expander("View Full GeoJSON Structure"):
                st.json(geo_data)
            
            # Download Button
            with open(selected_geo, "rb") as f:
                st.download_button(
                    label="Download GeoJSON",
                    data=f,
                    file_name=os.path.basename(selected_geo),
                    mime="application/geo+json"
                )
                
        except json.JSONDecodeError:
            st.error("Error: This file is not valid JSON/GeoJSON.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("No GeoJSON files found in this project.")

# --- TAB 4: CODE ---
with tab_code:
    py_files = glob.glob(os.path.join(project_path, "*.py"))
    py_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    if py_files:
        for py in py_files:
            display_code, is_core_only = extract_core_code(py)
            
            if is_core_only:
                label = f"{os.path.basename(py)} (Core Logic Only)"
            else:
                label = f"{os.path.basename(py)} (Full Script)"
                
            with st.expander(label, expanded=False):
                if not is_core_only:
                    st.caption("ℹ️ No core markers found. Displaying full script (filtered).")
                
                st.code(display_code, language='python')
                
                with open(py, "rb") as f:
                    st.download_button(
                        label="Download Full Script",
                        data=f,
                        file_name=os.path.basename(py),
                        mime='text/plain',
                        key=f"dl_code_{py}"
                    )
    else:
        st.info("No Python source code (`.py`) found in this project.")
