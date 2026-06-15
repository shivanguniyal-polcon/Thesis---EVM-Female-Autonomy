import streamlit as st
import pandas as pd
import os
import glob
import re

# --- Helper Function for Extracting Core Code ---
def extract_core_code(file_path):
    """
    Extracts code between '# [CORE START]' and '# [CORE END]' markers.
    If markers are not found, it falls back to returning the entire file.
    Also filters out streamlit commands and plt.show() to prevent layout breaking.
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
    """Helper to sort strings with numbers naturally (e.g., Step1, Step2, Step10)."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def parse_description_content(content):
    """
    Parses the description.md content to separate:
    1. The Overview (first non-empty line, usually bold summary)
    2. The Summary Outcome (Numbers, Proves, Needs Proving)
    3. The Detailed Analysis (everything else)
    
    Assumes a specific structure in the markdown:
    - Line 1: Overview
    - Blank line
    - "### Summary Outcome (Data)" header
    - Content for Numbers, Proves, Needs Proving
    - Blank line
    - "### Detailed Sequential Analysis" header (optional, rest is detailed)
    """
    lines = content.split('\n')
    
    overview = ""
    summary_outcome = ""
    detailed_analysis = ""
    
    # State machine to parse sections
    state = "overview" # overview, summary, detailed
    current_section_lines = []
    
    # Skip empty lines at the very start
    while lines and not lines[0].strip():
        lines.pop(0)
        
    if not lines:
        return "", "", content

    # The first line is the Overview
    overview = lines[0].strip()
    lines = lines[1:]
    
    # Skip empty lines after overview
    while lines and not lines[0].strip():
        lines.pop(0)
        
    # Now look for sections
    in_summary = False
    in_detailed = False
    
    summary_lines = []
    detailed_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        if stripped.startswith("### Summary Outcome (Data)"):
            in_summary = True
            in_detailed = False
            continue
        elif stripped.startswith("### Detailed Sequential Analysis"):
            in_summary = False
            in_detailed = True
            continue
            
        if in_summary:
            summary_lines.append(line)
        elif in_detailed:
            detailed_lines.append(line)
        else:
            # If no headers found, assume everything remaining is detailed
            detailed_lines.append(line)
            
    summary_outcome = "\n".join(summary_lines)
    detailed_analysis = "\n".join(detailed_lines)
    
    return overview, summary_outcome, detailed_analysis

# --- 1. Page Configuration (Aesthetic Base) ---
st.set_page_config(
    page_title="Project Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Header ---
st.title("📊 Project Showcase Dashboard")
st.markdown("Browse through readymade analyses, datasets, visualizations, and core code logic.")

# --- 3. Directory Setup ---
BASE_DIR = "./projects"

if not os.path.exists(BASE_DIR):
    st.warning(f"Please create a folder named `{BASE_DIR}` and add your project subfolders to it.")
    st.stop()

project_folders = sorted([f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))])

if not project_folders:
    st.info("No projects found. Add a subfolder to the `projects` directory to get started!")
    st.stop()

# --- 4. Sidebar Navigation (Interactivity) ---
st.sidebar.header("📂 Navigation")
selected_project = st.sidebar.selectbox("Choose a project to view:", project_folders)
project_path = os.path.join(BASE_DIR, selected_project)

# --- 5. Project Header & Description ---
# Display a clean, standardized header based on the folder name
st.header(f"📁 {selected_project.replace('_', ' ').title()}")

description_file = os.path.join(project_path, "description.md")
if os.path.exists(description_file):
    with open(description_file, "r", encoding="utf-8") as f:
        raw_content = f.read()
    
    # Parse the content into three parts
    overview, summary_outcome, detailed_analysis = parse_description_content(raw_content)
    
    # 1. Always Visible: Overview
    if overview:
        st.markdown(f"**{overview}**")
        st.markdown("---") # Separator after overview

    # 2. Always Visible: Summary Outcome (Data)
    if summary_outcome:
        st.subheader("Summary Outcome (Data)")
        st.markdown(summary_outcome)
        st.markdown("---") # Separator before collapsible section

    # 3. Collapsible: Detailed Sequential Analysis
    if detailed_analysis:
        with st.expander("🔍 Detailed Sequential Analysis (Click to Expand)", expanded=False):
            st.markdown(detailed_analysis)
else:
    st.info("No `description.md` found for this project.")

# --- 6. Main Content Tabs (Aesthetic & Organization) ---
tab_data, tab_plots, tab_code = st.tabs(["📊 Datasets (CSV)", "🖼️ Visualizations (PNG)", "💻 Source Code"])

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
                    label="📥 Download Dataset",
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
                            label="📥 Download Image",
                            data=f,
                            file_name=os.path.basename(png),
                            mime="image/png",
                            key=f"dl_img_{i}_{j}"
                        )
    else:
        st.info("No PNG visualizations found in this project.")

# --- TAB 3: CODE ---
with tab_code:
    py_files = glob.glob(os.path.join(project_path, "*.py"))
    py_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    if py_files:
        for py in py_files:
            display_code, is_core_only = extract_core_code(py)
            
            if is_core_only:
                label = f"✨ {os.path.basename(py)} (Core Logic Only)"
            else:
                label = f"📄 {os.path.basename(py)} (Full Script)"
                
            with st.expander(label, expanded=False):
                if not is_core_only:
                    st.caption("ℹ️ No core markers found. Displaying full script.")
                
                st.code(display_code, language='python')
                
                with open(py, "rb") as f:
                    st.download_button(
                        label="📥 Download Full Script",
                        data=f,
                        file_name=os.path.basename(py),
                        mime='text/plain',
                        key=f"dl_code_{py}"
                    )
    else:
        st.info("No Python source code (`.py`) found in this project.")
