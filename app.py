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
    """Helper to sort strings with numbers naturally (e.g., Step1, Step2, Step10)."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def parse_description(content):
    """
    Parses the description.md content into three parts:
    1. process_summary: The first non-empty paragraph (before the first double newline or '##').
    2. outcome_data: The section between '## Summary Outcome (Data)' and '## Detailed Sequential Analysis'.
    3. detailed_analysis: Everything after '## Detailed Sequential Analysis'.
    """
    lines = content.split('\n')
    
    process_summary = []
    outcome_data = []
    detailed_analysis = []
    
    current_section = 'process'
    
    for line in lines:
        stripped = line.strip()
        
        # Detect section headers
        if stripped.startswith("## Summary Outcome (Data)"):
            current_section = 'outcome'
            continue
        elif stripped.startswith("## Detailed Sequential Analysis"):
            current_section = 'detailed'
            continue
            
        # Assign lines to sections
        if current_section == 'process':
            # Stop collecting process summary if we hit a header or empty line after some text
            if stripped.startswith("# ") and not stripped.startswith("##"):
                continue # Skip main H1 if present
            if stripped == "" and process_summary:
                # If we have content and hit an empty line, check if next is a header
                # For simplicity, we just take the first block of text as process summary
                pass 
            process_summary.append(line)
        elif current_section == 'outcome':
            outcome_data.append(line)
        elif current_section == 'detailed':
            detailed_analysis.append(line)
            
    # Clean up empty lines at start/end of sections
    def clean_section(section_lines):
        text = "\n".join(section_lines)
        return text.strip()

    return clean_section(process_summary), clean_section(outcome_data), clean_section(detailed_analysis)

# --- 1. Page Configuration ---
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

# --- 4. Sidebar Navigation ---
st.sidebar.header("📂 Navigation")
selected_project = st.sidebar.selectbox("Choose a project to view:", project_folders)
project_path = os.path.join(BASE_DIR, selected_project)

# --- 5. Project Header & Description Parsing ---
st.header(f"📁 {selected_project.replace('_', ' ').title()}")

description_file = os.path.join(project_path, "description.md")
if os.path.exists(description_file):
    with open(description_file, "r", encoding="utf-8") as f:
        raw_content = f.read()
    
    proc_sum, out_data, det_anal = parse_description(raw_content)
    
    # 1. Summary of the Process (Always Visible)
    if proc_sum:
        st.markdown(proc_sum)
        st.markdown("---") # Separator
    
    # 2. Summary of Outcome (Data) (Always Visible)
    if out_data:
        st.subheader("📊 Summary Outcome (Data)")
        st.markdown(out_data)
        st.markdown("---") # Separator

    # 3. Detailed Sequential Analysis (Collapsible)
    if det_anal:
        with st.expander("🔍 Detailed Sequential Analysis", expanded=False):
            st.markdown(det_anal)
else:
    st.info("No `description.md` found for this project.")

# --- 6. Main Content Tabs ---
tab_data, tab_plots, tab_code = st.tabs(["📊 Datasets (CSV)", "🖼️ Visualizations (PNG)", "💻 Source Code"])

# --- TAB 1: DATA ---
with tab_
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
                    st.caption("ℹ️ No core markers found. Displaying full script (UI commands filtered).")
                
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
