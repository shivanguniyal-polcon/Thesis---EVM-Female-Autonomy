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

def clean_description_header(content):
    """
    Removes the first H1 header line from the markdown content 
    to prevent duplicate titles in the dashboard.
    """
    lines = content.split('\n')
    if lines and lines[0].strip().startswith('# '):
        return '\n'.join(lines[1:])
    return content

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
        description = f.read()
    
    # Clean the description to remove duplicate H1 headers
    clean_desc = clean_description_header(description)
    
    # Split content into Summary (always visible) and Detailed Analysis (collapsible)
    # We assume the format: **Bold Summary** followed by detailed text.
    # We'll split by the first occurrence of a double newline after the summary paragraph.
    # Or simpler: The user requested "summary should be always visible" and "detailed overview... collapsible".
    # Let's assume the first paragraph (or up to the first blank line) is the summary if it's short, 
    # but the prompt implies the whole top part is the summary.
    # Let's parse: The first block of text before any "###" or specific marker could be summary.
    # However, the new description.md files likely have a specific structure now.
    # Let's assume the first non-header paragraph is the summary.
    
    lines = clean_desc.split('\n')
    summary_lines = []
    detail_lines = []
    in_detail = False
    
    for line in lines:
        if not in_detail:
            # Heuristic: If we hit a header (###) or a specific marker, switch to detail
            # Or if the summary block (usually 1-2 paragraphs) is done.
            # Let's use a explicit split if the user formatted it that way, 
            # otherwise, let's just put the whole thing in the expander if no clear split?
            # No, the prompt says "only the summary should be always visible".
            # Let's assume the first paragraph (empty line separated) is the summary.
            if line.strip() == "" and len(summary_lines) > 0:
                in_detail = True
                continue
            summary_lines.append(line)
        else:
            detail_lines.append(line)
            
    summary_text = "\n".join(summary_lines).strip()
    detail_text = "\n".join(detail_lines).strip()
    
    # Display Summary
    if summary_text:
        st.markdown(summary_text)
    
    # Display Detailed Analysis in Expander
    if detail_text:
        with st.expander("🔍 View Sequential Data Analysis & Step-by-Step Findings"):
            st.markdown(detail_text)

else:
    st.info("No `description.md` found for this project.")

st.markdown("---")

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
