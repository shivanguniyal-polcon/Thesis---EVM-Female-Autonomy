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
    
    for line in lines:
        if START_MARKER in line:
            in_core_block = True
            has_markers = True
            continue # Skip the marker comment itself
        elif END_MARKER in line:
            in_core_block = False
            continue # Skip the marker comment itself
            
        if in_core_block:
            core_lines.append(line)
            
    if has_markers and core_lines:
        # Return the extracted core code and a flag that it is partial
        return "".join(core_lines), True
    else:
        # Fallback: Return full code if no markers were found
        return "".join(lines), False

def natural_sort_key(s):
    """Helper to sort strings with numbers naturally (e.g., Step1, Step2, Step10)."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

# --- 1. Page Configuration (Aesthetic Base) ---
st.set_page_config(
    page_title="Project Dashboard",
    page_icon="📊",
    layout="wide",              # Uses the full screen width
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
st.header(f"📁 {selected_project.replace('_', ' ').title()}")

description_file = os.path.join(project_path, "description.md")
if os.path.exists(description_file):
    with open(description_file, "r", encoding="utf-8") as f:
        description = f.read()
    # st.container(border=True) creates a nice, clean card-like aesthetic
    with st.container(border=True):
        st.markdown(description)
else:
    st.info("No `description.md` found for this project.")

st.markdown("---")

# --- 6. Main Content Tabs (Aesthetic & Organization) ---
tab_data, tab_plots, tab_code = st.tabs(["📊 Datasets (CSV)", "🖼️ Visualizations (PNG)", "💻 Source Code"])

# --- TAB 1: DATA ---
with tab_data:
    csv_files = glob.glob(os.path.join(project_path, "*.csv"))
    # Sort naturally so Step1 comes before Step10
    csv_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    if csv_files:
        selected_csv = st.selectbox("Select a dataset to preview:", csv_files, key="csv_select")
        try:
            df = pd.read_csv(selected_csv)
            # Interactive Dataframe (allows sorting/searching without server-side recalculation)
            st.dataframe(df, use_container_width=True, height=400)
            
            # Download Button
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
        # Dynamic columns: max 3 per row
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
            # 1. Extract the code using our helper function
            display_code, is_core_only = extract_core_code(py)
            
            # 2. Dynamic UI Labeling based on what was extracted
            if is_core_only:
                label = f"✨ {os.path.basename(py)} (Core Logic Only)"
            else:
                label = f"📄 {os.path.basename(py)} (Full Script)"
                
            with st.expander(label, expanded=False):
                # 3. Add a helpful note if we are showing the full script because markers were missing
                if not is_core_only:
                    st.caption("ℹ️ No core markers found. Displaying full script.")
                
                # 4. Display the extracted code
                st.code(display_code, language='python')
                
                # 5. Download Button (Always downloads the FULL original file so it runs locally)
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
