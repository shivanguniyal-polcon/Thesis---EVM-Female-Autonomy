import streamlit as st
import pandas as pd
import os
import glob
import re

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
    """Helper to sort strings with numbers naturally."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def parse_description_content(content):
    """
    Parses the description.md content into three parts:
    1. Process Summary (First paragraph before 'Outcome' or 'Numbers')
    2. Outcome Summary (The structured Numbers/Proves/Needs Proving section)
    3. Detailed Analysis (The rest, intended for the collapsible section)
    """
    lines = content.split('\n')
    
    process_summary_lines = []
    outcome_summary_lines = []
    detailed_analysis_lines = []
    
    state = 'process' # 'process', 'outcome', 'detailed'
    
    # Heuristics to detect sections
    # We assume the file structure is:
    # [Process Summary Paragraphs]
    # [Outcome Summary Header & Content]
    # [Detailed Sequential Analysis Header & Content]
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Detect Outcome Section Start
        if "Summary of Outcome" in line or "Numbers:" in line or line.startswith("### Numbers"):
            state = 'outcome'
            # Don't skip the line, add it to outcome
            outcome_summary_lines.append(lines[i])
            i += 1
            continue
            
        # Detect Detailed Analysis Section Start
        if "Detailed Sequential Analysis" in line or line.startswith("## Detailed"):
            state = 'detailed'
            # Skip the header itself if we want to render it inside the expander, 
            # or keep it. Let's keep it for context inside the expander.
            detailed_analysis_lines.append(lines[i])
            i += 1
            continue
            
        if state == 'process':
            # If we hit an empty line after some text, it might be the end of the summary
            # But we rely on the explicit headers above mostly.
            # If we encounter a header that isn't outcome/detailed, it's likely part of process or detailed
            if line.startswith('#') and "Outcome" not in line and "Detailed" not in line:
                 # If it's a subheader like "### Step 1", it usually belongs to detailed analysis 
                 # UNLESS it's clearly part of the intro. 
                 # Given the prompt, let's assume everything before "Outcome" is process summary.
                 pass 
            process_summary_lines.append(lines[i])
        elif state == 'outcome':
            outcome_summary_lines.append(lines[i])
        elif state == 'detailed':
            detailed_analysis_lines.append(lines[i])
            
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

    lines = md_content.split('\n')
    
    # We will process lines to ensure headers are rendered correctly
    # Streamlit's st.markdown handles #, ##, ### automatically.
    # The issue might be that the whole block was passed as one code block previously.
    # Passing directly to st.markdown should work if the syntax is correct.
    
    # Let's just render it directly. If the MD has '### Step X', st.markdown makes it a subheader.
    st.markdown(md_content, unsafe_allow_html=True)

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
        content = f.read()
    
    process_summary, outcome_summary, detailed_analysis = parse_description_content(content)
    
    # 1. Always Visible: Process Summary
    if process_summary:
        st.markdown(process_summary)
        st.markdown("---") # Separator
    
    # 2. Always Visible: Outcome Summary (Data)
    if outcome_summary:
        st.subheader("📊 Summary of Outcome (Data)")
        st.markdown(outcome_summary)
        st.markdown("---") # Separator

    # 3. Collapsible: Detailed Sequential Analysis
    if detailed_analysis:
        with st.expander("🔍 Detailed Sequential Analysis (Click to Expand)", expanded=False):
            render_detailed_analysis(detailed_analysis)
else:
    st.info("No `description.md` found for this project.")

# --- 6. Main Content Tabs ---
tab_data, tab_plots, tab_code = st.tabs(["📊 Datasets (CSV)", "🖼️ Visualizations (PNG)", "💻 Source Code"])

# --- TAB 1: DATA ---
with tab_:
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
                    st.caption("ℹ️ No core markers found. Displaying full script (filtered).")
                
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
