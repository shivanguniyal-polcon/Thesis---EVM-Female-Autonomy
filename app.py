import streamlit as st
import pandas as pd
import os
import glob
import re
import datetime

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
    Parses the description.md content into three parts.
    """
    lines = content.split('\n')
    
    process_summary_lines = []
    outcome_summary_lines = []
    detailed_analysis_lines = []
    
    state = 'process' 
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if "Summary of Outcome" in line or "Numbers:" in line or line.startswith("### Numbers"):
            state = 'outcome'
            outcome_summary_lines.append(lines[i])
            i += 1
            continue
            
        if "Detailed Sequential Analysis" in line or line.startswith("## Detailed"):
            state = 'detailed'
            detailed_analysis_lines.append(lines[i])
            i += 1
            continue
            
        if state == 'process':
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
    if not md_content:
        st.info("No detailed analysis available.")
        return
    st.markdown(md_content, unsafe_allow_html=True)

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Project Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State for Feedback
if 'feedback_log' not in st.session_state:
    st.session_state.feedback_log = []

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
    
    if process_summary:
        st.markdown(process_summary)
        st.markdown("---")
    
    if outcome_summary:
        st.subheader("📊 Summary of Outcome (Data)")
        st.markdown(outcome_summary)
        st.markdown("---")

    if detailed_analysis:
        with st.expander("🔍 Detailed Sequential Analysis (Click to Expand)", expanded=False):
            render_detailed_analysis(detailed_analysis)
else:
    st.info("No `description.md` found for this project.")

# --- 6. Main Content Tabs (Including Feedback) ---
tab_data, tab_plots, tab_code, tab_feedback = st.tabs([
    "📊 Datasets (CSV)", 
    "🖼️ Visualizations (PNG)", 
    "💻 Source Code",
    "💬 Guide Feedback & Notes"
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

# --- TAB 4: FEEDBACK & NOTES ---
with tab_feedback:
    st.subheader("📝 Guide's Query & Feedback Log")
    st.markdown(f"**Current Project:** `{selected_project}`")
    st.info("Use this section to record observations, queries, or required changes. Data is stored temporarily for this session. Download the CSV to save your notes permanently.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        feedback_category = st.selectbox(
            "Category",
            ["Data Quality", "Methodology", "Visualization", "Code Logic", "General Query", "Next Steps"]
        )
        feedback_text = st.text_area(
            "Enter your feedback or query here:",
            height=150,
            placeholder="e.g., The coefficient in Step 3 seems unstable. Check robustness..."
        )
    
    with col2:
        st.write("### Actions")
        if st.button("➕ Add to Log", type="primary", use_container_width=True):
            if feedback_text:
                new_entry = {
                    "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Project": selected_project,
                    "Category": feedback_category,
                    "Feedback": feedback_text
                }
                st.session_state.feedback_log.append(new_entry)
                st.success("Added to session log!")
                st.rerun()
            else:
                st.warning("Please enter some text before adding.")
        
        if st.session_state.feedback_log:
            if st.button("🗑️ Clear Log", use_container_width=True):
                st.session_state.feedback_log = []
                st.rerun()

    # Display Current Session Log
    if st.session_state.feedback_log:
        st.markdown("---")
        st.subheader("📋 Current Session Log")
        
        # Filter log to show current project first, but show all for context if needed
        df_log = pd.DataFrame(st.session_state.feedback_log)
        
        # Display table
        st.dataframe(df_log, use_container_width=True, hide_index=True)
        
        # Download Button for the log
        csv_log = df_log.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Feedback Log (CSV)",
            data=csv_log,
            file_name=f"feedback_log_{selected_project}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="download_feedback"
        )
    else:
        st.info("No feedback entries added yet for this session.")
