"""
Project Sam - Modular Video Annotation Pipeline
Main Streamlit Application Entry Point
"""
import streamlit as st
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # Load .env file (HF token, etc.)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration
from config.settings import PROJECTS_DIR, DEFAULT_CONFIDENCE
from config.logging_config import setup_logging

# Import managers
from managers.project_manager import ProjectManager
from managers.video_manager import VideoManager

# Setup logging
log = setup_logging("app")

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="SAM3 Video Annotation Pipeline",
    layout="wide"
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'page' not in st.session_state:
    st.session_state.page = 'upload'
if 'project_manager' not in st.session_state:
    st.session_state.project_manager = None
if 'extracted_frames' not in st.session_state:
    st.session_state.extracted_frames = []
if 'text_prompts' not in st.session_state:
    st.session_state.text_prompts = []  # List of {prompt_text, class_name, class_id}
if 'annotations' not in st.session_state:
    st.session_state.annotations = {}
if 'video_thresholds' not in st.session_state:
    st.session_state.video_thresholds = {}  # {video_name: {class_name: threshold}}
if 'selected_images' not in st.session_state:
    st.session_state.selected_images = set()
if 'sam3_annotator' not in st.session_state:
    st.session_state.sam3_annotator = None
if 'sample_results' not in st.session_state:
    st.session_state.sample_results = {}
if 'dataset_ready' not in st.session_state:
    st.session_state.dataset_ready = False
if 'trained_model_path' not in st.session_state:
    st.session_state.trained_model_path = None

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
from ui.components.sidebar import render_sidebar
render_sidebar()

# ============================================================================
# MAIN CONTENT HEADER
# ============================================================================
st.title("SAM3 Video Annotation Pipeline")

# ============================================================================
# PAGE ROUTING
# ============================================================================
if st.session_state.page == 'upload':
    from ui.pages.upload_page import render_upload_page
    render_upload_page()

elif st.session_state.page == 'prompts':
    from ui.pages.prompts_page import render_prompts_page
    render_prompts_page()

elif st.session_state.page == 'sample_test':
    from ui.pages.sample_test_page import render_sample_test_page
    render_sample_test_page()

elif st.session_state.page == 'annotate':
    from ui.pages.annotate_page import render_annotate_page
    render_annotate_page()

elif st.session_state.page == 'organize':
    from ui.pages.organize_page import render_organize_page
    render_organize_page()

elif st.session_state.page == 'summary':
    from ui.pages.summary_page import render_summary_page
    render_summary_page()

elif st.session_state.page == 'train':
    from ui.pages.train_page import render_train_page
    render_train_page()
