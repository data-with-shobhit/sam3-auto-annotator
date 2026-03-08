"""
Summary Page
"""
import streamlit as st
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.logging_config import setup_logging

log = setup_logging("summary_page")

def render_summary_page():
    st.header("5. Summary")
    
    if st.session_state.project_manager is None:
        st.warning("No project loaded")
        return
    
    pm = st.session_state.project_manager
    dataset_dir = pm.dataset_dir
    
    # Check if dataset exists
    if not (dataset_dir / "data.yaml").exists():
        st.warning("Dataset not generated yet. Please complete the Organize step.")
        return
    
    st.success(f"Dataset ready at: `{dataset_dir}`")
    
    # Statistics
    st.subheader("Dataset Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    # Count files in each split
    for split, col in [("train", col1), ("valid", col2), ("test", col3)]:
        images_dir = dataset_dir / split / "images"
        count = len(list(images_dir.glob("*"))) if images_dir.exists() else 0
        col.metric(f"{split.capitalize()} Images", count)
    
    # Display data.yaml
    st.subheader("data.yaml")
    yaml_path = dataset_dir / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            st.code(f.read(), language="yaml")
    
    # Download button
    if yaml_path.exists():
        with open(yaml_path, "rb") as f:
            st.download_button(
                "Download data.yaml",
                f.read(),
                "data.yaml",
                "application/x-yaml"
            )
    
    # Training command
    st.subheader("Training Command")
    st.code(f"yolo train data={dataset_dir}/data.yaml model=yolo11n.pt epochs=100", language="bash")
    
    log.info("Displayed summary page")
