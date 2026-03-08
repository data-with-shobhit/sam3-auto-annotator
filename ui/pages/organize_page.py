"""
Organize Dataset Page
With multiple export format options
"""
import streamlit as st
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.logging_config import setup_logging
from config.settings import DEFAULT_TRAIN_SPLIT, DEFAULT_VALID_SPLIT

log = setup_logging("organize_page")

# Supported export formats
EXPORT_FORMATS = {
    "YOLOv8": "YOLO v8 format (Ultralytics)",
    "YOLOv9": "YOLO v9 format (Ultralytics)",
    "YOLOv10": "YOLO v10 format (Ultralytics)",
    "YOLOv11": "YOLO v11 format (Ultralytics)",
    "YOLO12": "YOLO 12 format (Ultralytics)",
    "YOLO26": "YOLO 26 format (Ultralytics)",
    "RT-DETR": "RT-DETR format (Real-Time Detection Transformer)"
}

def render_organize_page():
    st.header("5. Organize Dataset")
    
    if not st.session_state.selected_images:
        st.warning("Please select images in the Annotate step first")
        return
    
    pm = st.session_state.project_manager
    
    st.info(f"{len(st.session_state.selected_images)} images selected for export")
    
    # Export format selection
    st.subheader("Export Format")
    export_format = st.selectbox(
        "Select Format",
        list(EXPORT_FORMATS.keys()),
        format_func=lambda x: f"{x} - {EXPORT_FORMATS[x]}"
    )
    
    # Split configuration
    st.subheader("Configure Dataset Split")
    
    col1, col2, col3 = st.columns(3)
    train_pct = col1.slider("Train %", 0, 100, DEFAULT_TRAIN_SPLIT, 5)
    valid_pct = col2.slider("Valid %", 0, 100, DEFAULT_VALID_SPLIT, 5)
    test_pct = 100 - train_pct - valid_pct
    col3.metric("Test %", test_pct)
    
    if test_pct < 0:
        st.error("Train + Valid cannot exceed 100%")
        return
    
    # Calculate counts
    total = len(st.session_state.selected_images)
    train_count = int(total * train_pct / 100)
    valid_count = int(total * valid_pct / 100)
    test_count = total - train_count - valid_count
    
    st.write(f"**Split Preview**: Train={train_count}, Valid={valid_count}, Test={test_count}")
    
    # Generate button
    if st.button(f"Generate {export_format} Dataset", type="primary"):
        from managers.dataset_manager import DatasetManager
        
        progress = st.progress(0)
        status = st.empty()
        
        # Get selected frames with annotations
        status.text("Preparing data...")
        
        selected_data = {}
        for fp in st.session_state.selected_images:
            if fp not in st.session_state.annotations:
                continue
            
            # Get video name for this frame
            frame_name = Path(fp).stem
            video_name = "_".join(frame_name.split("_")[:-2]) or "images"
            
            # Get video name for this frame
            frame_name = Path(fp).stem
            video_name = "_".join(frame_name.split("_")[:-2]) or "images"
            
            # Get thresholds for this video
            thresholds = st.session_state.video_thresholds.get(video_name, {})
            
            # Filter detections by per-video thresholds
            filtered_dets = []
            for det in st.session_state.annotations[fp].get('detections', []):
                class_name = st.session_state.text_prompts[det['class_id']]['class_name']
                threshold = thresholds.get(class_name, 0.3)
                
                if det['score'] >= threshold:
                    filtered_dets.append(det)
            
            selected_data[fp] = filtered_dets
        
        progress.progress(0.2)
        status.text("Splitting dataset...")
        
        # Split
        splits = DatasetManager.split_dataset(list(selected_data.keys()), train_pct, valid_pct)
        
        progress.progress(0.4)
        status.text(f"Generating {export_format} format...")
        
        # Get class names (not prompts!)
        class_names = [p['class_name'] for p in st.session_state.text_prompts]
        
        # Save dataset
        DatasetManager.save_to_dataset(
            str(pm.dataset_dir),
            selected_data,
            splits,
            class_names,
            export_format
        )
        
        progress.progress(1.0)
        status.empty()
        
        st.session_state.dataset_ready = True
        log.info(f"Generated {export_format} dataset with {total} images")
        st.success(f"{export_format} dataset generated successfully!")
        
        # Show dataset info
        st.subheader("Dataset Location")
        st.code(str(pm.dataset_dir))
    
    # Next button if dataset is ready
    if st.session_state.get('dataset_ready', False):
        st.markdown("---")
        col1, col2 = st.columns(2)
        if col1.button("Next → Summary", type="secondary", use_container_width=True):
            st.session_state.page = 'summary'
            st.rerun()
        if col2.button("Next → Train Model", type="primary", use_container_width=True):
            st.session_state.page = 'train'
            st.rerun()
