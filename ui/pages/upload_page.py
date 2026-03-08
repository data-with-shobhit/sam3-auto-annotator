"""
Upload & Extract Page - Videos and Images
"""
import streamlit as st
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from managers.project_manager import ProjectManager
from managers.video_manager import VideoManager
from config.logging_config import setup_logging

log = setup_logging("upload_page")

def render_upload_page():
    st.header("1. Upload Videos/Images & Extract Frames")
    
    # Project Selection
    if st.session_state.project_manager is None:
        st.subheader("Project Selection")
        
        project_mode = st.radio(
            "Select Mode",
            ["Create New Project", "Load Existing Project"],
            horizontal=True
        )
        
        if project_mode == "Create New Project":
            project_name = st.text_input(
                "Enter Project Name",
                placeholder="e.g., construction_site_2024"
            )
            if st.button("Create Project", type="primary"):
                if project_name:
                    pm = ProjectManager()
                    if pm.create(project_name):
                        st.session_state.project_manager = pm
                        log.info(f"Created project: {project_name}")
                        st.rerun()
                else:
                    st.error("Please enter a project name")
        
        else:  # Load Existing
            existing = ProjectManager.list_projects()
            if not existing:
                st.warning("No existing projects found")
            else:
                selected = st.selectbox("Select Project", existing)
                
                col_load, col_delete = st.columns([3, 1])
                
                with col_load:
                    if st.button("Load Project", type="primary", use_container_width=True):
                        pm = ProjectManager()
                        if pm.load(selected):
                            st.session_state.project_manager = pm
                            st.session_state.text_prompts = pm.get_prompts()
                            log.info(f"Loaded project: {selected}")
                            st.rerun()
                
                with col_delete:
                    # Delete with confirmation
                    if f"confirm_delete_{selected}" not in st.session_state:
                        st.session_state[f"confirm_delete_{selected}"] = False
                    
                    if not st.session_state[f"confirm_delete_{selected}"]:
                        if st.button("Delete", type="secondary", use_container_width=True):
                            st.session_state[f"confirm_delete_{selected}"] = True
                            st.rerun()
                    else:
                        st.warning(f"Delete **{selected}**?")
                        c1, c2 = st.columns(2)
                        if c1.button("Yes", key="confirm_yes"):
                            if ProjectManager.delete_project(selected):
                                st.success(f"Deleted project: {selected}")
                                st.session_state[f"confirm_delete_{selected}"] = False
                                st.rerun()
                            else:
                                st.error("Failed to delete project")
                        if c2.button("No", key="confirm_no"):
                            st.session_state[f"confirm_delete_{selected}"] = False
                            st.rerun()
        
        st.stop()
    
    # Show current project
    pm = st.session_state.project_manager
    st.success(f"Project: **{pm.project_name}**")
    
    # =========================================================================
    # FRAME RECOVERY - detect existing frames
    # =========================================================================
    existing_frames = []
    if pm.frames_dir.exists():
        existing_frames = sorted([
            str(f) for f in pm.frames_dir.iterdir() 
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
        ])
    
    if existing_frames and not st.session_state.extracted_frames:
        st.info(f"Found **{len(existing_frames)}** previously extracted frames in this project.")
        if st.button("Recover Previous Frames", type="secondary"):
            st.session_state.extracted_frames = existing_frames
            log.info(f"Recovered {len(existing_frames)} existing frames")
            st.rerun()
        st.markdown("---")
    
    # =========================================================================
    # FILE UPLOAD - Videos and Images
    # =========================================================================
    uploaded_files = st.file_uploader(
        "Upload Videos or Images",
        type=['mp4', 'avi', 'mov', 'mkv', 'jpg', 'jpeg', 'png'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Separate videos and images
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
        image_extensions = {'.jpg', '.jpeg', '.png'}
        
        video_files = [f for f in uploaded_files if Path(f.name).suffix.lower() in video_extensions]
        image_files = [f for f in uploaded_files if Path(f.name).suffix.lower() in image_extensions]
        
        if video_files:
            st.success(f"{len(video_files)} video(s) uploaded")
        if image_files:
            st.success(f"{len(image_files)} image(s) uploaded")
        
        # =====================================================================
        # HANDLE IMAGES - save directly to frames directory
        # =====================================================================
        if image_files:
            if st.button(f"Add {len(image_files)} Image(s) as Frames", type="primary"):
                added_count = 0
                for img_file in image_files:
                    dest_path = pm.frames_dir / img_file.name
                    if not dest_path.exists():
                        with open(dest_path, 'wb') as f:
                            f.write(img_file.read())
                        
                        frame_path_str = str(dest_path)
                        if frame_path_str not in st.session_state.extracted_frames:
                            st.session_state.extracted_frames.append(frame_path_str)
                            added_count += 1
                
                log.info(f"Added {added_count} images as frames")
                st.success(f"Added {added_count} image(s) as frames!")
                st.rerun()
        
        # =====================================================================
        # HANDLE VIDEOS - standard extraction flow
        # =====================================================================
        if video_files:
            video_configs = []
            for idx, uploaded_file in enumerate(video_files):
                video_path = pm.videos_dir / uploaded_file.name
                
                if not video_path.exists():
                    with open(video_path, 'wb') as f:
                        f.write(uploaded_file.read())
                
                info = VideoManager.get_video_info(str(video_path))
                
                with st.expander(f"{uploaded_file.name}", expanded=(idx == 0)):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Duration", f"{info['duration']:.1f}s")
                    col2.metric("FPS", f"{info['fps']:.1f}")
                    col3.metric("Frames", info['total_frames'])
                    col4.metric("Resolution", f"{info['width']}x{info['height']}")
                    
                    num_frames = st.slider(
                        f"Frames to extract from {uploaded_file.name}",
                        min_value=1,
                        max_value=info['total_frames'],
                        value=min(100, info['total_frames']),
                        key=f"frames_{idx}"
                    )
                    
                    video_configs.append({
                        'path': str(video_path),
                        'name': uploaded_file.name,
                        'num_frames': num_frames
                    })
            
            # Extraction settings
            st.subheader("Extraction Settings")
            method = st.radio(
                "Sampling method",
                ['uniform', 'sequential'],
                help="Uniform: evenly spaced | Sequential: first N frames"
            )
            
            total_frames = sum(cfg['num_frames'] for cfg in video_configs)
            
            if st.button(f"Extract {total_frames} Frames", type="primary"):
                progress = st.progress(0)
                all_frames = []
                
                for i, cfg in enumerate(video_configs):
                    frames = VideoManager.extract_frames(
                        cfg['path'],
                        str(pm.frames_dir),
                        cfg['num_frames'],
                        method
                    )
                    all_frames.extend(frames)
                    progress.progress((i + 1) / len(video_configs))
                
                # Merge with any already-recovered frames (avoid duplicates)
                existing_set = set(st.session_state.extracted_frames)
                for f in all_frames:
                    if f not in existing_set:
                        st.session_state.extracted_frames.append(f)
                
                log.info(f"Extracted {len(all_frames)} frames, total now: {len(st.session_state.extracted_frames)}")
                st.success(f"Extracted {len(all_frames)} new frames!")
    
    # Show extracted frames count and Next button
    if st.session_state.extracted_frames:
        st.info(f"{len(st.session_state.extracted_frames)} frames ready for annotation")
        
        st.markdown("---")
        if st.button("Next → Configure Prompts", type="primary", use_container_width=True):
            st.session_state.page = 'prompts'
            st.rerun()
