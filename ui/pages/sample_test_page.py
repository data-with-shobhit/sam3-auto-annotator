"""
Sample Test Page - Test on 4 frames per video
"""
import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.logging_config import setup_logging
from config.settings import DEFAULT_CONFIDENCE

log = setup_logging("sample_test_page")

def render_sample_test_page():
    st.header("3. Sample Test (4 frames per video)")
    
    if not st.session_state.extracted_frames:
        st.warning("Please extract frames first")
        return
    
    if not st.session_state.text_prompts:
        st.warning("Please add prompts first")
        return
    
    pm = st.session_state.project_manager
    
    # Group frames by video source
    frames_by_video = {}
    for frame_path in st.session_state.extracted_frames:
        # Extract video name from frame filename (format: videoname_frame_000123.jpg)
        frame_name = Path(frame_path).stem
        video_name = "_".join(frame_name.split("_")[:-2])  # Remove _frame_XXXXXX
        if not video_name:
            video_name = "images"
        
        if video_name not in frames_by_video:
            frames_by_video[video_name] = []
        frames_by_video[video_name].append(frame_path)
    
    st.info(f"Found {len(frames_by_video)} video source(s)")
    
    # Initialize sample results in session state
    if 'sample_results' not in st.session_state:
        st.session_state.sample_results = {}
    
    # Initialize per-video thresholds
    if 'video_thresholds' not in st.session_state:
        st.session_state.video_thresholds = {}

    # Detect prompt changes — invalidate stale sample results
    current_prompt_fingerprint = tuple(sorted(
        (p['prompt_text'], p['class_name']) for p in st.session_state.text_prompts
    ))
    if 'sample_prompt_fingerprint' not in st.session_state:
        st.session_state.sample_prompt_fingerprint = None
    
    if (
        st.session_state.sample_results
        and st.session_state.sample_prompt_fingerprint != current_prompt_fingerprint
    ):
        st.warning("Prompts have changed since the last sample test. Re-running...")
        st.session_state.sample_results = {}
        st.session_state.force_sample_test = True
        st.rerun()

    # Run Sample Test Button
    if not st.session_state.sample_results:
        # Check if we should auto-run (because of invalidate or "Re-run" click)
        force_run = st.session_state.get('force_sample_test', False)
        
        if force_run or st.button("Run Sample Test", type="primary"):
            # Clear the flag so it doesn't auto-run next time
            st.session_state.force_sample_test = False
            
            from managers.annotation_manager import SAM3Annotator
            
            progress = st.progress(0)
            status = st.empty()
            
            annotator = SAM3Annotator()
            annotator.initialize()
            st.session_state.sam3_annotator = annotator
            
            sample_results = {}
            prompts = [p['prompt_text'] for p in st.session_state.text_prompts]
            
            video_list = list(frames_by_video.keys())
            for v_idx, video_name in enumerate(video_list):
                status.text(f"Testing {video_name}...")
                frames = frames_by_video[video_name]
                
                # Get 4 evenly spaced samples
                indices = np.linspace(0, len(frames)-1, min(4, len(frames)), dtype=int)
                sample_frames = [frames[i] for i in indices]
                
                video_samples = []
                for frame_path in sample_frames:
                    result = annotator.annotate_single_image(frame_path, prompts, DEFAULT_CONFIDENCE)
                    video_samples.append({
                        'path': frame_path,
                        'detections': result['detections'],
                        'image': result['original_image']
                    })
                
                sample_results[video_name] = video_samples
                progress.progress((v_idx + 1) / len(video_list))
            
            st.session_state.sample_results = sample_results
            
            # Initialize thresholds for each video
            for video_name in sample_results.keys():
                if video_name not in st.session_state.video_thresholds:
                    st.session_state.video_thresholds[video_name] = {
                        p['class_name']: DEFAULT_CONFIDENCE 
                        for p in st.session_state.text_prompts
                    }
            
            # Save fingerprint of prompts used for this run
            st.session_state.sample_prompt_fingerprint = current_prompt_fingerprint
            log.info(f"Sample test completed for {len(sample_results)} videos")
            st.rerun()
    
    else:
        col_status, col_rerun = st.columns([4, 1])
        col_status.success(f"Sample test completed for {len(st.session_state.sample_results)} video(s)")
        if col_rerun.button("Re-run", use_container_width=True):
            st.session_state.sample_results = {}
            st.session_state.sample_prompt_fingerprint = None
            st.session_state.force_sample_test = True
            st.rerun()
        
        # Video tabs
        video_names = list(st.session_state.sample_results.keys())
        tabs = st.tabs(video_names)
        
        for tab, video_name in zip(tabs, video_names):
            with tab:
                st.subheader(f"{video_name}")
                
                # Threshold sliders for this video
                st.markdown("**Adjust Thresholds:**")
                cols = st.columns(len(st.session_state.text_prompts))
                
                for i, prompt_data in enumerate(st.session_state.text_prompts):
                    class_name = prompt_data['class_name']
                    current = st.session_state.video_thresholds.get(video_name, {}).get(class_name, DEFAULT_CONFIDENCE)
                    
                    new_val = cols[i].slider(
                        f"{class_name}",
                        0.0, 1.0, current, 0.05,
                        key=f"thresh_{video_name}_{class_name}"
                    )
                    
                    if video_name not in st.session_state.video_thresholds:
                        st.session_state.video_thresholds[video_name] = {}
                    st.session_state.video_thresholds[video_name][class_name] = new_val
                
                # Save thresholds to project
                pm.set_video_thresholds(video_name, st.session_state.video_thresholds[video_name])
                
                # Display sample images
                samples = st.session_state.sample_results[video_name]
                img_cols = st.columns(min(4, len(samples)))
                
                for col, sample in zip(img_cols, samples):
                    img = sample['image'].copy()
                    thresholds = st.session_state.video_thresholds.get(video_name, {})
                    
                    # Build prompt → class_name lookup for display labels
                    prompt_to_class = {
                        p['prompt_text']: p['class_name']
                        for p in st.session_state.text_prompts
                    }
                    
                    # Filter and draw detections
                    det_count = 0
                    for det in sample['detections']:
                        # det['class_name'] is the raw prompt; resolve to user-defined class name
                        prompt_text = det['class_name']
                        class_name = prompt_to_class.get(prompt_text, prompt_text)
                        threshold = thresholds.get(class_name, DEFAULT_CONFIDENCE)
                        
                        if det['score'] >= threshold:
                            det_count += 1
                            box = det['box_xyxy']
                            cv2.rectangle(img, 
                                (int(box[0]), int(box[1])), 
                                (int(box[2]), int(box[3])), 
                                (0, 255, 0), 2)
                            label = f"{class_name}: {det['score']:.2f}"
                            cv2.putText(img, label, 
                                (int(box[0]), int(box[1])-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    col.image(img_rgb, caption=f"Detections: {det_count}", use_container_width=True)
        
        st.markdown("---")
        st.info("Adjust thresholds per video above. When satisfied, proceed to Full Annotation.")
        
        if st.button("Next -> Full Annotation", type="primary", use_container_width=True):
            st.session_state.page = 'annotate'
            st.rerun()

