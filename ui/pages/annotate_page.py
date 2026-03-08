"""
Annotate & Preview Page
With video selector and per-video threshold adjustment
"""
import streamlit as st
import cv2
import gc
import json
import os
import pickle
import signal
import subprocess
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch

from config.logging_config import setup_logging
from config.settings import DEFAULT_CONFIDENCE, IMAGES_PER_PAGE
from managers.project_manager import ProjectManager

log = setup_logging("annotate_page")

def render_annotate_page():
    st.header("4. Full Annotation & Preview")
    
    pm = st.session_state.project_manager
    if pm is None:
        st.warning("No project loaded. Please go to Upload page.")
        return

    # AUTO-RESTORE SESSION IF EMPTY (e.g. after refresh)
    
    # Restore SELECTION first
    if 'selected_images' not in st.session_state:
        st.session_state.selected_images = set()
        # Try to load from project config
        saved_selection = pm.get_selected_images()
        if saved_selection:
             st.session_state.selected_images = set(saved_selection)
             st.info(f"Restored {len(saved_selection)} selected images from project.")

    if not st.session_state.extracted_frames:
        # Try to reload frames from frames_dir
        frames_dir = pm.frames_dir
        if frames_dir.exists():
            frames = sorted([str(f) for f in frames_dir.glob("*.jpg")])
            if frames:
                st.session_state.extracted_frames = frames
                st.info(f"Restored {len(frames)} frames from project.")
    
    if not st.session_state.text_prompts:
        # Try to reload prompts
        prompts = pm.get_prompts()
        if prompts:
            st.session_state.text_prompts = prompts
            st.info(f"Restored {len(prompts)} prompts from project.")

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
        frame_name = Path(frame_path).stem
        video_name = "_".join(frame_name.split("_")[:-2])
        if not video_name:
            video_name = "images"
        
        if video_name not in frames_by_video:
            frames_by_video[video_name] = []
        frames_by_video[video_name].append(frame_path)
    
    # Run full annotation
    temp_dir = pm.project_dir / "temp_annot"
    
    # 1. RECOVERY CHECK (Independent Block)
    # Only show if we have NO annotations in memory.
    # If we have annotations, we are active and don't need to recover from disk.
    if temp_dir.exists() and len(st.session_state.annotations) == 0:
         import pickle as pkl_mod
         files = list(temp_dir.glob("output_*.pkl"))
         if files:
            st.warning(f"Found {len(files)} partial result files from a previous run.")
            if st.button("Recover Previous Results"):
                progress = st.progress(0)
                recovered_results = {}
                for i, pkl in enumerate(files):
                    try:
                        with open(pkl, 'rb') as f:
                            res = pickle.load(f)
                            
                            # CRITICAL MEMORY FIX:
                            # The pickle files contain full numpy images (2GB each!). 
                            # We MUST strip them immediately or RAM will explode (56 * 2GB = 112GB).
                            # We only need coordinates/labels. Images can be lazy-loaded from disk.
                            keys_to_remove = ['original_image', 'annotated_image']
                            for frame_key in res:
                                for k in keys_to_remove:
                                    if k in res[frame_key]:
                                        del res[frame_key][k]
                                        
                            recovered_results.update(res)
                            
                            # Free memory immediately
                            del res
                            gc.collect()
                            
                    except Exception as e:
                        log.error(f"Failed to recover {pkl}: {e}")
                    progress.progress((i+1)/len(files))
                
                st.session_state.annotations = recovered_results
                st.success(f"Recovered {len(recovered_results)} annotations! (Memory Optimized)")
                st.rerun()

    total_frames = len(st.session_state.extracted_frames)
    annotated_count = len(st.session_state.annotations)
    
    # 2. PARALLEL RUNNER (Show if incomplete)
    # If we have work to do, show the runner.
    if annotated_count < total_frames:
        st.info(f"{total_frames} frames total. {annotated_count} annotated. {total_frames - annotated_count} remaining.")
        
        # Parallel Settings (Indent this block)
        with st.expander("Parallel Execution Settings / Resume", expanded=True):

            col1, col2 = st.columns(2)
            workers = col1.slider("Number of Workers (GPUs/Processes)", 1, 4, 1, 
                                 help="Spawns separate models. Warning: Requires ~N x VRAM.")
            batch_size = col2.slider("Batch Size per Job", 10, 500, 200, 
                                     help="Number of images processed per worker job.")
            
            if st.button("Start Parallel Annotation", type="primary"):
                import asyncio

                # CRITICAL: Free up VRAM from main process before spawning workers
                if st.session_state.get('sam3_annotator') is not None:
                    status_text = st.empty()
                    status_text.text("Freeing GPU memory for parallel workers...")
                    
                    # Delete the annotator object
                    del st.session_state.sam3_annotator
                    st.session_state.sam3_annotator = None
                    
                    # Force garbage collection
                    gc.collect()
                    
                    # Clear CUDA cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    log.info("Released main process SAM3 model from memory")
                
                # Temporary directories for IPC
                temp_dir = pm.project_dir / "temp_annot"
                # Only clean if we are NOT resuming? 
                # SAFETY: Do NOT delete existing temp_dir if we want to resume
                # Just ensure it exists
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                # Save prompts
                prompts_file = temp_dir / "prompts.json"
                with open(prompts_file, 'w') as f:
                    json.dump(st.session_state.text_prompts, f)
                    
                # Split frames into batches
                all_frames = st.session_state.extracted_frames
                
                # RESUME LOGIC: Filter out frames that are already in st.session_state.annotations
                frames_to_process = [
                    f for f in all_frames 
                    if f not in st.session_state.annotations
                ]
                
                if len(frames_to_process) == 0:
                    st.success("All frames are already annotated!")
                elif len(frames_to_process) < len(all_frames):
                    st.info(f"Resuming: Skipping {len(all_frames) - len(frames_to_process)} already annotated frames. Processing {len(frames_to_process)} remaining.")
                
                if frames_to_process:
                    batches = [frames_to_process[i:i + batch_size] for i in range(0, len(frames_to_process), batch_size)]
                    
                    log.info(f"Split {len(frames_to_process)} images into {len(batches)} batches (Size: {batch_size})")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Job tracking
                    pending_batches = list(enumerate(batches)) # (index, batch_list)
                    running_jobs = [] # (process, batch_index, output_file)
                    completed_results = {}
                    
                    total_batches = len(batches)
                    completed_count = 0
                    
                    # pm.project_dir is usually .../Project_Sam/projects/ProjectName
                    # scripts is at .../Project_Sam/scripts
                    # so we need .parent.parent / "scripts"
                    script_path = pm.project_dir.parent.parent / "scripts" / "annotate_worker.py"
                    
                    if not script_path.exists():
                        # Fallback check
                        # Maybe pm.project_dir is just .../Project_Sam (if running differently?)
                        # Let's check relative to file
                        script_path = Path(__file__).parent.parent.parent / "scripts" / "annotate_worker.py"
                    
                    if not script_path.exists():
                        st.error(f"Worker script not found at {script_path}")
                        st.stop()
                    
                    # Prepare configs
                    worker_tasks = [[] for _ in range(workers)]
                    batch_files_map = {} # output_pkl -> batch_idx
                    
                    import time as time_mod
                    timestamp = int(time.time())
                    
                    for i, batch_frames in enumerate(batches):
                        # Unique ID for batch
                        batch_id = f"{timestamp}_{i}"
                        input_json = temp_dir / f"batch_{batch_id}.json"
                        output_pkl = temp_dir / f"output_{timestamp}_{batch_id}.pkl"
                        
                        # Write batch content immediately
                        with open(input_json, 'w') as f:
                            json.dump(batch_frames, f)
                            
                        # Assign to worker (Round Robin)
                        worker_id = i % workers
                        worker_tasks[worker_id].append({
                            "input_json": str(input_json),
                            "output_pkl": str(output_pkl)
                        })
                        
                        batch_files_map[str(output_pkl)] = i

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    active_procs = []
                    
                    # Spawn Persistent Workers
                    for w_id in range(workers):
                        tasks = worker_tasks[w_id]
                        if not tasks:
                            continue
                            
                        config_file = temp_dir / f"worker_config_{timestamp}_{w_id}.json"
                        with open(config_file, 'w') as f:
                            json.dump({
                                "prompts_file": str(prompts_file),
                                "tasks": tasks
                            }, f)
                            
                        cmd = [
                            sys.executable, str(script_path),
                            "--worker_config", str(config_file)
                        ]
                        
                        proc = subprocess.Popen(cmd)
                        active_procs.append(proc)
                        log.info(f"Spawned Worker {w_id} (PID: {proc.pid}) with {len(tasks)} tasks")

                    status_text.text(f"Started {len(active_procs)} persistent workers for {len(batches)} batches...")
                    
                    # Monitor Loop
                    total_batches = len(batches)
                    
                    while any(p.poll() is None for p in active_procs):
                        # Count completed output files from map
                        completed = 0
                        for out_pkl in batch_files_map.keys():
                            if Path(out_pkl).exists():
                                completed += 1
                                
                        progress = min(1.0, completed / max(1, total_batches))
                        progress_bar.progress(progress)
                        status_text.text(f"Processing... {completed}/{total_batches} batches completed.")
                        
                        time.sleep(1.0)
                        
                    # Final check
                    completed = 0
                    for out_pkl in batch_files_map.keys():
                        if Path(out_pkl).exists():
                            completed += 1
                            
                    if completed >= total_batches:
                        st.success("All batches processed!")
                    else:
                        st.warning(f"Workers finished but only {completed}/{total_batches} batches found. Some may have failed. Check logs/terminal.")
                    
                    # Reload ALL results from disk
                    completed_results = {}
                    for out_pkl in batch_files_map.keys():
                        if Path(out_pkl).exists():
                             try:
                                with open(out_pkl, 'rb') as f:
                                    res = pickle.load(f)
                                    # MEMORY OPTIMIZATION (Same as Recovery)
                                    keys_to_remove = ['original_image', 'annotated_image']
                                    for frame_key in res:
                                        for k in keys_to_remove:
                                            if k in res[frame_key]:
                                                del res[frame_key][k]
                                    completed_results.update(res)
                                    
                                    # GC
                                    del res
                                    gc.collect()
                             except: pass
                    
                    # Cleanup - OPTIONAL: Don't delete immediately so user can recover if this step crashes?
                    # Let's keep it for now, user can manually delete or next run will delete.
                    # try:
                    #     import shutil
                    #     shutil.rmtree(temp_dir)
                    # except: pass
                    
                    # Store in session state
                    
                    status_text.text("Finalizing results...")
                    
                    for fp, res in completed_results.items():
                         # Load image for the result object.
                         img = cv2.imread(fp)
                         if img is not None:
                             res['original_image'] = img

                    st.session_state.annotations.update(completed_results) # Update existing
                    log.info(f"Annotated {len(completed_results)} frames total")
                    st.success("Annotation Complete!")
                    st.rerun()
    
    # 3. PREVIEW & ACTIONS (Show if ANY annotations exist)
    if annotated_count > 0:
        st.success(f"{len(st.session_state.annotations)} frames annotated")
        
        # GLOBAL ACTIONS
        col1, col2 = st.columns(2)
        if col1.button("Select ALL Annotated Images", type="primary"):
            st.session_state.selected_images = set(st.session_state.annotations.keys())
            # Sync checkbox widget keys so they render as checked on rerun
            for fp in st.session_state.annotations.keys():
                st.session_state[f"sel_{fp}"] = True
            pm.set_selected_images(list(st.session_state.selected_images))
            st.rerun()
            
        if col2.button("Deselect ALL"):
            # Sync checkbox widget keys so they render as unchecked on rerun
            for fp in st.session_state.selected_images:
                st.session_state[f"sel_{fp}"] = False
            st.session_state.selected_images.clear()
            pm.set_selected_images([])
            st.rerun()
            
        # Video selector
        video_names = list(frames_by_video.keys())
        selected_video = st.selectbox("Select Video", video_names)
        
        # Threshold adjustment for selected video
        st.subheader(f"Thresholds for: {selected_video}")
        
        # Initialize if needed
        if 'video_thresholds' not in st.session_state:
            st.session_state.video_thresholds = {}
        if selected_video not in st.session_state.video_thresholds:
            st.session_state.video_thresholds[selected_video] = {
                p['class_name']: DEFAULT_CONFIDENCE 
                for p in st.session_state.text_prompts
            }
        
        # Threshold sliders
        cols = st.columns(len(st.session_state.text_prompts))
        for i, prompt_data in enumerate(st.session_state.text_prompts):
            class_name = prompt_data['class_name']
            current = st.session_state.video_thresholds[selected_video].get(class_name, DEFAULT_CONFIDENCE)
            
            new_val = cols[i].slider(
                f"{class_name}",
                0.0, 1.0, current, 0.05,
                key=f"annot_thresh_{selected_video}_{class_name}"
            )
            st.session_state.video_thresholds[selected_video][class_name] = new_val
        
        # Save to project
        pm.set_video_thresholds(selected_video, st.session_state.video_thresholds[selected_video])
        
        st.markdown("---")
        
        # Preview frames from selected video
        st.subheader(f"Preview: {selected_video}")
        
        video_frames = frames_by_video[selected_video]
        thresholds = st.session_state.video_thresholds[selected_video]
        
        # Pagination
        total_pages = max(1, len(video_frames) // IMAGES_PER_PAGE + 1)
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        start_idx = (page - 1) * IMAGES_PER_PAGE
        end_idx = min(start_idx + IMAGES_PER_PAGE, len(video_frames))
        
        page_frames = video_frames[start_idx:end_idx]
        
        # Select all / deselect all for this video
        col1, col2 = st.columns(2)
        if col1.button(f"Select All ({selected_video})"):
            for fp in video_frames:
                st.session_state.selected_images.add(fp)
                st.session_state[f"sel_{fp}"] = True
            st.rerun()
        if col2.button(f"Deselect All ({selected_video})"):
            for fp in video_frames:
                st.session_state.selected_images.discard(fp)
                st.session_state[f"sel_{fp}"] = False
            st.rerun()
        
        # Grid display
        img_cols = st.columns(4)
        for i, frame_path in enumerate(page_frames):
            result = st.session_state.annotations.get(frame_path, {})
            # LAZY LOAD: If original_image is missing (due to OOM fix), load it now
            img = result.get('original_image')
            if img is None:
               img = cv2.imread(frame_path)
            
            if img is None:
                continue
            
            img = img.copy()
            
            # Filter and draw
            det_count = 0
            for det in result.get('detections', []):
                class_name = st.session_state.text_prompts[det['class_id']]['class_name']
                threshold = thresholds.get(class_name, DEFAULT_CONFIDENCE)
                
                if det['score'] >= threshold:
                    det_count += 1
                    box = det['box_xyxy']
                    cv2.rectangle(img, 
                        (int(box[0]), int(box[1])), 
                        (int(box[2]), int(box[3])), 
                        (0, 255, 0), 2)
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            with img_cols[i % 4]:
                st.image(img_rgb, caption=f"Det: {det_count}", use_container_width=True)
                
                # Initialize checkbox key from selected_images if not already set
                cb_key = f"sel_{frame_path}"
                if cb_key not in st.session_state:
                    st.session_state[cb_key] = frame_path in st.session_state.selected_images
                
                selected = st.checkbox(
                    "Select", 
                    key=cb_key
                )
                if selected:
                    st.session_state.selected_images.add(frame_path)
                else:
                    st.session_state.selected_images.discard(frame_path)
        
        st.markdown("---")
        st.info(f"{len(st.session_state.selected_images)} total images selected for dataset")
        
        # Always show Next button
        if st.button("Next -> Organize Dataset", type="primary", use_container_width=True):
            # If nothing selected, assume Select ALL
            if len(st.session_state.selected_images) == 0:
                 st.info("No selection made. Auto-selecting ALL annotated images...")
                 for fp in st.session_state.annotations.keys():
                    st.session_state.selected_images.add(fp)
            
            # SAVE SELECTION PERSISTENTLY
            pm.set_selected_images(list(st.session_state.selected_images))
            st.session_state.page = 'organize'
            st.rerun()

