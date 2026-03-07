"""
Training Page - Advanced Features
Live dashboard, Background Training, Controls, Exports
"""
import streamlit as st
import time
import subprocess
import signal
import os
import shutil
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.logging_config import setup_logging
from utils.log_parser import parse_training_log, get_latest_metrics

log = setup_logging("train_page")

# Model options
MODEL_OPTIONS = {
    "YOLO26": {"sizes": ["n", "s", "m", "l", "x"], "prefix": "yolo26", "suffix": ".pt"},
    "YOLO12": {"sizes": ["n", "s", "m", "l", "x"], "prefix": "yolo12", "suffix": ".pt"},
    "YOLOv11": {"sizes": ["n", "s", "m", "l", "x"], "prefix": "yolo11", "suffix": ".pt"},
    "YOLOv10": {"sizes": ["n", "s", "m", "l", "b", "x"], "prefix": "yolov10", "suffix": ".pt"},
    "YOLOv9": {"sizes": ["t", "s", "m", "c", "e"], "prefix": "yolov9", "suffix": ".pt"},
    "YOLOv8": {"sizes": ["n", "s", "m", "l", "x"], "prefix": "yolov8", "suffix": ".pt"},
    "RT-DETR": {"sizes": ["l", "x"], "prefix": "rtdetr-", "suffix": ".pt"}
}

def render_train_page():
    st.header("7. Train Model")
    
    pm = st.session_state.project_manager
    if pm is None:
        st.warning("Please load a project first")
        return
    
    # State initialization
    if 'training_pid' not in st.session_state:
        st.session_state.training_pid = None
        st.session_state.training_cmd = None
        st.session_state.training_start_time = None
        st.session_state.training_output_dir = None
    
    # Tabs
    tab_train, tab_history, tab_resume = st.tabs(["Start Training", "History & Export", "Resume"])
    
    with tab_train:
        render_start_training(pm)
        
    with tab_history:
        render_history(pm)

    with tab_resume:
        render_resume(pm)

def render_start_training(pm):
    # Check for active training
    if st.session_state.training_pid is not None:
        render_training_monitor(pm)
        return

    # Show completion message if training just finished
    if st.session_state.get('training_completed'):
        st.success("**Training Complete!** Your model has finished training.")
        out_dir = st.session_state.get('training_output_dir')
        if out_dir and Path(str(out_dir)).exists():
            best_pt = Path(str(out_dir)) / "weights" / "best.pt"
            if best_pt.exists():
                st.info(f"Best model saved at: `{best_pt}`")
            # Show final metrics if available
            csv_path = Path(str(out_dir)) / "results.csv"
            if csv_path.exists():
                from utils.log_parser import parse_training_log, get_latest_metrics
                df = parse_training_log(csv_path)
                if df is not None and not df.empty:
                    metrics = get_latest_metrics(df)
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Final mAP50", f"{metrics.get('mAP50', 0):.3f}")
                    m2.metric("Final mAP50-95", f"{metrics.get('mAP50-95', 0):.3f}")
                    m3.metric("Precision", f"{metrics.get('Precision', 0):.3f}")
                    m4.metric("Recall", f"{metrics.get('Recall', 0):.3f}")
        if st.button("Dismiss"):
            st.session_state.training_completed = False
            st.rerun()
        st.markdown("---")

    data_yaml = pm.dataset_dir / "data.yaml"
    if not data_yaml.exists():
        st.warning("Please generate dataset first (Step 5)")
        return
    
    st.success(f"Dataset: {pm.dataset_dir}")

    # =========================================================================
    # 2. MODEL SELECTION
    # =========================================================================
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    model_family = col1.selectbox("Model Family", list(MODEL_OPTIONS.keys()))
    model_size = col2.selectbox("Model Size", MODEL_OPTIONS[model_family]["sizes"])
    
    prefix = MODEL_OPTIONS[model_family]["prefix"]
    suffix = MODEL_OPTIONS[model_family]["suffix"]
    model_name = f"{prefix}{model_size}{suffix}"
    
    # Parameters
    st.subheader("Parameters")
    col1, col2, col3 = st.columns(3)
    col1, col2, col3 = st.columns(3)
    epochs = col1.number_input("Epochs", 1, 10000, 100)
    batch_size = col2.number_input("Batch Size", 1, 512, 128, help="Try 128-256 for A100. Lower if OOM.")
    img_size = col3.selectbox("Image Size", [320, 640, 1024, 1280], index=1)
    
    # Advanced Params
    with st.expander("Advanced Options & Augmentation"):
        col1, col2 = st.columns(2)
        patience = col1.number_input("Patience", 0, 500, 50, help="Epochs to wait for improvement before stopping.")
        workers = col2.number_input("Workers", 0, 128, 32, help="Dataloader threads. High (16-32) is good for EPYC.")
        
        st.markdown("### Augmentations")
        from config.augmentations import AUGMENTATION_INFO
        
        # Collect aug values
        aug_values = {}
        
        # Display in grid
        keys = list(AUGMENTATION_INFO.keys())
        # Split into columns for better layout
        cols = st.columns(2)
        
        for i, key in enumerate(keys):
            info = AUGMENTATION_INFO[key]
            current_col = cols[i % 2]
            
            with current_col:
                # Use a checkbox to enable/disable specific if needed, or just sliders? 
                # Ultralytics uses floats, so slider 0.0 effectively usually disables or neutralizes.
                # Adding description as caption
                aug_values[key] = st.slider(
                    label=info['label'], 
                    min_value=info['min'], 
                    max_value=info['max'], 
                    value=info['default'],
                    step=0.01 if key != 'degrees' else 1.0, 
                    help=info['desc']
                )
                st.caption(f"{info['desc']}")

    # =========================================================================
    # 2. DATA SCALING (OFFLINE)
    # =========================================================================
    st.markdown("---")
    st.subheader("Data Scaling")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("**Multiply Dataset Size**")
        scale_factor = st.slider("Target Size (x times original)", 1, 10, 1, help="Generate new images using the augmentations selected above.")
        
    with col2:
        st.markdown("**Select Augmentations for Generation:**")
        c1, c2, c3 = st.columns(3)
        use_rotate = c1.checkbox("Rotate", value=True)
        use_hflip = c1.checkbox("H-Flip", value=True)
        use_vflip = c2.checkbox("V-Flip", value=False)
        use_brightness = c2.checkbox("Brightness", value=True)
        use_noise = c3.checkbox("Noise", value=False)
        use_blur = c3.checkbox("Blur", value=False)
        use_clahe = c1.checkbox("CLAHE", value=False)

        if scale_factor > 1:
            st.info(f"Will generate **{scale_factor - 1}** new variants for every image.")
            if st.button("Generate Scaled Dataset Now", type="secondary"):
                from managers.augmentation_manager import AugmentationManager
                
                # Use explicit checkboxes
                aug_config_bools = {
                    'rotate': use_rotate,
                    'horizontal_flip': use_hflip,
                    'vertical_flip': use_vflip,
                    'brightness': use_brightness,
                    'noise': use_noise,
                    'blur': use_blur,
                    'clahe': use_clahe
                }
                
                am = AugmentationManager(pm.project_dir)
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_prog(current):
                    status_text.text(f"Generating image {current}...")
                
                output_path = am.generate_augmented_dataset(scale_factor, aug_config_bools, update_prog)
                
                progress_bar.progress(1.0)
                st.success(f"Generated {scale_factor}x Dataset at: {output_path.name}")
                st.session_state.use_augmented_data = True
                st.session_state.last_aug_path = str(output_path)
                st.rerun()

    # Check if we should use augmented data
    data_path_to_use = data_yaml
    if st.session_state.get('use_augmented_data') and scale_factor > 1:
        aug_path = pm.project_dir / "dataset_augmented" / "data.yaml"
        if aug_path.exists():
            data_path_to_use = aug_path
            st.info(f"Training on Scaled Dataset: `dataset_augmented` ({scale_factor}x)")
    
    # Output Config
    output_dir = pm.project_dir / "runs" / "train"
    experiment_name = st.text_input("Experiment Name", value=f"{model_family}_{model_size}_exp")
    
    full_output_path = output_dir / experiment_name
    
    if full_output_path.exists():
        st.warning(f"Experiment '{experiment_name}' already exists. Ultralytics will auto-increment (exp2, exp3...) or resume if specified.")
    
    # Start Button
    if st.button("Start Training (Background)", type="primary"):
        start_training(
            pm, model_name, str(data_path_to_use), str(output_dir), experiment_name,
            epochs, batch_size, img_size, patience, workers,
            **aug_values
        )
        st.rerun()

def start_training(pm, model, data, proj, name, epochs, batch, imgsz, pat, workers, **kwargs):
    # Ensure base model exists
    base_models_dir = pm.project_dir.parent.parent / "base_models"
    base_models_dir.mkdir(parents=True, exist_ok=True)
    
    # Ultralytics naming logic is complex, simplistic check:
    # If model is 'yolov8n.pt', look for it.
    target_path = base_models_dir / model
    
    try:
        if not target_path.exists():
            st.info(f"Downloading {model} to cache...")
            from ultralytics import YOLO
            import shutil
            
            # Download to CWD
            temp = YOLO(model) 
            
            # Look for downloaded file
            candidates = [Path.cwd() / model, Path.cwd() / f"{model}"]
            found = next((c for c in candidates if c.exists()), None)
            
            if found:
                shutil.move(str(found), str(target_path))
                st.success("Cached base model.")
            else:
                st.warning("Could not locate downloaded model to cache. Proceeding anyway.")
    except Exception as e:
        log.warning(f"Caching failed: {e}")

    # Construct arguments for worker script
    # Go up 2 levels: projects/MyProject -> projects -> root -> scripts
    script_path = pm.project_dir.parent.parent / "scripts" / "train_worker.py"
    
    cmd = [
        sys.executable, str(script_path),
        "--model_path", str(target_path) if target_path.exists() else model, 
        "--data_yaml", data,
        "--project_dir", proj,
        "--name", name,
        "--epochs", str(epochs),
        "--batch_size", str(batch),
        "--img_size", str(imgsz),
        "--patience", str(pat),
        "--workers", str(workers),
        "--mosaic", str(kwargs.get('mosaic', 1.0)),
        "--mixup", str(kwargs.get('mixup', 0.0)),
        "--degrees", str(kwargs.get('degrees', 0.0))
    ]
    
    # Launch subprocess
    process = subprocess.Popen(cmd)
    
    st.session_state.training_pid = process.pid
    st.session_state.training_cmd = cmd
    st.session_state.training_start_time = time.time()
    # Output dir is guessable: proj/name
    st.session_state.training_output_dir = Path(proj) / name 
    
    log.info(f"Started training process PID: {process.pid}")

def render_training_monitor(pm):
    st.subheader("Training in Progress...")
    
    pid = st.session_state.training_pid
    
    # Check if process is still running
    if not is_process_running(pid):
        st.session_state.training_pid = None
        st.session_state.training_completed = True
        st.rerun()
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("PID", pid)
    elapsed = int(time.time() - st.session_state.training_start_time)
    col2.metric("Elapsed Time", f"{elapsed//60}m {elapsed%60}s")
    
    # Control Buttons
    st.write("### Controls")
    c1, c2, c3 = st.columns(3)
    
    if c1.button("Early Stop & Finalize"):
        stop_process(pid)
        st.success("Stopped. Saving best model so far...")
        time.sleep(2)
        st.session_state.training_pid = None
        st.rerun()
        
    if c2.button("Stop & Discard"):
        stop_process(pid)
        # Cleanup
        out_dir = st.session_state.training_output_dir
        if out_dir and out_dir.exists():
            import shutil
            try:
                shutil.rmtree(out_dir)
            except: pass
        st.warning("Training discarded.")
        st.session_state.training_pid = None
        st.rerun()
        
    # Live Plots
    st.write("### Live Metrics")
    
    runs_dir = pm.project_dir / "runs" / "train"
    stats_dir = get_latest_run_dir(runs_dir)
    
    if stats_dir:
        csv_path = stats_dir / "results.csv"
        if csv_path.exists():
            df = parse_training_log(csv_path)
            if df is not None and not df.empty:
                # Metrics
                metrics = get_latest_metrics(df)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("mAP50", f"{metrics.get('mAP50', 0):.3f}")
                m2.metric("mAP50-95", f"{metrics.get('mAP50-95', 0):.3f}")
                m3.metric("Precision", f"{metrics.get('Precision', 0):.3f}")
                m4.metric("Recall", f"{metrics.get('Recall', 0):.3f}")
                
                # Charts
                st.line_chart(df, y=[c for c in df.columns if 'loss' in c.lower() and 'val' not in c])
                st.line_chart(df, y=[c for c in df.columns if 'map' in c.lower()])
    else:
        st.info("Waiting for first metrics...")
        
    # Auto-refresh
    time.sleep(2)
    st.rerun()

def render_history(pm):
    st.subheader("Training History")
    runs_dir = pm.project_dir / "runs" / "train"
    if not runs_dir.exists():
        st.info("No training runs found.")
        return
        
    runs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], key=os.path.getmtime, reverse=True)
    
    # =========================================================================
    # EXPORT CONFIGURATION
    # =========================================================================
    st.subheader("Export Configuration")
    
    export_format = st.selectbox("Export Format", ["onnx", "engine", "torchscript", "saved_model"], key="history_fmt")
    
    col_mode, col_prec = st.columns(2)
    
    with col_mode:
        export_mode = st.radio("Export Mode", ["Static", "Dynamic"], horizontal=True, key="export_mode",
                               help="Static: fixed input shape | Dynamic: variable batch size")
    
    with col_prec:
        precision = st.radio("Precision", ["FP32 (Full)", "FP16 (Half)"], horizontal=True, key="export_prec",
                             help="FP16 is faster but slightly less accurate. Recommended for inference.")
    
    use_half = "FP16" in precision
    use_dynamic = export_mode == "Dynamic"
    
    # Static shape configuration
    export_batch = 1
    export_imgsz = 640
    if not use_dynamic:
        st.markdown("**Static Shape Configuration**")
        sc1, sc2 = st.columns(2)
        export_batch = sc1.number_input("Batch Size", min_value=1, max_value=64, value=1, key="export_batch",
                                         help="Number of images per inference batch")
        export_imgsz = sc2.selectbox("Image Size", [320, 416, 512, 640, 768, 1024, 1280], index=3, key="export_imgsz")
    
    st.markdown("---")
    
    for run in runs:
        with st.expander(f"{run.name}", expanded=False):
            # Show metrics
            csv_path = run / "results.csv"
            if csv_path.exists():
                df = parse_training_log(csv_path)
                if df is not None:
                    latest = get_latest_metrics(df)
                    c1, c2 = st.columns(2)
                    c1.write(f"**Best mAP50**: {latest.get('mAP50', 0):.3f}")
                    c2.write(f"**Epochs**: {len(df)}")
            
            # Export Section
            st.write("#### Actions")
            best_pt = run / "weights" / "best.pt"
            
            mode_label = "Dynamic" if use_dynamic else f"Static {export_batch}x3x{export_imgsz}x{export_imgsz}"
            prec_label = "FP16" if use_half else "FP32"
            
            if best_pt.exists():
                if st.button(f"Export to {export_format.upper()} ({mode_label}, {prec_label})", key=f"exp_{run.name}"):
                    export_model(best_pt, export_format, dynamic=use_dynamic, half=use_half,
                                batch=export_batch, imgsz=export_imgsz)
            else:
                st.warning("No best.pt found")

def render_resume(pm):
    st.subheader("Resume Training")
    runs_dir = pm.project_dir / "runs" / "train"
    
    if not runs_dir.exists():
        return
        
    # Find active runs (has last.pt)
    resumable = []
    for run in runs_dir.iterdir():
        last_pt = run / "weights" / "last.pt"
        if last_pt.exists():
            resumable.append(run)
            
    if not resumable:
        st.info("No interrupted runs found to resume.")
        return
        
    selected_run = st.selectbox("Select Run to Resume", resumable, format_func=lambda x: x.name)
    
    if st.button("Resume Training", type="primary"):
        # Launch worker with resume=True
        weights_path = selected_run / "weights" / "last.pt"
        
        script_path = pm.project_dir.parent.parent / "scripts" / "train_worker.py"
        cmd = [
            sys.executable, str(script_path),
            "--model_path", str(weights_path),
            "--data_yaml", "dummy", 
            "--project_dir", str(runs_dir.parent), 
            "--name", "resume_run", 
            "--resume"
        ]
        
        process = subprocess.Popen(cmd)
        st.session_state.training_pid = process.pid
        st.session_state.training_start_time = time.time()
        st.session_state.training_output_dir = selected_run
        st.rerun()

def is_process_running(pid):
    try:
        # Standard check
        os.kill(pid, 0)
        
        # Check for zombie process on Linux
        try:
            with open(f"/proc/{pid}/stat", 'r') as f:
                stat = f.read()
                # 3rd field is state. Z = Zombie, X = Dead
                state = stat.split()[2]
                if state in ['Z', 'X']:
                    return False
        except FileNotFoundError:
            return False # Process gone
        except Exception:
            pass # Non-linux or permission error, assume running if kill(0) worked
            
    except OSError:
        return False
    return True

def stop_process(pid):
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception:
        pass

def get_latest_run_dir(parent_dir):
    try:
        subdirs = [d for d in parent_dir.iterdir() if d.is_dir()]
        if not subdirs: return None
        return max(subdirs, key=os.path.getmtime)
    except Exception:
        return None

def export_model(weights_path, format, dynamic=False, half=False, batch=1, imgsz=640):
    try:
        from ultralytics import YOLO
        model = YOLO(str(weights_path))
        
        mode_str = "Dynamic" if dynamic else f"Static {batch}x3x{imgsz}x{imgsz}"
        prec_str = "FP16" if half else "FP32"
        st.info(f"Exporting to {format} ({mode_str}, {prec_str})...")
        
        export_kwargs = {
            "format": format,
            "half": half,
            "imgsz": imgsz,
        }
        
        if dynamic:
            export_kwargs["dynamic"] = True
        else:
            export_kwargs["batch"] = batch
        
        out = model.export(**export_kwargs)
        st.success(f"Exported to: {out}")
    except Exception as e:
        st.error(f"Export failed: {e}")
