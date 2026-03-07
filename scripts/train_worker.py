"""
Training Worker Script
Executed as a subprocess by the Streamlit app.
"""
import sys
import argparse
import signal
from pathlib import Path
from ultralytics import YOLO
import threading
import time
import os

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.logging_config import setup_logging
log = setup_logging("train_worker")

def train_model(
    model_path: str,
    data_yaml: str,
    project_dir: str,
    name: str,
    epochs: int,
    batch_size: int,
    img_size: int,
    patience: int,
    workers: int,
    resume: bool,
    # Augmentation args
    hsv_h: float, hsv_s: float, hsv_v: float,
    degrees: float, translate: float, scale: float,
    flipud: float, fliplr: float,
    mosaic: float, mixup: float
):
    try:
        log.info(f"Worker starting training: {name}")
        model = YOLO(model_path)
        
        args = {
            "data": data_yaml,
            "project": project_dir,
            "name": name,
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": img_size,
            "patience": patience if patience > 0 else 0, # Ultralytics treats 0 as disabled
            "workers": workers,
            "exist_ok": True,
            "hsv_h": hsv_h,
            "hsv_s": hsv_s,
            "hsv_v": hsv_v,
            "degrees": degrees,
            "translate": translate,
            "scale": scale,
            "flipud": flipud,
            "fliplr": fliplr,
            "mosaic": mosaic,
            "mixup": mixup,
            "resume": resume
        }
        
        # Start training
        model.train(**args)
        
        log.info("Worker finished successfully")
        
    except Exception as e:
        log.error(f"Worker failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_yaml", required=True)
    parser.add_argument("--project_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    
    # Augmentations
    parser.add_argument("--hsv_h", type=float, default=0.015)
    parser.add_argument("--hsv_s", type=float, default=0.7)
    parser.add_argument("--hsv_v", type=float, default=0.4)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    
    args = parser.parse_args()
    
    # Handle graceful shutdown on SIGTERM (killed by UI)
    def signal_handler(signum, frame):
        log.info("Worker received kill signal. Exiting...")
        sys.exit(0)
        
    signal.signal(signal.SIGTERM, signal_handler)
    
    train_model(
        args.model_path, args.data_yaml, args.project_dir, args.name,
        args.epochs, args.batch_size, args.img_size, args.patience, args.workers,
        args.resume,
        args.hsv_h, args.hsv_s, args.hsv_v,
        args.degrees, args.translate, args.scale,
        args.flipud, args.fliplr,
        args.mosaic, args.mixup
    )
