"""
Annotation Worker Script
Executed as a subprocess by the Streamlit app for parallel processing.
"""
import sys
import argparse
import json
import cv2
import numpy as np
import torch
import pickle
from pathlib import Path
from typing import List, Dict

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from managers.annotation_manager import SAM3Annotator
from config.logging_config import setup_logging

log = setup_logging("annot_worker")

def process_persistent(worker_config_file: str):
    """
    Process multiple batches with a single model load (Persistent Mode).
    config_file contains:
    {
        "prompts_file": str,
        "tasks": [
            {"input_json": str, "output_pkl": str},
            ...
        ]
    }
    """
    try:
        with open(worker_config_file, 'r') as f:
            config = json.load(f)
            
        prompts_file = config['prompts_file']
        tasks = config['tasks']
        
        if not tasks:
            log.info("No tasks assigned to this worker.")
            return

        # Load prompts ONCE
        with open(prompts_file, 'r') as f:
            prompts_data = json.load(f)
            prompt_texts = [p['prompt_text'] for p in prompts_data]
            
        log.info(f"Worker started. Tasks: {len(tasks)}. Prompts: {len(prompt_texts)}")
        
        # Initialize Model ONCE
        annotator = SAM3Annotator()
        annotator.initialize() 
        
        # Loop through tasks
        for idx, task in enumerate(tasks):
            input_json = task['input_json']
            output_pkl = task['output_pkl']
            
            # Check if output already exists (Resume safety)
            if Path(output_pkl).exists():
                log.info(f"Skipping already completed task: {output_pkl}")
                continue
            
            try:
                # Load images for this batch
                with open(input_json, 'r') as f:
                    image_paths = json.load(f)
                
                log.info(f"Processing Task {idx+1}/{len(tasks)}: {len(image_paths)} images")
                
                batch_results = {}
                
                # Annotate images in this batch
                for i, img_path in enumerate(image_paths):
                    try:
                        res = annotator.annotate_single_image(img_path, prompt_texts)
                        
                        # Optimize storage (remove heavy original_image)
                        if 'original_image' in res:
                            del res['original_image'] 
                        
                        batch_results[img_path] = res
                    except Exception as e:
                        log.error(f"Error processing {img_path}: {e}")
                
                # Save batch results
                with open(output_pkl, 'wb') as f:
                    pickle.dump(batch_results, f)
                    
                log.info(f"Saved: {output_pkl}")
                
                # Optional: GC every few batches to prevent slow leaks?
                if idx % 5 == 0:
                    import gc
                    gc.collect()
                    
            except Exception as e:
                log.error(f"Failed task {input_json}: {e}")
                # Continue together tasks!
                
        log.info("Worker finished all tasks.")
        
    except Exception as e:
        log.error(f"Worker critical failure: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Old mode support (optional, but let's just switch to new mode)
    parser.add_argument("--worker_config", help="Path to worker config json")
    
    # Keep old args for backward compat if needed? 
    # Actually, we control the caller, so we can break compat.
    parser.add_argument("--images_json", required=False) # Deprecated
    parser.add_argument("--prompts_json", required=False) # Deprecated
    parser.add_argument("--output_res", required=False) # Deprecated
    
    args = parser.parse_args()
    
    if args.worker_config:
        process_persistent(args.worker_config)
    elif args.images_json:
        # Fallback to old mode (single batch)
        process_batch(args.images_json, args.prompts_json, args.output_res)
    else:
        log.error("Usage: --worker_config <file>")
