"""
SAM3 annotation backend for batch processing
"""
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image

# Add sam3 to path
SAM3_PATH = Path(__file__).parent.parent / "sam3"
if SAM3_PATH.exists():
    sys.path.insert(0, str(SAM3_PATH))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class SAM3Annotator:
    """SAM3 annotation handler with caching and GPU optimization"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        
    def initialize(self):
        """Load SAM3 model (call once and cache)"""
        if self.model is None:
            print("Loading SAM3 model...")
            self.model = build_sam3_image_model()
            
            # Explicitly move to CUDA if available
            if torch.cuda.is_available():
                print("CUDA is available. moving model to GPU...")
                self.model = self.model.cuda()
            else:
                print("CUDA not found. Keeping model on CPU.")
                
            self.processor = Sam3Processor(self.model)
            print(f"Model loaded on {self.model.device}")
        return self.model, self.processor
    
    def annotate_single_image(self, image_path: str, text_prompts: List[str], 
                             score_threshold: float = 0.3) -> Dict:
        """
        Annotate a single image with text prompts
        
        Returns:
            {
                'detections': [
                    {
                        'class_id': int,
                        'class_name': str,
                        'box_xyxy': [x1, y1, x2, y2],
                        'box_yolo': [cx, cy, w, h],  # normalized
                        'score': float
                    }
                ],
                'annotated_image': np.ndarray  # BGR image with boxes drawn
            }
        """
        # Ensure model is loaded
        self.initialize()
        
        # Load image
        pil_image = Image.open(image_path).convert("RGB")
        cv_image = cv2.imread(image_path)
        h_orig, w_orig = cv_image.shape[:2]
        
        # Initialize inference state ONCE
        inference_state = self.processor.set_image(pil_image)
        
        all_detections = []
        
        # Process each prompt
        for class_id, prompt in enumerate(text_prompts):
            # Reset prompts for new query (important!)
            self.processor.reset_all_prompts(inference_state)
            
            # Run inference - set_text_prompt returns the output directly
            output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)
            
            # Extract results
            masks = output["masks"]
            boxes = output["boxes"]
            scores = output["scores"]
            
            # Filter by score threshold
            if len(scores) > 0:
                valid_idx = scores > score_threshold
                boxes_filtered = boxes[valid_idx].cpu().numpy() if hasattr(boxes, 'cpu') else boxes[valid_idx]
                scores_filtered = scores[valid_idx].cpu().numpy() if hasattr(scores, 'cpu') else scores[valid_idx]
            else:
                boxes_filtered = np.array([])
                scores_filtered = np.array([])
            
            num_detections = len(scores_filtered) if hasattr(scores_filtered, '__len__') else 0
            
            # Convert to detections
            for i in range(num_detections):
                box_xyxy = boxes_filtered[i].tolist() if len(boxes_filtered) > 0 else None
                score = float(scores_filtered[i]) if len(scores_filtered) > 0 else 0.0
                
                if box_xyxy is not None:
                    # Convert to YOLO format (normalized)
                    x1, y1, x2, y2 = box_xyxy
                    bw = (x2 - x1) / w_orig
                    bh = (y2 - y1) / h_orig
                    bx = (x1 + x2) / 2 / w_orig
                    by = (y1 + y2) / 2 / h_orig
                    
                    all_detections.append({
                        'class_id': class_id,
                        'class_name': prompt,
                        'box_xyxy': box_xyxy,
                        'box_yolo': [bx, by, bw, bh],
                        'score': score
                    })
        
        # Draw bounding boxes on image
        annotated_image = self.draw_detections(cv_image.copy(), all_detections)
        
        return {
            'detections': all_detections,
            'annotated_image': annotated_image,
            'original_image': cv_image
        }
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict], 
                       show_labels: bool = True) -> np.ndarray:
        """Draw bounding boxes on image"""
        # Color palette for different classes
        colors = [
            (0, 0, 255),    # Red
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
            (128, 0, 255),  # Purple
            (0, 128, 255),  # Orange
        ]
        
        for det in detections:
            box = det['box_xyxy']
            class_id = det['class_id']
            class_name = det['class_name']
            score = det['score']
            
            color = colors[class_id % len(colors)]
            
            # Draw rectangle
            cv2.rectangle(image, 
                         (int(box[0]), int(box[1])), 
                         (int(box[2]), int(box[3])), 
                         color, 2)
            
            # Draw label
            if show_labels:
                label = f"{class_name} {score:.2f}"
                cv2.putText(image, label, 
                           (int(box[0]), int(box[1]) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image
    
    def filter_by_confidence(self, detections: List[Dict], 
                            class_thresholds: Dict[str, float]) -> List[Dict]:
        """
        Filter detections based on per-class confidence thresholds
        
        Args:
            detections: List of detection dicts
            class_thresholds: {'class_name': threshold, ...}
        
        Returns:
            Filtered list of detections
        """
        filtered = []
        for det in detections:
            class_name = det['class_name']
            threshold = class_thresholds.get(class_name, 0.3)
            if det['score'] >= threshold:
                filtered.append(det)
        return filtered


def annotate_batch(frame_paths: List[str], text_prompts: List[str], 
                   score_threshold: float = 0.3, 
                   progress_callback=None,
                   batch_size: int = None,
                   num_models: int = None) -> Dict[str, Dict]:
    """
    Annotate multiple frames with multi-model parallel processing
    
    Args:
        frame_paths: List of image paths
        text_prompts: List of text prompts
        score_threshold: Minimum confidence score
        progress_callback: Function to call with progress updates
        batch_size: Number of images to process in parallel (auto-detect if None)
        num_models: Number of SAM3 model instances to spawn (auto-detect if None)
    
    Returns:
        {frame_path: annotation_result, ...}
    """
    import torch
    
    # Get GPU memory if available
    gpu_mem_gb = 0
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    # Auto-detect number of models based on GPU VRAM
    if num_models is None:
        if torch.cuda.is_available():
            if gpu_mem_gb >= 40:  # A100 40GB - spawn 2 models
                num_models = 2
            elif gpu_mem_gb >= 24:  # RTX 3090/4090 - can fit 2 models
                num_models = 2
            else:
                num_models = 1
        else:
            num_models = 1
    
    # Auto-detect batch size based on GPU VRAM
    if batch_size is None:
        if torch.cuda.is_available():
            if gpu_mem_gb >= 32:  # A100 40GB or similar
                batch_size = 16  # Doubled from 8 to fully utilize 40GB VRAM
            elif gpu_mem_gb >= 16:  # V100 or similar
                batch_size = 8
            elif gpu_mem_gb >= 8:
                batch_size = 4
            else:
                batch_size = 2
        else:
            batch_size = 1  # CPU fallback
    
    print(f"🚀 Spawning {num_models} SAM3 model(s) for parallel processing")
    print(f"Using batch size: {batch_size} (GPU VRAM: {gpu_mem_gb:.1f}GB)" if torch.cuda.is_available() else "Using CPU (batch_size=1)")
    
    results = {}
    total = len(frame_paths)
    
    if num_models > 1:
        # Multi-model parallel processing with ProcessPoolExecutor
        print(f"Using ProcessPoolExecutor with {num_models} workers")
        results = _process_with_multiple_models(
            frame_paths, text_prompts, score_threshold, 
            num_models, batch_size, progress_callback
        )
    else:
        # Single model processing (original)
        annotator = SAM3Annotator()
        annotator.initialize()
        
        # Process in batches
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_frames = frame_paths[batch_start:batch_end]
            
            # Process batch in parallel (if batch_size > 1)
            if batch_size > 1 and torch.cuda.is_available():
                # GPU batch processing within single model context
                batch_results = _process_batch_parallel(
                    annotator, 
                    batch_frames, 
                    text_prompts, 
                    score_threshold
                )
            else:
                # Sequential processing (CPU or batch_size=1)
                batch_results = {}
                for frame_path in batch_frames:
                    result = annotator.annotate_single_image(frame_path, text_prompts, score_threshold)
                    batch_results[frame_path] = result
            
            results.update(batch_results)
            
            if progress_callback:
                progress_callback(batch_end, total)
    
    return results


def _worker_process_chunk(chunk_frames: List[str], text_prompts: List[str], 
                          score_threshold: float) -> Dict[str, Dict]:
    """
    Worker function for multiprocessing. 
    Must be at module level to be picklable.
    """
    # Create a fresh annotator instance in this process
    try:
        annotator = SAM3Annotator()
        annotator.initialize()
        
        chunk_results = {}
        for frame_path in chunk_frames:
            try:
                result = annotator.annotate_single_image(frame_path, text_prompts, score_threshold)
                chunk_results[frame_path] = result
            except Exception as e:
                print(f"Error processing {frame_path}: {e}")
                chunk_results[frame_path] = {
                    'detections': [],
                    'annotated_image': None,
                    'original_image': None
                }
        return chunk_results
    except Exception as e:
        print(f"Critical worker error: {e}")
        return {}


def _process_with_multiple_models(frame_paths: List[str], text_prompts: List[str],
                                  score_threshold: float, num_models: int,
                                  batch_size: int, progress_callback) -> Dict[str, Dict]:
    """
    Process frames using multiple SAM3 model instances in separate PROCESSES.
    This guarantees separate VRAM validation and true parallelism.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    
    # Split frames across models
    chunk_size = len(frame_paths) // num_models
    model_chunks = []
    
    # Handle remainder frames
    remainder = len(frame_paths) % num_models
    
    start_idx = 0
    for i in range(num_models):
        # Distribute remainder frames one by one
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        
        if start_idx < len(frame_paths):
            model_chunks.append(frame_paths[start_idx:end_idx])
        start_idx = end_idx
    
    all_results = {}
    processed_count = 0
    
    # specialized start method for CUDA
    ctx = multiprocessing.get_context('spawn')
    
    print(f"🚀 Starting {len(model_chunks)} worker processes...")
    
    with ProcessPoolExecutor(max_workers=num_models, mp_context=ctx) as executor:
        futures = [
            executor.submit(_worker_process_chunk, chunk, text_prompts, score_threshold) 
            for chunk in model_chunks
        ]
        
        for future in as_completed(futures):
            try:
                chunk_results = future.result()
                all_results.update(chunk_results)
                processed_count += len(chunk_results)
                
                if progress_callback:
                    progress_callback(processed_count, len(frame_paths))
            except Exception as e:
                print(f"Error in process chunk: {e}")
                import traceback
                traceback.print_exc()
    
    return all_results


def _process_batch_parallel(annotator: SAM3Annotator, 
                            frame_paths: List[str],
                            text_prompts: List[str],
                            score_threshold: float) -> Dict[str, Dict]:
    """
    Process a batch of images in parallel on GPU (Single Model context)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    batch_results = {}
    
    max_workers = len(frame_paths) * 2 if len(frame_paths) < 8 else 16
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(
                annotator.annotate_single_image, 
                frame_path, 
                text_prompts, 
                score_threshold
            ): frame_path 
            for frame_path in frame_paths
        }
        
        for future in as_completed(future_to_path):
            frame_path = future_to_path[future]
            try:
                result = future.result()
                batch_results[frame_path] = result
            except Exception as e:
                print(f"Error processing {frame_path}: {e}")
                batch_results[frame_path] = {
                    'detections': [],
                    'annotated_image': None,
                    'original_image': None
                }
    
    return batch_results
