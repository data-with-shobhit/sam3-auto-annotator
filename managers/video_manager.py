"""
Video Manager - Handles video processing and frame extraction
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.logging_config import setup_logging

log = setup_logging("video_manager")

class VideoManager:
    """Handles video info extraction and frame sampling."""
    
    @staticmethod
    def get_video_info(video_path: str) -> Dict:
        """Get video metadata."""
        cap = cv2.VideoCapture(video_path)
        
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': 0
        }
        
        if info['fps'] > 0:
            info['duration'] = info['total_frames'] / info['fps']
        
        cap.release()
        log.debug(f"Video info for {video_path}: {info}")
        return info
    
    @staticmethod
    def extract_frames(
        video_path: str,
        output_dir: str,
        num_frames: int = 100,
        method: str = 'uniform'
    ) -> List[str]:
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            num_frames: Number of frames to extract
            method: 'uniform' for evenly spaced, 'sequential' for first N
        
        Returns:
            List of extracted frame paths
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine frame indices
        num_frames = min(num_frames, total_frames)
        
        if method == 'uniform':
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:  # sequential
            indices = list(range(num_frames))
        
        extracted = []
        video_name = Path(video_path).stem
        output_dir = Path(output_dir)
        
        log.info(f"Extracting {num_frames} frames from {video_path} using {method} method")
        
        # Sort indices to ensure sequential access
        indices = sorted(list(set(indices))) # unique and sorted
        indices_set = set(indices)
        max_idx = indices[-1]
        
        import gc
        from concurrent.futures import ThreadPoolExecutor
        
        # Thread pool for writing images to disk asynchronously
        writer_pool = ThreadPoolExecutor(max_workers=4)
        futures = []
        
        frame_idx = 0
        extracted_count = 0
        
        while cap.isOpened():
            # Optimization: If we passed the last frame we need, stop
            if frame_idx > max_idx:
                break
                
            # Efficiently read frames
            # grab() is fast (no decoding), retrieve() decodes only when needed
            if frame_idx in indices_set:
                ret = cap.grab()
                if not ret:
                    break
                ret, frame = cap.retrieve()
                
                if ret:
                    output_path = output_dir / f"{video_name}_frame_{frame_idx:06d}.jpg"
                    # Submit write task to thread pool
                    futures.append(writer_pool.submit(cv2.imwrite, str(output_path), frame))
                    extracted.append(str(output_path))
                    extracted_count += 1
                    
                    # Periodic memory cleanup every 1000 extracted frames
                    if extracted_count % 1000 == 0:
                        gc.collect()
                        # Wait for pending writes to prevent OOM if disk is slow
                        for f in futures:
                            f.result()
                        futures = []
                        log.info(f"Extracted {extracted_count} frames...")
            else:
                # Skip frame without decoding (fastest)
                ret = cap.grab()
                if not ret:
                    break
            
            frame_idx += 1
            
        # Wait for all writes to finish
        writer_pool.shutdown(wait=True)
        cap.release()
        
        log.info(f"Extracted {len(extracted)} frames to {output_dir}")
        return extracted
    
    @staticmethod
    def get_frame(video_path: str, frame_idx: int) -> np.ndarray:
        """Get a single frame from video."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None
