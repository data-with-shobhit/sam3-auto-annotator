"""
Augmentation Manager
Handles offline dataset augmentation using Albumentations.
Generates multiplied datasets (2x, 5x, 10x) based on user config.
"""
import albumentations as A
import cv2
import shutil
import yaml
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import sys

# Import logging
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.logging_config import setup_logging

log = setup_logging("aug_manager")

class AugmentationManager:
    """Manages generation of augmented datasets."""
    
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.dataset_dir = project_dir / "dataset"
        self.aug_dataset_dir = project_dir / "dataset_augmented"
        
    def generate_augmented_dataset(
        self,
        scaling_factor: int,
        aug_config: Dict[str, bool],
        progress_callback=None
    ) -> Path:
        """
        Generate a new dataset scaled by factor (e.g., 5x)
        
        Args:
            scaling_factor: Total size multiplier (2 = 1 original + 1 generated)
            aug_config: Dictionary of enabled augmentations
            progress_callback: Function(current, total)
        """
        log.info(f"Starting augmentation x{scaling_factor}")
        
        # 1. Clean/Create Output Directory
        if self.aug_dataset_dir.exists():
            shutil.rmtree(self.aug_dataset_dir)
        self.aug_dataset_dir.mkdir()
        
        # 2. Define Pipeline
        transform = self._build_pipeline(aug_config)
        
        # 3. Process Splits (train, valid, test)
        # Usually we only augment TRAIN. Valid/Test should remain pure for fair evaluation.
        # But we must copy valid/test to the new folder.
        
        splits = ['train', 'valid', 'test']
        total_files = sum(len(list((self.dataset_dir / s / 'images').glob('*'))) for s in splits)
        # We augment train set (N * factor), others are just copied (N)
        # Actually total ops = train_files * factor + valid_files + test_files
        
        processed_count = 0
        
        for split in splits:
            src_img_dir = self.dataset_dir / split / 'images'
            src_lbl_dir = self.dataset_dir / split / 'labels'
            
            dst_img_dir = self.aug_dataset_dir / split / 'images'
            dst_lbl_dir = self.aug_dataset_dir / split / 'labels'
            
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            dst_lbl_dir.mkdir(parents=True, exist_ok=True)
            
            if not src_img_dir.exists():
                continue
                
            frame_files = list(src_img_dir.glob('*'))
            
            for img_path in frame_files:
                # Always copy original first
                shutil.copy(img_path, dst_img_dir / img_path.name)
                
                lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
                if lbl_path.exists():
                    shutil.copy(lbl_path, dst_lbl_dir / lbl_path.name)
                
                # Augment all splits if requested (Train, Valid, Test)
                if scaling_factor > 1:
                    # Load image and labels once
                    image = cv2.imread(str(img_path))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    bboxes, class_ids = self._load_yolo_labels(lbl_path, image.shape)
                    
                    # Generate variants
                    for i in range(scaling_factor - 1):
                        try:
                            # Apply transform
                            transformed = transform(image=image, bboxes=bboxes, class_labels=class_ids)
                            trans_image = transformed['image']
                            trans_bboxes = transformed['bboxes']
                            trans_cls = transformed['class_labels']
                            
                            # Save new file
                            suffix = f"_aug_{i+1}"
                            new_name = f"{img_path.stem}{suffix}.jpg"
                            
                            # Save Image
                            save_img = cv2.cvtColor(trans_image, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(str(dst_img_dir / new_name), save_img)
                            
                            # Save Label
                            self._save_yolo_labels(
                                dst_lbl_dir / f"{img_path.stem}{suffix}.txt",
                                trans_bboxes,
                                trans_cls,
                                image.shape # Normalized requires orig shape? No, pipeline returns normalized usually if configured?
                                # Albumentations BboxParams: normalize requires 0-1.
                                # Let's assume we handle denormalize/normalize logic correctly below.
                            )
                            
                        except Exception as e:
                            log.warning(f"Augmentation failed for {img_path.name}: {e}")
                
                processed_count += 1
                if progress_callback:
                    progress_callback(processed_count) # Just tick steps
                    
        # 4. Copy data.yaml and update paths
        self._update_yaml()
        
        log.info(f"Augmentation complete. Saved to {self.aug_dataset_dir}")
        return self.aug_dataset_dir

    def _build_pipeline(self, config: Dict[str, bool]):
        """Build Albumentations pipeline based on config checkboxes."""
        augs = []
        
        # Mapping config keys to transforms
        if config.get('rotate', False):
            augs.append(A.Rotate(limit=30, p=0.5))
        
        if config.get('horizontal_flip', False):
            augs.append(A.HorizontalFlip(p=0.5))
            
        if config.get('vertical_flip', False):
            augs.append(A.VerticalFlip(p=0.5))
            
        if config.get('brightness', False):
            augs.append(A.RandomBrightnessContrast(p=0.5))
            
        if config.get('noise', False):
            augs.append(A.GaussNoise(p=0.2))
            
        if config.get('blur', False):
            augs.append(A.Blur(blur_limit=3, p=0.2))
            
        if config.get('clahe', False):
            augs.append(A.CLAHE(p=0.2))
            
        # Add BboxParams
        # YOLO format is [x_center, y_center, width, height] normalized
        # Albumentations 'yolo' format expects normalized
        return A.Compose(
            augs,
            bbox_params=A.BboxParams(
                format='yolo', 
                label_fields=['class_labels'],
                min_visibility=0.1
            )
        )

    def _load_yolo_labels(self, label_path: Path, img_shape: Tuple) -> Tuple[List, List]:
        """Load YOLO labels from txt file."""
        bboxes = []
        class_ids = []
        
        if not label_path.exists():
            return [], []
            
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    coords = [float(x) for x in parts[1:5]]
                    
                    # Robust Clamping: Convert to corners -> Clamp -> Convert back
                    # This handles edge cases where cy + h/2 > 1.0 due to float precision
                    cx, cy, w, h = coords
                    
                    x1 = cx - w/2
                    y1 = cy - h/2
                    x2 = cx + w/2
                    y2 = cy + h/2
                    
                    x1 = max(0.0, min(1.0, x1))
                    y1 = max(0.0, min(1.0, y1))
                    x2 = max(0.0, min(1.0, x2))
                    y2 = max(0.0, min(1.0, y2))
                    
                    # New w, h
                    w_new = x2 - x1
                    h_new = y2 - y1
                    cx_new = (x1 + x2) / 2
                    cy_new = (y1 + y2) / 2
                    
                    # Verify box is still valid (has area)
                    if w_new > 0.00001 and h_new > 0.00001:
                        bboxes.append([cx_new, cy_new, w_new, h_new])
                        class_ids.append(cls)
                        
        return bboxes, class_ids

    def _save_yolo_labels(self, path: Path, bboxes, class_ids, shape):
        """Save coordinates back to YOLO format."""
        with open(path, 'w') as f:
            for bbox, cls in zip(bboxes, class_ids):
                # Clamp again just to be safe before writing
                cx, cy, w, h = [max(0.0, min(1.0, x)) for x in bbox]
                f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    def _update_yaml(self):
        """Create a new data.yaml pointing to the augmented dataset."""
        orig_yaml = self.dataset_dir / "data.yaml"
        if not orig_yaml.exists():
            return
            
        with open(orig_yaml) as f:
            data = yaml.safe_load(f)
            
        # Update path
        data['path'] = str(self.aug_dataset_dir.absolute())
        
        with open(self.aug_dataset_dir / "data.yaml", 'w') as f:
            yaml.dump(data, f, sort_keys=False)
