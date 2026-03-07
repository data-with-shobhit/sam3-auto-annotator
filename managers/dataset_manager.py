"""
Dataset Manager - Handles dataset splitting and export
Supports multiple export formats: YOLOv8, YOLOv11, YOLOv12, YOLO26, RT-DETR
"""
import os
import json
import shutil
import random
import yaml
from pathlib import Path
from typing import List, Dict, Tuple
import cv2


class DatasetManager:
    """Manager for dataset operations."""
    
    @staticmethod
    def split_dataset(frame_paths: List[str], train_pct: int, valid_pct: int) -> Dict[str, List[str]]:
        """
        Split frames into train/valid/test sets.
        
        Args:
            frame_paths: List of frame paths
            train_pct: Training percentage (0-100)
            valid_pct: Validation percentage (0-100)
        
        Returns:
            {'train': [...], 'valid': [...], 'test': [...]}
        """
        frames = list(frame_paths)
        random.shuffle(frames)
        
        total = len(frames)
        train_count = int(total * train_pct / 100)
        valid_count = int(total * valid_pct / 100)
        
        return {
            'train': frames[:train_count],
            'valid': frames[train_count:train_count + valid_count],
            'test': frames[train_count + valid_count:]
        }

    @staticmethod
    def save_to_dataset(output_dir: str, annotations: Dict[str, List[Dict]], 
                        splits: Dict[str, List[str]], class_names: List[str],
                        export_format: str = "YOLOv8"):
        """
        Save annotations to dataset in specified format.
        
        Args:
            output_dir: Output directory
            annotations: {frame_path: [detections]}
            splits: {'train': [...], 'valid': [...], 'test': [...]}
            class_names: List of class names (for YAML)
            export_format: Export format (YOLOv8, YOLOv11, YOLOv12, YOLO26, RT-DETR)
        """
        output_dir = Path(output_dir)
        
        # Create directories
        for split in ['train', 'valid', 'test']:
            (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Process each split
        for split_name, frame_paths in splits.items():
            for frame_path in frame_paths:
                if frame_path not in annotations:
                    continue
                
                # Copy image
                src = Path(frame_path)
                dst_img = output_dir / split_name / 'images' / src.name
                shutil.copy(src, dst_img)
                
                # Create label file
                detections = annotations[frame_path]
                
                if export_format in ["YOLOv8", "YOLOv9", "YOLOv10", "YOLOv11", "YOLO12", "YOLO26"]:
                    # YOLO format: class_id cx cy w h (normalized)
                    DatasetManager._save_yolo_labels(output_dir / split_name / 'labels', src.stem, detections)
                elif export_format == "RT-DETR":
                    # RT-DETR uses same format as YOLO
                    DatasetManager._save_yolo_labels(output_dir / split_name / 'labels', src.stem, detections)
        
        # Create data.yaml
        DatasetManager._create_yaml(output_dir, class_names, export_format)

    @staticmethod
    def _save_yolo_labels(labels_dir: Path, frame_stem: str, detections: List[Dict]):
        """Save detections in YOLO format."""
        label_file = labels_dir / f"{frame_stem}.txt"
        
        with open(label_file, 'w') as f:
            for det in detections:
                class_id = det['class_id']
                box_yolo = det.get('box_yolo', [])
                
                if len(box_yolo) == 4:
                    cx, cy, w, h = box_yolo
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    @staticmethod
    def _create_yaml(output_dir: Path, class_names: List[str], export_format: str):
        """Create data.yaml configuration file."""
        
        yaml_data = {
            'path': str(output_dir.absolute()),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'nc': len(class_names),
            'names': {i: name for i, name in enumerate(class_names)}
        }
        
        # Format-specific additions
        if export_format == "RT-DETR":
            yaml_data['model'] = 'rtdetr-l.pt'
        elif export_format == "YOLOv8":
            yaml_data['model'] = 'yolov8n.pt'
        elif export_format == "YOLOv9":
            yaml_data['model'] = 'yolov9c.pt'
        elif export_format == "YOLOv10":
            yaml_data['model'] = 'yolov10n.pt'
        elif export_format == "YOLOv11":
            yaml_data['model'] = 'yolo11n.pt'
        elif export_format == "YOLO12":
            yaml_data['model'] = 'yolo12n.pt'
        elif export_format == "YOLO26":
            yaml_data['model'] = 'yolo26n.pt'
        
        yaml_file = output_dir / 'data.yaml'
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def get_dataset_stats(dataset_dir: str) -> Dict:
        """Get statistics for a dataset."""
        dataset_dir = Path(dataset_dir)
        
        stats = {
            'train_images': 0,
            'valid_images': 0,
            'test_images': 0,
            'total_labels': 0
        }
        
        for split in ['train', 'valid', 'test']:
            images_dir = dataset_dir / split / 'images'
            labels_dir = dataset_dir / split / 'labels'
            
            if images_dir.exists():
                stats[f'{split}_images'] = len(list(images_dir.glob('*')))
            
            if labels_dir.exists():
                for label_file in labels_dir.glob('*.txt'):
                    with open(label_file) as f:
                        stats['total_labels'] += len(f.readlines())
        
        return stats

    @staticmethod
    def get_class_distribution(dataset_dir: str, class_names: List[str]) -> Dict[str, Dict[str, int]]:
        """Get class distribution per split."""
        dataset_dir = Path(dataset_dir)
        
        distribution = {split: {name: 0 for name in class_names} for split in ['train', 'valid', 'test']}
        
        for split in ['train', 'valid', 'test']:
            labels_dir = dataset_dir / split / 'labels'
            
            if not labels_dir.exists():
                continue
            
            for label_file in labels_dir.glob('*.txt'):
                with open(label_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            if class_id < len(class_names):
                                distribution[split][class_names[class_id]] += 1
        
        return distribution

def save_project_config(project_dir: str, config: Dict):
    """Save project configuration."""
    config_file = Path(project_dir) / 'config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

def load_project_config(project_dir: str) -> Dict:
    """Load project configuration."""
    config_file = Path(project_dir) / 'config.json'
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return {}



