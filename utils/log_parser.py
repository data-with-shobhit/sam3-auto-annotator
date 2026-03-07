"""
Training Log Parser
Parses Ultralytics results.csv for live plotting
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

def parse_training_log(results_csv: Path) -> Optional[pd.DataFrame]:
    """
    Parse YOLO results.csv into a DataFrame suitable for plotting.
    Normalizes column names nicely.
    """
    if not results_csv.exists():
        return None
    
    try:
        # Read CSV (skipping bad lines if any)
        df = pd.read_csv(results_csv)
        
        # Strip whitespace from column names
        df.columns = [c.strip() for c in df.columns]
        
        # Standardize column names for easier plotting
        # YOLOv8 format usually: epoch, train/box_loss, ..., metrics/mAP50(B), ...
        
        return df
    except Exception:
        return None

def get_latest_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Get the latest metrics from the dataframe."""
    if df is None or df.empty:
        return {}
    
    latest = df.iloc[-1]
    
    # Try to find standard metrics (names vary slightly by YOLO version)
    metrics = {}
    
    # mAP50
    map50_col = next((c for c in df.columns if 'map50' in c.lower() and '95' not in c), None)
    if map50_col:
        metrics['mAP50'] = latest[map50_col]
        
    # mAP50-95
    map95_col = next((c for c in df.columns if 'map50-95' in c.lower()), None)
    if map95_col:
        metrics['mAP50-95'] = latest[map95_col]
        
    # Precision
    prec_col = next((c for c in df.columns if 'precision' in c.lower()), None)
    if prec_col:
        metrics['Precision'] = latest[prec_col]
        
    # Recall
    rec_col = next((c for c in df.columns if 'recall' in c.lower()), None)
    if rec_col:
        metrics['Recall'] = latest[rec_col]
        
    return metrics
