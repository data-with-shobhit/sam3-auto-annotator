"""
Image Utilities
"""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple

def resize_for_display(image: np.ndarray, max_width: int = 800) -> np.ndarray:
    """Resize image for UI display."""
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_size = (max_width, int(h * scale))
        return cv2.resize(image, new_size)
    return image

def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert RGB to BGR."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def numpy_to_pil(image: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = bgr_to_rgb(image)
    return Image.fromarray(image)

def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array."""
    return np.array(image)

def draw_box(image: np.ndarray, box: Tuple[int, int, int, int], 
             color: Tuple[int, int, int] = (0, 255, 0), 
             thickness: int = 2) -> np.ndarray:
    """Draw bounding box on image."""
    x1, y1, x2, y2 = [int(v) for v in box]
    return cv2.rectangle(image.copy(), (x1, y1), (x2, y2), color, thickness)

def draw_label(image: np.ndarray, text: str, position: Tuple[int, int],
               color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Draw label text on image."""
    return cv2.putText(image.copy(), text, position, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
