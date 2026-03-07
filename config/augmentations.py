"""
Augmentation Configuration Data
"""

AUGMENTATION_INFO = {
    "hsv_h": {
        "label": "HSV-Hue",
        "default": 0.015,
        "min": 0.0,
        "max": 1.0,
        "desc": "Adjusts image hue (color fraction). Helps model verify objects in different lighting/color conditions. Higher = more color variation."
    },
    "hsv_s": {
        "label": "HSV-Saturation",
        "default": 0.7,
        "min": 0.0,
        "max": 1.0,
        "desc": "Adjusts color intensity. Useful if objects appear in dull vs vibrant environments. 0 = grayscale, 1 = super vivid."
    },
    "hsv_v": {
        "label": "HSV-Value",
        "default": 0.4,
        "min": 0.0,
        "max": 1.0,
        "desc": "Adjusts brightness. Critical for detecting objects in shadows or bright sunlight. Higher = more brightness variation."
    },
    "degrees": {
        "label": "Rotation (°)",
        "default": 0.0,
        "min": 0.0,
        "max": 180.0,
        "desc": "Rotates image randomly. Essential if objects can appear at any angle (e.g. aerial views). Keep 0 for upright objects (e.g. pedestrians)."
    },
    "translate": {
        "label": "Translate",
        "default": 0.1,
        "min": 0.0,
        "max": 1.0,
        "desc": "Shifts image horizontally/vertically. Helps handling off-center objects or partial occlusions. 0.1 means +/- 10% shift."
    },
    "scale": {
        "label": "Scale",
        "default": 0.5,
        "min": 0.0,
        "max": 1.0,
        "desc": "Zooms in/out. Critical for detecting objects at different distances. 0.5 means image can be 50% smaller or larger."
    },
    "flipud": {
        "label": "Flip Up-Down",
        "default": 0.0,
        "min": 0.0,
        "max": 1.0,
        "desc": "Flips image vertically. Enabling is good for satellite/microscope data. BAD for real-world upright objects like cars/people."
    },
    "fliplr": {
        "label": "Flip Left-Right",
        "default": 0.5,
        "min": 0.0,
        "max": 1.0,
        "desc": "Flips image horizontally. Almost always good (e.g. a car facing left is same as right). Set 0.5 (50% chance)."
    },
    "mosaic": {
        "label": "Mosaic",
        "default": 1.0,
        "min": 0.0,
        "max": 1.0,
        "desc": "Combines 4 images into 1. The 'secret sauce' of YOLO. forces model to handle complex scenes and small objects. Keep at 1.0 usually."
    },
    "mixup": {
        "label": "MixUp",
        "default": 0.0,
        "min": 0.0,
        "max": 1.0,
        "desc": "Blends 2 images together transparently. Good for reducing overfitting on large datasets. Usually 0.0-0.2 is enough."
    }
}
