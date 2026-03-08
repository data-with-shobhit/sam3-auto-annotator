"""
Application Settings
"""
from pathlib import Path
import os

# Base Paths
BASE_DIR = Path(__file__).parent.parent
SAM3_PATH = BASE_DIR / "sam3"
PROJECTS_DIR = BASE_DIR / "projects"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
PROJECTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# App Defaults
DEFAULT_CONFIDENCE = 0.3
IMAGES_PER_PAGE = 12
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv']
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.webp']

# Dataset Split Defaults
DEFAULT_TRAIN_SPLIT = 70
DEFAULT_VALID_SPLIT = 20
DEFAULT_TEST_SPLIT = 10

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
