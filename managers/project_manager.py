"""
Project Manager - Handles project CRUD operations
Updated to support per-video thresholds and prompt/class separation
"""
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import PROJECTS_DIR
from config.logging_config import setup_logging

log = setup_logging("project_manager")

class ProjectManager:
    """Manages project creation, loading, and configuration."""
    
    def __init__(self, project_name: str = None):
        self.project_name = project_name
        self.project_dir = None
        self.config = {}
        
        if project_name:
            self.project_dir = PROJECTS_DIR / project_name
    
    @staticmethod
    def list_projects() -> List[str]:
        """Get list of existing projects."""
        if not PROJECTS_DIR.exists():
            return []
        # Strip whitespace from names; rename folders with leading/trailing spaces on the fly
        projects = []
        for p in PROJECTS_DIR.iterdir():
            if p.is_dir():
                clean_name = p.name.strip().replace(" ", "_")
                if clean_name != p.name:
                    # Rename the folder to the clean name
                    try:
                        p.rename(p.parent / clean_name)
                        log.info(f"Renamed project folder '{p.name}' -> '{clean_name}'")
                    except Exception as e:
                        log.warning(f"Could not rename '{p.name}': {e}")
                        clean_name = p.name  # fall back to original
                projects.append(clean_name)
        return projects
    
    @staticmethod
    def delete_project(name: str) -> bool:
        """Delete a project and its entire folder."""
        project_path = PROJECTS_DIR / name
        if not project_path.exists():
            log.error(f"Project not found for deletion: {name}")
            return False
        try:
            shutil.rmtree(project_path)
            log.info(f"Deleted project: {name}")
            return True
        except Exception as e:
            log.error(f"Failed to delete project {name}: {e}")
            return False
    
    def create(self, name: str) -> bool:
        """Create a new project with required folders."""
        # Sanitize: strip whitespace, replace spaces with underscores
        name = name.strip().replace(" ", "_")
        self.project_name = name
        self.project_dir = PROJECTS_DIR / name
        
        try:
            # Create folder structure
            (self.project_dir / "temp" / "uploaded_videos").mkdir(parents=True, exist_ok=True)
            (self.project_dir / "temp" / "uploaded_images").mkdir(parents=True, exist_ok=True)
            (self.project_dir / "temp" / "extracted_frames").mkdir(parents=True, exist_ok=True)
            (self.project_dir / "annotations").mkdir(exist_ok=True)
            (self.project_dir / "dataset" / "train" / "images").mkdir(parents=True, exist_ok=True)
            (self.project_dir / "dataset" / "train" / "labels").mkdir(parents=True, exist_ok=True)
            (self.project_dir / "dataset" / "valid" / "images").mkdir(parents=True, exist_ok=True)
            (self.project_dir / "dataset" / "valid" / "labels").mkdir(parents=True, exist_ok=True)
            (self.project_dir / "dataset" / "test" / "images").mkdir(parents=True, exist_ok=True)
            (self.project_dir / "dataset" / "test" / "labels").mkdir(parents=True, exist_ok=True)
            
            # Initialize config with per-video thresholds structure
            self.config = {
                "project_name": name,
                "prompts": [],  # List of {prompt_text, class_name, class_id}
                "video_thresholds": {},  # {video_name: {class_name: threshold}}
                "global_thresholds": {}  # Fallback thresholds
            }
            self.save_config()
            
            log.info(f"Created project: {name}")
            return True
        except Exception as e:
            log.error(f"Failed to create project: {e}")
            return False
    
    def load(self, name: str) -> bool:
        """Load an existing project."""
        # Sanitize name (handles legacy projects with spaces too)
        name = name.strip().replace(" ", "_")
        self.project_name = name
        self.project_dir = PROJECTS_DIR / name
        
        if not self.project_dir.exists():
            log.error(f"Project not found: {name}")
            return False
        
        # Load config
        config_file = self.project_dir / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                self.config = json.load(f)
        else:
            self.config = {
                "project_name": name, 
                "prompts": [], 
                "video_thresholds": {},
                "global_thresholds": {}
            }
        
        log.info(f"Loaded project: {name}")
        return True
    
    def save_config(self):
        """Save project configuration."""
        if not self.project_dir:
            return
        
        config_file = self.project_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        log.debug(f"Saved config for {self.project_name}")
    
    def get_prompts(self) -> List[Dict]:
        """Get list of prompts with class names."""
        return self.config.get("prompts", [])
    
    def set_prompts(self, prompts: List[Dict]):
        """Set prompts with class names."""
        self.config["prompts"] = prompts
        self.save_config()
        log.info(f"Updated prompts: {len(prompts)} items")
    
    def get_video_thresholds(self, video_name: str) -> Dict[str, float]:
        """Get thresholds for a specific video."""
        return self.config.get("video_thresholds", {}).get(video_name, {})
    
    def set_video_thresholds(self, video_name: str, thresholds: Dict[str, float]):
        """Set thresholds for a specific video."""
        if "video_thresholds" not in self.config:
            self.config["video_thresholds"] = {}
        self.config["video_thresholds"][video_name] = thresholds
        self.save_config()
        log.info(f"Updated thresholds for {video_name}")
    
    def get_all_video_names(self) -> List[str]:
        """Get list of all videos with configured thresholds."""
        return list(self.config.get("video_thresholds", {}).keys())
    
    def get_selected_images(self) -> List[str]:
        """Get list of selected image paths."""
        return self.config.get("selected_images", [])

    def set_selected_images(self, images: List[str]):
        """Set list of selected image paths."""
        self.config["selected_images"] = images
        self.save_config()
        log.info(f"Updated selected images: {len(images)} items")

    @property
    def temp_dir(self) -> Path:
        return self.project_dir / "temp"
    
    @property
    def frames_dir(self) -> Path:
        return self.project_dir / "temp" / "extracted_frames"
    
    @property
    def videos_dir(self) -> Path:
        return self.project_dir / "temp" / "uploaded_videos"
    
    @property
    def images_dir(self) -> Path:
        return self.project_dir / "temp" / "uploaded_images"
    
    @property
    def dataset_dir(self) -> Path:
        return self.project_dir / "dataset"
