"""
File Utilities
"""
from pathlib import Path
import shutil
from typing import List

def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def list_files(directory: Path, extensions: List[str] = None) -> List[Path]:
    """List files in directory, optionally filtered by extension."""
    directory = Path(directory)
    if not directory.exists():
        return []
    
    files = list(directory.iterdir())
    if extensions:
        files = [f for f in files if f.suffix.lower() in extensions]
    
    return sorted(files)

def copy_file(src: Path, dst: Path) -> bool:
    """Copy file from src to dst."""
    try:
        shutil.copy(src, dst)
        return True
    except Exception:
        return False

def safe_delete(path: Path) -> bool:
    """Safely delete file or directory."""
    try:
        path = Path(path)
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        return True
    except Exception:
        return False
