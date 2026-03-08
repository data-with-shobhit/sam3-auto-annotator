# SAM3 Auto-Annotator Architecture

## System Overview
The SAM3 Auto-Annotator is an end-to-end computer vision pipeline designed to replace manual bounding-box annotation with Zero-Shot natural language prompting via Meta's Segment Anything 3 (SAM3) model. 

The entire system is orchestrated via a Streamlit frontend, while heavy processing (SAM3 inference and YOLO training) is offloaded to parallel GPU-bound background subprocesses.

---

## 🏗️ Core Architecture Components

### 1. The Presentation Layer (Frontend)
- **Framework**: Streamlit (`app.py`)
- **State Management**: Heavy reliance on `st.session_state` to persist projects, Extracted Frames, Prompts, and loaded managers across page reruns.
- **Routing**: Sidebar navigation extracting UI logic into modular components (`ui/pages/` and `ui/components/`).

### 2. The Management Layer (Business Logic)
Located in `managers/`, these classes act as the central brain orchestrating data flow between the UI and the Data Layer.
- **ProjectManager**: Handles CRUD operations and JSON persistence for user projects (`projects/{project_name}/config.json`).
- **VideoManager**: Utilizes OpenCV for fast, memory-safe frame extraction directly to disk, avoiding high-RAM Numpy caching.
- **AnnotationManager**: The core SAM3 wrapper. Maps text prompts to bounding boxes. Spawns multi-process GPU workers via `ProcessPoolExecutor(mp_context=multiprocessing.get_context('spawn'))`.
- **AugmentationManager**: Uses Albumentations to offline-scale datasets (Nx multiplier) across all splits (brightness, flip, noise, CLAHE) while automatically recalculating YOLO format bounding boxes.
- **DatasetManager**: Handles PyTorch/YOLO/RT-DETR formatted directory structuring (Train/Valid/Test splits) and `data.yaml` generation.

### 3. The Execution Layer (Background Workers)
Heavy GPU operations are strictly decoupled from the main Streamlit process to prevent UI freezing and CUDA initialization crashes.
- **`scripts/annotate_worker.py`**: A headless script invoked by the `AnnotationManager`. It receives a chunk of images (via JSON), loads a dedicated SAM3 model instance into GPU VRAM, processes the images in pure PyTorch/CUDA, and saves the resulting annotations to disk.
- **`scripts/train_worker.py`**: A headless Ultralytics training script. Writes metrics to a `results.csv` file which the Streamlit `train_page.py` reads via a `while` loop to render real-time charts.

---

## 💾 Data & Storage Architecture

```
Project_Sam/
├── projects/
│   └── {project_name}/
│       ├── config.json               # State & Prompt definitions
│       ├── temp/
│       │   ├── uploaded_videos/      # Raw MP4s
│       │   └── extracted_frames/     # Extracted JPGs
│       ├── annotations/              # Raw SAM3 outputs (JSON)
│       ├── dataset_augmented/        # Processed Albumentation duplicates
│       └── dataset/                  # YOLO-ready Export
│           ├── data.yaml
│           ├── train/
│           ├── valid/
│           └── test/
```

### Memory Optimization Design
- **IPC (Inter-Process Communication) Minimization**: Frames are NOT passed as Numpy arrays from the Main Process to the Background Workers. Instead, physical file paths are passed via lightweight JSON.
- **Disk-Caching**: Annotations are streamed to disk instead of being held in RAM. 

---

## ⚡ Inference & Parallelization Model
1. The user selects $N$ frames to annotate.
2. `AnnotationManager` queries system hardware to evaluate available GPU VRAM.
3. The workload is chunked into $W$ batches (where $W=$ number of safe parallel workers).
4. $W$ distinct `annotate_worker.py` subprocesses are spawned using Python's `subprocess.Popen` or `ProcessPoolExecutor`.
5. Each worker loads SAM3 independently.
6. Results are merged upon worker completion.
