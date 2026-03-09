# SAM3 Auto-Annotation & Training Pipeline

> An end-to-end, zero-manual-annotation pipeline that uses **Meta's SAM3 (Segment Anything 3)** foundation model to auto-annotate video/image datasets via natural language prompts, then trains and exports production-ready YOLO/RT-DETR object detection models — all through an interactive Streamlit web UI.

---

## The Problem

Traditional object detection workflows require **thousands of hours of manual bounding-box annotation**. Labeling 10,000 frames by hand takes ~40 hours and costs $5,000–$50,000 on platforms like Scale AI or Labelbox.

**This pipeline replaces manual annotation with natural language.** Instead of drawing boxes, you type: *"yellow excavator on a construction site"* — and SAM3 auto-detects and labels every instance across thousands of frames in minutes.

### How It Impacts the Business
- **Accelerated Time-to-Market**: Slashes the data preparation phase from weeks to hours, allowing rapid deployment of computer vision models.
- **Massive Cost Reduction**: Eliminates the need for expensive third-party manual labeling services ($5k–$50k per dataset) and endless SaaS subscription fees.
- **Data Sovereignty & Security**: Keeps highly proprietary business data (e.g., manufacturing lines, defense footage, medical imaging) completely on-premise, bypassing the security risks of uploading to external SaaS platforms.
- **Agile Model Iteration**: Allows for same-day retraining on edge cases. If a deployed model fails in a new environment, new data can be collected, auto-annotated, and a fixed model can be deployed within hours, not weeks.
- **Retail & Quick Commerce Scaling (e.g., Swiggy Instamart, Zomato, Blinkit)**: Radically speeds up product cataloging and visual auditing. Instead of manually annotating thousands of new SKUs every week, teams can use zero-shot prompts to detect brand-new packaging designs instantly on warehouse shelves.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Streamlit Web UI (app.py)                  │
├──────┬──────────┬───────────┬──────────┬─────────┬───────────┤
│Upload│ Prompts  │Sample Test│Full Annot│Organize │  Train    │
│Video │ (NLP)    │(4 frames) │(Parallel)│(Split)  │(Background│
│Image │          │           │          │         │  + Export) │
└──┬───┴────┬─────┴─────┬─────┴────┬─────┴────┬────┴─────┬────┘
   │        │           │          │          │          │
   ▼        ▼           ▼          ▼          ▼          ▼
 Video    Project    SAM3 Model  Parallel   Dataset   YOLO/DETR
 Manager  Manager   (Foundation) Workers   Manager   Training
                    on GPU       (Multi-   (YOLO     (Background
                                 Process)   Format)   Subprocess)
```
---

## Project Screenshots

<p align="center">
  <img src="https://raw.githubusercontent.com/data-with-shobhit/sam3-auto-annotator/main/assets/img1.jpg" width="30%">
  <img src="https://raw.githubusercontent.com/data-with-shobhit/sam3-auto-annotator/main/assets/img2.jpg" width="30%">
  <img src="https://raw.githubusercontent.com/data-with-shobhit/sam3-auto-annotator/main/assets/img3.jpg" width="30%">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/data-with-shobhit/sam3-auto-annotator/main/assets/img4.jpg" width="30%">
  <img src="https://raw.githubusercontent.com/data-with-shobhit/sam3-auto-annotator/main/assets/img5.jpg" width="30%">
  <img src="https://raw.githubusercontent.com/data-with-shobhit/sam3-auto-annotator/main/assets/img6.jpg" width="30%">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/data-with-shobhit/sam3-auto-annotator/main/assets/img7.jpg" width="30%">
  <img src="https://raw.githubusercontent.com/data-with-shobhit/sam3-auto-annotator/main/assets/img8.jpg" width="30%">
  <img src="https://raw.githubusercontent.com/data-with-shobhit/sam3-auto-annotator/main/assets/img9.jpg" width="30%">
</p>


---

## Pipeline Steps

| Step | Page | Description |
|------|------|-------------|
| **1** | Upload & Extract | Upload videos or images; extract frames with uniform/sequential sampling |
| **2** | Configure Prompts | Define detection classes via natural language (e.g., *"red hard hat"* → class: `hard_hat`) |
| **3** | Sample Test | Run SAM3 on 4 sample frames to validate prompts & tune confidence thresholds |
| **4** | Full Annotation | Auto-annotate all frames using parallel multi-process GPU workers |
| **5** | Organize Dataset | Split into train/valid/test, export as YOLO or RT-DETR format |
| **6** | Summary | Dataset statistics and class distribution review |
| **7** | Train Model | Background YOLO training with live dashboard, data scaling, and ONNX/TensorRT export |

---

## Key Technical Highlights

### Foundation Model Integration (SAM3)
- Integrated **Meta's SAM3** — a state-of-the-art vision foundation model for text-prompted zero-shot object detection
- Secure gated model access via HuggingFace Hub with `.env`-based token management

### Parallel GPU Processing
- **Multi-process annotation** using `ProcessPoolExecutor` with CUDA `spawn` context
- Each worker spawns its own SAM3 model instance for true parallelism
- **Memory-optimized IPC**: strips full numpy images from pickle files to prevent 100GB+ RAM usage

### Smart Prompt-to-Label Mapping
- Decouples natural language prompts from class labels (prompt: *"a yellow excavator with a boom arm"* → label: `excavator`)
- **Prompt fingerprinting** auto-detects prompt changes and invalidates stale predictions
- Per-video, per-class confidence thresholds with real-time visual feedback

### Offline Data Augmentation Engine
- Albumentations-based pipeline: rotate, flip, brightness, noise, blur, CLAHE
- **Nx dataset scaling** (2x–10x) across all splits with proper YOLO bbox coordinate transformations

### Background Training & Export
- Non-blocking subprocess training with **live metrics dashboard** (mAP, precision, recall, loss curves)
- Supports **7 model families**: YOLOv8, YOLOv9, YOLOv10, YOLOv11, YOLO12, YOLO26, RT-DETR
- One-click export to **ONNX** and **TensorRT** (FP16/FP32, static/dynamic shapes)
- Resume training from checkpoints after interruptions

### Production Engineering
- Modular manager architecture (Project, Video, Annotation, Augmentation, Dataset)
- Session state persistence — configs, annotations, and selections survive page refreshes
- GPU-aware auto-configuration — batch size and worker count tuned based on detected VRAM
- Graceful error handling with SIGTERM handlers and zombie process detection

---

## Project Structure

```
Project_Sam/
├── app.py                      # Main Streamlit entry point
├── config/
│   ├── settings.py             # Paths, defaults, supported formats
│   ├── logging_config.py       # Structured logging setup
│   └── augmentations.py        # Augmentation parameter definitions
├── managers/
│   ├── project_manager.py      # Project CRUD, config persistence
│   ├── video_manager.py        # Video info extraction, frame sampling
│   ├── annotation_manager.py   # SAM3 inference wrapper + parallel processing
│   ├── dataset_manager.py      # YOLO/RT-DETR dataset generation
│   └── augmentation_manager.py # Offline data scaling with Albumentations
├── ui/pages/
│   ├── upload_page.py          # Upload videos/images, extract frames
│   ├── prompts_page.py         # Configure text prompts and class names
│   ├── sample_test_page.py     # Quick 4-frame test per video
│   ├── annotate_page.py        # Full parallel annotation with preview
│   ├── organize_page.py        # Train/valid/test split and export
│   ├── summary_page.py         # Dataset stats and data.yaml download
│   └── train_page.py           # Training, monitoring, export, and resume
├── scripts/
│   ├── annotate_worker.py      # Subprocess worker for parallel annotation
│   └── train_worker.py         # Subprocess worker for background training
├── utils/
│   ├── file_utils.py           # File system helpers
│   ├── image_utils.py          # Image processing utilities
│   └── log_parser.py           # Training log/CSV parsing
├── sam3/                       # SAM3 model source
├── base_models/                # Pretrained model weights
├── projects/                   # User project data (auto-created)
└── logs/                       # Application log files (auto-created)
```

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Frontend** | Streamlit |
| **Foundation Model** | Meta SAM3 (Segment Anything 3) |
| **Object Detection** | Ultralytics (YOLOv8–YOLO26, RT-DETR) |
| **Deep Learning** | PyTorch, CUDA, TensorRT, ONNX Runtime |
| **Computer Vision** | OpenCV, Pillow, Albumentations |
| **Infrastructure** | HuggingFace Hub, python-dotenv, uv |

---

## Quick Start

```bash
# 1. Clone and enter project
git clone https://github.com/YOUR_USERNAME/sam3-auto-annotator.git
cd sam3-auto-annotator

# 2. Create .env file with your HuggingFace token and Log Level
echo -e "HF_TOKEN=hf_your_token_here\nLOG_LEVEL=INFO" > .env
# 3. Install dependencies
uv sync
# or: pip install -r requirements.txt

# 4. Run the app
uv run streamlit run app.py --server.headless true
```

## Configuration

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | HuggingFace access token for SAM3 model download |
| `LOG_LEVEL` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Settings (`config/settings.py`)

- `PROJECTS_DIR` — Base directory for user projects
- `DEFAULT_CONFIDENCE` — Default detection confidence threshold
- `IMAGES_PER_PAGE` — Pagination size for annotation preview

---

## Impact

| Metric | Manual Workflow | This Pipeline |
|--------|----------------|---------------|
| **10K frames annotation** | ~40 hours | ~1 hour (auto-labelled) |
| **10K frames training** | — | ~1 hour on A100 |
| **Annotation cost** | $5K–$50K | $0 |
| **Manual labels needed** | Thousands | Zero |
| **Video → Deployed model** | Days–Weeks | ~2 hours |

### Cost Comparison vs SaaS Platforms

Unlike Roboflow ($400/month subscription — even months you don't train), this pipeline runs on **pay-per-use cloud GPUs**:

| GPU | Cost (₹/hr) | Full Pipeline (2 hrs) | Providers |
|-----|------------|----------------------|-----------|
| **NVIDIA L4** | ₹40–50/hr | **₹80–100** (~$1) | Jarvislabs, Lambda, Vast.ai |
| **NVIDIA A100** | ₹180–200/hr | **₹360–400** (~$4.50) | Jarvislabs, Lambda, RunPod, AWS |

> **Bottom line**: A complete 10K-image annotation + training run costs **₹100–400** on-demand. No subscriptions, no idle costs — you only pay when you actually train.

### 🔒 Data Privacy & Full Control

Most importantly — **your data never leaves your system**. Unlike SaaS platforms where your proprietary images/videos are uploaded to third-party servers:

- All annotation, training, and export happens **locally on your GPU**
- No data is sent to any external API — the SAM3 model runs **entirely on-device**
- You own and control the full pipeline: data, models, weights, and exports
- Ideal for **sensitive/proprietary datasets** (medical, defense, industrial) where data sovereignty matters

### Model Export Formats

After training, models can be exported in multiple formats optimized for different deployment scenarios:

| Format | Extension | Execution Backend | Speed | Best For |
|--------|-----------|-------------------|-------|----------|
| **PyTorch** | `.pt` | GPU CUDA Cores | Baseline | Research, fine-tuning, prototyping |
| **ONNX** | `.onnx` | ONNX Runtime + CUDA | ~2x faster | Cross-platform deployment, edge devices |
| **TensorRT** | `.engine` | NVIDIA Tensor Cores | ~5–10x faster | Maximum inference speed, production deployment |

- **`.pt`** — Native PyTorch weights. Uses standard GPU CUDA cores. Best for continued training and experimentation.
- **`.onnx`** — Open Neural Network Exchange format. Runs on ONNX Runtime with CUDA acceleration. Portable across frameworks and hardware.
- **`.engine`** — NVIDIA TensorRT compiled engine. Leverages dedicated **Tensor Cores** for FP16/INT8 inference. Delivers the highest possible throughput — **deployment-ready** for real-time production systems.

**Static vs Dynamic Export:**

| Mode | Description | Use Case |
|------|-------------|----------|
| **Static** | Fixed input shape (e.g., `640x640`). Maximum speed — TensorRT fully optimizes every layer for the exact shape. | Fixed-resolution cameras, production pipelines |
| **Dynamic** | Flexible input shapes (e.g., `1-8 × 3 × 480-1280 × 480-1280`). Slightly slower but handles variable resolutions. | Multi-camera setups, varying input sources |

**Throughput Benchmarks (640×640, FP16, batch=1):**

| Model         | GPU  | `.pt` (PyTorch) | `.onnx` (ONNX RT) | `.engine` (TensorRT) |
| ------------- | ---- | --------------- | ----------------- | -------------------- |
| **YOLOv11x**  | L4   | ~30–35 FPS      | ~65–70 FPS        | ~150–180 FPS         |
| **YOLOv11x**  | A100 | ~80–85 FPS      | ~150–160 FPS      | ~400–450 FPS         |
| **RT-DETR-x** | L4   | ~15–20 FPS      | ~40–45 FPS        | ~100–120 FPS         |
| **RT-DETR-x** | A100 | ~45–50 FPS      | ~100–110 FPS      | ~280–300 FPS         |

---

## 🚧 Roadmap & Future Development

This project is under **active development** with ongoing bug fixes and feature additions:

- [ ] **Click-to-Mask Annotation** — Leverage SAM3's interactive segmentation mode for precise mask generation via point clicks
- [ ] **Improved Memory & GPU Management** — Smarter VRAM allocation, automatic model offloading, and multi-GPU load balancing
- [ ] **Enhanced Batch Processing** — Optimized batched inference for faster annotation on large-scale datasets
- [ ] **Model Pruning & Retraining** — Structured pruning to reduce model size and inference latency while maintaining accuracy
- [ ] **Active Learning Loop** — Flag low-confidence detections for human review instead of discarding
- [ ] **SAM3 Video Tracking** — Temporal consistency across video frames using SAM3's video mode
- [ ] **Evaluation Dashboard** — Confusion matrices, per-class AP curves, and model comparison tools
- [ ] **Next.js Frontend Migration** — Replace Streamlit with a full Next.js + FastAPI stack for a production-grade UI with better state management, real-time WebSocket updates, and deployment flexibility
- [ ] **Layer 2: CLIP — Label Correction / Niche Annotation**
  - **Take cropped images** from SAM3 bounding boxes.
  - **Define candidate labels** (e.g. *"Kurkure Solid Masti"*, *"Kurkure Tangy Tomato"*, *"Lays Chips"*).
  - **Use CLIP similarity matching** to compare each crop to candidate text labels and assign the highest probability label.
  - **Update Annotations** by replacing generic SAM3 labels with high-precision, niche variant labels.
  - **Purpose**: High precision differentiation of visually similar products via zero-shot classification, requiring zero extra training.

---

## 🛠️ Built With AI

This project was built using AI coding tools — and I'm transparent about it.

| Role | Who |
|------|-----|
| **Planning & Architecture** | Me — system design, model selection, pipeline architecture, prompt engineering |
| **Execution Plan & Debugging** | Me — defining the workflow, diagnosing GPU/memory issues, making engineering trade-offs |
| **Code Generation** | AI Tools — Antigravity IDE, Google Gemini, ChatGPT, Claude |

> AI tools are power tools — they write the code, but **I designed the system, made the engineering decisions, and debugged every production issue.** A nail gun doesn't make you an architect.

---

## 🤝 Contributing

Open for **suggestions and contributions**! Whether it's a bug fix, new feature, or performance improvement — all contributions are welcome.

---

## 📬 Contact

**Shobhit Mohadikar**

- 💼 LinkedIn: [linkedin.com/in/shobhit-mohadikar](https://www.linkedin.com/in/shobhit-mohadikar/)
- 📧 Email: [shobhitmohadikar@gmail.com](mailto:shobhitmohadikar@gmail.com)
