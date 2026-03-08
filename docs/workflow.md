# End-to-End Pipeline Workflow

This document traces the exact data flow of a user interacting with the SAM3 Auto-Annotator, from zero data to a deployed edge model.

---

## Step 1: Ingestion (`upload_page.py` + `VideoManager`)
1. User uploads a `.mp4` video or a batch of `.jpg`/`.png` images.
2. Data is securely saved to `projects/{name}/temp/uploaded_videos`.
3. `VideoManager.extract_frames()` is invoked. It uses `cv2.VideoCapture` to sample frames (either sequentially or uniformly based on FPS duration).
4. Extracted frames are stored instantly in the `temp/extracted_frames/` directory to preserve RAM.

---

## Step 2: NLP Prompt Mapping (`prompts_page.py`)
1. User enters natural language queries describing what they want to track (e.g., *"white hard hat worn by a construction worker"*).
2. User maps the NLP query to a standardized Class Label (e.g., `hard_hat`).
3. `ProjectManager` saves this mapping to `config.json`.

---

## Step 3: Zero-Shot Validation (`sample_test_page.py`)
1. To ensure the text prompt correctly isolates the target object without hallucinating background noise, a "Trial Test" is run.
2. 4 random frames are selected.
3. `AnnotationManager` runs SAM3 synchronously in the main thread (acceptable due to small batch size).
4. Streamlit renders the resulting Bounding Boxes over the images.
5. User adjusts the Confidence Threshold slider until false positives disappear. This threshold is bound specifically to that prompt and saved to state.

---

## Step 4: Scale Out Annotation (`annotate_page.py` + `annotate_worker.py`)
1. User clicks "Annotate All".
2. `AnnotationManager` splits the total pool of un-annotated frames into equal-sized batches based on CPU/GPU core count.
3. It writes temporary `config_{worker_id}.json` files containing paths to the frames.
4. Python's `multiprocessing` spawns multiple headless `annotate_worker.py` processes. 
5. Subprocesses chunk through images in parallel, utilizing hardware acceleration (CUDA).
6. Streamlit loops, reading the output directories to display a real-time progress bar.

---

## Step 5: Dataset Finalization & Splits (`organize_page.py` + `DatasetManager` / `AugmentationManager`)
1. Raw dictionary annotations are converted into normalized YOLO format: `class_id center_x center_y width height`.
2. `DatasetManager` splits the data randomly into `Train (70%)`, `Validation (20%)`, and `Test (10%)` directories.
3. **Optional**: User triggers Data Scaling (e.g., 3x). `AugmentationManager` passes the Train split through Albumentations (Flip, Rotate, CLAHE) and generates `_aug_1.jpg`, `_aug_2.jpg`, etc., while automatically transforming the bounding box coordinates to match the distorted images.
4. A standard YOLO `data.yaml` is generated.

---

## Step 6: Background Training & Hardware Export (`train_page.py` + `train_worker.py`)
1. User selects a base foundation architecture (YOLOv8, YOLO11, RT-DETR) and sets Epochs/Batch Size.
2. A detached subprocess `train_worker.py` is invoked using `subprocess.Popen`. This completely unblocks the Streamlit UI, allowing the user to close their browser.
3. `YOLO().train()` runs in the background. Every epoch, it writes to `runs/detect/train/results.csv`.
4. If the user opens the UI, `train_page.py` reads that CSV and plots dynamic Line Charts for `loss`, `mAP50`, and `precision`.
5. Upon completion, the model is automatically exported to ONNX (`.onnx`) and highly optimized TensorRT (`.engine`) graphs for immediate deployment.
