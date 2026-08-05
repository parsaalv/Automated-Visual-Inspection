# Model Training & Inference Architecture

This directory houses the machine learning ecosystem, dataset processing pipelines, and deep neural network training logs. The primary codebase for data generation, YOLOv8 training, and PatchCore inference tuning is contained within:

📓 **`MVTecAD_Project.ipynb`**

---

## 🧠 Model Overview

The training pipeline builds a **two-stage industrial inspection system** on top of the **MVTec AD** dataset, restricted to six component classes: `bottle`, `cable`, `metal_nut`, `screw`, `toothbrush`, and `transistor`.

### Stage 1 — Dataset Engineering
- The MVTec AD dataset is downloaded via the Kaggle API and filtered down to the six selected classes to reduce processing overhead.
- All random operations (weight init, shuffling, augmentation, etc.) are locked to a fixed seed (`seed=42`) via a `set_seed()` utility and deterministic CuDNN settings, ensuring fully reproducible experiments.
- The **transistor** class masks originally suffered from structural annotation errors (merged/overlapping defect regions). To fix this, raw transistor images were exported to **Roboflow**, where the **SAM 3** foundation model was used for automated polygon segmentation (Auto Label), which was then converted to standard YOLO bounding boxes and merged back into the main dataset.

### Stage 2 — Object Localization (YOLOv8n)
- Bounding boxes for all classes are **auto-generated** using a classical CV pipeline (Canny edge detection → morphological dilation → contour extraction → bounding rectangle fitting), avoiding costly manual annotation.
- A **YOLOv8n** (Nano) model is fine-tuned from COCO pre-trained weights for **30 epochs** at `imgsz=640`, using physically-informed augmentations (full 180° rotation, horizontal/vertical flips, HSV jitter, and Mosaic augmentation to compensate for class imbalance — e.g., only 48 `toothbrush` training images).
- **Result:** ~0.995 mAP50 and 0.879 mAP50-95 across all six classes, with 100% recall — critical for industrial QC, where a missed defect is unacceptable.

### Stage 3 — Anomaly Detection (PatchCore)
- Detected parts are cropped (with a safety padding margin) from the raw frame, and their corresponding ground-truth masks are cropped using **identical coordinates** to guarantee pixel-level alignment.
- The **PatchCore** algorithm is trained per-class, exclusively on nominal (defect-free) images, using deep feature extraction and **Coreset Subsampling** to build a compact memory bank of "normal" patterns.
- Two backbone configurations were evaluated:
  - **ResNet-18** (`coreset_sampling_ratio=0.1`) — lightweight, fast, ideal for real-time edge inference.
  - **Wide ResNet-50-2** (`coreset_sampling_ratio=0.15`) — higher representational capacity, better accuracy in low-data or visually complex classes (e.g., `toothbrush`, `cable`, `transistor`), at a significantly higher computational cost.
- Models are evaluated using **Image/Pixel AUROC**, **Image F1-Score**, and **Pixel Dice (Sørensen–Dice coefficient)**, achieving an overall mean IoU of ~85% (ResNet-18) on bounding-box-cropped anomaly localization.
- Trained models are exported to **TorchScript (`.pt`)** format, bundling weights, architecture, and pre-processing into a single deployable artifact.

> ⚡ **Which model powers the Simulator?**
> The digital twin simulation (`Simulator/`) uses the **lightweight configuration**: **YOLOv8n** for part localization and **PatchCore with a ResNet-18 backbone** for anomaly detection. This combination was selected because it delivers real-time inference speed (sub-6ms per frame) suitable for continuous conveyor-belt operation, while still maintaining strong accuracy (>97% pixel AUROC across all classes). The higher-capacity Wide ResNet-50-2 models were trained and evaluated for comparison purposes only, and are not used at simulation runtime.

---

## 🛠️ Installation & Running

> ⚠️ **Important:** The notebook (`MVTecAD_Project.ipynb`) provided in this directory is primarily intended to **demonstrate the full training methodology** — dataset curation, auto-annotation, YOLO training, and PatchCore training — rather than to be a plug-and-play script. It was originally developed and executed in **Google Colab**.

To run the notebook cells **as-is** (i.e., downloading the dataset directly via the Kaggle API, as implemented in the code), you will need:

- A **Kaggle account** with a generated **`kaggle.json`** API token (Account → Settings → API → *Create New Token*). The notebook will prompt you to upload this file, which it then places in the `~/.kaggle` directory for authentication.
- A **Hugging Face account** (with an access token), since several cells rely on the `huggingface-hub` / `transformers` ecosystem for model and dataset utilities.

### Local Environment Setup

If you'd like to explore or modify the notebook locally instead of on Colab, you can set up a local environment automatically:

▶️ **[`run_model_setup.bat`](run_model_setup.bat)**

This script will:
1. Create a local Python virtual environment (`.venv`) in the project root.
2. Install all dependencies listed in `requirements.txt`.
3. Register a dedicated Jupyter kernel: **`Python (RV Model Env)`**.
4. Launch the Jupyter Notebook interface.

Once Jupyter opens, load `MVTecAD_Project.ipynb` and make sure the **`Python (RV Model Env)`** kernel is selected before running any cells.

> Note: Because the notebook contains `!pip install` commands designed for cloud environments like Google Colab, you do **not** need to execute those installation cells when running locally — `run_model_setup.bat` already handles all dependencies.

---

## 🔄 Alternative Method: Using Google Drive Instead of Kaggle

If you don't have (or don't want to use) a Kaggle API token, you can substitute the Kaggle download step with your own **Google Drive**:

1. Manually download the MVTec AD dataset (or your own dataset) and upload it to a folder in your Google Drive.
2. In the notebook, use the **Google Drive mounting cell** (`google.colab.drive.mount`) instead of the Kaggle download cell.
3. Update the **`PROJECT_DIR`** (or equivalent dataset path variable) in that cell to point to the folder you created in your Drive.
4. Run the cell — its output should print or evaluate to **`True`**, confirming that Google Drive was mounted successfully and the target path exists (verified internally via `os.path.exists`).
5. Continue running the remaining cells as normal; all downstream processing steps (filtering, auto-annotation, training) work identically regardless of the data source.
