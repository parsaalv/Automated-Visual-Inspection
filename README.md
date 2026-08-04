# Automated Visual Inspection: From Object Detection to Anomaly Segmentation

**Courses:** Robotic & Computer Vision | Introduction to Machine Learning
**Professors:** Dr. Hamidreza Taghirad, Amirhossein Nikoofard, Mohammad Javad Ahmadi

**Team Members:**
- Mostafa Latifian — 40122193
- Parsa Alavinikoo — 40120993
- Hamidreza Abedini — 40120633

---

## 📖 Project Overview

This repository presents a complete **Digital Twin framework** for automated industrial visual inspection, combining a physics-based simulation environment (PyBullet) with a two-stage deep learning inspection pipeline.

In real production lines, defective components are rare and highly diverse in appearance, making traditional supervised classification impractical. To address this, the project follows a modern **data-centric, unsupervised anomaly detection** approach built on top of the **MVTec AD** dataset, covering six industrial component classes: `bottle`, `cable`, `metal_nut`, `screw`, `toothbrush`, and `transistor`.

The end-to-end pipeline consists of:

1. **Dataset Engineering** — Automated ingestion, filtering, and curation of the MVTec AD dataset, including a Roboflow-based re-annotation pass to fix structurally flawed transistor labels.
2. **Object Localization (YOLOv8n)** — A lightweight object detector trained (via classical CV auto-annotation) to locate and crop industrial parts from the camera frame, isolating them from background noise.
3. **Anomaly Detection & Segmentation (PatchCore)** — A memory-bank-based, unsupervised anomaly detection algorithm trained exclusively on nominal (defect-free) samples, capable of flagging and localizing unseen defect types with pixel-level precision.
4. **Digital Twin Simulation (PyBullet)** — A virtual conveyor-belt factory line where simulated parts are inspected in real time by the trained models, and a robotic arm automatically sorts defective parts into **Repair** or **Scrap** bins based on the inspection result.

For a full technical deep-dive into the dataset curation, training methodology, evaluation metrics, and experimental results, please refer to our detailed reports:

📄 **English Report:** [Docs/EN_report.pdf](Docs/EN_report.pdf)

📄 **Persian Report (گزارش فارسی):** [Docs/FA_report.pdf](Docs/FA_report.pdf)

For component-specific documentation, see:
- 📓 **[Model/README.md](Model/README.md)** — Model training pipeline (YOLOv8n + PatchCore).
- 🤖 **[Simulator/README.md](Simulator/README.md)** — PyBullet digital twin and real-time inference engine.

---

## ⚙️ Setup and Installation

### 1. Get the Repository

You can obtain a local copy of this project in one of two ways:

**Option A — Clone with Git**
```bash
git clone https://github.com/parsaalv/Automated-Visual-Inspection.git
cd Automated-Visual-Inspection
```

**Option B — Download as ZIP**
1. Navigate to the repository page on GitHub.
2. Click the green **Code** button, then select **Download ZIP**.
3. Extract the archive to a folder of your choice.

### 2. Run the Simulator

Once you have the repository locally, simply launch the root batch script:

▶️ **[`run_sim.bat`](run_sim.bat)**

This script automatically creates a virtual environment, installs all dependencies, downloads the pre-trained model weights, and starts the simulation. See **[Simulator/README.md](Simulator/README.md)** for full details and troubleshooting steps.

> 💡 **Testing with Your Own Images**
> Want to try the system with parts other than the default MVTec AD samples? Simply drop your own images (belonging to one of the trained classes: `screw`, `metal_nut`, `transistor`, `cable`, `bottle`, `toothbrush`) into the **`Simulator/data`** directory. When run in `real` vision mode, the simulator will automatically pick up any images placed there and feed them through the YOLO + PatchCore inspection pipeline — no code changes required.

### 3. (Optional) Explore or Retrain the Models

If you'd like to inspect, modify, or retrain the underlying detection and anomaly models, see **[Model/README.md](Model/README.md)** for instructions on setting up the training notebook environment..
