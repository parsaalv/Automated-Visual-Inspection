# Robotic Vision & Simulation Project (Digital Twin Framework)

Welcome to the **Robotic Vision & Simulation Project**! This repository hosts a complete Digital Twin framework integrating rigid-body physics simulation (PyBullet) with state-of-the-art Deep Learning models for real-time industrial component inspection and defect detection.

For a comprehensive technical deep-dive into the theoretical and engineering aspects of this project, please refer to the detailed documentation located at:
📄 **[Docs/RV-E/main.tex](Docs/RV-E/main.tex)**

---

## 🚀 Quick Start (Simulation)

To launch the simulation environment with all necessary dependencies and AI models automatically handled, simply run the root batch script:

▶️ **[`run_sim.bat`](run_sim.bat)**

This script will seamlessly set up a virtual environment, install all requirements, and download the pre-trained weights from Google Drive. 
*If you encounter any issues during startup (such as missing C++ Build Tools for PyBullet), please read the [Simulator README](Simulator/README.md) for troubleshooting.*

---

## 🧠 AI Architecture: YOLOv8 & PatchCore

Our computer vision pipeline utilizes a **two-stage inspection architecture** to ensure robust anomaly detection while maintaining real-time execution speeds suitable for industrial conveyor belts.

### 1. Object Localization (YOLOv8)
Before inspecting for defects, the system must accurately locate the component on the moving conveyor. We utilize **YOLOv8n**, a lightweight and highly efficient object detection network trained on a custom, Roboflow-refined dataset. YOLO extracts the exact region of interest (ROI) and crops the component out of the noisy background.

### 2. Anomaly Detection & Segmentation (PatchCore)
For identifying microscopic surface defects, scratches, and deformations, we employ the **PatchCore** algorithm. Instead of analyzing raw images, PatchCore uses pre-trained Convolutional Neural Networks (CNNs) to extract deep spatial features, storing nominal patterns in a memory bank using Coreset Subsampling.

**Why ResNet-18 for the Simulator?**
While our extensive training experiments evaluated advanced, high-capacity models (like Wide ResNet-50) for extreme precision, **we selected the lightweight ResNet-18 backbone for the PyBullet Simulator.** 
ResNet-18 strikes the perfect engineering balance—it is deep enough to capture complex textures, yet fast enough to guarantee real-time Frame Per Second (FPS) inference rates without blocking the physics engine's main loop.

---

## 📓 Model Training & Notebook Environment

If you want to view, retrain, or experiment with the model architectures directly, the complete training pipeline is available in the Jupyter Notebook located in the `Model/` directory.

We have automated the environment setup so you **do not** need to manually execute the `!pip install` cells inside the notebook again. 

To easily set up your notebook environment, simply run:
▶️ **[`Model/run_model_setup.bat`](Model/run_model_setup.bat)**

For more details on running the notebook and managing its dependencies, please read the **[Model README](Model/README.md)**.
