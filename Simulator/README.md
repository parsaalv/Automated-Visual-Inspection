# Simulator

This directory contains the core PyBullet physics simulation and the real-time Vision Inference Module for the digital twin environment.

## 🚀 How to Run

To run the simulator, you **do not** need to manually install dependencies or fetch weights. Simply execute the batch script provided:

▶️ **[`run_sim.bat`](run_sim.bat)**

### What `run_sim.bat` does automatically:
1. Creates a local Python virtual environment (`.venv`).
2. Installs all required packages defined in `requirements.txt`.
3. Downloads the YOLOv8 object detection model weights (`industrial_parts_det-2`).
4. Downloads the PatchCore anomaly detection model weights (`anomalib_outputs`).
5. Launches `main.py` directly inside the virtual environment.

---

## ⚠️ Troubleshooting PyBullet (C++ Build Tools Error)

PyBullet is a robust physics engine, but installing it on Windows via `pip` occasionally requires compiling native C++ code. If you encounter an error during the `pybullet` installation phase complaining about missing build tools or compilers, follow these steps:

1. **Download Microsoft C++ Build Tools:**
   Navigate to the official Microsoft portal:  
   🔗 [https://visualstudio.microsoft.com/downloads/?q=build+tools](https://visualstudio.microsoft.com/downloads/?q=build+tools)

2. **Install "Desktop development with C++":**
   - Run the downloaded installer.
   - In the workloads tab, check the box for **"Desktop development with C++"**.
   - Ensure the default optional components (like the Windows 10/11 SDK and MSVC compiler) remain checked.
   - Click **Install**.

3. **Rerun the Simulator:**
   Once the installation is complete, simply execute `run_sim.bat` again. PyBullet will now compile and install successfully.
