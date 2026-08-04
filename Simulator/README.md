# Simulator

This directory contains the core PyBullet physics simulation and the real-time Vision Inference Module for the digital twin environment.

---

## 🏗️ Architecture Overview

The simulator models a small industrial conveyor line as a **Digital Twin**: a static camera watches parts move down a belt, a two-stage AI vision pipeline (YOLOv8n + PatchCore) inspects each part, and a robotic arm sorts defective components into **Repair** or **Scrap** bins — all inside a PyBullet physics session.

```
main.py  →  simulation.py (SimulationManager)
                 │
                 ├── config.py            (global parameters)
                 ├── logging_config.py    (shared logger)
                 ├── urdf_utils.py        (procedural asset generation)
                 ├── conveyor.py          (ConveyorController)
                 ├── objects.py           (ObjectManager / TrackedObject)
                 ├── camera.py            (CameraAPI)
                 ├── vision.py            (MockVisionModule / RealVisionModule)
                 ├── robot.py             (RobotController, Franka Panda arm)
                 └── controller.py        (CentralController — ties everything together)
```

Below is a breakdown of what each module is responsible for.

### `config.py` — Central Configuration
A single static `Config` class acting as the **single source of truth** for every tunable parameter in the system: physics settings (gravity, timestep, step budget), conveyor geometry and speed, robot base position and drop-off coordinates, camera resolution/FOV/placement, object spawn timing, and the AI vision pipeline's file paths and decision thresholds (`ANOMALY_CONFIDENCE_THRESHOLD`, `DEFECT_AREA_SCRAP_THRESHOLD`). Nearly every other module imports from this file, so it's the first place to look when adjusting simulation behavior.

### `logging_config.py` — Centralized Logging
Configures Python's standard `logging` module exactly once (using a run-once/Singleton pattern) and exposes a shared `logger` under the `"DigitalTwin"` namespace. Every module logs through this single instance, giving consistent, timestamped output instead of scattered `print()` calls.

### `urdf_utils.py` — Procedural Asset Generation
Rather than shipping static 3D asset files, this module generates them **on the fly** and writes them to disk for PyBullet to load:
- `create_camera_stand_urdf` — builds the static camera mount (base, pole, arm, camera body).
- `create_magnetic_tool_urdf` — builds the robot's end-effector attachment used to "grasp" parts.
- `create_bucket` — programmatically constructs an open-top collection bin (used for both the Repair and Scrap stations) directly via low-level PyBullet API calls, with a lowered front lip so parts can enter freely.
- `create_textured_plane_obj` — synthesizes a custom quad mesh with exact 0–1 UV coordinates, so real industrial part photos are mapped onto surfaces without the distortion/cropping that PyBullet's default box texturing would introduce.

### `conveyor.py` — Conveyor Belt Physics
The `ConveyorController` class creates a static (mass-zero) box body representing the belt. Rather than simulating a moving mesh (which would be computationally expensive and unstable), it takes a simpler and more robust approach: on every physics step, it scans for contact points between the belt and dynamic bodies, filters out the ground plane and static geometry, and directly overwrites each contacting object's X-axis velocity to match the belt's effective speed — leaving Y/Z motion and rotation untouched for natural physical settling.

### `objects.py` — Object Lifecycle Management
Implements the full lifecycle of a part moving through the scene:
- **`create_dummy_texture`** — generates simple placeholder images (used only in mock testing modes).
- **`ObjectState`** — an FSM-style enum: `SPAWNED` → `ON_CONVEYOR` → `REMOVED` / `FELL_OFF`.
- **`TrackedObject`** — a thin wrapper around a raw PyBullet body ID that exposes live position/orientation and a human-readable label.
- **`ObjectManager`** — the central engine that spawns new parts (building a box collision shape paired with the custom textured visual mesh from `urdf_utils`), tracks all active objects every step, automatically garbage-collects any part that falls off the line (`Z < -1.0`), and detects when an object crosses a registered **X-axis checkpoint** (e.g., the camera zone or the robot pick zone), emitting an event for the controller to react to.

### `camera.py` — Overhead Camera Abstraction
The `CameraAPI` class models a fixed overhead camera perpendicular to the conveyor. It precomputes and caches the PyBullet view/projection matrices at startup (based on `config.py` parameters), then exposes two capture methods:
- **`get_frame()`** — renders a frame and returns it as a standard OpenCV-compatible **BGR** NumPy array (with automatic hardware/software renderer selection depending on whether a GUI is active).
- **`get_depth()`** — returns the raw Z-buffer as a 2D array, available for future depth-based processing.

### `vision.py` — AI Inspection Pipeline
Contains two interchangeable vision backends, selected via `Config.VISION_MODE`:
- **`MockVisionModule`** — a lightweight stand-in that returns deterministic (`scenario`) or seeded-random classifications without loading any deep learning frameworks. Useful for testing conveyor/robot mechanics in isolation.
- **`RealVisionModule`** — the production pipeline. On first use it preloads the **YOLOv8n** detector and all six per-class **PatchCore** anomaly models into GPU memory. For each captured frame it:
  1. Runs YOLO to detect and crop the highest-confidence part (with a small safety padding margin).
  2. Feeds the crop into the class-specific PatchCore `TorchInferencer` to obtain an anomaly score and (if above threshold) a binary defect mask.
  3. Converts the mask into contours and a defect-area percentage.
  4. Applies the decision thresholds from `config.py` to output a final `VisionClass` — **GOOD**, **REPAIR**, or **SCRAP**.

  A fail-safe path ensures that any missing frame, failed detection, or missing model defaults the part to `GOOD` rather than halting the line.

### `robot.py` — Robotic Arm Control (Franka Panda)
The `RobotController` class drives a Franka Panda arm through a **Finite State Machine** (`RobotState`: `IDLE → MOVE_TO_PICK → GRASP → LIFT → MOVE_TO_DROP → LOWER_TO_DROP → RELEASE → LIFT_FROM_DROP → RETURN_HOME`). Motion is resolved via PyBullet's built-in **Inverse Kinematics** solver (`move_ik`). Rather than simulating realistic finger-friction grasping (which tends to be unstable in rigid-body engines), the arm "grasps" objects using a **magnetic tool**: a fixed constraint is created between the tool tip and the target object's exact relative transform, guaranteeing zero-slip transport until the constraint is removed at the drop-off point.

### `controller.py` — Central Coordination Logic
The `CentralController` class acts as the **mediator** between all subsystems, keeping the camera, vision pipeline, and robot decoupled from one another. Its responsibilities:
- At the **camera checkpoint**, it captures a frame, runs vision inference, caches the resulting classification per object label, and (if `CAMERA_LIVE_VIEW` is enabled) renders a live OpenCV window with the YOLO bounding box (green) and PatchCore defect contours (red) overlaid.
- At the **pick checkpoint**, it looks up the cached decision: `GOOD` parts are left alone, while `REPAIR`/`SCRAP` parts trigger a robot dispatch to the corresponding bin — and the conveyor is paused (`set_speed(0.0)`) to keep the target part within the robot's reach.
- On every tick, it advances the robot's state machine and automatically **resumes the conveyor** once the robot returns to `IDLE`.

### `simulation.py` — Simulation Lifecycle Owner
The `SimulationManager` (a Singleton) owns the entire PyBullet session lifecycle:
- **`start()` / `disconnect()`** — connect/disconnect from the physics server (GUI or headless `DIRECT` mode) and clean up any OpenCV windows.
- **`load_environment()`** — instantiates the ground plane, conveyor, camera stand, robot, sorting bins, object manager, camera, vision module, and central controller, wiring them all together (dependency injection).
- **`step()`** — the main loop tick: updates conveyor physics, advances PyBullet's rigid-body solver, updates object tracking, feeds checkpoint events to the controller, advances the robot state machine, and (throttled) refreshes the live camera feed window.

### `main.py` — Entry Point
The executable script that ties everything together:
- Forces UTF-8 console output to avoid crashes on non-ASCII log text.
- Builds the list of parts to spawn — either **real photographs** read from the `Simulator/data` directory (in `"real"` vision mode) or **generated placeholder textures** (in mock/`"scenario"` mode).
- Spawns parts onto the conveyor at fixed time intervals (`Config.SPAWN_INTERVAL`) to avoid physical overlap.
- Runs the main simulation loop until every spawned part has exited the line (`FELL_OFF`).
- Guarantees a clean shutdown via a `try/except/finally` block: the PyBullet connection is always closed and any temporary mock texture files are always removed, even on error or manual interruption (`Ctrl+C`).

> 💡 **Tip:** To test the system with your own parts, drop images belonging to one of the trained classes (`screw`, `metal_nut`, `transistor`, `cable`, `bottle`, `toothbrush`) into the `Simulator/data` directory and set `Config.VISION_MODE = "real"`. `main.py` will automatically discover and spawn them.

---

## 🚀 How to Run

To run the simulator, you **do not** need to manually install dependencies or fetch weights. Simply execute the batch script provided:

▶️ **[`run_sim.bat`](run_sim.bat)**

### What `run_sim.bat` does automatically:
1. Creates a local Python virtual environment (`.venv`).
2. Installs all required packages defined in `requirements.txt`.
3. Downloads the YOLOv8 object detection model weights (`yolo_model`).
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

