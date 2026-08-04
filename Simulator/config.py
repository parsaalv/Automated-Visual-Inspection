"""Central configuration for the Digital Twin PyBullet simulation.

All tunable constants for the physics simulation, conveyor belt, robot arm,
tooling, camera, object spawning, and the vision pipeline are grouped in a
single :class:`Config` class so the rest of the codebase can be tuned from
one place.
"""

import os


class Config:
    """Static configuration container for the entire simulation.

    Every value below is defined as a class attribute, so it can be
    accessed either directly on the class (``Config.SOME_VALUE``) or on an
    instance (``Config().SOME_VALUE``) with identical behavior.
    """

    # --- Simulation ---
    USE_GUI = True
    TIME_STEP = 1. / 240.
    GRAVITY = -9.81
    SIMULATION_STEPS = 6000   # Length of the demo run (previously 2000; now roughly 3x longer)

    # --- Conveyor ---
    CONVEYOR_SPEED_DEFAULT = 0.5
    CONVEYOR_LENGTH = 8.0
    CONVEYOR_WIDTH  = 0.6
    CONVEYOR_HEIGHT = 0.5
    CONVEYOR_POS    = [0, 0, 0.25]

    # --- Robot Arm ---
    ROBOT_BASE_POS = [0.1, -0.6, CONVEYOR_HEIGHT - 0.25]
    ROBOT_PICK_X = 0.1
    ROBOT_REPAIR_DROP_POS = [-0.4, -0.6, CONVEYOR_HEIGHT + 0.15]
    ROBOT_SCRAP_DROP_POS  = [ 0.6, -0.6, CONVEYOR_HEIGHT + 0.15]

    # --- Tools / Objects ---
    TOOL_THICKNESS = 0.015
    TOOL_BASE_SIZE = 0.15
    TOOL_MASS      = 0.5

    # --- Camera ---
    CAMERA_WIDTH  = 320
    CAMERA_HEIGHT = 320
    CAMERA_FOV    = 35
    CAMERA_Z_OFFSET   = 0.4
    CAMERA_NEAR_VAL   = 0.1
    CAMERA_FAR_VAL    = 5.0
    # Note: the camera checkpoint must be positioned before the sorting gates
    # (Repair/Scrap) so that vision inference completes before the gate
    # decision needs to be made.
    CAMERA_TARGET_POS = [-1.5, 0, CONVEYOR_HEIGHT]

    # --- Live Camera View ---
    CAMERA_LIVE_VIEW               = True   # Show a real-time camera feed window (cv2.imshow)
    CAMERA_LIVE_VIEW_EVERY_N_STEPS = 10      # Capture a live-view frame every N physics steps

    # --- Object Spawning ---
    SPAWN_INTERVAL = 350  # Steps between object spawns (~1.45s apart at 240Hz). Change this to adjust spacing between objects.

    # --- Vision (Mock, used until the real YOLO model is ready) ---
    VISION_MODE          = "real"   # "random", "scenario", or "real"
    VISION_SCENARIO       = {}        # Only used in "scenario" mode; key = object label (e.g. "object_001")
    VISION_RANDOM_SEED    = None

    # --- Real Vision Pipeline (YOLOv8 part detection -> PatchCore defect detection) ---
    YOLO_WEIGHTS_PATH   = os.path.join("yolo_model", "weights", "best.pt")
    # Folder containing the anomaly-detection models (base/ResNet-18 version);
    # expected structure: anomalib_outputs/{class_name}/**/*.pt
    ANOMALY_BASE_DIR    = "anomalib_outputs"
    YOLO_CONF_THRESHOLD = 0.3   # Matches the cropping stage used in the notebook (conf=0.3)
    CROP_PADDING        = 0    # Pixels; matches get_crop_coords in the notebook
    # Decision thresholds applied to PatchCore's pred_score output (range 0 to 1)
    ANOMALY_CONFIDENCE_THRESHOLD = 0.7    # >= this value -> inspect defect area
    DEFECT_AREA_SCRAP_THRESHOLD  = 0.15   # >= this area percentage -> SCRAP, otherwise -> REPAIR
    # Classes that YOLO/Anomaly were trained on (used to preload models at startup)
    SELECTED_CLASSES = ['screw', 'metal_nut', 'transistor', 'cable', 'bottle', 'toothbrush']
