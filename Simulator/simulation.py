"""Top-level simulation lifecycle owner: connects PyBullet and wires all subsystems together."""

import os
import time

import cv2
import pybullet as p
import pybullet_data

from camera import CameraAPI
from controller import CentralController
from conveyor import ConveyorController
from objects import ObjectManager
from robot import RobotController
from urdf_utils import create_bucket, create_camera_stand_urdf
from vision import MockVisionModule, RealVisionModule


class SimulationManager:
    """Singleton owner of the PyBullet session and every simulation subsystem.

    Responsible for connecting/disconnecting from PyBullet, loading the
    static environment (ground, conveyor, camera stand, robot, buckets),
    and driving the per-step update loop that ties the conveyor, object
    manager, controller, and live camera view together.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.client_id = -1
            cls._instance.is_running = False
        return cls._instance

    def __init__(self, config):
        if not hasattr(self, 'initialized'):
            self.config = config
            self.conveyor = None
            self.robot = None
            self.camera_stand_id = -1
            self.object_manager = None
            self.camera = None
            self.vision = None
            self.controller = None
            self._step_count = 0
            self.initialized = True

    def start(self):
        """Connect to PyBullet (GUI or headless), configure the timestep, and load the environment."""
        self.disconnect()
        mode = p.GUI if self.config.USE_GUI else p.DIRECT
        self.client_id = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(self.config.TIME_STEP, physicsClientId=self.client_id)
        self.setup_gui_camera()
        self.load_environment()
        self.is_running = True
        print("[SimManager] PyBullet Started Successfully.")

    def disconnect(self):
        """Disconnect the current PyBullet session (if any) and close any OpenCV windows."""
        if self.client_id >= 0:
            try:
                p.disconnect(physicsClientId=self.client_id)
            except:
                pass
            self.client_id = -1
            self.is_running = False
            try:
                cv2.destroyAllWindows()
            except:
                pass
            print("[SimManager] Disconnected from PyBullet.")

    def reset(self):
        """Reset the simulation: (re)start PyBullet if disconnected, otherwise reload the environment in place."""
        if self.client_id < 0:
            self.start()
        else:
            p.resetSimulation(physicsClientId=self.client_id)
            self.load_environment()
            print("[SimManager] Simulation Reset.")

    def stop(self):
        """Pause the simulation loop and stop the conveyor."""
        self.is_running = False
        if self.conveyor: self.conveyor.stop()
        print("[SimManager] Simulation Paused.")

    def load_environment(self):
        """Load the ground plane, conveyor, camera stand, robot arm, buckets, and all controllers."""
        p.setGravity(0, 0, self.config.GRAVITY, physicsClientId=self.client_id)
        if self.config.USE_GUI:
            p.resetDebugVisualizerCamera(cameraDistance=4.5, cameraYaw=10, cameraPitch=-45, cameraTargetPosition=[0, 0, 0], physicsClientId=self.client_id)

        ground_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        self.conveyor = ConveyorController(self.config, self.client_id, ground_id=ground_id)

        urdf_file = create_camera_stand_urdf()
        self.camera_stand_id = p.loadURDF(urdf_file, basePosition=[self.config.CAMERA_TARGET_POS[0], -0.5, 0], useFixedBase=True, physicsClientId=self.client_id)
        if os.path.exists(urdf_file): os.remove(urdf_file)

        self.robot = RobotController(self.config, self.client_id)
        self.buckets = {
            "repair": create_bucket(self.client_id, position=[self.config.ROBOT_REPAIR_DROP_POS[0], self.config.ROBOT_REPAIR_DROP_POS[1], 0.01], color=[0.9, 0.75, 0.1, 1], size=(0.6, 0.6, 0.25)),
            "scrap":  create_bucket(self.client_id, position=[self.config.ROBOT_SCRAP_DROP_POS[0], self.config.ROBOT_SCRAP_DROP_POS[1], 0.01], color=[0.85, 0.15, 0.15, 1], size=(0.6, 0.6, 0.25))
        }
        self.object_manager = ObjectManager(self.config, self.client_id)
        self.object_manager.add_checkpoint("Camera_Zone", self.config.CAMERA_TARGET_POS[0])
        self.object_manager.add_checkpoint("Robot_Pick_Zone", self.config.ROBOT_PICK_X)
        self.camera = CameraAPI(self.config, self.client_id)
        if self.config.VISION_MODE == "real":
            self.vision = RealVisionModule(self.config)
            self.vision.preload_all_models()
        else:
            self.vision = MockVisionModule(
                mode=self.config.VISION_MODE,
                scenario=self.config.VISION_SCENARIO,
                seed=self.config.VISION_RANDOM_SEED
            )
        self.controller = CentralController(self.config, self.object_manager, self.robot, self.vision, camera=self.camera, conveyor=self.conveyor)
        self._step_count = 0
        print("[SimManager] Environment & Controllers Loaded.")

    def setup_gui_camera(self):
        """Position the PyBullet debug-view camera when running with a GUI."""
        if self.config.USE_GUI:
            p.resetDebugVisualizerCamera(
                cameraDistance=3.5,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0],
                physicsClientId=self.client_id
            )

    def step(self):
        """Advance the simulation by one physics step and run all per-step subsystem updates.

        Returns
        -------
        list[dict]
            Checkpoint-crossing events produced by the object manager this
            step (empty if the simulation isn't running).
        """
        if not self.is_running or self.client_id < 0:
            return []
        if not p.isConnected(physicsClientId=self.client_id):
            self.is_running = False
            return []
        if self.conveyor:
            self.conveyor.update_physics()
        p.stepSimulation(physicsClientId=self.client_id)
        self._step_count += 1
        events = []
        if self.object_manager:
            events = self.object_manager.update_tracking()
        if self.controller:
            self.controller.process_events(events, self._step_count)
            self.controller.update_robot(self._step_count)
        # Live camera feed display (independent of the inference moment; purely for real-time monitoring)
        if self.config.CAMERA_LIVE_VIEW and self.camera is not None:
            if self._step_count % self.config.CAMERA_LIVE_VIEW_EVERY_N_STEPS == 0:
                live_frame = self.camera.get_frame()
                try:
                    cv2.imshow("Camera - Live Feed", live_frame)
                    cv2.waitKey(1)
                except Exception:
                    pass
        if self.config.USE_GUI:
            time.sleep(self.config.TIME_STEP)
        return events
