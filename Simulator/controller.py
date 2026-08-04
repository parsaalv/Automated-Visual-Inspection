"""Central coordination logic: links camera/vision checkpoints to robot dispatch."""

import cv2
import numpy as np

from logging_config import logger
from robot import RobotState
from vision import VisionClass


class CentralController:
    """Coordinates checkpoint events, vision inference, and robot dispatch.

    Two checkpoints matter here: the camera checkpoint (where a frame is
    captured and classified) and the pick checkpoint (where, based on the
    cached classification, the robot arm is dispatched to remove
    defective parts into the repair or scrap bucket).
    """

    CAMERA_CHECKPOINT = "Camera_Zone"

    def __init__(self, config, object_manager, robot, vision_module, camera=None, conveyor=None):
        self.config = config
        self.object_manager = object_manager
        self.robot = robot
        self.vision = vision_module
        self.camera = camera
        self.conveyor = conveyor
        self.checkpoint_for_pick = "Robot_Pick_Zone"
        self.decisions = {}
        self.dispatched = set()

    def _capture_and_infer(self, obj_id, obj_label, current_step):
        """Capture a camera frame, run vision inference, cache the decision, and optionally display it.

        Returns
        -------
        The resolved class_id for the object.
        """
        frame = self.camera.get_frame() if self.camera is not None else None
        logger.info(f"[Camera] Shot captured for '{obj_label}' (ID {obj_id}) at step {current_step}")
        result = self.vision.infer(obj_id, obj_label, frame=frame)
        self.decisions[obj_label] = result["class_id"]
        logger.info(
            f"[Controller] {obj_label} (ID {obj_id}): class={result['class_id']} (conf={result['confidence']:.3f})"
        )
        # Real-time display of the captured shot with a green bounding box for the
        # detected object (YOLO) and red contours for any detected defect (Anomalib).
        if self.config.CAMERA_LIVE_VIEW and frame is not None:
            display_frame = frame.copy()
            # 1. Green box for the object detected by YOLO
            yolo_bbox = result.get("yolo_bbox")
            if yolo_bbox is not None:
                x1, y1, x2, y2 = yolo_bbox
                yolo_cls = result.get("yolo_class", "Object")
                yolo_conf = result.get("yolo_conf", 0.0)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{yolo_cls} ({yolo_conf:.2f})", (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 2. Red contours/curves around the defect area found by Anomalib
            defect_contours = result.get("defect_contours_frame")
            if defect_contours and len(defect_contours) > 0:
                cv2.drawContours(display_frame, defect_contours, -1, (0, 0, 255), 2)
                all_pts = np.vstack(defect_contours)
                min_x, min_y = np.min(all_pts[:, 0, :], axis=0)
                pct = result.get("defect_percentage", 0.0)
                cv2.putText(display_frame, f"DEFECT ({pct:.1f}%)", (min_x, max(15, min_y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow("Camera - Last Detection Shot", display_frame)
            cv2.waitKey(1)
        return self.decisions[obj_label]

    def process_events(self, events, current_step):
        """Handle checkpoint-crossing events: trigger vision inference and robot dispatch.

        At the camera checkpoint, a frame is captured and classified (if
        not already done for that object). At the pick checkpoint, the
        cached classification determines whether the robot arm is
        dispatched to the repair or scrap bucket; GOOD parts are left on
        the conveyor.
        """
        for event in events:
            obj_id = event["object_id"]
            checkpoint = event["checkpoint"]
            if obj_id not in self.object_manager.objects:
                continue
            obj = self.object_manager.objects[obj_id]
            obj_label = obj.label
            # Camera stage: capture a frame and classify before the sorting gates
            if checkpoint == self.CAMERA_CHECKPOINT:
                if obj_label not in self.decisions:
                    self._capture_and_infer(obj_id, obj_label, current_step)
                continue

            if checkpoint == self.checkpoint_for_pick:
                class_id = self.decisions.get(obj_label)
                if class_id is None:
                    class_id = self._capture_and_infer(obj_id, obj_label, current_step)
                if class_id == VisionClass.GOOD.value:
                    continue
                if obj_label in self.dispatched:
                    continue

                drop_pos = self.config.ROBOT_REPAIR_DROP_POS if class_id == VisionClass.REPAIR.value else self.config.ROBOT_SCRAP_DROP_POS

                if self.robot.dispatch(obj_id, obj_label, drop_pos):
                    self.dispatched.add(obj_label)
                    print(f"[Controller] Robot dispatched for {obj_label}")
                    if self.conveyor:
                        self.conveyor.set_speed(0.0)
                        print(f"[Controller] Conveyor paused for picking.")
                else:
                    print(f"[Controller] Robot is busy, ignored {obj_label}")

    def update_robot(self, current_step):
        """Advance the robot by one step and resume the conveyor once it returns to idle."""
        self.robot.step()
        if self.robot.state == RobotState.IDLE and self.conveyor and self.conveyor.current_speed == 0.0:
            self.conveyor.set_speed(self.config.CONVEYOR_SPEED_DEFAULT)
            print("[Controller] Robot returned to IDLE, Conveyor resumed.")

    def reset(self):
        """Clear cached vision decisions and dispatch history."""
        self.decisions = {}
        self.dispatched = set()
