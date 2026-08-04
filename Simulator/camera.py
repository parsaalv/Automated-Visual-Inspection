"""Overhead camera abstraction for capturing color and depth frames."""

import cv2
import numpy as np
import pybullet as p


class CameraAPI:
    """Overhead camera mounted perpendicular to the conveyor surface.

    Returns frames directly as NumPy arrays (BGR) for processing with
    OpenCV/YOLO.
    """

    def __init__(self, config, client_id):
        self.config = config
        self.client_id = client_id
        self.width = config.CAMERA_WIDTH
        self.height = config.CAMERA_HEIGHT
        target = config.CAMERA_TARGET_POS
        self.eye_position = [target[0], target[1], target[2] + config.CAMERA_Z_OFFSET]
        self.target_position = target
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.eye_position,
            cameraTargetPosition=self.target_position,
            cameraUpVector=[0, 1, 0],
            physicsClientId=self.client_id
        )
        aspect = self.width / self.height
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=config.CAMERA_FOV,
            aspect=aspect,
            nearVal=config.CAMERA_NEAR_VAL,
            farVal=config.CAMERA_FAR_VAL,
            physicsClientId=self.client_id
        )
        print(f"[CameraAPI] Initialized. Eye: {self.eye_position} -> Target: {self.target_position}")

    def get_frame(self):
        """Capture one frame and return it as a BGR array (OpenCV-compatible)."""
        # Explicitly choose the renderer: hardware rendering (faster and more
        # stable) is used when running with a GUI.
        renderer = p.ER_BULLET_HARDWARE_OPENGL if self.config.USE_GUI else p.ER_TINY_RENDERER
        img = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=renderer,
            physicsClientId=self.client_id
        )
        rgb_array = np.reshape(img[2], (self.height, self.width, 4))[:, :, :3].astype(np.uint8)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        return bgr_array

    def get_depth(self):
        """Capture one frame and return its raw depth buffer as a 2D array."""
        renderer = p.ER_BULLET_HARDWARE_OPENGL if self.config.USE_GUI else p.ER_TINY_RENDERER
        img = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=renderer,
            physicsClientId=self.client_id
        )
        return np.reshape(img[3], (self.height, self.width))
