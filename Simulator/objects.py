"""Object lifecycle management: spawning, tracking, and checkpoint events."""

import os
from enum import Enum

import cv2
import numpy as np
import pybullet as p

from urdf_utils import create_textured_plane_obj


def create_dummy_texture(filename, text, bg_color):
    """Generate a simple placeholder texture image with a label and border, and save it to disk.

    Used only for the mock vision scenarios (GOOD/REPAIR/SCRAP) where no
    real part photographs are available.
    """
    img = np.ones((256, 256, 3), dtype=np.uint8); img[:] = bg_color
    font = cv2.FONT_HERSHEY_SIMPLEX
    ts = cv2.getTextSize(text, font, 1, 2)[0]
    cv2.putText(img, text, ((256-ts[0])//2, (256+ts[1])//2), font, 1, (0,0,0), 2)
    cv2.rectangle(img, (5, 5), (250, 250), (50, 50, 50), 4)
    cv2.imwrite(filename, img)
    return filename


class ObjectState(Enum):
    """Lifecycle states of a tracked part moving through the simulation."""
    SPAWNED = "SPAWNED"
    ON_CONVEYOR = "ON_CONVEYOR"
    REMOVED = "REMOVED"
    FELL_OFF = "FELL_OFF"


class TrackedObject:
    """Lightweight wrapper around a PyBullet body representing one tracked part."""

    def __init__(self, client_id, obj_id, obj_type, label=None):
        self.client_id = client_id
        self.object_id = obj_id
        self.object_type = obj_type
        self.label = label if label is not None else str(obj_id)
        self.state = ObjectState.SPAWNED
        self.last_position = self.position

    @property
    def position(self):
        """Current world-space position of the object, read live from PyBullet."""
        pos, _ = p.getBasePositionAndOrientation(self.object_id, physicsClientId=self.client_id)
        return pos

    @property
    def orientation(self):
        """Current world-space orientation (quaternion) of the object, read live from PyBullet."""
        _, ori = p.getBasePositionAndOrientation(self.object_id, physicsClientId=self.client_id)
        return ori


class ObjectManager:
    """Spawns, tracks, and removes parts on the conveyor, and raises checkpoint-crossing events."""

    def __init__(self, config, client_id):
        self.config = config
        self.client_id = client_id
        self.objects = {}
        self.checkpoints = {}
        self.spawn_counter = 0

    def add_checkpoint(self, name, x_position):
        """Register a named X-axis checkpoint that objects will be detected crossing."""
        self.checkpoints[name] = x_position
        print(f"[ObjManager] Added Checkpoint '{name}' at X={x_position}")

    def spawn_object(self, obj_type, image_path, start_pos):
        """Spawn a new textured part body on the conveyor.

        Builds a simple box collision shape paired with a custom textured
        quad mesh (see :func:`create_textured_plane_obj`) so the supplied
        image is mapped onto the part without distortion.

        Parameters
        ----------
        obj_type : str
            Category/type label for the object (e.g. a YOLO class name).
        image_path : str
            Path to the texture image to apply to the part's top surface.
        start_pos : list[float]
            Initial [x, y, z] spawn position.

        Returns
        -------
        TrackedObject
            The newly created tracked object wrapper.
        """
        if not os.path.exists(image_path):
            # Previously, if the image path was wrong, p.loadTexture would
            # usually return an empty/white texture without raising an
            # error, and the problem would only surface much later (and
            # much less clearly) on the part's surface. Now we stop here
            # immediately with a clear error instead.
            raise FileNotFoundError(
                f"[ObjManager] Image file not found for '{obj_type}': {image_path}"
            )
        extents = [self.config.TOOL_BASE_SIZE/2, self.config.TOOL_BASE_SIZE/2, self.config.TOOL_THICKNESS/2]
        tex_id = p.loadTexture(image_path, physicsClientId=self.client_id)
        # Collision shape remains a simple box (sufficient for physics/contact purposes)
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=extents, physicsClientId=self.client_id)
        # Visual shape: instead of the default GEOM_BOX, build a custom quad with exact
        # 0..1 UV mapping so the real image is not cropped/zoomed and sits exactly on
        # the part's top face.
        obj_filename = f"_part_visual_{self.spawn_counter}.obj"
        obj_file = create_textured_plane_obj(
            filename=obj_filename,
            size_x=self.config.TOOL_BASE_SIZE,
            size_y=self.config.TOOL_BASE_SIZE,
            z_offset=self.config.TOOL_THICKNESS / 2.0  # On the box's top surface
        )
        vis = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=obj_file,
            rgbaColor=[1, 1, 1, 1],
            physicsClientId=self.client_id
        )
        if os.path.exists(obj_file):
            os.remove(obj_file)
        body_id = p.createMultiBody(
            baseMass=self.config.TOOL_MASS,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=start_pos,
            physicsClientId=self.client_id
        )
        p.changeVisualShape(body_id, -1, textureUniqueId=tex_id, physicsClientId=self.client_id)
        p.changeDynamics(body_id, -1, lateralFriction=0.8, physicsClientId=self.client_id)
        self.spawn_counter += 1
        label = f"object_{self.spawn_counter:03d}"
        tracked_obj = TrackedObject(self.client_id, body_id, obj_type, label)
        self.objects[body_id] = tracked_obj
        print(f"[ObjManager] Spawned '{obj_type}' (label={label}) with ID {body_id} at {start_pos}")
        return tracked_obj

    def remove_object(self, obj_id):
        """Remove a tracked object's PyBullet body and stop tracking it."""
        if obj_id in self.objects:
            obj = self.objects[obj_id]
            obj.state = ObjectState.REMOVED
            p.removeBody(obj_id, physicsClientId=self.client_id)
            del self.objects[obj_id]
            print(f"[ObjManager] Removed object ID {obj_id}")

    def reset_all(self):
        """Remove every currently tracked object and reset the spawn counter."""
        for obj_id in list(self.objects.keys()):
            self.remove_object(obj_id)
        self.spawn_counter = 0
        print("[ObjManager] System Reset: All objects cleared.")

    def update_tracking(self):
        """Update tracked-object positions/states for one step and report checkpoint crossings.

        Detects objects that have fallen off the conveyor (z below -1.0)
        and removes them automatically. For objects still on the
        conveyor, checks whether their X position has just crossed any
        registered checkpoint since the last update.

        Returns
        -------
        list[dict]
            One event dict per checkpoint crossing detected this step,
            each containing ``object_id``, ``object_type``, and
            ``checkpoint``.
        """
        events = []
        ids_to_remove = []
        for obj_id, obj in self.objects.items():
            curr_pos = obj.position
            if curr_pos[2] < -1.0:
                obj.state = ObjectState.FELL_OFF
                ids_to_remove.append(obj_id)
                continue
            else:
                obj.state = ObjectState.ON_CONVEYOR
            for cp_name, cp_x in self.checkpoints.items():
                if obj.last_position[0] < cp_x and curr_pos[0] >= cp_x:
                    events.append({
                        "object_id": obj_id,
                        "object_type": obj.object_type,
                        "checkpoint": cp_name
                    })
            obj.last_position = curr_pos
        for dropped_id in ids_to_remove:
            print(f"[ObjManager] Auto-removing dropped object ID {dropped_id}")
            self.remove_object(dropped_id)
        return events
