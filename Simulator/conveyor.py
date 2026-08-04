"""Conveyor belt body and physics-driving logic."""

import pybullet as p


class ConveyorController:
    """Creates the conveyor belt's static physics body and drives objects along it.

    The conveyor itself does not move; instead, on every physics step it
    scans for bodies in contact with its top surface and directly sets
    their linear velocity in the belt's travel direction, simulating a
    moving surface.
    """

    def __init__(self, config, client_id, ground_id=None):
        self.config = config
        self.client_id = client_id
        # ID of the ground/plane body that must be excluded from contact
        # filtering. Previously this was hardcoded to 0, which only
        # happened to be correct because plane.urdf is always the first
        # body loaded. It is now passed in explicitly instead.
        self.ground_id = ground_id
        self.is_running = False
        self.current_speed = 0.0
        self.id = self._create_body()
        self.reset()

    def _create_body(self):
        """Create the static (mass=0) box body representing the conveyor belt."""
        extents = [self.config.CONVEYOR_LENGTH/2, self.config.CONVEYOR_WIDTH/2, self.config.CONVEYOR_HEIGHT/2]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=extents, physicsClientId=self.client_id)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=extents, rgbaColor=[0.2, 0.2, 0.2, 1], physicsClientId=self.client_id)
        body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=self.config.CONVEYOR_POS, physicsClientId=self.client_id)
        p.changeDynamics(body, -1, lateralFriction=1.0, physicsClientId=self.client_id)
        return body

    def start(self):
        """Set the conveyor to the running state (its speed becomes effective again)."""
        self.is_running = True
        print("[Conveyor] State: STARTED")

    def stop(self):
        """Stop the conveyor (effective speed becomes 0 regardless of current_speed)."""
        self.is_running = False
        print("[Conveyor] State: STOPPED")

    def set_speed(self, speed):
        """Set the conveyor's target speed in meters/second."""
        self.current_speed = float(speed)
        print(f"[Conveyor] Speed set to: {self.current_speed} m/s")

    def get_speed(self):
        """Return the effective speed: current_speed if running, else 0."""
        return self.current_speed if self.is_running else 0.0

    def reset(self):
        """Reset the conveyor to its default running state and default speed."""
        self.is_running = True
        self.current_speed = self.config.CONVEYOR_SPEED_DEFAULT
        print(f"[Conveyor] RESET to default (Running, Speed: {self.current_speed} m/s)")

    def update_physics(self):
        """Drive every object currently resting on the conveyor at the belt's effective speed.

        Called once per physics step. Inspects all contact points against
        the conveyor body and pushes each contacting dynamic body forward
        along the belt's travel axis, while leaving static bodies (mass 0)
        and the ground plane untouched.
        """
        effective_speed = self.get_speed()
        contacts = p.getContactPoints(bodyA=self.id, physicsClientId=self.client_id)
        for contact in contacts:
            body_b = contact[2]
            if self.ground_id is not None and body_b == self.ground_id:
                continue
            if contact[8] > 0.005:
                continue
            # Do not move static objects (like buckets or pushers) if they accidentally touch the conveyor
            mass = p.getDynamicsInfo(body_b, -1, physicsClientId=self.client_id)[0]
            if mass == 0:
                continue
            lin_vel, ang_vel = p.getBaseVelocity(body_b, physicsClientId=self.client_id)
            target_vel = [effective_speed, lin_vel[1], lin_vel[2]]
            p.resetBaseVelocity(body_b, target_vel, ang_vel, physicsClientId=self.client_id)
