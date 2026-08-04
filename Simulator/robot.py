"""Robot arm state machine: pick-and-place logic for the Franka Panda arm."""

from enum import Enum

import pybullet as p

from urdf_utils import create_magnetic_tool_urdf


class RobotState(Enum):
    """Discrete states of the robot arm's pick-and-place cycle."""
    IDLE = 0
    MOVE_TO_PICK = 1
    GRASP = 2
    LIFT = 3
    MOVE_TO_DROP = 4
    LOWER_TO_DROP = 7
    RELEASE = 5
    LIFT_FROM_DROP = 8
    RETURN_HOME = 6


class RobotController:
    """Drives a Franka Panda arm through a pick-lift-drop-return state machine.

    The arm uses inverse kinematics (:func:`move_ik`) to reach target poses
    and a fixed constraint (rather than true gripper friction) to "grasp"
    objects via an attached magnetic tool.
    """

    def __init__(self, config, client_id):
        self.config = config
        self.client_id = client_id
        self.id = p.loadURDF("franka_panda/panda.urdf", basePosition=config.ROBOT_BASE_POS, useFixedBase=True, physicsClientId=self.client_id)
        self.num_joints = p.getNumJoints(self.id, physicsClientId=self.client_id)

        self.ee_link_idx = 11
        self.gripper_indices = [9, 10]

        self.rest_poses = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0, 0.0, 0.0, 0.0, 0.0]
        for i in range(self.num_joints):
            p.resetJointState(self.id, i, self.rest_poses[i] if i < len(self.rest_poses) else 0.0, physicsClientId=self.client_id)
            p.setJointMotorControl2(self.id, i, p.POSITION_CONTROL, targetPosition=self.rest_poses[i] if i < len(self.rest_poses) else 0.0, force=500, physicsClientId=self.client_id)

        self.state = RobotState.IDLE
        self.target_pos = None
        self.target_drop_pos = None
        self.state_step = 0
        self.gripper_target = 0.04
        self.target_obj_id = None
        self.target_obj_label = None
        self.grasp_constraint = None

        tool_urdf = create_magnetic_tool_urdf("magnetic_tool.urdf")
        self.tool_id = p.loadURDF(tool_urdf, useFixedBase=False, physicsClientId=self.client_id)

        p.createConstraint(
            parentBodyUniqueId=self.id,
            parentLinkIndex=self.ee_link_idx,
            childBodyUniqueId=self.tool_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0.04],
            childFramePosition=[0, 0, 0],
            physicsClientId=self.client_id
        )

    def dispatch(self, obj_id, obj_label, drop_pos):
        """Assign the arm a new pick-and-place job, if it is currently idle.

        Parameters
        ----------
        obj_id : int
            PyBullet body ID of the object to pick up.
        obj_label : str
            Human-readable label for the object (used in logging).
        drop_pos : list[float]
            Target [x, y, z] position to drop the object at.

        Returns
        -------
        bool
            True if the job was accepted (arm was idle), False if the arm
            is already busy.
        """
        if self.state != RobotState.IDLE:
            return False
        self.target_obj_id = obj_id
        self.target_obj_label = obj_label
        pos, _ = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.client_id)
        self.target_pos = [pos[0], pos[1], pos[2] + 0.15]
        self.target_drop_pos = drop_pos
        self.state = RobotState.MOVE_TO_PICK
        self.state_step = 0
        self.gripper_target = 0.04
        print(f"[Robot] Dispatched to pick object {obj_label} at {self.target_pos}")
        return True

    def open_gripper(self):
        """Command the gripper fingers to their open position."""
        self.gripper_target = 0.04
        for idx in self.gripper_indices:
            p.setJointMotorControl2(self.id, idx, p.POSITION_CONTROL, targetPosition=self.gripper_target, force=100, physicsClientId=self.client_id)

    def close_gripper(self):
        """Command the gripper fingers to their closed position."""
        self.gripper_target = 0.0
        for idx in self.gripper_indices:
            p.setJointMotorControl2(self.id, idx, p.POSITION_CONTROL, targetPosition=self.gripper_target, force=100, physicsClientId=self.client_id)

    def move_ik(self, pos, ori=None):
        """Move the end-effector toward a target pose using inverse kinematics.

        Parameters
        ----------
        pos : list[float]
            Target [x, y, z] position for the end-effector.
        ori : list[float], optional
            Target orientation as a quaternion. Defaults to pointing
            straight down.
        """
        if ori is None:
            ori = p.getQuaternionFromEuler([3.1415, 0, 0])
        joint_poses = p.calculateInverseKinematics(self.id, self.ee_link_idx, pos, ori, maxNumIterations=100, physicsClientId=self.client_id)
        for i in range(len(joint_poses)):
            if i not in self.gripper_indices:
                p.setJointMotorControl2(self.id, i, p.POSITION_CONTROL, targetPosition=joint_poses[i], force=500, physicsClientId=self.client_id)

    def step(self):
        """Advance the robot's pick-and-place state machine by one physics step."""
        self.open_gripper()

        if self.state == RobotState.IDLE:
            self.move_ik([self.config.ROBOT_BASE_POS[0], self.config.ROBOT_BASE_POS[1] + 0.3, self.config.ROBOT_BASE_POS[2] + 0.5])
            return

        self.state_step += 1

        if self.state == RobotState.MOVE_TO_PICK:
            if self.target_obj_id is not None:
                try:
                    pos, _ = p.getBasePositionAndOrientation(self.target_obj_id, physicsClientId=self.client_id)
                    self.target_pos = [pos[0], pos[1], pos[2] + 0.15]
                except:
                    pass
            self.move_ik(self.target_pos)
            if self.state_step > 120:
                self.state = RobotState.GRASP
                self.state_step = 0

        elif self.state == RobotState.GRASP:
            if self.target_obj_id is not None:
                try:
                    pos, _ = p.getBasePositionAndOrientation(self.target_obj_id, physicsClientId=self.client_id)
                    self.target_pos = [pos[0], pos[1], pos[2] + 0.08]
                except:
                    pass
            self.move_ik(self.target_pos)

            if self.state_step == 60 and self.target_obj_id is not None:
                try:
                    tool_pos, tool_orn = p.getBasePositionAndOrientation(self.tool_id, physicsClientId=self.client_id)
                    obj_pos, obj_orn = p.getBasePositionAndOrientation(self.target_obj_id, physicsClientId=self.client_id)

                    inv_tool_pos, inv_tool_orn = p.invertTransform(tool_pos, tool_orn)
                    _, rel_orn = p.multiplyTransforms(inv_tool_pos, inv_tool_orn, obj_pos, obj_orn)

                    self.grasp_constraint = p.createConstraint(
                        parentBodyUniqueId=self.tool_id,
                        parentLinkIndex=-1,
                        childBodyUniqueId=self.target_obj_id,
                        childLinkIndex=-1,
                        jointType=p.JOINT_FIXED,
                        jointAxis=[0, 0, 0],
                        parentFramePosition=[0, 0, 0.04],  # Tip of the cylinder
                        childFramePosition=[0, 0, 0],
                        parentFrameOrientation=rel_orn,
                        physicsClientId=self.client_id
                    )
                except Exception as e:
                    print(f"[Robot] Constraint error: {e}")

            if self.state_step > 60:
                self.state = RobotState.LIFT
                self.state_step = 0

        elif self.state == RobotState.LIFT:
            lift_pos = [self.target_pos[0], self.target_pos[1], self.target_pos[2] + 0.3]
            self.move_ik(lift_pos)

            if self.state_step > 60:
                self.state = RobotState.MOVE_TO_DROP
                self.state_step = 0

        elif self.state == RobotState.MOVE_TO_DROP:
            # Move ABOVE the bucket first
            above_drop_pos = [self.target_drop_pos[0], self.target_drop_pos[1], self.target_drop_pos[2] + 0.3]
            self.move_ik(above_drop_pos)

            if self.state_step > 100:
                self.state = RobotState.LOWER_TO_DROP
                self.state_step = 0

        elif self.state == RobotState.LOWER_TO_DROP:
            # Slowly go down into the bucket
            self.move_ik(self.target_drop_pos)

            if self.state_step > 60:
                self.state = RobotState.RELEASE
                self.state_step = 0

        elif self.state == RobotState.RELEASE:
            self.move_ik(self.target_drop_pos)

            if self.state_step == 10 and self.grasp_constraint is not None:
                try:
                    p.removeConstraint(self.grasp_constraint, physicsClientId=self.client_id)
                except:
                    pass
                self.grasp_constraint = None

            if self.state_step > 40:
                self.state = RobotState.LIFT_FROM_DROP
                self.state_step = 0

        elif self.state == RobotState.LIFT_FROM_DROP:
            # Go UP after releasing
            above_drop_pos = [self.target_drop_pos[0], self.target_drop_pos[1], self.target_drop_pos[2] + 0.3]
            self.move_ik(above_drop_pos)

            if self.state_step > 60:
                self.state = RobotState.RETURN_HOME
                self.state_step = 0

        elif self.state == RobotState.RETURN_HOME:
            self.move_ik([self.config.ROBOT_BASE_POS[0], self.config.ROBOT_BASE_POS[1] + 0.3, self.config.ROBOT_BASE_POS[2] + 0.5])
            if self.state_step > 80:
                self.state = RobotState.IDLE
                self.state_step = 0
                self.target_obj_id = None
                self.target_obj_label = None

    def reset(self):
        """Return the arm to IDLE and release any active grasp constraint."""
        self.state = RobotState.IDLE
        self.state_step = 0
        self.target_obj_id = None
        self.target_obj_label = None
        if hasattr(self, 'grasp_constraint') and self.grasp_constraint is not None:
            try:
                p.removeConstraint(self.grasp_constraint, physicsClientId=self.client_id)
            except:
                pass
            self.grasp_constraint = None
