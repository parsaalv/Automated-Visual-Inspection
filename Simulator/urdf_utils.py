"""Procedural asset generation helpers.

This module builds small, disposable URDF and OBJ asset files on disk
(camera stand, magnetic pickup tool, collection buckets, textured quads)
that are loaded into PyBullet at runtime. Generating them procedurally
avoids shipping static asset files alongside the project.
"""

import pybullet as p


def create_camera_stand_urdf(filename="camera_stand.urdf"):
    """Write a simple static camera-stand URDF (base + pole + arm + camera body) to disk.

    Parameters
    ----------
    filename : str
        Path to write the generated URDF file to.

    Returns
    -------
    str
        The filename that was written (same as the input).
    """
    urdf_content = """<?xml version="1.0"?>
    <robot name="camera_stand">
      <link name="base">
        <visual><geometry><box size="0.2 0.2 0.05"/></geometry><material name="gray"><color rgba="0.5 0.5 0.5 1"/></material></visual>
        <collision><geometry><box size="0.2 0.2 0.05"/></geometry></collision>
        <inertial><mass value="0"/><inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial>
      </link>
      <link name="pole">
        <visual><origin xyz="0 0 0.45"/><geometry><cylinder radius="0.02" length="0.9"/></geometry><material name="gray"/></visual>
        <collision><origin xyz="0 0 0.45"/><geometry><cylinder radius="0.02" length="0.9"/></geometry></collision>
      </link>
      <joint name="base_to_pole" type="fixed">
        <parent link="base"/>
        <child link="pole"/>
        <origin xyz="0 0 0"/>
      </joint>
      <link name="arm">
        <visual><origin xyz="0 0.25 0.9"/><geometry><box size="0.04 0.5 0.04"/></geometry><material name="gray"/></visual>
        <collision><origin xyz="0 0.25 0.9"/><geometry><box size="0.04 0.5 0.04"/></geometry></collision>
      </link>
      <joint name="pole_to_arm" type="fixed">
        <parent link="pole"/>
        <child link="arm"/>
        <origin xyz="0 0 0"/>
      </joint>
      <link name="camera_body">
        <visual><origin xyz="0 0 0"/><geometry><box size="0.08 0.08 0.08"/></geometry><material name="black"><color rgba="0.1 0.1 0.1 1"/></material></visual>
      </link>
      <joint name="arm_to_camera" type="fixed">
        <parent link="arm"/>
        <child link="camera_body"/>
        <origin xyz="0 0.5 0.9"/>
      </joint>
    </robot>
    """
    with open(filename, "w") as f:
        f.write(urdf_content)
    return filename


def create_magnetic_tool_urdf(filename):
    """Write a small cylindrical "magnetic tool" URDF (the robot's end-effector attachment) to disk.

    Parameters
    ----------
    filename : str
        Path to write the generated URDF file to.

    Returns
    -------
    str
        The filename that was written (same as the input).
    """
    urdf_content = """<?xml version="1.0"?>
<robot name="magnetic_tool">
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.03" length="0.08"/>
      </geometry>
      <material name="magnet_color">
        <color rgba="0.2 0.2 0.8 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.03" length="0.08"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>
</robot>
"""
    with open(filename, "w") as f:
        f.write(urdf_content)
    return filename


def create_bucket(client_id, position, color, size=(0.6, 0.6, 0.4), wall_thickness=0.02):
    """
    Creates a physical 3D open-top collection bucket in PyBullet.
    Uses baseMass=0 to make it completely fixed so it cannot move or crash the simulation.
    """
    sx, sy, sz = size
    t = wall_thickness

    # Base is the floor
    base_extents = [sx/2.0, sy/2.0, t/2.0]
    col_base = p.createCollisionShape(p.GEOM_BOX, halfExtents=base_extents, physicsClientId=client_id)
    vis_base = p.createVisualShape(p.GEOM_BOX, halfExtents=base_extents, rgbaColor=color, physicsClientId=client_id)

    # Links are the 4 walls
    half_extents_list = [
        [t/2.0, sy/2.0, sz],                         # Left wall (-x) - TALLER
        [t/2.0, sy/2.0, sz],                         # Right wall (+x) - TALLER
        [sx/2.0, t/2.0, sz],                         # Back wall (+y) - TALLER (Backboard)
        [sx/2.0, t/2.0, (sz*0.3)/2.0],               # Front lip (-y) - shorter to allow objects in
    ]

    pos_list = [
        [-sx/2.0 + t/2.0, 0, sz],                    # Left
        [sx/2.0 - t/2.0, 0, sz],                     # Right
        [0, sy/2.0 - t/2.0, sz],                     # Back
        [0, -sy/2.0 + t/2.0, (sz*0.3)/2.0],          # Front lip
    ]

    col_shapes = [p.createCollisionShape(p.GEOM_BOX, halfExtents=h, physicsClientId=client_id) for h in half_extents_list]
    vis_shapes = [p.createVisualShape(p.GEOM_BOX, halfExtents=h, rgbaColor=color, physicsClientId=client_id) for h in half_extents_list]

    body_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col_base,
        baseVisualShapeIndex=vis_base,
        linkMasses=[0]*4,
        linkCollisionShapeIndices=col_shapes,
        linkVisualShapeIndices=vis_shapes,
        linkPositions=pos_list,
        linkOrientations=[[0, 0, 0, 1]]*4,
        linkInertialFramePositions=[[0, 0, 0]]*4,
        linkInertialFrameOrientations=[[0, 0, 0, 1]]*4,
        linkParentIndices=[0]*4,
        linkJointTypes=[p.JOINT_FIXED]*4,
        linkJointAxis=[[0, 0, 1]]*4,
        basePosition=position,
        physicsClientId=client_id
    )
    p.changeDynamics(body_id, -1, lateralFriction=0.8, physicsClientId=client_id)
    for i in range(4):
        p.changeDynamics(body_id, i, lateralFriction=0.8, physicsClientId=client_id)
    return body_id


def create_textured_plane_obj(filename, size_x, size_y, z_offset=0.0):
    """
    Builds a flat quad mesh with exact UV mapping (0 to 1 across the whole
    plane). This is used instead of the default GEOM_BOX because PyBullet
    tiles/crops a box's texture based on its physical size (in meters)
    rather than a normalized 0..1 mapping, which makes a real photo look
    "zoomed in" or misaligned with the part. With this custom mesh, the
    full image is placed exactly and without distortion onto the plane.
    z_offset is typically half the part's thickness, so the plane sits
    exactly on the part's top surface.
    """
    hx, hy = size_x / 2.0, size_y / 2.0
    obj_content = f"""# Auto-generated textured quad (exact 0..1 UV mapping)
v {-hx} {-hy} {z_offset}
v {hx} {-hy} {z_offset}
v {hx} {hy} {z_offset}
v {-hx} {hy} {z_offset}
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 0 1
f 1/1/1 2/2/1 3/3/1
f 1/1/1 3/3/1 4/4/1
"""
    with open(filename, "w") as f:
        f.write(obj_content)
    return filename
