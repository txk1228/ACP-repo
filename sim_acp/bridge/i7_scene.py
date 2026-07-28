"""生成 / 解析 i7 ACP 翻方块场景：桌 + 可翻立方体 + tip 球 + 腕部相机。

主接触：acp_tip_ball；夹爪 mesh 次要碰撞。
RGB 推理/录制：Gripper_Right 上的 acp_wrist_cam（非固定外置机位）。
生成文件写在 robot-control 的 mjcf/ 下。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

CUBE_HALF = 0.025
TABLE_POS = (0.62, -0.55, 0.0)
TABLE_TOP_CENTER_Z = 0.75
TABLE_HALF_Z = 0.03
CUBE_XY = (0.70, -0.55)
TOOL_OFFSET_LOCAL = (0.0, 0.0, 0.012)
TIP_RADIUS = 0.012
# 腕部相机（Gripper_Right 局部系）：略靠腕侧、俯视接触区
WRIST_CAM_POS = (0.018, 0.010, 0.052)
WRIST_CAM_FOVY = 55.0
# MuJoCo 离屏渲染上限（<visual><global offwidth/offheight>）
OFFSCREEN_MAX_W = 2560
OFFSCREEN_MAX_H = 1440

_EE_MESH_GEOMS = (
    "Gripper_Right_visual",
    "falan_Right_visual",
    "Wrist_Pitch_Right_visual",
)


def model_root() -> Path:
    return Path(
        os.environ.get(
            "ACP_I7_MODEL_ROOT",
            "/home/zj/robot-control-v1.5/model_new",
        )
    )


def table_top_z_nominal() -> float:
    return TABLE_TOP_CENTER_Z + TABLE_HALF_Z


def cube_spawn_xyz() -> tuple[float, float, float]:
    return (CUBE_XY[0], CUBE_XY[1], table_top_z_nominal() + CUBE_HALF + 0.001)


def wrist_cam_quat_wxyz() -> tuple[float, float, float, float]:
    """腕部相机姿态：局部 -Z 对准 tip→方块方向（home 附近）。"""
    # home 下 tip→cube 在夹爪系 ≈ [0.04, 0.32, -0.95]
    view = np.array([0.04, 0.32, -0.95], dtype=float)
    view /= np.linalg.norm(view)
    z_axis = -view  # MuJoCo 相机沿 -Z 看；+Z 轴与视线相反
    x_axis = np.cross(np.array([0.0, 0.0, 1.0]), z_axis)
    if float(np.linalg.norm(x_axis)) < 1e-6:
        x_axis = np.cross(np.array([1.0, 0.0, 0.0]), z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rot = np.column_stack([x_axis, y_axis, z_axis])
    # scipy 可选；手算 wxyz 避免额外依赖
    tr = float(np.trace(rot))
    if tr > 0.0:
        s = float(np.sqrt(tr + 1.0) * 2.0)
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(rot)))
        if i == 0:
            s = float(np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0)
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
        elif i == 1:
            s = float(np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0)
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = float(np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0)
            w = (rot[1, 0] - rot[0, 1]) / s
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    q /= np.linalg.norm(q)
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def ensure_i7_acp_scene() -> Path:
    root = model_root()
    mjcf_dir = root / "mjcf"
    urdf_mjcf = mjcf_dir / "URDF_V1.5_0416.xml"
    if not urdf_mjcf.is_file():
        raise FileNotFoundError(
            f"找不到 {urdf_mjcf}\n请设置 ACP_I7_MODEL_ROOT"
        )
    if not (root / "meshes").is_dir():
        raise FileNotFoundError(f"找不到 meshes: {root / 'meshes'}")

    out = mjcf_dir / "_acp_i7_scene.xml"
    cx, cy, cz = cube_spawn_xyz()
    tx, ty, _ = TABLE_POS
    half = CUBE_HALF
    xml = f"""<!-- ACP sim：tip 球主接触翻方块；可删 -->
<mujoco model="i7_acp_flip">
  <include file="URDF_V1.5_0416.xml"/>

  <statistic center="0.65 -0.55 0.85" extent="0.6" meansize="0.05"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.15 0.15 0.15" specular="0 0 0"/>
    <global offwidth="{OFFSCREEN_MAX_W}" offheight="{OFFSCREEN_MAX_H}" azimuth="145" elevation="-22"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2"
             width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
    <material name="grid" texture="grid" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" directional="true"/>
    <light name="acp_key" pos="0.75 -0.25 1.9" dir="-0.1 0 -1" diffuse="0.9 0.9 0.9"/>

    <body name="acp_table" pos="{tx} {ty} 0">
      <geom name="acp_table_top" type="box" size="0.28 0.20 {TABLE_HALF_Z}"
            pos="0 0 {TABLE_TOP_CENTER_Z}"
            rgba="0.55 0.40 0.25 1" contype="1" conaffinity="1" condim="3"
            friction="1.6 0.8 0.02" solref="0.008 1"/>
      <geom name="acp_table_leg" type="box" size="0.03 0.03 0.36" pos="0 0 0.36"
            rgba="0.35 0.25 0.15 1" contype="0" conaffinity="0"/>
    </body>

    <body name="acp_obj" pos="{cx} {cy} {cz}">
      <freejoint name="acp_obj_free"/>
      <inertial pos="0 0 0" mass="0.10"
               diaginertia="4.2e-5 4.2e-5 4.2e-5"/>
      <geom name="acp_obj_geom" type="box" size="{half} {half} {half}"
            rgba="0.90 0.25 0.15 1" contype="1" conaffinity="1" condim="3"
            friction="1.5 0.7 0.02" solref="0.008 1"/>
    </body>
  </worldbody>
</mujoco>
"""
    out.write_text(xml, encoding="utf-8")
    return out


def compile_i7_acp_model():
    import mujoco

    path = ensure_i7_acp_scene()
    spec = mujoco.MjSpec.from_file(str(path))
    gripper = spec.body("Gripper_Right")
    if gripper is None:
        raise RuntimeError("Gripper_Right not found in i7 scene")

    for geom in spec.geoms:
        if geom.name not in _EE_MESH_GEOMS:
            continue
        geom.contype = 1
        geom.conaffinity = 1
        geom.condim = 3
        geom.friction = [0.5, 0.25, 0.01]
        geom.solref = [0.01, 1.0]

    has_site = False
    try:
        has_site = spec.site("acp_ee_site") is not None
    except Exception:
        has_site = False
    if not has_site:
        gripper.add_site(
            name="acp_ee_site",
            pos=list(TOOL_OFFSET_LOCAL),
            size=[0.004, 0.0, 0.0],
            rgba=[0.0, 0.0, 0.0, 0.0],
        )

    has_tip = False
    try:
        has_tip = spec.geom("acp_tip_ball") is not None
    except Exception:
        has_tip = False
    if not has_tip:
        tip = gripper.add_geom(
            name="acp_tip_ball",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[TIP_RADIUS, 0.0, 0.0],
            pos=list(TOOL_OFFSET_LOCAL),
            rgba=[0.1, 0.85, 0.3, 0.8],
        )
        tip.contype = 1
        tip.conaffinity = 1
        tip.condim = 3
        tip.friction = [1.5, 0.7, 0.02]
        tip.solref = [0.006, 1.0]

    has_wcam = False
    try:
        has_wcam = spec.camera("acp_wrist_cam") is not None
    except Exception:
        has_wcam = False
    if not has_wcam:
        qw, qx, qy, qz = wrist_cam_quat_wxyz()
        wcam = gripper.add_camera(
            name="acp_wrist_cam",
            pos=list(WRIST_CAM_POS),
        )
        wcam.quat = [qw, qx, qy, qz]
        wcam.fovy = float(WRIST_CAM_FOVY)

    model = spec.compile()

    for name in _EE_MESH_GEOMS:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            continue
        model.geom_contype[gid] = 1
        model.geom_conaffinity[gid] = 1
        model.geom_condim[gid] = 3
        model.geom_friction[gid, :] = (0.5, 0.25, 0.01)
        model.geom_solref[gid, :] = (0.01, 1.0)
        mid = int(model.geom_dataid[gid])
        if int(model.mesh_graphadr[mid]) < 0:
            raise RuntimeError(f"mesh {name} 无碰撞凸包")

    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "acp_tip_ball")
    if tip_id >= 0:
        model.geom_contype[tip_id] = 1
        model.geom_conaffinity[tip_id] = 1
        model.geom_condim[tip_id] = 3
        model.geom_friction[tip_id, :] = (1.5, 0.7, 0.02)
        model.geom_solref[tip_id, :] = (0.006, 1.0)

    return model, path
