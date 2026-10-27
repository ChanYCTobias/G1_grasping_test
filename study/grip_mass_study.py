#!/usr/bin/env python3
# Copyright (c) 2026. Apache License 2.0.
"""Grip torque versus mass for the G1 Dex3 in unitree_mujoco.

The object is two vertical handles on one rigid body. Fingers close to the
surface pose saved in surface_grasp_config.json and stop there. Grip G is a
cap on the finger motors, in N·m, at that fixed pose. A deeper target is not
used: driving through the mesh overlaps the posts and is not a grasp.
"""
import csv
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "unitree_robots" / "g1" / "scene_handles.xml"
OUT = ROOT / "study"

# Official close direction from the joint ranges in g1_29dof_with_hand.xml.
# Left middle/index close toward negative; right fingers are mirrored.
# Near the joint limits, so a handle that stops the finger early keeps the motor at the torque cap.
LEFT_CLOSE = {
    "left_hand_thumb_0_joint": 0.90,
    "left_hand_thumb_1_joint": 0.90,
    "left_hand_thumb_2_joint": 1.50,
    "left_hand_middle_0_joint": -1.40,
    "left_hand_middle_1_joint": -1.50,
    "left_hand_index_0_joint": -1.40,
    "left_hand_index_1_joint": -1.50,
}
RIGHT_CLOSE = {
    "right_hand_thumb_0_joint": -0.90,
    "right_hand_thumb_1_joint": -0.90,
    "right_hand_thumb_2_joint": -1.50,
    "right_hand_middle_0_joint": 1.40,
    "right_hand_middle_1_joint": 1.50,
    "right_hand_index_0_joint": 1.40,
    "right_hand_index_1_joint": 1.50,
}
# Moderate curl used only to find where the handles sit. The motors use LEFT/RIGHT_CLOSE.
PLACE_CLOSE = {
    "left_hand_thumb_0_joint": 0.6,
    "left_hand_thumb_1_joint": 0.6,
    "left_hand_thumb_2_joint": 0.8,
    "left_hand_middle_0_joint": -1.0,
    "left_hand_middle_1_joint": -1.0,
    "left_hand_index_0_joint": -1.0,
    "left_hand_index_1_joint": -1.0,
    "right_hand_thumb_0_joint": -0.6,
    "right_hand_thumb_1_joint": -0.6,
    "right_hand_thumb_2_joint": -0.8,
    "right_hand_middle_0_joint": 1.0,
    "right_hand_middle_1_joint": 1.0,
    "right_hand_index_0_joint": 1.0,
    "right_hand_index_1_joint": 1.0,
}
# Chest-height reach. Right arm mirrors roll and yaw.
LEFT_ARM = {
    "left_shoulder_pitch_joint": -0.9,
    "left_shoulder_roll_joint": 0.20,
    "left_shoulder_yaw_joint": 0.15,
    "left_elbow_joint": 1.05,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": -0.40,
    "left_wrist_yaw_joint": -0.60,
}
RIGHT_ARM = {
    "right_shoulder_pitch_joint": -0.9,
    "right_shoulder_roll_joint": -0.20,
    "right_shoulder_yaw_joint": -0.15,
    "right_elbow_joint": 1.05,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": -0.40,
    "right_wrist_yaw_joint": 0.60,
}

# Surface friction hold. Seat offset is applied on top of place_between_hands.
# The posts start just clear of the open hand. Thumbs are abducted so the thumb
# mesh is not inside the post. Index and middle stop at first contact instead
# of driving through the mesh. A deep close at the old seat (-0.03, 0, -0.02)
# left the posts 7–14 mm inside the finger meshes; that pose is not used.
POST_CENTER_Z = 0.07
SEAT_OFFSET = np.array([-0.04, 0.0, -0.02])
THUMB_ASIDE = {
    "left_hand_thumb_0_joint": 0.0,
    "left_hand_thumb_1_joint": -0.60,
    "left_hand_thumb_2_joint": 0.0,
    "right_hand_thumb_0_joint": 0.0,
    "right_hand_thumb_1_joint": 0.60,
    "right_hand_thumb_2_joint": 0.0,
}
THUMB_SCALE = 0.0
HOLD_GRIP_NM = 1.0
SPIN_DAMPING = 0.3
NOSLIP_ITERATIONS = 50
# Solver compression at the surface. Deeper than this is mesh overlap.
MAX_OVERLAP_M = 0.0015

# Fraction past the surface pose. The post blocks the finger, so the motor sits at the torque cap.
SQUEEZE_SCALE = 1.05
# Slow ladder. 5 kg is left out of the fit and predicted from the others.
STUDY_MASSES = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
HOLD_OUT_MASS = 5.0
# Finger joints cap at 1.4 N·m. Thumbs are held aside and are not the squeeze.
STUDY_GRIPS = (0.05, 0.15, 0.25, 0.40, 0.60, 0.90, 1.40)
REPEATS = 3
STATIC_S = 1.5
HOLD_DROP_M = 0.02
HOLD_TILT_DEG = 6.0
FIT_MASSES = tuple(m for m in STUDY_MASSES if m != HOLD_OUT_MASS)
HOLD_MASSES = (HOLD_OUT_MASS,)
GRIPS_NM = STUDY_GRIPS
LIFT_S = 2.5
LIFT_DPITCH = -0.08
SLIP_BEHIND = 0.03
SLIP_DIST = 0.28
SLIP_Z = 0.25
CONFIG_PATH = OUT / "surface_grasp_config.json"


def joint_id(model, name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise KeyError(name)
    return jid


def set_hinge(model, data, name, value):
    jid = joint_id(model, name)
    data.qpos[model.jnt_qposadr[jid]] = value
    data.qvel[model.jnt_dofadr[jid]] = 0.0


def hinge(model, data, name):
    jid = joint_id(model, name)
    return data.qpos[model.jnt_qposadr[jid]], data.qvel[model.jnt_dofadr[jid]]


def body_pos(model, data, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xpos[bid].copy()


def lock_pose(model, data):
    """Pose of every joint that is not a Dex3 finger, captured after the arms reach."""
    locked = {}
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        if "hand_" in name:
            continue
        locked[name] = float(data.qpos[model.jnt_qposadr[jid]])
    return locked


def lock_arms(model, data, targets):
    """Hold legs, waist, and arms. Only the Dex3 motors push."""
    pose = dict(BODY_LOCK)
    for name, value in targets.items():
        if "hand_" in name:
            continue
        pose[name] = value
    for name, value in pose.items():
        set_hinge(model, data, name, value)


def apply_pd(model, data, targets, grip_nm):
    ctrl = np.zeros(model.nu)
    for i in range(model.nu):
        jid = int(model.actuator_trnid[i, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if "hand_" not in name:
            continue
        q, qd = hinge(model, data, name)
        q_des = targets.get(name, 0.0)
        jfr = model.jnt_actfrcrange[jid]
        hard = float(jfr[1]) if jfr[1] > jfr[0] else 1.4
        hard = min(hard, float(grip_nm))
        # kp is high enough that a finger stopped by the handle saturates at the torque cap.
        tau = 20.0 * (q_des - q) - 0.2 * qd
        ctrl[i] = np.clip(tau, -hard, hard)
    data.ctrl[:] = ctrl


def set_payload(model, data, pos, mass):
    jid = joint_id(model, "payload_free")
    adr = model.jnt_qposadr[jid]
    data.qpos[adr:adr + 3] = pos
    data.qpos[adr + 3:adr + 7] = (1.0, 0.0, 0.0, 0.0)
    vadr = model.jnt_dofadr[jid]
    data.qvel[vadr:vadr + 6] = 0.0
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "payload")
    model.body_mass[bid] = mass
    # Keep a plausible inertia so a heavy object does not spin from a tiny torque.
    i = 0.02 * mass
    model.body_inertia[bid] = (i, 0.25 * i, i)
    saved_q = data.qpos.copy()
    saved_v = data.qvel.copy()
    mujoco.mj_setConst(model, data)
    # mj_setConst restores qpos0 for the whole model, including the arms.
    data.qpos[:] = saved_q
    data.qvel[:] = saved_v
    data.qpos[adr:adr + 3] = pos
    data.qpos[adr + 3:adr + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[vadr:vadr + 6] = 0.0


BODY_LOCK = {}


def stand_and_reach(model, data):
    global BODY_LOCK
    for name, value in {**LEFT_ARM, **RIGHT_ARM}.items():
        set_hinge(model, data, name, value)
    mujoco.mj_forward(model, data)
    # Freeze legs and waist at this stance. The arm targets override those joints.
    BODY_LOCK = lock_pose(model, data)


def finger_damping(model):
    """Implicit joint damping. Without it the light finger links slam through the handles."""
    for name in finger_targets():
        jid = joint_id(model, name)
        model.dof_damping[int(model.jnt_dofadr[jid])] = 0.4


def finger_targets():
    return {**LEFT_CLOSE, **RIGHT_CLOSE}


def handle_contact_dists(model, data):
    """Signed distances for handle contacts on the fingers. Wrist and palm are excluded."""
    dists = []
    for i in range(data.ncon):
        con = data.contact[i]
        names = "".join(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            for g in (con.geom1, con.geom2)
        )
        if "handle" not in names:
            continue
        bodies = " ".join(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
            for g in (con.geom1, con.geom2)
        )
        if "wrist" in bodies or "forearm" in bodies:
            continue
        dists.append(float(con.dist))
    return dists


def stiffen_grasp(model):
    """Keep a loaded contact from sinking the post into the finger mesh."""
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i])) or ""
        if name.startswith("handle") or ("hand_" in body and int(model.geom_contype[i]) > 0):
            model.geom_solref[i] = (0.002, 1.0)
            model.geom_solimp[i] = (0.99, 0.999, 0.0001, 0.5, 2.0)


def _lock_hinge_at(model, data, name, half_width):
    q, _ = hinge(model, data, name)
    jid = joint_id(model, name)
    model.jnt_limited[jid] = 1
    model.jnt_range[jid] = (q - half_width, q + half_width)


def close_to_surface(model, data, place, mass):
    """Curl the fingers onto the posts and stop at the surface.

    The object is held upright only while the fingers are closing. Callers then
    step the simulation freely. Returns the free-joint qpos address and the
    original finger ranges so the grasp can be opened again.
    """
    mujoco.mj_resetData(model, data)
    stand_and_reach(model, data)
    for name, value in THUMB_ASIDE.items():
        set_hinge(model, data, name, value)
    finger_ranges = {}
    for name in finger_targets():
        jid = joint_id(model, name)
        finger_ranges[name] = (float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1]))
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        if "hand_" in name or int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        q = float(data.qpos[model.jnt_qposadr[jid]])
        model.jnt_limited[jid] = 1
        model.jnt_range[jid] = (q - 1e-5, q + 1e-5)
    for name in THUMB_ASIDE:
        _lock_hinge_at(model, data, name, 1e-4)
    seat = np.asarray(place, float) + SEAT_OFFSET
    set_payload(model, data, seat, mass)
    mujoco.mj_forward(model, data)
    jid = joint_id(model, "payload_free")
    adr = int(model.jnt_qposadr[jid])
    vadr = int(model.jnt_dofadr[jid])
    model.opt.noslip_iterations = NOSLIP_ITERATIONS
    model.dof_damping[vadr + 3:vadr + 6] = SPIN_DAMPING
    saved = data.qpos[adr:adr + 7].copy()
    curling = {name: goal for name, goal in finger_targets().items() if "thumb" not in name}
    arm = {**LEFT_ARM, **RIGHT_ARM}
    dt = model.opt.timestep
    pose = None
    for blend in np.linspace(0.0, 1.0, 81):
        targets = {**arm, **THUMB_ASIDE, **{name: float(blend) * goal for name, goal in curling.items()}}
        for _ in range(int(0.012 / dt)):
            data.qpos[adr:adr + 7] = saved
            data.qvel[vadr:vadr + 6] = 0.0
            apply_pd(model, data, targets, HOLD_GRIP_NM)
            mujoco.mj_step(model, data)
            data.qpos[adr:adr + 7] = saved
            data.qvel[vadr:vadr + 6] = 0.0
        mujoco.mj_forward(model, data)
        dists = handle_contact_dists(model, data)
        if dists and min(dists) < 0.0002:
            pose = {name: float(hinge(model, data, name)[0]) for name in curling}
            break
    if pose is None:
        raise RuntimeError("fingers closed fully without touching the posts")
    for _ in range(40):
        dists = handle_contact_dists(model, data)
        if not dists or min(dists) >= -0.0008:
            break
        for name in list(pose):
            pose[name] *= 0.98
            set_hinge(model, data, name, pose[name])
        data.qpos[adr:adr + 7] = saved
        mujoco.mj_forward(model, data)
    dists = handle_contact_dists(model, data)
    deepest = min(dists) if dists else 1.0
    if deepest < -MAX_OVERLAP_M:
        raise RuntimeError(f"refusing the grasp: posts overlap the fingers by {-deepest * 1000:.1f} mm")
    for name, q in pose.items():
        j = joint_id(model, name)
        model.jnt_limited[j] = 1
        model.jnt_range[j] = (q - 1e-4, q + 1e-4)
        set_hinge(model, data, name, q)
    data.qpos[adr:adr + 7] = saved
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return adr, finger_ranges


def finger_stall(model, data, targets):
    shorts = []
    for name, goal in targets.items():
        q, _ = hinge(model, data, name)
        if abs(goal) < 1e-4:
            continue
        if q * goal > 0:
            shorts.append(max(0.0, abs(goal) - abs(q)))
        else:
            shorts.append(abs(goal))
    return float(np.mean(shorts)) if shorts else 0.0


def object_state(model, data):
    pos = body_pos(model, data, "payload")
    left = body_pos(model, data, "left_wrist_yaw_link")
    right = body_pos(model, data, "right_wrist_yaw_link")
    dist = max(np.linalg.norm(pos - left), np.linalg.norm(pos - right))
    return pos, left, right, dist


def slipped(z, z0, hand_z, hand0, dist, lifting):
    if z < SLIP_Z or dist > SLIP_DIST:
        return True
    if lifting and (hand_z - hand0) - (z - z0) > SLIP_BEHIND:
        return True
    if (not lifting) and (z0 - z) > SLIP_BEHIND:
        return True
    return False


def settle_fingers(model, data, grip, seconds, on_step=None):
    """Close the fingers while the object is held still, then let it go."""
    targets = finger_targets()
    n = int(seconds / model.opt.timestep)
    pin = int(0.45 / model.opt.timestep)
    jid = joint_id(model, "payload_free")
    adr = model.jnt_qposadr[jid]
    vadr = model.jnt_dofadr[jid]
    saved_q = data.qpos[adr:adr + 7].copy()
    arm = {**LEFT_ARM, **RIGHT_ARM}
    for step in range(n):
        if step < pin:
            data.qpos[adr:adr + 7] = saved_q
            data.qvel[vadr:vadr + 6] = 0.0
        lock_arms(model, data, arm)
        apply_pd(model, data, {**arm, **targets}, grip)
        mujoco.mj_step(model, data)
        lock_arms(model, data, arm)
        if step < pin:
            data.qpos[adr:adr + 7] = saved_q
            data.qvel[vadr:vadr + 6] = 0.0
        if on_step is not None:
            on_step("closing, object held in place" if step < pin else "closing, object free")
    return finger_stall(model, data, targets)


def run_phase(model, data, grip, seconds, lifting, on_step=None):
    targets = {**LEFT_ARM, **RIGHT_ARM, **finger_targets()}
    n = int(seconds / model.opt.timestep)
    mujoco.mj_forward(model, data)
    pos, left, right, dist0 = object_state(model, data)
    z0 = float(pos[2])
    hand0 = float(0.5 * (left[2] + right[2]))
    slip_t = 0.0 if slipped(z0, z0, hand0, hand0, dist0, False) else None
    z = z0
    hand_z = hand0
    dist = dist0
    dt = model.opt.timestep
    for k in range(n):
        t = (k + 1) / n
        if lifting:
            targets["left_shoulder_pitch_joint"] = LEFT_ARM["left_shoulder_pitch_joint"] + LIFT_DPITCH * t
            targets["right_shoulder_pitch_joint"] = RIGHT_ARM["right_shoulder_pitch_joint"] + LIFT_DPITCH * t
        lock_arms(model, data, targets)
        apply_pd(model, data, targets, grip)
        mujoco.mj_step(model, data)
        lock_arms(model, data, targets)
        pos, left, right, dist = object_state(model, data)
        z = float(pos[2])
        hand_z = float(0.5 * (left[2] + right[2]))
        if slip_t is None and slipped(z, z0, hand_z, hand0, dist, lifting):
            slip_t = t * seconds
        if on_step is not None:
            on_step("lift" if lifting else "static hold")
    return {
        "slip": int(slip_t is not None),
        "slip_t": -1.0 if slip_t is None else slip_t,
        "hand_dz": hand_z - hand0,
        "obj_dz": z - z0,
        "obj_z": z,
        "dist": float(dist),
        "stall": finger_stall(model, data, finger_targets()),
    }


def reset_trial(model, data, mass, grip, place):
    mujoco.mj_resetData(model, data)
    stand_and_reach(model, data)
    set_payload(model, data, place, mass)
    mujoco.mj_forward(model, data)
    # Close onto the handles before the timed phase, at this trial's torque cap.
    settle_fingers(model, data, grip, 0.6)


def post_tilt_deg(model, data):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "handle_l")
    axis = data.geom_xmat[gid].reshape(3, 3)[:, 2]
    return float(np.degrees(np.arccos(np.clip(abs(float(axis[2])), 0.0, 1.0))))


def save_surface_config(pose):
    """Write the grasp that the weight sweep is not allowed to change."""
    payload = {
        "name": "surface_friction_hold",
        "scene": "unitree_robots/g1/scene_handles.xml",
        "seat_offset_m": [float(v) for v in SEAT_OFFSET],
        "post_center_z_m": POST_CENTER_Z,
        "thumb_aside_rad": {k: float(v) for k, v in THUMB_ASIDE.items()},
        "left_arm_rad": {k: float(v) for k, v in LEFT_ARM.items()},
        "right_arm_rad": {k: float(v) for k, v in RIGHT_ARM.items()},
        "surface_pose_rad": {k: float(v) for k, v in pose.items()},
        "squeeze_scale": SQUEEZE_SCALE,
        "finger_kp": 20.0,
        "finger_kd": 0.2,
        "finger_damping": 0.4,
        "spin_damping": SPIN_DAMPING,
        "noslip_iterations": NOSLIP_ITERATIONS,
        "contact_solref": [0.002, 1.0],
        "contact_solimp": [0.99, 0.999, 0.0001, 0.5, 2.0],
        "max_overlap_m": MAX_OVERLAP_M,
        "hold_drop_m": HOLD_DROP_M,
        "hold_tilt_deg": HOLD_TILT_DEG,
        "hold_seconds": STATIC_S,
        "notes": (
            "Posts stand on a non-colliding bar. Thumbs are abducted so they are not inside the posts. "
            "Index and middle stop at first contact. Grip is a torque cap at 1.05 times that pose, "
            "not a deeper close. The rejected seat (-0.03, 0, -0.02) left the posts 7-14 mm inside the meshes."
        ),
    }
    OUT.mkdir(exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    return CONFIG_PATH


def prepare_surface(model, data, place):
    """Close once to the surface and remember that pose for every weight."""
    adr, ranges = close_to_surface(model, data, place, 1.0)
    pose = {name: float(hinge(model, data, name)[0]) for name in ranges if "thumb" not in name}
    for name, (lo, hi) in ranges.items():
        if "thumb" in name:
            continue
        jid = joint_id(model, name)
        model.jnt_limited[jid] = 1
        model.jnt_range[jid] = (lo, hi)
    targets = {**LEFT_ARM, **RIGHT_ARM, **THUMB_ASIDE}
    for name, q in pose.items():
        lo, hi = ranges[name]
        targets[name] = float(np.clip(q * SQUEEZE_SCALE, lo, hi))
    return {
        "adr": adr,
        "qpos": data.qpos.copy(),
        "seat": data.qpos[adr:adr + 3].copy(),
        "targets": targets,
        "pose": pose,
    }


def run_static(model, data, ready, mass, grip, jitter):
    """Hold the saved pose at this torque cap. The object is free the whole time."""
    data.qpos[:] = ready["qpos"]
    data.qvel[:] = 0.0
    set_payload(model, data, ready["seat"], mass)
    adr = ready["adr"]
    data.qpos[adr] += jitter
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    z0 = float(data.qpos[adr + 2])
    worst_gap = min(handle_contact_dists(model, data) or [1.0])
    worst_tilt = post_tilt_deg(model, data)
    n = int(STATIC_S / model.opt.timestep)
    for _ in range(n):
        apply_pd(model, data, ready["targets"], grip)
        mujoco.mj_step(model, data)
        if float(data.qpos[adr + 2]) > 0.75:
            worst_tilt = max(worst_tilt, post_tilt_deg(model, data))
            dists = handle_contact_dists(model, data)
            if dists:
                worst_gap = min(worst_gap, min(dists))
    drop = z0 - float(data.qpos[adr + 2])
    overlap = worst_gap < -MAX_OVERLAP_M
    held = drop <= HOLD_DROP_M and worst_tilt <= HOLD_TILT_DEG and not overlap
    return {
        "slip": 0 if held else 1,
        "drop": float(drop),
        "max_tilt": float(worst_tilt),
        "min_gap_mm": float(worst_gap * 1000.0),
        "obj_z": float(data.qpos[adr + 2]),
        "overlap": int(overlap),
    }


def place_between_hands(model, data):
    """Put the handle origin where the closed index meshes meet the cylinders."""
    mujoco.mj_resetData(model, data)
    stand_and_reach(model, data)
    for name, value in PLACE_CLOSE.items():
        set_hinge(model, data, name, value)
    mujoco.mj_forward(model, data)
    def colliding_geom(mesh_name):
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, mesh_name)
        for i in range(model.ngeom):
            if int(model.geom_dataid[i]) == mid and int(model.geom_contype[i]) > 0:
                return data.geom_xpos[i].copy()
        raise KeyError(mesh_name)

    left = colliding_geom("left_hand_index_1_link")
    right = colliding_geom("right_hand_index_1_link")
    mid = 0.5 * (left + right)
    # Body origin is the bottom of the posts. Shift down so the post centers
    # and the center of mass sit in the finger curl, not below it.
    place = np.array([float(mid[0]), 0.0, float(mid[2]) - POST_CENTER_Z])
    print(f"[mujoco] closed index mesh L {left} R {right} place {place}")
    return place


def _majority_holds(rows, mass, grip):
    cell = [r for r in rows if r["mass"] == mass and r["grip"] == grip]
    return sum(1 for r in cell if r["slip"] == 0), cell


def _g_star(rows, mass):
    """Lowest torque cap with a 2-of-3 hold. A higher cap can still tilt the posts."""
    for grip in STUDY_GRIPS:
        if _majority_holds(rows, mass, grip)[0] >= 2:
            return grip
    return None


def _video_cells(rows):
    """One slip and one hold per mass, in increasing weight."""
    cells = []
    for mass in STUDY_MASSES:
        star = _g_star(rows, mass)
        if star is None:
            cells.append((mass, 0.40))
            cells.append((mass, STUDY_GRIPS[-1]))
            continue
        below = [g for g in STUDY_GRIPS if g < star]
        if below:
            cells.append((mass, below[-1]))
        cells.append((mass, star))
    return cells


def record_weight_videos(model, data, ready, cells):
    """Side-view clips. Fingers are already on the surface when each clip starts."""
    import time
    from PIL import Image, ImageDraw
    import imageio.v2 as imageio

    width, height, fps = 960, 720, 30
    lookat = np.array([0.38, 0.0, 0.62])
    distance, azimuth, elevation = 1.25, 90.0, -8.0
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    stride = max(1, int(round((1.0 / fps) / model.opt.timestep)))
    frames = []
    viewer = None
    try:
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = lookat
        viewer.cam.distance = distance
        viewer.cam.azimuth = azimuth
        viewer.cam.elevation = elevation
    except Exception as exc:
        print(f"[mujoco] viewer unavailable ({exc}); recording offscreen")
        viewer = None

    def stamp(image, text):
        picture = Image.fromarray(image)
        draw = ImageDraw.Draw(picture)
        draw.rectangle((0, 0, width, 36), fill=(0, 0, 0))
        draw.text((12, 8), text, fill=(255, 255, 255))
        return np.asarray(picture)

    try:
        adr = ready["adr"]
        for mass, grip in cells:
            data.qpos[:] = ready["qpos"]
            data.qvel[:] = 0.0
            set_payload(model, data, ready["seat"], mass)
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            z0 = float(data.qpos[adr + 2])
            worst_gap = min(handle_contact_dists(model, data) or [1.0])
            worst_tilt = post_tilt_deg(model, data)
            clip = []
            wall = time.perf_counter()
            sim0 = float(data.time)
            n = int(STATIC_S / model.opt.timestep)
            for step in range(n):
                apply_pd(model, data, ready["targets"], grip)
                mujoco.mj_step(model, data)
                if float(data.qpos[adr + 2]) > 0.75:
                    worst_tilt = max(worst_tilt, post_tilt_deg(model, data))
                    dists = handle_contact_dists(model, data)
                    if dists:
                        worst_gap = min(worst_gap, min(dists))
                if step % stride == 0:
                    cam.lookat[:] = lookat
                    cam.distance = distance
                    cam.azimuth = azimuth
                    cam.elevation = elevation
                    renderer.update_scene(data, cam)
                    clip.append(renderer.render().copy())
                if viewer is not None and step % 20 == 0:
                    viewer.cam.lookat[:] = lookat
                    viewer.cam.distance = distance
                    viewer.cam.azimuth = azimuth
                    viewer.cam.elevation = elevation
                    viewer.sync()
                    ahead = (data.time - sim0) - (time.perf_counter() - wall)
                    if ahead > 0:
                        time.sleep(min(ahead, 0.03))
            drop = z0 - float(data.qpos[adr + 2])
            held = drop <= HOLD_DROP_M and worst_tilt <= HOLD_TILT_DEG and worst_gap >= -MAX_OVERLAP_M
            verdict = "HOLD" if held else "SLIP"
            label = (
                f"side   {mass:.0f} kg   grip {grip:.2f} Nm   {verdict}"
                f"   drop {drop * 100:.1f} cm   tilt {worst_tilt:.1f} deg"
            )
            print(f"[mujoco] video {label}")
            frames.extend(stamp(image, label) for image in clip)
    finally:
        if viewer is not None:
            viewer.close()
    out = OUT / "weight_grip.mp4"
    imageio.mimsave(out, frames, fps=fps)
    print(f"[mujoco] wrote {out} ({len(frames)} frames)")
    return out


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    finger_damping(model)
    stiffen_grasp(model)
    data = mujoco.MjData(model)
    place = place_between_hands(model, data)
    ready = prepare_surface(model, data, place)
    config = save_surface_config(ready["pose"])
    print(f"[mujoco] saved {config}")
    print(f"[mujoco] surface pose { {k: round(v, 3) for k, v in ready['pose'].items()} }")

    jitters = (0.0, 0.001, -0.001)
    rows = []
    total = len(STUDY_MASSES) * len(STUDY_GRIPS) * REPEATS
    done = 0
    for mass in STUDY_MASSES:
        for grip in STUDY_GRIPS:
            for rep, jitter in enumerate(jitters):
                result = run_static(model, data, ready, mass, grip, jitter)
                row = {"mass": mass, "grip": grip, "repeat": rep, **result}
                rows.append(row)
                done += 1
                print(
                    f"[mujoco] {done}/{total} m={mass:.0f} G={grip:.2f} rep={rep} "
                    f"slip={row['slip']} drop={row['drop']:.3f} tilt={row['max_tilt']:.1f} "
                    f"gap={row['min_gap_mm']:+.2f} mm"
                )

    OUT.mkdir(exist_ok=True)
    csv_path = OUT / "weight_grip_results.csv"
    fields = ["mass", "grip", "repeat", "slip", "drop", "max_tilt", "min_gap_mm", "obj_z", "overlap"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})

    g_of = {mass: _g_star(rows, mass) for mass in STUDY_MASSES}
    xs = [m for m in FIT_MASSES if g_of[m] is not None]
    ys = [g_of[m] for m in xs]
    lines = [
        "Surface grasp saved in surface_grasp_config.json.",
        "G1 Dex3, two vertical posts, bar does not collide. Thumbs abducted.",
        "Fingers stop at the surface. Grip G is the torque cap at 1.05 times that pose.",
        f"Hold: drop <= {HOLD_DROP_M * 100:.0f} cm, tilt <= {HOLD_TILT_DEG:.0f} deg, "
        f"overlap <= {MAX_OVERLAP_M * 1000:.1f} mm, for {STATIC_S:.1f} s.",
        "Repeats shift the object 0, +1 mm, and -1 mm along X.",
        "G* is the lowest cap with at least 2 of 3 holds.",
    ]
    for mass in STUDY_MASSES:
        for grip in STUDY_GRIPS:
            holds, cell = _majority_holds(rows, mass, grip)
            drop = sum(r["drop"] for r in cell) / len(cell)
            tilt = sum(r["max_tilt"] for r in cell) / len(cell)
            gap = min(r["min_gap_mm"] for r in cell)
            lines.append(
                f"{mass:.0f} kg at {grip:.2f} Nm: {holds}/3 held, "
                f"mean drop {drop * 100:.1f} cm, mean tilt {tilt:.1f} deg, deepest {gap:+.2f} mm"
            )
    for mass in STUDY_MASSES:
        grip = g_of[mass]
        tag = " held out of the fit" if mass == HOLD_OUT_MASS else ""
        lines.append(f"G*({mass:.0f} kg)=" + ("none" if grip is None else f"{grip:.2f} Nm") + tag)
    if len(xs) >= 2:
        alpha, beta = np.polyfit(np.array(xs, float), np.array(ys, float), 1)
        lines.append(f"fit on {', '.join(f'{m:.0f}' for m in xs)} kg: G* = {alpha:.3f} * mass_kg + {beta:.3f}")
        predicted = list(HOLD_MASSES) + [m for m in STUDY_MASSES if g_of[m] is None and m not in HOLD_MASSES]
        for mass in predicted:
            pred = alpha * mass + beta
            obs = g_of[mass]
            lines.append(
                f"predicted G*({mass:.0f} kg)={pred:.2f} Nm, observed "
                + ("none" if obs is None else f"{obs:.2f} Nm")
            )
    else:
        lines.append("Linear fit not available: fewer than two fit masses held.")
    text = "\n".join(lines) + "\n"
    report = OUT / "weight_grip_study.txt"
    report.write_text(text)
    print(text)
    print(f"wrote {csv_path}")
    print(f"wrote {report}")
    record_weight_videos(model, data, ready, _video_cells(rows))


if __name__ == "__main__":
    main()
