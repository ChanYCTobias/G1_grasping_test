#!/usr/bin/env python3
# Copyright (c) 2026. Apache License 2.0.
"""Full Dex3 close at the real torque limits. The object is free. Joints are not frozen."""
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

import grip_mass_study as study

OUT = Path(__file__).resolve().parents[1] / "video" / "physical_grasp.mp4"
# Flatter than the saved -0.40 rad wrist, so the thumb meets the side of the post.
WRIST_PITCH = -0.15
# Thicker than the old 16 mm posts, so the fingers stall against them.
ROD_RADIUS = 0.022
SEAT = np.array([-0.038, 0.0, 0.02])
SOLREF = (0.005, 1.0)
# Above every Dex3 joint cap, so apply_pd stops at the actuator limit.
TORQUE_CAP = 2.45


def _arms():
    arms = {**study.LEFT_ARM, **study.RIGHT_ARM}
    arms["left_wrist_pitch_joint"] = WRIST_PITCH
    arms["right_wrist_pitch_joint"] = WRIST_PITCH
    return arms


_BASE_FRICTION = None


def _prepare(model, mu=None):
    global _BASE_FRICTION
    model.opt.noslip_iterations = study.NOSLIP_ITERATIONS
    model.opt.iterations = 40
    study.finger_damping(model)
    if _BASE_FRICTION is None:
        _BASE_FRICTION = model.geom_friction.copy()
    model.geom_friction[:] = _BASE_FRICTION
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i])) or ""
        if name.startswith("handle") or ("hand_" in body and int(model.geom_contype[i]) > 0):
            model.geom_solref[i] = SOLREF
            model.geom_solimp[i] = (0.95, 0.99, 0.001, 0.5, 2.0)
            if mu is not None:
                model.geom_friction[i, 0] = mu


def _stalls(model, data, targets):
    count = 0
    peak = 0.0
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        if "hand_" not in name:
            continue
        cap = float(model.jnt_actfrcrange[int(model.actuator_trnid[i, 0]), 1])
        tau = abs(float(data.ctrl[i]))
        peak = max(peak, tau)
        goal = targets[name]
        q = study.hinge(model, data, name)[0]
        if tau > 0.85 * cap and abs(goal - q) > 0.05:
            count += 1
    return count, peak


def set_radius(model, radius=ROD_RADIUS):
    for name in ("handle_l", "handle_r"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        half = float(model.geom_size[gid, 1])
        model.geom_size[gid, 0] = radius
        model.geom_rbound[gid] = float(np.sqrt(radius * radius + half * half))


def freeze_rod(model, data, adr):
    """Hold the rod still. Gravity stays off until the hands have closed."""
    eid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "rod_freeze")
    seat = data.qpos[adr:adr + 3].copy()
    model.eq_data[eid, 3:6] = -seat
    model.eq_data[eid, 6:10] = (1.0, 0.0, 0.0, 0.0)
    data.eq_active[eid] = 1
    model.opt.gravity[:] = 0.0
    return eid


def release_rod(model, data, eid):
    data.eq_active[eid] = 0
    vadr = int(model.jnt_dofadr[study.joint_id(model, "payload_free")])
    data.qvel[vadr:vadr + 6] = 0.0
    model.opt.gravity[:] = (0.0, 0.0, -9.81)


def settle(model, data, place, mass, mu=None):
    """Open hand, free object, then command the official close at the real torque caps."""
    set_radius(model)
    _prepare(model, mu)
    mujoco.mj_resetData(model, data)
    arms = _arms()
    full = study.finger_targets()
    for name, value in arms.items():
        study.set_hinge(model, data, name, value)
    for name, value in study.THUMB_ASIDE.items():
        study.set_hinge(model, data, name, value)
    mujoco.mj_forward(model, data)
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        if "hand_" in name or int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        q = float(data.qpos[model.jnt_qposadr[jid]])
        model.jnt_limited[jid] = 1
        model.jnt_range[jid] = (q - 1e-5, q + 1e-5)
    study.set_payload(model, data, np.asarray(place, float) + SEAT, mass)
    for name, value in arms.items():
        study.set_hinge(model, data, name, value)
    for name, value in study.THUMB_ASIDE.items():
        study.set_hinge(model, data, name, value)
    for name in full:
        if "thumb" not in name:
            study.set_hinge(model, data, name, 0.0)
    mujoco.mj_forward(model, data)
    adr = int(model.jnt_qposadr[study.joint_id(model, "payload_free")])
    vadr = int(model.jnt_dofadr[study.joint_id(model, "payload_free")])
    model.dof_damping[vadr + 3:vadr + 6] = 0.1
    data.qvel[:] = 0.0
    targets = {**arms, **full}
    return adr, targets


def step_close(model, data, targets, seconds, on_step=None):
    adr = int(model.jnt_qposadr[study.joint_id(model, "payload_free")])
    z0 = float(data.qpos[adr + 2])
    worst = min(study.handle_contact_dists(model, data) or [1.0])
    n = int(seconds / model.opt.timestep)
    for _ in range(n):
        study.apply_pd(model, data, targets, TORQUE_CAP)
        mujoco.mj_step(model, data)
        if float(data.qpos[adr + 2]) > 0.55:
            dists = study.handle_contact_dists(model, data)
            if dists:
                worst = min(worst, min(dists))
        if on_step is not None:
            on_step()
    drop = z0 - float(data.qpos[adr + 2])
    stalls, peak = _stalls(model, data, targets)
    return {
        "drop": float(drop),
        "tilt": study.post_tilt_deg(model, data),
        "gap_mm": float(worst * 1000.0),
        "stalls": stalls,
        "peak_nm": peak,
        "z": float(data.qpos[adr + 2]),
    }


def record():
    import imageio.v2 as imageio

    width, height, fps = 960, 720, 30
    lookat = np.array([0.40, 0.0, 0.96])
    model = mujoco.MjModel.from_xml_path(str(study.SCENE))
    data = mujoco.MjData(model)
    place = study.place_between_hands(model, data)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    stride = max(1, int(round((1.0 / fps) / model.opt.timestep)))
    frames = []
    renderer = None

    def stamp(image, text):
        picture = Image.fromarray(image)
        draw = ImageDraw.Draw(picture)
        draw.rectangle((0, 0, width, 36), fill=(0, 0, 0))
        draw.text((12, 8), text, fill=(255, 255, 255))
        return np.asarray(picture)

    clips = (
        (1.0, None, "friction"),
        (1.0, 0.0, "no friction"),
    )
    for mass, mu, tag in clips:
        adr, targets = settle(model, data, place, mass, mu)
        if renderer is None:
            renderer = mujoco.Renderer(model, height=height, width=width)
        open_gap = min(study.handle_contact_dists(model, data) or [1.0])
        print(f"[phys] {mass:.0f} kg {tag} open gap {open_gap * 1000:+.2f} mm")
        if open_gap < -0.0005:
            raise RuntimeError("posts start inside the hand")
        eid = freeze_rod(model, data, adr)
        close_frames = []
        hold_frames = []

        def shoot(bucket):
            cam.lookat[:] = lookat
            cam.distance = 0.55
            cam.azimuth = 110.0
            cam.elevation = -18.0
            renderer.update_scene(data, cam)
            bucket.append(renderer.render().copy())

        for step in range(int(1.3 / model.opt.timestep)):
            study.apply_pd(model, data, targets, TORQUE_CAP)
            mujoco.mj_step(model, data)
            if step % stride == 0:
                shoot(close_frames)
        release_rod(model, data, eid)
        z0 = float(data.qpos[adr + 2])
        worst = min(study.handle_contact_dists(model, data) or [1.0])
        for step in range(int(2.0 / model.opt.timestep)):
            study.apply_pd(model, data, targets, TORQUE_CAP)
            mujoco.mj_step(model, data)
            if float(data.qpos[adr + 2]) > 0.5:
                dists = study.handle_contact_dists(model, data)
                if dists:
                    worst = min(worst, min(dists))
            if step % stride == 0:
                shoot(hold_frames)
        drop = z0 - float(data.qpos[adr + 2])
        tilt = study.post_tilt_deg(model, data)
        stalls, peak = _stalls(model, data, targets)
        print(
            f"[phys] {mass:.0f} kg {tag} drop {drop * 100:.1f} cm tilt {tilt:.1f} "
            f"gap {worst * 1000:+.1f} mm stall {stalls} peak {peak:.1f}"
        )
        frames.extend(stamp(image, f"{mass:.0f} kg  {tag}  no gravity  hands closing") for image in close_frames)
        frames.extend(
            stamp(
                image,
                f"{mass:.0f} kg  {tag}  gravity on  drop {drop * 100:.1f} cm  tilt {tilt:.0f} deg  gap {worst * 1000:+.1f} mm",
            )
            for image in hold_frames
        )
    OUT.parent.mkdir(exist_ok=True)
    imageio.mimsave(OUT, frames, fps=fps)
    print(f"[phys] wrote {OUT} ({len(frames)} frames)")


if __name__ == "__main__":
    record()
