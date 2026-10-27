#!/usr/bin/env python3
# Copyright (c) 2026. Apache License 2.0.
"""Open the MuJoCo window and record the same trials as the grip study."""
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image, ImageDraw

import grip_mass_study as study

OUT = Path(__file__).resolve().parents[1] / "video" / "grasp_open.mp4"
FPS = 30
WIDTH = 960
HEIGHT = 720
# Side view, wide enough to see the hands and the posts falling to the floor.
LOOKAT = (0.38, 0.0, 0.62)
DISTANCE = 1.25
VIEWS = (
    ("side", 90.0, -8.0),
)


def caption(frame, text):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 36), fill=(0, 0, 0))
    draw.text((12, 8), text, fill=(255, 255, 255))
    return np.asarray(image)


class Recorder:
    def __init__(self, model, data, viewer, renderer, cam, mass, grip, view):
        self.model = model
        self.data = data
        self.viewer = viewer
        self.renderer = renderer
        self.cam = cam
        self.mass = mass
        self.grip = grip
        self.view_name, self.azimuth, self.elevation = view
        self.frames = []
        self.steps = 0
        self.stride = max(1, int(round((1.0 / FPS) / model.opt.timestep)))
        self.wall = time.perf_counter()
        self.sim0 = data.time

    def on_step(self, phase):
        self.steps += 1
        if self.steps % self.stride == 0:
            z = study.body_pos(self.model, self.data, "payload")[2]
            label = f"{self.view_name}   {self.mass:.0f} kg   grip {self.grip:.2f} Nm   z={z:.2f} m"
            self.cam.lookat[:] = LOOKAT
            self.cam.distance = DISTANCE
            self.cam.azimuth = self.azimuth
            self.cam.elevation = self.elevation
            self.renderer.update_scene(self.data, self.cam)
            self.frames.append(caption(self.renderer.render(), label))
        if self.viewer is not None and self.steps % 20 == 0:
            self.viewer.cam.lookat[:] = LOOKAT
            self.viewer.cam.distance = DISTANCE
            self.viewer.cam.azimuth = self.azimuth
            self.viewer.cam.elevation = self.elevation
            self.viewer.sync()
            ahead = (self.data.time - self.sim0) - (time.perf_counter() - self.wall)
            if ahead > 0:
                time.sleep(min(ahead, 0.03))


def _tilt(model, data):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "handle_l")
    axis = data.geom_xmat[gid].reshape(3, 3)[:, 2]
    return float(np.degrees(np.arccos(np.clip(abs(axis[2]), 0.0, 1.0))))


def play(model, data, viewer, renderer, cam, frames, mass, grip, place, view):
    """Clip starts with the fingers already on the surface, then they open."""
    adr, finger_ranges = study.close_to_surface(model, data, place, mass)
    tilt0 = _tilt(model, data)
    z0 = float(data.qpos[adr + 2])
    index0 = study.hinge(model, data, "left_hand_index_0_joint")[0]
    dists = study.handle_contact_dists(model, data)
    deepest = min(dists) if dists else 1.0
    print(
        f"[gui] start z {z0:.3f} tilt {tilt0:.2f} deg index {index0:.3f} "
        f"gap {deepest * 1000:+.2f} mm quat {np.round(data.qpos[adr+3:adr+7], 3)}"
    )
    if tilt0 > 1.0 or index0 > -0.8 or deepest < -study.MAX_OVERLAP_M:
        raise RuntimeError("refusing to record: posts are tilted, open, or inside the fingers")
    rec = Recorder(model, data, viewer, renderer, cam, mass, grip, view)
    rec.sim0 = data.time
    rec.wall = time.perf_counter()
    dt = model.opt.timestep
    worst = tilt0
    worst_gap = deepest
    for _ in range(int(1.0 / dt)):
        mujoco.mj_step(model, data)
        worst = max(worst, _tilt(model, data))
        gap = study.handle_contact_dists(model, data)
        if gap and float(data.qpos[adr + 2]) > 0.7:
            worst_gap = min(worst_gap, min(gap))
        rec.on_step("held on the surface")
    z1 = float(data.qpos[adr + 2])
    print(f"[gui] held z {z1:.3f} drop {z0 - z1:.3f} tilt {worst:.2f} deepest {worst_gap * 1000:+.2f} mm")
    if z0 - z1 > 0.02 or worst > 3.0 or worst_gap < -study.MAX_OVERLAP_M:
        raise RuntimeError("refusing to keep the video: the posts slipped, leaned, or overlapped")
    # Restore the finger ranges and drive them open. Thumbs stay aside.
    curling = {name: 0.0 for name in finger_ranges if "thumb" not in name}
    for name, (lo, hi) in finger_ranges.items():
        if "thumb" in name:
            continue
        jid = study.joint_id(model, name)
        model.jnt_limited[jid] = 1
        model.jnt_range[jid] = (lo, hi)
    arm = {**study.LEFT_ARM, **study.RIGHT_ARM}
    targets = {**arm, **study.THUMB_ASIDE, **curling}
    for _ in range(int(1.6 / dt)):
        study.apply_pd(model, data, targets, grip)
        mujoco.mj_step(model, data)
        rec.on_step("fingers opening")
    z2 = float(data.qpos[adr + 2])
    index1 = study.hinge(model, data, "left_hand_index_0_joint")[0]
    print(f"[gui] opened z {z2:.3f} drop {z0 - z2:.3f} index {index1:.3f}")
    if z0 - z2 < 0.2 or index1 < -0.4:
        raise RuntimeError("refusing to keep the video: the fingers did not let go")
    frames.extend(rec.frames)


def _frictionless_falls(place):
    """The same surface pose must drop when finger and handle friction are zero."""
    model = mujoco.MjModel.from_xml_path(str(study.SCENE))
    study.finger_damping(model)
    study.stiffen_grasp(model)
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i])) or ""
        if name.startswith("handle") or ("hand_" in body and int(model.geom_contype[i]) > 0):
            model.geom_friction[i, 0] = 0.0
    data = mujoco.MjData(model)
    adr, _ = study.close_to_surface(model, data, place, 1.0)
    z0 = float(data.qpos[adr + 2])
    for _ in range(int(1.2 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    drop = z0 - float(data.qpos[adr + 2])
    print(f"[gui] frictionless drop {drop:.3f} m")
    if drop < 0.3:
        raise RuntimeError("the posts stay up with friction off, so this is not a friction hold")


def main():
    model = mujoco.MjModel.from_xml_path(str(study.SCENE))
    study.finger_damping(model)
    study.stiffen_grasp(model)
    data = mujoco.MjData(model)
    place = study.place_between_hands(model, data)
    _frictionless_falls(place)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = LOOKAT
    cam.distance = DISTANCE
    cam.azimuth = VIEWS[0][1]
    cam.elevation = VIEWS[0][2]
    frames = []

    def run(viewer):
        if viewer is not None:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        for view in VIEWS:
            cam.azimuth = view[1]
            cam.elevation = view[2]
            if viewer is not None:
                viewer.cam.lookat[:] = cam.lookat
                viewer.cam.distance = cam.distance
                viewer.cam.azimuth = cam.azimuth
                viewer.cam.elevation = cam.elevation
            play(model, data, viewer, renderer, cam, frames, 1.0, study.HOLD_GRIP_NM, place, view)

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            run(viewer)
    except Exception as exc:
        print(f"[gui] viewer unavailable ({exc}); recording offscreen")
        run(None)

    import imageio.v2 as imageio
    imageio.mimsave(OUT, frames, fps=FPS)
    print(f"[gui] wrote {OUT} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
