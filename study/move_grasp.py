#!/usr/bin/env python3
# Copyright (c) 2026. Apache License 2.0.
"""Move a grasped rod up, down, left, and right. Grip is the finger torque cap."""
import csv
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

import grip_mass_study as study
import physical_grasp as phys

OUT = Path(__file__).resolve().parent
# Hand motion, in joint radians, from the closed pose in the first two seconds.
MOVES = {
    "up": {"left_shoulder_pitch_joint": -0.22, "right_shoulder_pitch_joint": -0.22},
    "down": {"left_shoulder_pitch_joint": 0.12, "right_shoulder_pitch_joint": 0.12},
    "left": {"left_shoulder_roll_joint": -0.16, "right_shoulder_roll_joint": -0.16},
    "right": {"left_shoulder_roll_joint": 0.16, "right_shoulder_roll_joint": 0.16},
}
MASSES = (1.0, 2.0, 4.0, 5.0)
GRIPS = (0.40, 0.70, 1.00, 1.40)
MOVE_S = 1.6
# The rod slipped if it does not travel with the hand.
FOLLOW_M = 0.03


def _arm_q(model, data):
    pose = {}
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        if "hand_" in name or int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        pose[name] = float(data.qpos[model.jnt_qposadr[jid]])
    return pose


def _set_arm(model, pose):
    for name, q in pose.items():
        jid = study.joint_id(model, name)
        model.jnt_range[jid] = (q - 2e-3, q + 2e-3)


def _normal_force(model, data):
    total = 0.0
    for k in range(data.ncon):
        con = data.contact[k]
        names = "".join(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            for g in (con.geom1, con.geom2)
        )
        if "handle" not in names:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, k, force)
        total += abs(float(force[0]))
    return total


def establish(mass):
    """The closed grasp from the first two seconds, then gravity comes on."""
    model = mujoco.MjModel.from_xml_path(str(study.SCENE))
    data = mujoco.MjData(model)
    place = study.place_between_hands(model, data)
    adr, targets = phys.settle(model, data, place, mass)
    eid = phys.freeze_rod(model, data, adr)
    dt = model.opt.timestep
    for _ in range(int(1.3 / dt)):
        study.apply_pd(model, data, targets, phys.TORQUE_CAP)
        mujoco.mj_step(model, data)
    phys.release_rod(model, data, eid)
    for _ in range(int(0.4 / dt)):
        study.apply_pd(model, data, targets, phys.TORQUE_CAP)
        mujoco.mj_step(model, data)
    return model, data, adr, targets, _arm_q(model, data)


def run_move(mass, grip, direction):
    model, data, adr, targets, base = establish(mass)
    # The close used the full motor cap. The trial uses this grip from here on.
    dt = model.opt.timestep
    hand0 = study.body_pos(model, data, "left_wrist_yaw_link")
    rod0 = data.qpos[adr:adr + 3].copy()
    n = int(MOVE_S / dt)
    worst = 0.0
    forces = []
    deltas = MOVES[direction]
    for k in range(n):
        blend = (k + 1) / n
        pose = {name: q + blend * deltas.get(name, 0.0) for name, q in base.items()}
        _set_arm(model, pose)
        study.apply_pd(model, data, targets, grip)
        mujoco.mj_step(model, data)
        hand = study.body_pos(model, data, "left_wrist_yaw_link")
        rod = data.qpos[adr:adr + 3]
        worst = max(worst, float(np.linalg.norm((rod - rod0) - (hand - hand0))))
        if k > n // 3:
            forces.append(_normal_force(model, data))
    hand = study.body_pos(model, data, "left_wrist_yaw_link") - hand0
    rod = data.qpos[adr:adr + 3] - rod0
    fell = float(data.qpos[adr + 2]) < 0.55
    held = (not fell) and worst <= FOLLOW_M
    return {
        "mass": mass,
        "grip": grip,
        "direction": direction,
        "held": int(held),
        "follow_m": worst,
        "hand_dz": float(hand[2]),
        "rod_dz": float(rod[2]),
        "hand_dy": float(hand[1]),
        "rod_dy": float(rod[1]),
        "force_n": float(np.mean(forces) if forces else 0.0),
        "tilt": study.post_tilt_deg(model, data),
        "z": float(data.qpos[adr + 2]),
    }


def minimum_grip(rows, mass, direction):
    for grip in GRIPS:
        cell = [r for r in rows if r["mass"] == mass and r["direction"] == direction and r["grip"] == grip]
        if cell and cell[0]["held"]:
            return grip
    return None


def record(rows):
    import imageio.v2 as imageio

    width, height, fps = 960, 720, 30
    lookat = np.array([0.40, 0.0, 0.96])
    # One held move per direction at 1 kg and at 4 kg, using the lowest grip that worked.
    clips = []
    for direction in MOVES:
        for mass in (1.0, 4.0):
            grip = minimum_grip(rows, mass, direction)
            if grip is None:
                grip = GRIPS[-1]
            clips.append((mass, grip, direction, minimum_grip(rows, mass, direction) is not None))
    model = mujoco.MjModel.from_xml_path(str(study.SCENE))
    data = mujoco.MjData(model)
    place = study.place_between_hands(model, data)
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    frames = []

    def stamp(image, text):
        picture = Image.fromarray(image)
        draw = ImageDraw.Draw(picture)
        draw.rectangle((0, 0, width, 40), fill=(0, 0, 0))
        draw.text((12, 10), text, fill=(255, 255, 255))
        return np.asarray(picture)

    for mass, grip, direction, ok in clips:
        adr, targets = phys.settle(model, data, place, mass)
        eid = phys.freeze_rod(model, data, adr)
        dt = model.opt.timestep
        stride = max(1, int(round((1.0 / fps) / dt)))
        # Hands close with the rod fixed and gravity off. Not shown; this is the known good start.
        for _ in range(int(1.3 / dt)):
            study.apply_pd(model, data, targets, phys.TORQUE_CAP)
            mujoco.mj_step(model, data)
        phys.release_rod(model, data, eid)
        for _ in range(int(0.3 / dt)):
            study.apply_pd(model, data, targets, grip)
            mujoco.mj_step(model, data)
        base = _arm_q(model, data)
        clip = []
        n = int(MOVE_S / dt)
        verdict = "HELD" if ok else "SLIP"
        label = f"{mass:.0f} kg   {direction}   grip {grip:.2f} Nm   {verdict}"
        for k in range(n):
            blend = (k + 1) / n
            pose = {name: q + blend * MOVES[direction].get(name, 0.0) for name, q in base.items()}
            _set_arm(model, pose)
            study.apply_pd(model, data, targets, grip)
            mujoco.mj_step(model, data)
            if k % stride == 0:
                cam.lookat[:] = lookat
                cam.distance = 0.7
                cam.azimuth = 120.0
                cam.elevation = -16.0
                renderer.update_scene(data, cam)
                clip.append(renderer.render().copy())
        print("[move] video", label, "frames", len(clip))
        frames.extend(stamp(image, label) for image in clip)
    path = OUT / "move_grasp.mp4"
    imageio.mimsave(path, frames, fps=fps)
    print(f"[move] wrote {path} ({len(frames)} frames)")


def main():
    rows = []
    total = len(MASSES) * len(GRIPS) * len(MOVES)
    done = 0
    for mass in MASSES:
        for direction in MOVES:
            for grip in GRIPS:
                row = run_move(mass, grip, direction)
                rows.append(row)
                done += 1
                print(
                    f"[move] {done}/{total} {mass:.0f} kg {direction:5} G {grip:.2f} "
                    f"{'HOLD' if row['held'] else 'slip'} follow {row['follow_m']:.3f} m "
                    f"force {row['force_n']:.0f} N hand_dz {row['hand_dz']:+.3f} rod_dz {row['rod_dz']:+.3f}"
                )
    csv_path = OUT / "move_grasp_results.csv"
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "Closed grasp from the first two seconds (22 mm rods, gravity off while the hands close).",
        "Then the hands move the rod. Grip is the finger torque cap. Force is the mean contact normal force.",
        f"Held means the rod stays within {FOLLOW_M * 100:.0f} cm of the hand motion.",
    ]
    for mass in MASSES:
        for direction in MOVES:
            grip = minimum_grip(rows, mass, direction)
            if grip is None:
                lines.append(f"{mass:.0f} kg {direction}: no grip up to {GRIPS[-1]:.2f} Nm kept the rod with the hands")
                continue
            cell = next(r for r in rows if r["mass"] == mass and r["direction"] == direction and r["grip"] == grip)
            lines.append(
                f"{mass:.0f} kg {direction}: minimum grip {grip:.2f} Nm, contact force {cell['force_n']:.0f} N"
            )
    text = "\n".join(lines) + "\n"
    report = OUT / "move_grasp_study.txt"
    report.write_text(text)
    print(text)
    record(rows)


if __name__ == "__main__":
    main()
