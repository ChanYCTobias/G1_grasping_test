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

OUT = Path(__file__).resolve().parents[1] / "video"
# Hand motion, in joint radians, from the closed pose in the first two seconds.
MOVES = {
    "up": {"left_shoulder_pitch_joint": -0.35, "right_shoulder_pitch_joint": -0.35},
    "down": {"left_shoulder_pitch_joint": 0.20, "right_shoulder_pitch_joint": 0.20},
    "left": {"left_shoulder_roll_joint": -0.28, "right_shoulder_roll_joint": -0.28},
    "right": {"left_shoulder_roll_joint": 0.28, "right_shoulder_roll_joint": 0.28},
}
# Which way the rod must travel if the hands are actually carrying it.
AXIS = {"up": 2, "down": 2, "left": 1, "right": 1}
SIGN = {"up": 1.0, "down": -1.0, "left": -1.0, "right": 1.0}
MASSES = (1.0, 2.0, 4.0, 5.0)
GRIPS = (0.20, 0.40, 0.70, 1.00, 1.40)
MOVE_S = 1.8
# Rod and hand may differ by this much along the move and still count as carried.
FOLLOW_M = 0.035
# The hands have to actually go somewhere. A rod that never moves is stuck, not held.
MIN_TRAVEL_M = 0.05


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


def _gap(model, data):
    dists = study.handle_contact_dists(model, data)
    return min(dists) if dists else 1.0


def _weld_off(model, data):
    eid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "rod_freeze")
    return int(data.eq_active[eid]) == 0


def run_move(mass, grip, direction):
    model, data, adr, targets, base = establish(mass)
    # The close used the full motor cap. The trial uses this grip from here on.
    dt = model.opt.timestep
    hand0 = study.body_pos(model, data, "left_wrist_yaw_link")
    rod0 = data.qpos[adr:adr + 3].copy()
    n = int(MOVE_S / dt)
    axis = AXIS[direction]
    sign = SIGN[direction]
    worst_axis = 0.0
    deepest = _gap(model, data)
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
        worst_axis = max(worst_axis, abs(float((rod - rod0)[axis] - (hand - hand0)[axis])))
        if float(data.qpos[adr + 2]) > 0.5:
            deepest = min(deepest, _gap(model, data))
        if k > n // 3:
            forces.append(_normal_force(model, data))
    hand = study.body_pos(model, data, "left_wrist_yaw_link") - hand0
    rod = data.qpos[adr:adr + 3] - rod0
    hand_axis = float(hand[axis])
    rod_axis = float(rod[axis])
    traveled = abs(hand_axis) >= MIN_TRAVEL_M and abs(rod_axis) >= MIN_TRAVEL_M
    same_way = hand_axis * sign > 0 and rod_axis * sign > 0
    fell = float(data.qpos[adr + 2]) < 0.55
    # A few millimetres of contact compression is a grasp. Centimetres is the mesh swallowing the rod.
    clear = deepest >= -0.004
    held = _weld_off(model, data) and (not fell) and traveled and same_way and worst_axis <= FOLLOW_M and clear
    return {
        "mass": mass,
        "grip": grip,
        "direction": direction,
        "held": int(held),
        "follow_m": worst_axis,
        "hand_dz": float(hand[2]),
        "rod_dz": float(rod[2]),
        "hand_dy": float(hand[1]),
        "rod_dy": float(rod[1]),
        "force_n": float(np.mean(forces) if forces else 0.0),
        "gap_mm": float(deepest * 1000.0),
        "tilt": study.post_tilt_deg(model, data),
        "z": float(data.qpos[adr + 2]),
    }


def friction_required(mass):
    """Hands close on a fixed rod, then friction is removed before the rod is released."""
    model = mujoco.MjModel.from_xml_path(str(study.SCENE))
    data = mujoco.MjData(model)
    place = study.place_between_hands(model, data)
    adr, targets = phys.settle(model, data, place, mass)
    eid = phys.freeze_rod(model, data, adr)
    dt = model.opt.timestep
    for _ in range(int(1.3 / dt)):
        study.apply_pd(model, data, targets, phys.TORQUE_CAP)
        mujoco.mj_step(model, data)
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i])) or ""
        if name.startswith("handle") or ("hand_" in body and int(model.geom_contype[i]) > 0):
            model.geom_friction[i, 0] = 0.0
    for _ in range(int(0.15 / dt)):
        study.apply_pd(model, data, targets, phys.TORQUE_CAP)
        mujoco.mj_step(model, data)
    phys.release_rod(model, data, eid)
    z0 = float(data.qpos[adr + 2])
    for _ in range(int(1.5 / dt)):
        study.apply_pd(model, data, targets, phys.TORQUE_CAP)
        mujoco.mj_step(model, data)
    drop = z0 - float(data.qpos[adr + 2])
    return drop > 0.2, drop


def minimum_grip(rows, mass, direction):
    for grip in GRIPS:
        cell = [r for r in rows if r["mass"] == mass and r["direction"] == direction and r["grip"] == grip]
        if cell and cell[0]["held"]:
            return grip
    return None


def record(rows):
    import imageio.v2 as imageio

    width, height, fps = 960, 720, 30
    # Side view for up and down, front view for left and right.
    # Up looks higher so the raised rods stay in frame.
    cameras = {
        "up": (np.array([0.42, 0.0, 1.08]), 90.0, -14.0, 0.95),
        "down": (np.array([0.40, 0.0, 0.88]), 90.0, -8.0, 0.75),
        "left": (np.array([0.38, 0.0, 0.95]), 180.0, -10.0, 0.8),
        "right": (np.array([0.38, 0.0, 0.95]), 180.0, -10.0, 0.8),
    }
    # One weight at a time, then the four directions at the lowest grip that carried the rod.
    clips = []
    for mass in MASSES:
        for direction in MOVES:
            grip = minimum_grip(rows, mass, direction)
            clips.append((mass, grip if grip is not None else GRIPS[-1], direction, grip is not None))
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
        cell = next(
            (r for r in rows if r["mass"] == mass and r["direction"] == direction and r["grip"] == grip),
            None,
        )
        force = 0.0 if cell is None else cell["force_n"]
        label = f"{mass:.0f} kg   {direction}   grip {grip:.2f} Nm   contact {force:.0f} N   {verdict}"
        lookat, azimuth, elevation, distance = cameras[direction]
        for k in range(n):
            blend = (k + 1) / n
            pose = {name: q + blend * MOVES[direction].get(name, 0.0) for name, q in base.items()}
            _set_arm(model, pose)
            study.apply_pd(model, data, targets, grip)
            mujoco.mj_step(model, data)
            if k % stride == 0:
                cam.lookat[:] = lookat
                cam.distance = distance
                cam.azimuth = azimuth
                cam.elevation = elevation
                renderer.update_scene(data, cam)
                clip.append(renderer.render().copy())
        print("[move] video", label, "frames", len(clip))
        frames.extend(stamp(image, label) for image in clip)
    path = OUT / "move_grasp.mp4"
    OUT.mkdir(exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)
    print(f"[move] wrote {path} ({len(frames)} frames)")


def main():
    rows = []
    total = len(MASSES) * len(GRIPS) * len(MOVES)
    done = 0
    for mass in MASSES:
        drops = [friction_required(mass)[1] for _ in range(2)]
        drop = max(drops)
        loose = drop > 0.05
        print(f"[move] {mass:.0f} kg with friction removed drops {drop * 100:.1f} cm ({'grasp' if loose else 'STUCK'})")
        if not loose:
            raise RuntimeError(f"{mass:.0f} kg stays up with no friction, so this is not a grasp")
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
    csv_path = Path(__file__).resolve().parent / "move_grasp_results.csv"
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "Closed grasp from the first two seconds (22 mm rods, gravity off while the hands close).",
        "The rod weld is off before any move. With friction removed, each weight drops, so the rods are not stuck.",
        "Then the hands move the rod up, down, left, and right. Grip is the finger torque cap.",
        "Contact force is the mean normal force on the rods during the move.",
        f"Held means both the hands and the rods travel at least {MIN_TRAVEL_M * 100:.0f} cm the same way,",
        f"and the rod stays within {FOLLOW_M * 100:.0f} cm of the hand along that direction.",
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
    report = Path(__file__).resolve().parent / "move_grasp_study.txt"
    report.write_text(text)
    print(text)
    record(rows)


if __name__ == "__main__":
    main()
