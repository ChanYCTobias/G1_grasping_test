# How much grip a G1 needs to carry two rods

A short simulation study. The hands close first, with gravity off. Gravity is turned on only after the fingers are shut. The arms then move the rods up, down, left, and right at several weights. The question is how hard the fingers have to push so the rods stay with the hands.

## Setup

The robot is the Unitree G1, 29 degrees of freedom, with the official Dex3 hands, in MuJoCo. The legs and pelvis do not move. The object is one rigid body: two vertical rods, radius 22 mm, 140 mm tall, 164 mm apart, with a bar between them that does not collide. The bar cannot act as a shelf. Each rod sits in one hand.

Finger motors use the Dex3 torque limits: 1.40 N·m on the finger joints and 2.45 N·m at the thumb base. During a trial the grip is a cap on those motors. A higher cap is a harder squeeze. The fingers are still commanded to the full close, so a rod that blocks a finger holds that motor at the cap.

The rods start clear of the open hands. Gravity is off and the rods are held fixed while the hands close. The rods are then released and gravity is turned on. Without friction, the same closed hands drop a 1 kg pair. With friction, that pair stays in the hands (about 2 cm of sag, about 2° of lean). The hold is friction from the fingers, not a lock and not the rods sitting inside the meshes. The deepest contact in that hold is about 2 mm.

## Motions

From that closed grasp the arms move for 1.6 s:

- up and down, by both shoulder-pitch joints
- left and right, by both shoulder-roll joints

The hands travel about 5 cm. Four weights are tested: 1, 2, 4, and 5 kg. Four grip caps are tested: 0.40, 0.70, 1.00, and 1.40 N·m. A trial counts as held when the rods stay within 3 cm of the hand motion. The contact number reported below is the mean size of the normal forces on the rods during the later part of the move. It sums every contact point, so it is a load on the grasp, not a single pinch reading.

## Results

Up, left, and right are held at the lowest cap tested, 0.40 N·m, at every weight from 1 kg to 5 kg. That cap was the bottom of the grid, so the true minimum for those directions may be lower. It does not rise with weight inside the range tested.

Down is the direction that gets harder. The rods are already being pulled out of the hands by gravity, and the arms move the same way.

| Weight | Lowest cap that still lowers the rods | Contact load at that cap |
| --- | --- | --- |
| 1 kg | 0.40 N·m | 29 N |
| 2 kg | 0.40 N·m | 31 N |
| 4 kg | 1.00 N·m | 93 N |
| 5 kg | 1.40 N·m | 146 N |

At 4 kg, 0.40 and 0.70 N·m let the rods lag the hands. At 5 kg, every cap below 1.40 N·m does the same. 1.40 N·m is the finger limit, so a 5 kg lowering sits at the top of what these motors are allowed to command.

Raising the cap also raises the contact load. On the way down, the load at the lowest successful cap goes from about 30 N at 1–2 kg to 93 N at 4 kg and 146 N at 5 kg. Upward moves produce a large contact load (about 230 N already at 0.40 N·m) because the fingers stay pressed into the rods while the arms lift. That load is high, but a stronger cap is not required for the rods to follow.

## Limits

Each cell was run once. The 0.40 N·m floor means the easy directions are not fully resolved. The arms are moved by sliding their joint limits, not by a free arm controller, so this is the grip needed for a slow prescribed hand motion, not for a walk or a fast swing. The rods are smooth cylinders. A different shape would change the contact.

## Conclusion

After a real close, with gravity applied only once the hands are shut, the G1 can carry 1–5 kg up, left, and right at a 0.40 N·m finger cap. Lowering them is the costly motion. The cap has to rise from 0.40 N·m at 1–2 kg, to 1.00 N·m at 4 kg, and to the 1.40 N·m finger limit at 5 kg. The contact load on that downward move rises with the weight.

The numbers are in `move_grasp_results.csv`. The run is `move_grasp.py`, starting from the close in `physical_grasp.py`.
