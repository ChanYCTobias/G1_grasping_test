# How much grip a G1 needs to carry two rods

A short simulation study. The hands close first, with gravity off. Gravity is turned on only after the fingers are shut. The arms then move the rods up, down, left, and right at several weights. The question is how hard the fingers have to push so the rods stay with the hands.

## Setup

The robot is the Unitree G1, 29 degrees of freedom, with the official Dex3 hands, in MuJoCo. The legs and pelvis do not move. The object is one rigid body: two vertical rods, radius 22 mm, 140 mm tall, 164 mm apart, with a bar between them that does not collide. The bar cannot act as a shelf. Each rod sits in one hand.

Finger motors use the Dex3 torque limits: 1.40 N·m on the finger joints and 2.45 N·m at the thumb base. During a trial the grip is a cap on those motors. A higher cap is a harder squeeze. The fingers are still commanded to the full close, so a rod that blocks a finger holds that motor at the cap.

The rods start clear of the open hands. Gravity is off and the rods are held fixed while the hands close. The rods are then released and gravity is turned on. Without friction, the same closed hands drop a 1 kg pair. With friction, that pair stays in the hands (about 2 cm of sag, about 2° of lean). The hold is friction from the fingers, not a lock and not the rods sitting inside the meshes. The deepest contact in that hold is about 2 mm.

## Motions

From that closed grasp the arms move for 1.8 s. Up and down use both shoulder-pitch joints. Left and right use both shoulder-roll joints. The hands travel about 7–11 cm, and a hold is counted only if the rods travel with them by at least 5 cm in that same direction and stay within 3.5 cm of the hand along it. A rod that does not move is not counted as held.

Before any of those moves, friction on the fingers and the rods is set to zero and the rods are released. They slip down (1 kg by 54 cm, 2 kg by 6 cm, 4 kg by 27 cm, 5 kg by 90 cm). The weld that held them still during the close is off. So the later carries are the fingers, not a stuck contact.

Four weights: 1, 2, 4, and 5 kg. Five grip caps: 0.20, 0.40, 0.70, 1.00, and 1.40 N·m. The contact number below is the mean size of the normal forces on the rods during the later part of the move. It sums every contact point, so it is a load on the grasp, not one pinch reading.

## Results

Up, left, and right are carried at the lowest cap tested, 0.20 N·m, at every weight from 1 kg to 5 kg. That cap was the bottom of the grid, so the true minimum for those directions may be lower. It does not rise with weight inside the range tested. The rods move with the hands: about 10 cm up, and about 9 cm to either side.

Down is the direction that gets harder. The rods are already being pulled out of the hands by gravity, and the arms move the same way.

| Weight | Lowest cap that still lowers the rods | Contact load at that cap |
| --- | --- | --- |
| 1 kg | 0.20 N·m | 15 N |
| 2 kg | 0.20 N·m | 30 N |
| 4 kg | 1.00 N·m | 84 N |
| 5 kg | 1.40 N·m | 125 N |

At 4 kg, 0.20, 0.40, and 0.70 N·m let the rods lag the hands on the way down. At 5 kg, every cap below 1.40 N·m does the same. 1.40 N·m is the finger limit, so a 5 kg lowering sits at the top of what these motors are allowed to command.

On the way down, the contact load at the lowest successful cap goes from 15 N at 1 kg to 30 N at 2 kg, 84 N at 4 kg, and 125 N at 5 kg. Upward moves produce a larger contact load (about 300 N already at 0.20 N·m) because many finger contacts stay pressed in while the arms lift. That load is high, but a stronger cap is not required for the rods to follow.

## Limits

Each cell was run once. The 0.20 N·m floor means the easy directions are not fully resolved. The arms are moved by sliding their joint limits, not by a free arm controller, so this is the grip needed for a slow prescribed hand motion, not for a walk or a fast swing. The rods are smooth cylinders. A different shape would change the contact. The no-friction check for 2 kg only slipped 6 cm; the other weights fell much farther.

## Conclusion

After a real close, with gravity applied only once the hands are shut, the G1 can carry 1–5 kg up, left, and right at a 0.20 N·m finger cap. The rods move with the hands, and they slip when friction is removed, so they are not stuck in the fingers. Lowering them is the costly motion. The cap has to rise from 0.20 N·m at 1–2 kg, to 1.00 N·m at 4 kg, and to the 1.40 N·m finger limit at 5 kg. The contact load on that downward move rises with the weight.

The numbers are in `move_grasp_results.csv`. The run is `move_grasp.py`, starting from the close in `physical_grasp.py`.
