# renee_trajectory_generation: documentation

Missions for the RB-Vogui+ with a UR5e working around the Campetella:
scanning, cleaning and screw detection. Nav2 and MoveIt execute the large
motions, and a whole-body HQP controller refines poses and runs the passes
along lines. Everything runs in Gazebo today and is designed so the same
node and YAML run on the real robot.

The package [README](../README.md) covers building, the dependencies, and
the experiment YAML format. These pages describe how the system works, how
to run it, and where each part stands.

| Page | Contents |
|---|---|
| [architecture.md](architecture.md) | Who moves the robot (Nav2, MoveIt, HQP), the modules, the ROS interface, sim vs real |
| [navigation.md](navigation.md) | Boat-mode loop, counterclockwise-only constraint, base stop placement, shifts, overshoot, localization safeguards |
| [running_in_sim.md](running_in_sim.md) | Starting the sim, running an experiment, the video, monitoring, the outputs, fixing localization |
| [missions/cleaning.md](missions/cleaning.md) | The two cleaning passes along the transversal rail (45° out, perpendicular back in reverse) |
| [missions/screw_detection.md](missions/screw_detection.md) | Camera pass straight above the row of holes, with RGB-D captures |
| [run_log.md](run_log.md) | Dated results of the sim runs and what each one changed |

## Current status (2026-09-28)

| Experiment | Status in sim | Last result |
|---|---|---|
| `cleaning_sim` | Works | Both passes complete with the arm fixed: 45° 2.20/2.20 m, perpendicular in reverse 2.35/2.35 m |
| `screw_detection_sim` | Works | 0.87/0.87 m, 10/10 captures, cross-track 0.3 cm |
| `defect_detection_sim` | Not run yet | Simplified (plans/defect_detection_simple.md): Nav2 for the base, MoveIt for the arm, loop generated from the boxes; state machine checked with a simulated `io` only, IK of every target still to check in sim |

Real robot: not tested yet. The YAMLs contain the real-robot node parameters
as comments (see [architecture.md](architecture.md#sim-vs-real-robot)).
