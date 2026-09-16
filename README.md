# renee_trajectory_generation

ROS 2 package containing the Python HQP optimizer used to generate offline
inspection trajectories. It searches stationary mobile-base poses and optimizes
six-axis arm trajectories without commanding the robot or the navigation stack.

## Components

| Module | Responsibility |
| --- | --- |
| `scan_geometry.py` | Loads URDF/Xacro structure geometry and samples real surfaces |
| `station_planner.py` | Checks the base footprint, selects stations and builds scan tracts |
| `robot_model.py` | Provides Pinocchio kinematics and Coal collision distances |
| `hqp_optimizer.py` | Solves the lexicographic QP cascade and verifies nonlinear steps |
| `constraints.py` | Defines depth, pointing, joint-limit and collision constraints |
| `scan_optimizer.py` | Validates YAML input, runs the planner and prints JSON output |

`config/hqp_scan.yaml` is the documented configuration example. Its map and
structure pose must represent the same scene. Relative paths are resolved from
the YAML location, while `package://` paths require a sourced ROS workspace.

## Dependencies

Build and run the package inside the ROS Jazzy container. Install the numerical
dependencies in a virtual environment with access to the system ROS packages:

```bash
source /opt/ros/jazzy/setup.bash
python3 -m venv --system-site-packages /tmp/renee-hqp-venv
source /tmp/renee-hqp-venv/bin/activate
python -m pip install -r src/renee_simulation/renee_trajectory_generation/requirements-hqp.txt
```

The version bounds in `requirements-hqp.txt` keep Pinocchio, Coal, urdfdom and
TinyXML binary-compatible with the Python 3.12 environment used by Jazzy.

## Build

From the workspace root:

```bash
source /opt/ros/jazzy/setup.bash
source /tmp/renee-hqp-venv/bin/activate
colcon build --packages-select renee_trajectory_generation --cmake-args -DBUILD_TESTING=OFF
source install/setup.bash
```

## Usage

Check the map and robot/structure models without running the station search:

```bash
ros2 run renee_trajectory_generation hqp_scan_optimizer.py \
  --config src/renee_simulation/renee_trajectory_generation/config/hqp_scan.yaml \
  --check-inputs
```

Run the complete optimizer:

```bash
ros2 run renee_trajectory_generation hqp_scan_optimizer.py \
  --config src/renee_simulation/renee_trajectory_generation/config/hqp_scan.yaml
```

The same pipeline can run directly as a Python module:

```bash
python -m renee_trajectory_generation.scan_optimizer \
  --config /absolute/path/to/hqp_scan.yaml
```

An optional one-shot ROS node reads the configuration from a parameter:

```bash
ros2 run renee_trajectory_generation hqp_scan_optimizer.py --ros \
  --ros-args -p config_file:=/absolute/path/to/hqp_scan.yaml
```

The command prints JSON. Exit code `0` means complete coverage or successful
input validation, `2` means partial coverage, and `1` means invalid input or a
dependency/solver error. A partial result must not be treated as an executable
plan for the complete structure.

## HQP behavior

The optimizer changes joint positions rather than editing camera poses directly.
At every linearization it solves the configured least-squares tasks in priority
order and locks the achieved result of each higher-priority level. Trust-region
steps and backtracking are checked against the original nonlinear constraints.

`constraints.build_constraints(...)` enforces camera depth and pointing, URDF
joint bounds and signed collision clearance. `hqp_optimizer.solve_hqp(...)`
provides the generic matrix interface, using tasks `(A, b)` and constraints
`lower <= C @ x <= upper`.

Collision checking includes URDF geometry, SRDF exclusions, the floor, structure
geometry and occupied or unknown map cells extruded as static columns. Segment
validation is discrete at `solver.validation_step_rad`; it is not a continuous
collision certificate.

The package performs offline planning only. It does not generate timing, command
motion, plan navigation transfers, handle dynamic obstacles or consume live TF.
